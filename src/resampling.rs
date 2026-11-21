use std::error::Error;

use rubato::{
    Resampler as _, SincFixedOut, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use crate::buffer::{AudioBuffer, AudioBufferOptions, ChannelData};
use crate::AudioBufferIter;

/// Sinc filter length (in input frames per stage) for streaming sample-rate
/// conversion.
///
/// For upsampling and downsampling up to 2x (`ratio >= 0.5`), a single 64-tap
/// squared Blackman-Harris (`BlackmanHarris2`) windowed sinc stage preserves
/// phase continuity across chunk boundaries while keeping lookahead latency
/// low (32 input frames). For steeper downsampling (`ratio < 0.5`), exact 2:1
/// pre-decimation stages are chained ahead of the final stage so every stage
/// operates at `ratio >= 0.5` across the supported sample-rate range
/// (`3 kHz` to `768 kHz`).
const SINC_LEN: usize = 64;
/// Half-filter causal startup delay (in input frames per stage). Pre-seeding
/// each stage with `LOOKAHEAD_FRAMES` leading zeros satisfies `SincFixedOut`'s
/// initial lookahead without polling extra upstream packets on quantum 0
/// (preventing spurious silence insertion from non-blocking live sources).
const LOOKAHEAD_FRAMES: usize = SINC_LEN / 2;
const SINC_CUTOFF: f32 = 0.95;
const SINC_OVERSAMPLING_FACTOR: usize = 128;
/// Exact 2:1 pre-decimation steps by integer stride `2.0`, so nearest-phase
/// lookup with a minimal table has zero fractional-phase interpolation error.
const DECIMATION_OVERSAMPLING_FACTOR: usize = 2;

/// Maximum consecutive zero-length input chunks skipped within one
/// `Resampler::next` call before returning buffered output (padded) or silence.
const MAX_CONSECUTIVE_EMPTY_CHUNKS: usize = 16;

/// Threshold above which consumed prefix samples in `SincStage::input_fifo` are
/// compacted once at least half of the vector has been consumed.
const FIFO_COMPACT_THRESHOLD: usize = 4096;

/// One windowed-sinc conversion stage operating at `ratio >= 0.5`.
struct SincStage {
    ratio: f64,
    resampler: SincFixedOut<f32>,
    input_fifo: Vec<Vec<f32>>,
    input_read_pos: usize,
    scratch_in: Vec<Vec<f32>>,
    scratch_out: Vec<Vec<f32>>,
    input_frames_pushed: u64,
    output_frames_emitted: u64,
}

impl SincStage {
    fn new(
        ratio: f64,
        sample_len: usize,
        channels: usize,
        interpolation: SincInterpolationType,
        oversampling_factor: usize,
    ) -> Self {
        assert!(channels > 0, "SincStage requires at least one channel");
        let params = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            f_cutoff: SINC_CUTOFF,
            interpolation,
            oversampling_factor,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedOut::<f32>::new(ratio, 1.0, params, sample_len, channels)
            .expect("valid sample rates, chunk size, and channel count");

        let scratch_in = resampler.input_buffer_allocate(true);
        let scratch_out = resampler.output_buffer_allocate(true);

        Self {
            ratio,
            resampler,
            input_fifo: vec![vec![0.0; LOOKAHEAD_FRAMES]; channels],
            input_read_pos: 0,
            scratch_in,
            scratch_out,
            input_frames_pushed: LOOKAHEAD_FRAMES as u64,
            output_frames_emitted: 0,
        }
    }

    fn available_input(&self) -> usize {
        self.input_fifo[0].len() - self.input_read_pos
    }

    fn has_ready_input(&self) -> bool {
        self.available_input() >= self.resampler.input_frames_next()
    }

    fn expected_output(&self) -> u64 {
        (self.input_frames_pushed as f64 * self.ratio).ceil() as u64
    }

    fn is_flushed(&self) -> bool {
        self.output_frames_emitted >= self.expected_output()
    }

    fn push_audio_buffer(&mut self, buffer: &AudioBuffer) {
        let frames = buffer.length();
        if frames == 0 {
            return;
        }
        debug_assert_eq!(buffer.number_of_channels(), self.input_fifo.len());
        for (dst, src) in self.input_fifo.iter_mut().zip(buffer.channels()) {
            dst.extend_from_slice(src.as_slice());
        }
        self.input_frames_pushed += frames as u64;
    }

    fn push_slices(&mut self, channels: &[Vec<f32>], frames: usize) {
        if frames == 0 {
            return;
        }
        debug_assert_eq!(channels.len(), self.input_fifo.len());
        for (dst, src) in self.input_fifo.iter_mut().zip(channels.iter()) {
            dst.extend_from_slice(&src[..frames]);
        }
        self.input_frames_pushed += frames as u64;
    }

    fn consume_into_scratch(&mut self, count: usize) {
        let start = self.input_read_pos;
        let end = start + count;
        for (scratch, fifo) in self.scratch_in.iter_mut().zip(self.input_fifo.iter()) {
            scratch[..count].copy_from_slice(&fifo[start..end]);
        }
        self.input_read_pos = end;

        let total_len = self.input_fifo[0].len();
        if self.input_read_pos == total_len {
            for fifo in &mut self.input_fifo {
                fifo.clear();
            }
            self.input_read_pos = 0;
        } else if self.input_read_pos >= FIFO_COMPACT_THRESHOLD
            && self.input_read_pos * 2 >= total_len
        {
            for fifo in &mut self.input_fifo {
                fifo.drain(..self.input_read_pos);
            }
            self.input_read_pos = 0;
        }
    }

    fn process_one_ready(&mut self) -> usize {
        let needed = self.resampler.input_frames_next();
        debug_assert!(self.available_input() >= needed);

        for scratch in &mut self.scratch_in {
            scratch.resize(needed, 0.);
        }
        self.consume_into_scratch(needed);

        let (_, out_len) = self
            .resampler
            .process_into_buffer(&self.scratch_in, &mut self.scratch_out, None)
            .expect("pre-allocated resampler buffers match frame requirements");
        self.output_frames_emitted += out_len as u64;
        out_len
    }

    fn flush_one_step(&mut self, max_frames: Option<usize>) -> usize {
        let needed = self.resampler.input_frames_next();
        let avail = self.available_input().min(needed);
        for scratch in &mut self.scratch_in {
            scratch.resize(needed, 0.);
        }
        if avail > 0 {
            self.consume_into_scratch(avail);
        }
        if avail < needed {
            for scratch in &mut self.scratch_in {
                scratch[avail..needed].fill(0.);
            }
        }

        let (_, out_len) = self
            .resampler
            .process_into_buffer(&self.scratch_in, &mut self.scratch_out, None)
            .expect("pre-allocated resampler buffers match frame requirements");

        let take = max_frames.map_or(out_len, |limit| out_len.min(limit));
        self.output_frames_emitted += take as u64;
        take
    }
}

/// Stateful streaming sample-rate converter for a contiguous segment of uniform
/// `(source_sample_rate, channels)`.
struct StreamConverter {
    source_sample_rate: f32,
    target_sample_rate: f32,
    channels: usize,
    stages: Vec<SincStage>,
    input_frames_pushed: u64,
    output_frames_emitted: u64,
}

impl StreamConverter {
    fn new(
        source_sample_rate: f32,
        target_sample_rate: f32,
        sample_len: usize,
        channels: usize,
    ) -> Self {
        let mut remaining_ratio = f64::from(target_sample_rate) / f64::from(source_sample_rate);
        let mut stages = Vec::new();
        while remaining_ratio < 0.5 {
            // Size each 2:1 pre-decimation stage so one step consumes at most
            // one `sample_len` source packet, preventing cascaded startup
            // over-polling on live non-blocking streams.
            let stage_sample_len = (sample_len >> (stages.len() + 1)).max(1);
            stages.push(SincStage::new(
                0.5,
                stage_sample_len,
                channels,
                SincInterpolationType::Nearest,
                DECIMATION_OVERSAMPLING_FACTOR,
            ));
            remaining_ratio *= 2.0;
        }
        stages.push(SincStage::new(
            remaining_ratio,
            sample_len,
            channels,
            SincInterpolationType::Linear,
            SINC_OVERSAMPLING_FACTOR,
        ));

        Self {
            source_sample_rate,
            target_sample_rate,
            channels,
            stages,
            input_frames_pushed: 0,
            output_frames_emitted: 0,
        }
    }

    fn matches(&self, sample_rate: f32, channels: usize) -> bool {
        self.channels == channels && self.source_sample_rate.to_bits() == sample_rate.to_bits()
    }

    fn push_input(&mut self, buffer: &AudioBuffer) {
        let frames = buffer.length();
        if frames == 0 {
            return;
        }
        self.stages[0].push_audio_buffer(buffer);
        self.input_frames_pushed += frames as u64;
    }

    fn expected_total_output(&self) -> u64 {
        if self.stages.len() == 1 {
            (((self.input_frames_pushed + LOOKAHEAD_FRAMES as u64) as f64
                * f64::from(self.target_sample_rate))
                / f64::from(self.source_sample_rate))
            .ceil() as u64
        } else {
            let mut frames = self.input_frames_pushed as f64;
            for stage in &self.stages {
                frames = ((frames + LOOKAHEAD_FRAMES as f64) * stage.ratio).ceil();
            }
            frames as u64
        }
    }

    fn is_flushed(&self) -> bool {
        self.output_frames_emitted >= self.expected_total_output()
    }

    fn stage_exhausted(&self, idx: usize) -> bool {
        (0..=idx).all(|i| self.stages[i].is_flushed())
    }

    fn feed_stage(&mut self, idx: usize) -> bool {
        let needed = self.stages[idx].resampler.input_frames_next();
        if self.stages[idx].available_input() >= needed {
            return true;
        }
        if idx == 0 {
            return false;
        }
        while self.stages[idx].available_input() < needed {
            if !self.feed_stage(idx - 1) {
                return false;
            }
            let out_len = self.stages[idx - 1].process_one_ready();
            let (left, right) = self.stages.split_at_mut(idx);
            right[0].push_slices(&left[idx - 1].scratch_out, out_len);
        }
        true
    }

    fn feed_stage_flushing(&mut self, idx: usize) {
        if idx == 0 {
            return;
        }
        let needed = self.stages[idx].resampler.input_frames_next();
        while self.stages[idx].available_input() < needed && !self.stage_exhausted(idx - 1) {
            self.feed_stage_flushing(idx - 1);
            if self.stages[idx - 1].is_flushed() {
                break;
            }
            let remaining = (self.stages[idx - 1].expected_output()
                - self.stages[idx - 1].output_frames_emitted) as usize;
            let out_len = if self.stages[idx - 1].has_ready_input() {
                let produced = self.stages[idx - 1].process_one_ready();
                let expected = self.stages[idx - 1].expected_output();
                if self.stages[idx - 1].output_frames_emitted > expected {
                    let excess = (self.stages[idx - 1].output_frames_emitted - expected) as usize;
                    self.stages[idx - 1].output_frames_emitted = expected;
                    produced.saturating_sub(excess)
                } else {
                    produced
                }
            } else {
                self.stages[idx - 1].flush_one_step(Some(remaining))
            };
            if out_len > 0 {
                let (left, right) = self.stages.split_at_mut(idx);
                right[0].push_slices(&left[idx - 1].scratch_out, out_len);
            }
        }
    }

    fn process_one_step(&mut self, output: &mut Option<AudioBuffer>) -> bool {
        let last = self.stages.len() - 1;
        if !self.feed_stage(last) {
            return false;
        }
        let out_len = self.stages[last].process_one_ready();
        self.append_final_output(out_len, None, output);
        true
    }

    fn flush_one_step(&mut self, output: &mut Option<AudioBuffer>) -> bool {
        let target_total = self.expected_total_output();
        if self.output_frames_emitted >= target_total {
            return true;
        }
        let last = self.stages.len() - 1;
        self.feed_stage_flushing(last);
        let remaining = (target_total - self.output_frames_emitted) as usize;
        let out_len = if self.stages[last].has_ready_input() {
            self.stages[last].process_one_ready()
        } else {
            self.stages[last].flush_one_step(Some(remaining))
        };
        self.append_final_output(out_len, Some(remaining), output);
        self.is_flushed()
    }

    fn append_final_output(
        &mut self,
        out_len: usize,
        max_frames: Option<usize>,
        output: &mut Option<AudioBuffer>,
    ) {
        let take = max_frames.map_or(out_len, |limit| out_len.min(limit));
        if take == 0 {
            return;
        }

        let last = self.stages.len() - 1;
        let channels: Vec<_> = self.stages[last]
            .scratch_out
            .iter()
            .map(|ch| ChannelData::from(ch[..take].to_vec()))
            .collect();
        let produced = AudioBuffer::from_channels(channels, self.target_sample_rate);
        self.output_frames_emitted += take as u64;

        match output {
            Some(buf) => buf.extend(&produced),
            None => *output = Some(produced),
        }
    }
}

/// Sample rate converter and buffer chunk splitter.
///
/// A stream can be wrapped inside a `Resampler` to yield `AudioBuffer`s
/// of the desired `sample_rate` and `sample_len`.
///
/// When the input sample rate differs from the target `sample_rate`,
/// `Resampler` maintains a stateful windowed-sinc converter across input chunks
/// so chunk boundaries remain continuous.
pub(crate) struct Resampler<I> {
    /// desired sample rate
    sample_rate: f32,
    /// desired sample length
    sample_len: usize,
    /// input stream
    input: I,
    /// buffered output at `self.sample_rate` waiting to be sliced into `sample_len` chunks
    buffer: Option<AudioBuffer>,
    /// active stateful converter when resampling a non-matching input sample rate
    converter: Option<StreamConverter>,
    /// chunk held while draining a previous converter or channel layout
    pending_input: Option<AudioBuffer>,
    /// whether `self.input` has yielded `None`
    input_ended: bool,
    /// last observed non-zero channel count (used for bounded empty-chunk fallback)
    last_channels: usize,
}

impl<M: AudioBufferIter> Resampler<M> {
    pub fn new(sample_rate: f32, sample_len: usize, input: M) -> Self {
        assert!(sample_len > 0, "Resampler sample_len must be non-zero");
        Self {
            sample_rate,
            sample_len,
            input,
            buffer: None,
            converter: None,
            pending_input: None,
            input_ended: false,
            last_channels: 1,
        }
    }

    fn needs_resampling(&self, source_sample_rate: f32) -> bool {
        !float_eq::float_eq!(source_sample_rate, self.sample_rate, abs <= 0.1)
    }

    fn take_full_chunk(&mut self) -> Option<AudioBuffer> {
        let len = self.buffer.as_ref().map_or(0, AudioBuffer::length);
        if len < self.sample_len {
            return None;
        }
        let mut buf = self.buffer.take()?;
        if buf.length() > self.sample_len {
            self.buffer = Some(buf.split_off(self.sample_len));
        }
        Some(buf)
    }

    fn take_padded_remainder(&mut self) -> Option<AudioBuffer> {
        let mut buf = self.buffer.take()?;
        if buf.length() == 0 {
            return None;
        }
        if buf.length() < self.sample_len {
            let padding = AudioBuffer::new(AudioBufferOptions {
                number_of_channels: buf.number_of_channels(),
                length: self.sample_len - buf.length(),
                sample_rate: self.sample_rate,
            });
            buf.extend(&padding);
        } else if buf.length() > self.sample_len {
            self.buffer = Some(buf.split_off(self.sample_len));
        }
        Some(buf)
    }

    fn ingest_chunk(&mut self, mut chunk: AudioBuffer) {
        let in_channels = chunk.number_of_channels();
        if in_channels > 0 {
            self.last_channels = in_channels;
        }
        if chunk.length() == 0 {
            return;
        }

        let in_sr = chunk.sample_rate();
        if !self.needs_resampling(in_sr) {
            if self.converter.is_some()
                || self
                    .buffer
                    .as_ref()
                    .is_some_and(|b| b.number_of_channels() != in_channels)
            {
                self.pending_input = Some(chunk);
                return;
            }
            chunk.sample_rate = self.sample_rate;
            match &mut self.buffer {
                Some(buf) => buf.extend(&chunk),
                None => self.buffer = Some(chunk),
            }
            return;
        }

        if self
            .converter
            .as_ref()
            .is_some_and(|c| !c.matches(in_sr, in_channels))
            || self
                .buffer
                .as_ref()
                .is_some_and(|b| b.number_of_channels() != in_channels)
        {
            self.pending_input = Some(chunk);
            return;
        }

        let conv = self.converter.get_or_insert_with(|| {
            StreamConverter::new(in_sr, self.sample_rate, self.sample_len, in_channels)
        });
        conv.push_input(&chunk);
    }
}

impl<M: AudioBufferIter> Iterator for Resampler<M> {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut consecutive_empty = 0usize;

        loop {
            if let Some(chunk) = self.take_full_chunk() {
                return Some(Ok(chunk));
            }

            if (self.input_ended || self.pending_input.is_some()) && self.converter.is_some() {
                if let Some(ref mut conv) = self.converter {
                    if conv.flush_one_step(&mut self.buffer) {
                        self.converter = None;
                    }
                }
                continue;
            }

            if let Some(ref pending) = self.pending_input {
                if self
                    .buffer
                    .as_ref()
                    .is_some_and(|b| b.number_of_channels() != pending.number_of_channels())
                {
                    if let Some(buf) = self.take_padded_remainder() {
                        return Some(Ok(buf));
                    }
                }
                let chunk = self.pending_input.take().unwrap();
                self.ingest_chunk(chunk);
                continue;
            }

            if self.input_ended {
                if let Some(buf) = self.take_padded_remainder() {
                    return Some(Ok(buf));
                }
                // Allow restartable streams (such as a retained MediaElement
                // after a seek) to resume on subsequent polls.
                self.input_ended = false;
                return None;
            }

            if let Some(ref mut conv) = self.converter {
                if conv.process_one_step(&mut self.buffer) {
                    continue;
                }
            }

            match self.input.next() {
                None => self.input_ended = true,
                Some(Err(e)) => {
                    self.buffer = None;
                    self.converter = None;
                    self.pending_input = None;
                    return Some(Err(e));
                }
                Some(Ok(chunk)) => {
                    if chunk.number_of_channels() > 0 {
                        self.last_channels = chunk.number_of_channels();
                    }
                    if chunk.length() == 0 {
                        consecutive_empty += 1;
                        if consecutive_empty >= MAX_CONSECUTIVE_EMPTY_CHUNKS {
                            let out = self.take_padded_remainder().unwrap_or_else(|| {
                                AudioBuffer::new(AudioBufferOptions {
                                    number_of_channels: self.last_channels,
                                    length: self.sample_len,
                                    sample_rate: self.sample_rate,
                                })
                            });
                            return Some(Ok(out));
                        }
                        continue;
                    }
                    consecutive_empty = 0;
                    self.ingest_chunk(chunk);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use float_eq::assert_float_eq;

    use super::*;
    use crate::buffer::{AudioBuffer, ChannelData};
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::media_streams::{MediaStream, MediaStreamTrack};
    use crate::node::{
        AudioNode, MediaStreamAudioSourceNode, MediaStreamAudioSourceOptions,
        MediaStreamTrackAudioSourceNode, MediaStreamTrackAudioSourceOptions,
    };
    use crate::RENDER_QUANTUM_SIZE;

    fn sine_chunks(
        sample_rate: f32,
        freq_hz: f64,
        amplitude: f64,
        chunk_sizes: &[usize],
        total_frames: usize,
    ) -> Vec<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> {
        let mut chunks = Vec::new();
        let mut frame = 0usize;
        let mut idx = 0usize;
        while frame < total_frames {
            let len = chunk_sizes[idx % chunk_sizes.len()].min(total_frames - frame);
            idx += 1;
            let mut channel = Vec::with_capacity(len);
            for _ in 0..len {
                let t = frame as f64 / f64::from(sample_rate);
                channel.push(((2.0 * PI * freq_hz * t).sin() * amplitude) as f32);
                frame += 1;
            }
            chunks.push(Ok(AudioBuffer::from(vec![channel], sample_rate)));
        }
        chunks
    }

    /// Return `(max_adjacent_diff, fitted_amplitude, sinad_db)` on `samples`.
    fn measure_tone(samples: &[f32], sample_rate: f32, freq_hz: f64) -> (f32, f64, f64) {
        let max_diff = samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);

        let n = samples.len() as f64;
        let omega = 2.0 * PI * freq_hz / f64::from(sample_rate);
        let (mut sum_cos, mut sum_sin) = (0.0_f64, 0.0_f64);
        for (i, &s) in samples.iter().enumerate() {
            let phase = omega * i as f64;
            sum_cos += f64::from(s) * phase.cos();
            sum_sin += f64::from(s) * phase.sin();
        }
        let a = 2.0 * sum_cos / n;
        let b = 2.0 * sum_sin / n;
        let amp = a.hypot(b);

        let (mut sig_pow, mut err_pow) = (0.0_f64, 0.0_f64);
        for (i, &s) in samples.iter().enumerate() {
            let phase = omega * i as f64;
            let fitted = a * phase.cos() + b * phase.sin();
            sig_pow += fitted * fitted;
            let err = f64::from(s) - fitted;
            err_pow += err * err;
        }
        let sinad_db = 10.0 * (sig_pow / err_pow.max(1e-30)).log10();
        (max_diff, amp, sinad_db)
    }

    #[test]
    fn test_resampler_concat() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let input_buf = AudioBuffer::from_channels(vec![channel], 44_100.);
        let input = vec![input_buf; 3].into_iter().map(Ok);
        let mut resampler = Resampler::new(44_100., 10, input);

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 10);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[1., 2., 3., 4., 5., 1., 2., 3., 4., 5.,][..],
            abs_all <= 0.
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 10);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[1., 2., 3., 4., 5., 0., 0., 0., 0., 0.][..],
            abs_all <= 0.
        );

        assert!(resampler.next().is_none());
    }

    #[test]
    fn test_resampler_split() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let input_buf = Ok(AudioBuffer::from_channels(vec![channel], 44_100.));
        let input = vec![input_buf].into_iter();
        let mut resampler = Resampler::new(44_100., 5, input);

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 5);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[1., 2., 3., 4., 5.][..],
            abs_all <= 0.
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 5);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[6., 7., 8., 9., 10.][..],
            abs_all <= 0.
        );

        assert!(resampler.next().is_none());
    }

    #[test]
    fn test_resampler_continuous_across_chunks() {
        let freq_hz = 220.0;
        let amplitude = 0.15;

        for &(in_sr, out_sr) in &[
            (8_000.0_f32, 48_000.0_f32),
            (8_000.0_f32, 44_100.0_f32),
            (16_000.0_f32, 48_000.0_f32),
            (44_100.0_f32, 48_000.0_f32),
            (96_000.0_f32, 48_000.0_f32),
        ] {
            let chunks = sine_chunks(in_sr, freq_hz, amplitude, &[128], in_sr as usize);
            let resampler = Resampler::new(out_sr, RENDER_QUANTUM_SIZE, chunks.into_iter());
            let mut out = Vec::new();
            for buf in resampler {
                let buf = buf.unwrap();
                assert_float_eq!(buf.sample_rate(), out_sr, abs <= 0.);
                assert_eq!(buf.length(), RENDER_QUANTUM_SIZE);
                out.extend_from_slice(buf.get_channel_data(0));
            }

            let expected_valid = (((in_sr as usize + LOOKAHEAD_FRAMES) as f64 * f64::from(out_sr))
                / f64::from(in_sr))
            .ceil() as usize;
            let expected_padded =
                expected_valid.div_ceil(RENDER_QUANTUM_SIZE) * RENDER_QUANTUM_SIZE;
            assert_eq!(out.len(), expected_padded, "{in_sr}->{out_sr}: output len");
            assert!(
                out[expected_valid..].iter().all(|&v| v == 0.0),
                "{in_sr}->{out_sr}: expected zero padding after frame {expected_valid}"
            );

            // Exclude startup/tail filter transients when measuring interior continuity.
            let start = (0.05 * out_sr) as usize;
            let end = (0.90 * out_sr) as usize;
            let (max_diff, amp, sinad_db) = measure_tone(&out[start..end], out_sr, freq_hz);
            let theoretical_step = (2.0 * PI * freq_hz / f64::from(out_sr) * amplitude) as f32;

            assert!(
                max_diff <= theoretical_step * 1.25,
                "{in_sr}->{out_sr}: max_diff={max_diff:.6} > limit={:.6}",
                theoretical_step * 1.25
            );
            assert_float_eq!(amp, amplitude, abs <= 1e-3);
            assert!(
                sinad_db >= 60.0,
                "{in_sr}->{out_sr}: sinad_db={sinad_db:.2} < 60 dB"
            );
        }
    }

    #[test]
    fn test_resampler_stereo_and_variable_chunk_sizes() {
        let in_sr = 8_000.0_f32;
        let out_sr = 48_000.0_f32;
        let total_in = in_sr as usize;
        let chunk_sizes = [1, 17, 128, 3, 250, 64];

        let mut chunks = Vec::new();
        let mut frame = 0;
        let mut idx = 0;
        while frame < total_in {
            let len = chunk_sizes[idx % chunk_sizes.len()].min(total_in - frame);
            idx += 1;
            let mut left = Vec::with_capacity(len);
            let mut right = Vec::with_capacity(len);
            for _ in 0..len {
                let t = frame as f64 / f64::from(in_sr);
                left.push(((2.0 * PI * 220.0 * t).sin() * 0.25) as f32);
                right.push(((2.0 * PI * 440.0 * t).cos() * 0.20) as f32);
                frame += 1;
            }
            chunks.push(Ok(AudioBuffer::from(vec![left, right], in_sr)));
        }

        let resampler = Resampler::new(out_sr, RENDER_QUANTUM_SIZE, chunks.into_iter());
        let (mut left_out, mut right_out) = (Vec::new(), Vec::new());
        for buf in resampler {
            let buf = buf.unwrap();
            assert_eq!(buf.number_of_channels(), 2);
            assert_eq!(buf.length(), RENDER_QUANTUM_SIZE);
            left_out.extend_from_slice(buf.get_channel_data(0));
            right_out.extend_from_slice(buf.get_channel_data(1));
        }

        let start = (0.05 * out_sr) as usize;
        let end = (0.90 * out_sr) as usize;
        let (_, amp_l, sinad_l) = measure_tone(&left_out[start..end], out_sr, 220.0);
        let (_, amp_r, sinad_r) = measure_tone(&right_out[start..end], out_sr, 440.0);

        assert_float_eq!(amp_l, 0.25, abs <= 1e-3);
        assert_float_eq!(amp_r, 0.20, abs <= 1e-3);
        assert!(sinad_l >= 60.0, "left sinad={sinad_l:.2}");
        assert!(sinad_r >= 60.0, "right sinad={sinad_r:.2}");
    }

    #[test]
    fn test_resampler_downsampling_attenuates_above_nyquist() {
        // Verify anti-aliasing across single-stage 2x (96->48 kHz) and cascaded
        // multi-stage downsampling (48->8 kHz, 768->3 kHz), including tones that
        // would alias directly into the final passband if an intermediate 2:1
        // decimation filter were bypassed.
        let cases = [
            (96_000.0_f32, 48_000.0_f32, 30_000.0_f64, 48_000usize, 128),
            (48_000.0_f32, 8_000.0_f32, 6_000.0_f64, 24_000usize, 128),
            (48_000.0_f32, 8_000.0_f32, 23_000.0_f64, 24_000usize, 128),
            (768_000.0_f32, 3_000.0_f32, 3_000.0_f64, 384_000usize, 1024),
            (
                768_000.0_f32,
                3_000.0_f32,
                383_500.0_f64,
                384_000usize,
                1024,
            ),
        ];

        for (in_sr, out_sr, freq_hz, total_in, chunk_size) in cases {
            let mut chunks = Vec::new();
            let mut frame = 0;
            let phase_step = 2.0 * PI * freq_hz / f64::from(in_sr);
            while frame < total_in {
                let len = chunk_size.min(total_in - frame);
                let mut channel = Vec::with_capacity(len);
                for _ in 0..len {
                    channel.push((frame as f64 * phase_step).cos() as f32);
                    frame += 1;
                }
                chunks.push(Ok(AudioBuffer::from(vec![channel], in_sr)));
            }

            let out: Vec<f32> = Resampler::new(out_sr, RENDER_QUANTUM_SIZE, chunks.into_iter())
                .flat_map(|b| b.unwrap().get_channel_data(0).to_vec())
                .collect();

            let start = (out.len() / 4).max(128);
            let end = (out.len() * 3) / 4;
            let slice = &out[start..end];
            let rms = (slice
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / slice.len() as f64)
                .sqrt();
            assert!(
                rms < 0.01,
                "{in_sr}->{out_sr} @ {freq_hz} Hz: aliased rms={rms:.5} >= 0.01"
            );
        }

        // Verify passband preservation across an 8-stage 768->3 kHz cascade.
        let passband_chunks = sine_chunks(768_000.0, 400.0, 0.5, &[1024], 384_000);
        let passband_out: Vec<f32> =
            Resampler::new(3_000.0, RENDER_QUANTUM_SIZE, passband_chunks.into_iter())
                .flat_map(|b| b.unwrap().get_channel_data(0).to_vec())
                .collect();
        let (_, amp, sinad_db) = measure_tone(&passband_out[300..1200], 3_000.0, 400.0);
        assert_float_eq!(amp, 0.5, abs <= 2e-3);
        assert!(sinad_db >= 60.0, "768->3 kHz passband sinad={sinad_db:.2}");
    }

    #[test]
    fn test_resampler_format_changes_eof_and_edge_cases() {
        // Cascaded short-stream EOF flush: 240 frames @ 48->8 kHz (3 stages, each
        // with 32 input frames of causal startup delay) yields 78 valid frames
        // followed by zero padding to 128 frames.
        let short_dc = vec![Ok(AudioBuffer::from(vec![vec![0.5_f32; 240]], 48_000.))];
        let mut short_resampler = Resampler::new(8_000., RENDER_QUANTUM_SIZE, short_dc.into_iter());
        let first = short_resampler.next().unwrap().unwrap();
        let data = first.get_channel_data(0);
        assert!(data[37..77].iter().any(|&v| v > 0.4));
        assert!(data[78..].iter().all(|&v| v == 0.0));
        assert!(short_resampler.next().is_none());

        // Mid-stream rate change (8 kHz -> 16 kHz -> 48 kHz passthrough) preserves
        // cumulative segment duration including causal per-segment startup delay
        // ((128+32)*6 + (128+32)*3 = 1,440 frames before the 128-frame marker).
        let transition = vec![
            Ok(AudioBuffer::from(vec![vec![0.25_f32; 128]], 8_000.)),
            Ok(AudioBuffer::from(vec![vec![-0.25_f32; 128]], 16_000.)),
            Ok(AudioBuffer::from(vec![vec![1.0_f32; 128]], 48_000.)),
        ];
        let out: Vec<f32> = Resampler::new(48_000., RENDER_QUANTUM_SIZE, transition.into_iter())
            .flat_map(|b| b.unwrap().get_channel_data(0).to_vec())
            .collect();
        assert_eq!(out.len(), 1_664);
        assert!(out[1_440..1_568].iter().all(|&v| (v - 1.0).abs() <= 1e-6));
        assert!(out[1_568..].iter().all(|&v| v == 0.0));

        // Channel-layout change (mono -> stereo) flushes mono buffer cleanly before
        // switching to stereo output.
        let ch_change = vec![
            Ok(AudioBuffer::from(vec![vec![0.5_f32; 64]], 48_000.)),
            Ok(AudioBuffer::from(
                vec![vec![0.25_f32; 128], vec![-0.25_f32; 128]],
                48_000.,
            )),
        ];
        let mut ch_resampler = Resampler::new(48_000., RENDER_QUANTUM_SIZE, ch_change.into_iter());
        let q0 = ch_resampler.next().unwrap().unwrap();
        assert_eq!(q0.number_of_channels(), 1);
        assert!(q0.get_channel_data(0)[..64].iter().all(|&v| v == 0.5));
        assert!(q0.get_channel_data(0)[64..].iter().all(|&v| v == 0.0));
        let q1 = ch_resampler.next().unwrap().unwrap();
        assert_eq!(q1.number_of_channels(), 2);
        assert!(q1.get_channel_data(0).iter().all(|&v| v == 0.25));
        assert!(q1.get_channel_data(1).iter().all(|&v| v == -0.25));
        assert!(ch_resampler.next().is_none());

        // Finite non-integer conversion with irregular chunks (50 frames @ 8->44.1 kHz):
        // ceil((50 + 32) * 44,100 / 8,000) = 453 valid frames, padded to 4 * 128 = 512
        // with exact 59 trailing zero-pad frames.
        let irreg_chunks: Vec<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> = [7, 13, 19, 11]
            .into_iter()
            .map(|n| Ok(AudioBuffer::from(vec![vec![0.5_f32; n]], 8_000.)))
            .collect();
        let irreg_out: Vec<f32> =
            Resampler::new(44_100., RENDER_QUANTUM_SIZE, irreg_chunks.into_iter())
                .flat_map(|b| b.unwrap().get_channel_data(0).to_vec())
                .collect();
        assert_eq!(irreg_out.len(), 512);
        assert!(
            irreg_out[452].abs() > 0.1,
            "expected valid converted signal at final frame 452, got {}",
            irreg_out[452]
        );
        assert!(
            irreg_out[453..].iter().all(|&v| v == 0.0),
            "expected exact 59 trailing zero-pad frames after frame 453"
        );

        // Upstream errors arriving before a full output quantum is ready clear
        // buffered/converter state and surface immediately; subsequent packets
        // recover cleanly.
        let err_stream: Vec<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> = vec![
            Ok(AudioBuffer::from(vec![vec![0.5_f32; 16]], 8_000.)),
            Err("transient stream error".into()),
            Ok(AudioBuffer::from(vec![vec![-0.75_f32; 128]], 48_000.)),
        ];
        let mut err_resampler =
            Resampler::new(48_000., RENDER_QUANTUM_SIZE, err_stream.into_iter());
        assert!(err_resampler.next().unwrap().is_err());
        let recovered = err_resampler.next().unwrap().unwrap();
        assert!(recovered.get_channel_data(0).iter().all(|&v| v == -0.75));
        assert!(err_resampler.next().is_none());

        // Repeated empty chunks do not hang indefinitely.
        let empty = AudioBuffer::from(vec![Vec::new()], 48_000.);
        let mut empty_resampler = Resampler::new(
            48_000.,
            RENDER_QUANTUM_SIZE,
            std::iter::repeat_with(move || Ok(empty.clone())),
        );
        let silent = empty_resampler.next().unwrap().unwrap();
        assert_eq!(silent.length(), RENDER_QUANTUM_SIZE);
        assert!(silent.get_channel_data(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_media_stream_source_low_sample_rate_continuity() {
        // End-to-end reproduction of #577 through both MediaStreamAudioSourceNode
        // (8 kHz -> 48 kHz) and MediaStreamTrackAudioSourceNode (8 kHz -> 44.1 kHz).
        let freq_hz = 220.0;
        let amplitude = 0.15;

        let chunks_48 = sine_chunks(8_000., freq_hz, amplitude, &[128], 8_000);
        let track_48 = MediaStreamTrack::from_iter(chunks_48);
        let stream_48 = MediaStream::from_tracks(vec![track_48]);
        let mut ctx_48 = OfflineAudioContext::new(1, 48_000, 48_000.);
        let src_48 = MediaStreamAudioSourceNode::new(
            &ctx_48,
            MediaStreamAudioSourceOptions {
                media_stream: &stream_48,
            },
        );
        src_48.connect(&ctx_48.destination());
        let rendered_48 = ctx_48.start_rendering_sync();
        let (max_diff_48, amp_48, sinad_48) = measure_tone(
            &rendered_48.get_channel_data(0)[2_400..43_200],
            48_000.,
            freq_hz,
        );
        let limit_48 = (2.0 * PI * freq_hz / 48_000.0 * amplitude * 1.25) as f32;
        assert!(max_diff_48 <= limit_48, "max_diff={max_diff_48:.6}");
        assert_float_eq!(amp_48, amplitude, abs <= 1e-3);
        assert!(sinad_48 >= 60.0, "sinad={sinad_48:.2}");

        let chunks_441 = sine_chunks(8_000., freq_hz, amplitude, &[128], 8_000);
        let track_441 = MediaStreamTrack::from_iter(chunks_441);
        let mut ctx_441 = OfflineAudioContext::new(1, 44_100, 44_100.);
        let src_441 = MediaStreamTrackAudioSourceNode::new(
            &ctx_441,
            MediaStreamTrackAudioSourceOptions {
                media_stream_track: &track_441,
            },
        );
        src_441.connect(&ctx_441.destination());
        let rendered_441 = ctx_441.start_rendering_sync();
        let (max_diff_441, amp_441, sinad_441) = measure_tone(
            &rendered_441.get_channel_data(0)[2_205..39_690],
            44_100.,
            freq_hz,
        );
        let limit_441 = (2.0 * PI * freq_hz / 44_100.0 * amplitude * 1.25) as f32;
        assert!(max_diff_441 <= limit_441, "max_diff={max_diff_441:.6}");
        assert_float_eq!(amp_441, amplitude, abs <= 1e-3);
        assert!(sinad_441 >= 60.0, "sinad={sinad_441:.2}");
    }

    #[test]
    fn test_resampler_live_stream_arrival_timing_no_inserted_silence() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Model a non-blocking live source (like MicrophoneStream) arriving at
        // the true source-to-output rate ratio: by quantum q, only
        // `ceil((q + 1) * in_sr / out_sr)` 128-frame packets have arrived;
        // polling ahead of arrival yields a 128-frame zero buffer.
        struct LiveArrivalStream {
            sample_rate: f32,
            target_rate: f32,
            packets: std::collections::VecDeque<AudioBuffer>,
            delivered: Arc<AtomicUsize>,
            quantum: Arc<AtomicUsize>,
            silence_insertions: Arc<AtomicUsize>,
        }

        impl Iterator for LiveArrivalStream {
            type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

            fn next(&mut self) -> Option<Self::Item> {
                let q = self.quantum.load(Ordering::SeqCst);
                let arrived = (((q + 1) as f64) * f64::from(self.sample_rate)
                    / f64::from(self.target_rate))
                .ceil() as usize;
                let delivered = self.delivered.load(Ordering::SeqCst);
                if delivered < arrived {
                    let pkt = self.packets.pop_front()?;
                    self.delivered.fetch_add(1, Ordering::SeqCst);
                    Some(Ok(pkt))
                } else {
                    self.silence_insertions.fetch_add(1, Ordering::SeqCst);
                    Some(Ok(AudioBuffer::from(
                        vec![vec![0.0_f32; RENDER_QUANTUM_SIZE]],
                        self.sample_rate,
                    )))
                }
            }
        }

        let freq_hz = 220.0;
        let amplitude = 0.15;

        for &(in_sr, out_sr, expected_q0_polls) in &[
            (44_100.0_f32, 48_000.0_f32, 1usize),
            (96_000.0_f32, 48_000.0_f32, 2usize),
            (96_000.0_f32, 44_100.0_f32, 3usize),
            (48_000.0_f32, 8_000.0_f32, 6usize),
        ] {
            let quanta = (out_sr as usize).div_ceil(RENDER_QUANTUM_SIZE);
            let total_packets =
                ((quanta as f64) * f64::from(in_sr) / f64::from(out_sr)).ceil() as usize + 4;
            let packets: std::collections::VecDeque<AudioBuffer> = sine_chunks(
                in_sr,
                freq_hz,
                amplitude,
                &[RENDER_QUANTUM_SIZE],
                total_packets * RENDER_QUANTUM_SIZE,
            )
            .into_iter()
            .map(Result::unwrap)
            .collect();

            let quantum = Arc::new(AtomicUsize::new(0));
            let delivered = Arc::new(AtomicUsize::new(0));
            let silence_insertions = Arc::new(AtomicUsize::new(0));
            let stream = LiveArrivalStream {
                sample_rate: in_sr,
                target_rate: out_sr,
                packets,
                delivered: Arc::clone(&delivered),
                quantum: Arc::clone(&quantum),
                silence_insertions: Arc::clone(&silence_insertions),
            };

            let mut resampler = Resampler::new(out_sr, RENDER_QUANTUM_SIZE, stream);
            let mut out = Vec::with_capacity(quanta * RENDER_QUANTUM_SIZE);
            for q in 0..quanta {
                quantum.store(q, Ordering::SeqCst);
                let buf = resampler.next().unwrap().unwrap();
                if q == 0 {
                    assert_eq!(
                        delivered.load(Ordering::SeqCst),
                        expected_q0_polls,
                        "{in_sr}->{out_sr}: unexpected quantum-0 packet polls"
                    );
                }
                out.extend_from_slice(buf.get_channel_data(0));
            }

            assert_eq!(
                silence_insertions.load(Ordering::SeqCst),
                0,
                "{in_sr}->{out_sr}: live stream should not be over-polled into silence"
            );

            let start = (0.05 * out_sr) as usize;
            let end = (0.90 * out_sr) as usize;
            let (max_diff, amp, sinad_db) = measure_tone(&out[start..end], out_sr, freq_hz);
            let limit = (2.0 * PI * freq_hz / f64::from(out_sr) * amplitude * 1.25) as f32;
            assert!(
                max_diff <= limit,
                "{in_sr}->{out_sr}: max_diff={max_diff:.6}"
            );
            assert_float_eq!(amp, amplitude, abs <= 1e-3);
            assert!(sinad_db >= 60.0, "{in_sr}->{out_sr}: sinad={sinad_db:.2}");
        }
    }
}
