//! General purpose audio signal data structures
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use float_eq::{assert_float_eq, float_eq};

use crate::alloc::AudioBuffer as FixedAudioBuffer;
use crate::media::MediaStream;

#[derive(Debug, Clone, Copy)]
pub struct AudioBufferOptions {
    pub number_of_channels: Option<usize>,
    pub length: usize,
    pub sample_rate: f32,
}

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    sample_rate: f32,
    internal_data: Vec<ChannelData>,
}

use std::error::Error;

impl AudioBuffer {
    /// Allocate a silent audiobuffer with given channel and samples count.
    pub fn new(options: AudioBufferOptions) -> Self {
        let AudioBufferOptions {
            number_of_channels,
            length,
            sample_rate,
        } = options;

        let number_of_channels = number_of_channels.unwrap_or(1);

        let silence = ChannelData::new(length);

        Self {
            sample_rate,
            internal_data: vec![silence; number_of_channels],
        }
    }

    /// Create a multi-channel audiobuffer.
    /// todo replace with pub(crate)
    pub fn from_channels(channels: Vec<ChannelData>, sample_rate: f32) -> Self {
        Self {
            internal_data: channels,
            sample_rate,
        }
    }

    pub fn copy_to_channel(
        &mut self,
        source: &[f32],
        channel_number: usize,
        buffer_offset: Option<usize>,
    ) {
        let buffer_offset = buffer_offset.unwrap_or_default();
        let buffer_offset = usize::max(0, usize::min(self.length() - buffer_offset, source.len()));

        for (dest, src) in self.get_channel_data_mut(channel_number)[..]
            .iter_mut()
            .zip(&source[buffer_offset..])
        {
            *dest = *src;
        }
    }

    pub fn copy_from_channel(
        &mut self,
        destination: &mut [f32],
        channel_number: usize,
        buffer_offset: Option<usize>,
    ) {
        let buffer_offset = buffer_offset.unwrap_or_default();
        let buffer_offset = usize::max(
            0,
            usize::min(self.length() - buffer_offset, destination.len()),
        );

        for (src, dest) in self.get_channel_data_mut(channel_number)[..]
            .iter_mut()
            .zip(&mut destination[buffer_offset..])
        {
            *dest = *src;
        }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        self.internal_data.len()
    }

    /// Number of samples per channel in this AudioBuffer
    pub fn length(&self) -> usize {
        self.internal_data.get(0).map(ChannelData::len).unwrap_or(0)
    }

    /// Sample rate of this AudioBuffer in Hertz
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Duration in seconds of the AudioBuffer
    pub fn duration(&self) -> f64 {
        self.length() as f64 / f64::from(self.sample_rate)
    }

    /// Channel data as slice
    pub fn channels(&self) -> &[ChannelData] {
        &self.internal_data
    }

    /// Channel data as slice (mutable)
    pub fn channels_mut(&mut self) -> &mut [ChannelData] {
        &mut self.internal_data
    }

    /// Get the samples from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn get_channel_data(&self, index: usize) -> &ChannelData {
        &self.internal_data[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn get_channel_data_mut(&mut self, index: usize) -> &mut ChannelData {
        &mut self.internal_data[index]
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        // todo, optimize for Arcs that are equal
        self.internal_data.iter_mut().for_each(fun)
    }

    /// Extends an AudioBuffer with the contents of another.
    ///
    /// This function will panic if the sample_rate and channel_count are not equal
    pub fn extend(&mut self, other: &Self) {
        assert_float_eq!(self.sample_rate, other.sample_rate, ulps <= 0);
        assert_eq!(self.number_of_channels(), other.number_of_channels());

        let data = self.channels_mut();
        data.iter_mut()
            .zip(other.internal_data.iter())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend(&other_channel[..]);
            });
    }

    /// Extends an AudioBuffer with an [`FixedAudioBuffer`]
    ///
    /// This assumes the sample_rate matches. No up/down-mixing is performed
    pub fn extend_alloc(&mut self, other: &FixedAudioBuffer) {
        self.channels_mut()
            .iter_mut()
            .zip(other.channels())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend_from_slice(&other_channel[..]);
            })
    }

    /// Split an AudioBuffer in chunks with length `sample_len`.
    ///
    /// The last chunk may be shorter than `sample_len`
    pub fn split(mut self, sample_len: u32) -> Vec<AudioBuffer> {
        let sample_len = sample_len as usize;
        let total_len = self.length();
        let sample_rate = self.sample_rate();

        let mut channels: Vec<_> = self
            .channels_mut()
            .iter()
            .map(|channel_data| channel_data[..].chunks(sample_len))
            .collect();

        (0..total_len)
            .step_by(sample_len)
            .map(|_| {
                let cur: Vec<_> = channels
                    .iter_mut()
                    .map(|c| ChannelData::from(c.next().unwrap().to_vec()))
                    .collect();
                AudioBuffer::from_channels(cur, sample_rate)
            })
            .collect()
    }

    /// Split an AudioBuffer in two at the given index.
    pub fn split_off(&mut self, index: u32) -> AudioBuffer {
        let index = index as usize;
        let sample_rate = self.sample_rate();

        let channels: Vec<_> = self
            .channels_mut()
            .iter_mut()
            .map(|channel_data| Arc::make_mut(&mut channel_data.data).split_off(index))
            .map(ChannelData::from)
            .collect();

        AudioBuffer::from_channels(channels, sample_rate)
    }

    /// Resample to the desired sample rate.
    ///
    /// This changes the sample_length of the buffer.
    ///
    /// ```
    /// use web_audio_api::buffer::{ChannelData, AudioBuffer};
    ///
    /// let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
    /// let mut buffer = AudioBuffer::from_channels(vec![channel], 48_000.);
    ///
    /// // upmix from 48k to 96k Hertz sample rate
    /// buffer.resample(96_000.);
    ///
    /// assert_eq!(
    ///     buffer.get_channel_data(0),
    ///     &ChannelData::from(vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.,])
    /// );
    ///
    /// assert_eq!(buffer.sample_rate(), 96_000.0);
    /// ```
    pub fn resample(&mut self, sample_rate: f32) {
        if float_eq!(self.sample_rate(), sample_rate, ulps <= 0) {
            return;
        }

        let rate = sample_rate / self.sample_rate;
        self.modify_channels(|channel_data| {
            let mut current = 0;
            let resampled = channel_data
                .data
                .iter()
                .enumerate()
                .flat_map(|(i, v)| {
                    let target = ((i + 1) as f32 * rate) as usize;
                    let take = target - current.min(target);
                    current += take;
                    std::iter::repeat(*v).take(take)
                })
                .collect();
            channel_data.data = Arc::new(resampled);
        });

        self.sample_rate = sample_rate;
    }
}

/// Single channel audio samples, basically wraps a `Arc<Vec<f32>>`
///
/// ChannelData has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelData {
    data: Arc<Vec<f32>>,
}

impl ChannelData {
    pub fn new(length: usize) -> Self {
        let buffer = vec![0.; length];
        let data = Arc::new(buffer);

        Self { data }
    }

    pub fn from(data: Vec<f32>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<Idx> Index<Idx> for ChannelData
where
    Idx: SliceIndex<[f32]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[index]
    }
}

impl<Idx> IndexMut<Idx> for ChannelData
where
    Idx: SliceIndex<[f32]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut Arc::make_mut(&mut self.data)[index]
    }
}

/// How channels must be matched between the node's inputs and outputs.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChannelCountMode {
    /// `computedNumberOfChannels` is the maximum of the number of channels of all connections to an
    /// input. In this mode channelCount is ignored.
    Max,
    /// `computedNumberOfChannels` is determined as for "max" and then clamped to a maximum value of
    /// the given channelCount.
    ClampedMax,
    /// `computedNumberOfChannels` is the exact value as specified by the channelCount.
    Explicit,
}

impl From<u32> for ChannelCountMode {
    fn from(i: u32) -> Self {
        use ChannelCountMode::*;

        match i {
            0 => Max,
            1 => ClampedMax,
            2 => Explicit,
            _ => unreachable!(),
        }
    }
}

/// The meaning of the channels, defining how audio up-mixing and down-mixing will happen.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChannelInterpretation {
    Speakers,
    Discrete,
}

impl From<u32> for ChannelInterpretation {
    fn from(i: u32) -> Self {
        use ChannelInterpretation::*;

        match i {
            0 => Speakers,
            1 => Discrete,
            _ => unreachable!(),
        }
    }
}

/// Options for constructing ChannelConfig
#[derive(Clone, Debug)]
pub struct ChannelConfigOptions {
    pub count: usize,
    pub mode: ChannelCountMode,
    pub interpretation: ChannelInterpretation,
}

impl Default for ChannelConfigOptions {
    fn default() -> Self {
        Self {
            count: 2,
            mode: ChannelCountMode::Max,
            interpretation: ChannelInterpretation::Speakers,
        }
    }
}

/// Config for up/down-mixing of channels for audio nodes
#[derive(Clone, Debug)]
pub struct ChannelConfig {
    count: Arc<AtomicUsize>,
    mode: Arc<AtomicU32>,
    interpretation: Arc<AtomicU32>,
}

impl ChannelConfig {
    /// Represents an enumerated value describing the way channels must be matched between the
    /// node's inputs and outputs.
    pub fn count_mode(&self) -> ChannelCountMode {
        self.mode.load(Ordering::SeqCst).into()
    }
    pub fn set_count_mode(&self, v: ChannelCountMode) {
        self.mode.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    pub fn interpretation(&self) -> ChannelInterpretation {
        self.interpretation.load(Ordering::SeqCst).into()
    }
    pub fn set_interpretation(&self, v: ChannelInterpretation) {
        self.interpretation.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
    pub fn set_count(&self, v: usize) {
        self.count.store(v, Ordering::SeqCst)
    }
}

impl From<ChannelConfigOptions> for ChannelConfig {
    fn from(opts: ChannelConfigOptions) -> Self {
        ChannelConfig {
            count: Arc::new(AtomicUsize::from(opts.count)),
            mode: Arc::new(AtomicU32::from(opts.mode as u32)),
            interpretation: Arc::new(AtomicU32::from(opts.interpretation as u32)),
        }
    }
}

impl std::iter::FromIterator<AudioBuffer> for AudioBuffer {
    fn from_iter<I: IntoIterator<Item = AudioBuffer>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let mut collect: AudioBuffer = match iter.next() {
            None => {
                let options = AudioBufferOptions {
                    number_of_channels: Some(0),
                    length: 0,
                    sample_rate: 0.,
                };
                return AudioBuffer::new(options);
            }
            Some(first) => first,
        };

        for elem in iter {
            collect.extend(&elem);
        }

        collect
    }
}

/// Sample rate converter and buffer chunk splitter.
///
/// A `MediaElement` can be wrapped inside a `Resampler` to yield AudioBuffers of the desired sample_rate and length
///
/// ```
/// use web_audio_api::buffer::{ChannelData, AudioBuffer, Resampler};
///
/// // construct an input of 3 chunks of 5 samples
/// let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
/// let input_buf = AudioBuffer::from_channels(vec![channel], 44_100.);
/// let input = vec![input_buf; 3].into_iter().map(|b| Ok(b));
///
/// // resample to chunks of 10 samples
/// let mut resampler = Resampler::new(44_100., 10, input);
///
/// // first chunk contains 10 samples
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.length(), 10);
/// assert_eq!(next.get_channel_data(0), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
///     1., 2., 3., 4., 5.,
/// ]));
///
/// // second chunk contains 5 samples of signal, and 5 silent
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.length(), 10);
/// assert_eq!(next.get_channel_data(0), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
///     0., 0., 0., 0., 0.,
/// ]));
///
/// // no further chunks
/// assert!(resampler.next().is_none());
/// ```
pub struct Resampler<I> {
    /// desired sample rate
    sample_rate: f32,
    /// desired sample length
    sample_len: u32,
    /// input stream
    input: I,
    /// internal buffer
    buffer: Option<AudioBuffer>,
}

impl<M: MediaStream> Resampler<M> {
    pub fn new(sample_rate: f32, sample_len: u32, input: M) -> Self {
        Self {
            sample_rate,
            sample_len,
            input,
            buffer: None,
        }
    }
}

impl<M: MediaStream> Iterator for Resampler<M> {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = match self.buffer.take() {
            None => match self.input.next() {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(mut data)) => {
                    data.resample(self.sample_rate);
                    data
                }
            },
            Some(data) => data,
        };

        while (buffer.length() as u32) < self.sample_len {
            // buffer is smaller than desired len
            match self.input.next() {
                None => {
                    let options = AudioBufferOptions {
                        number_of_channels: Some(buffer.number_of_channels()),
                        length: self.sample_len as usize - buffer.length(),
                        sample_rate: self.sample_rate,
                    };
                    let padding = AudioBuffer::new(options);
                    buffer.extend(&padding);

                    return Some(Ok(buffer));
                }
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(mut data)) => {
                    data.resample(self.sample_rate);
                    buffer.extend(&data)
                }
            }
        }

        if buffer.length() as u32 == self.sample_len {
            return Some(Ok(buffer));
        }

        self.buffer = Some(buffer.split_off(self.sample_len));

        Some(Ok(buffer))
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_silent() {
        let options = AudioBufferOptions {
            number_of_channels: Some(2),
            length: 10,
            sample_rate: 44100.,
        };
        let b = AudioBuffer::new(options);

        assert_eq!(b.length(), 10);
        assert_eq!(b.number_of_channels(), 2);
        assert_float_eq!(b.sample_rate(), 44_100.0, ulps <= 0);
        assert_float_eq!(b.get_channel_data(0)[..], &[0.; 10][..], ulps_all <= 0);
        assert_float_eq!(b.get_channel_data(1)[..], &[0.; 10][..], ulps_all <= 0);
        assert_eq!(b.channels().get(2), None);
    }

    #[test]
    fn test_concat_split() {
        let options = AudioBufferOptions {
            number_of_channels: Some(2),
            length: 5,
            sample_rate: 44100.,
        };
        let mut b1 = AudioBuffer::new(options);
        let b2 = AudioBuffer::new(options);
        b1.extend(&b2);

        assert_eq!(b1.length(), 10);
        assert_eq!(b1.number_of_channels(), 2);
        assert_float_eq!(b1.sample_rate(), 44_100.0, ulps <= 0);

        let channel_data = ChannelData::from(vec![1.; 5]);
        let b3 = AudioBuffer::from_channels(vec![channel_data; 2], 44_100.);

        b1.extend(&b3);

        assert_eq!(b1.length(), 15);
        assert_eq!(b1.number_of_channels(), 2);
        assert_float_eq!(b1.sample_rate(), 44_100.0, ulps <= 0);
        assert_float_eq!(
            b1.get_channel_data(0)[..],
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
            ulps_all <= 0
        );

        let split = b1.split(8);
        assert_float_eq!(
            split[0].get_channel_data(0)[..],
            &[0., 0., 0., 0., 0., 0., 0., 0.][..],
            ulps_all <= 0
        );
        assert_float_eq!(
            split[1].get_channel_data(0)[..],
            &[0., 0., 1., 1., 1., 1., 1.][..],
            ulps_all <= 0
        );
    }

    #[test]
    fn test_resample_upmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], 100.);
        buffer.resample(200.);
        assert_float_eq!(
            buffer.get_channel_data(0)[..],
            &[1., 1., 2., 2., 3., 3., 4., 4., 5., 5.,][..],
            ulps_all <= 0
        );
        assert_float_eq!(buffer.sample_rate(), 200.0, ulps <= 0);
    }

    #[test]
    fn test_resample_downmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], 200.);
        buffer.resample(100.);
        assert_float_eq!(buffer.get_channel_data(0)[..], &[2., 4.][..], ulps_all <= 0);
        assert_float_eq!(buffer.sample_rate(), 100.0, ulps <= 0);
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
            next.get_channel_data(0)[..],
            &[1., 2., 3., 4., 5., 1., 2., 3., 4., 5.,][..],
            ulps_all <= 0
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 10);
        assert_float_eq!(
            next.get_channel_data(0)[..],
            &[1., 2., 3., 4., 5., 0., 0., 0., 0., 0.][..],
            ulps_all <= 0
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
            next.get_channel_data(0)[..],
            &[1., 2., 3., 4., 5.][..],
            ulps_all <= 0
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.length(), 5);
        assert_float_eq!(
            next.get_channel_data(0)[..],
            &[6., 7., 8., 9., 10.][..],
            ulps_all <= 0
        );

        assert!(resampler.next().is_none());
    }
}
