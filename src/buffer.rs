//! Audio signal data structures

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::media::MediaElement;
use crate::SampleRate;

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    data: AudioBufferType,
    sample_rate: SampleRate,
}

#[derive(Clone, Debug)]
enum AudioBufferType {
    /// n channels with m samples of silence
    Silence(usize, usize),
    /// n identical channels
    Mono(ChannelData, usize),
    /// n different channels
    Multi(Box<[ChannelData]>), // todo, make [ChannelData; 32] fixed size array
}

use AudioBufferType::*;

impl AudioBuffer {
    /// Create a silent audiobuffer with given channel and samples count.
    ///
    /// This function does not allocate.
    pub fn new(channels: usize, len: usize, sample_rate: SampleRate) -> Self {
        Self {
            data: Silence(channels, len),
            sample_rate,
        }
    }

    /// Create a mono audiobuffer (single channel)
    pub fn from_mono(data: ChannelData, sample_rate: SampleRate) -> Self {
        Self {
            data: Mono(data, 1),
            sample_rate,
        }
    }

    /// Create a multi-channel audiobuffer
    pub fn from_channels(data: Vec<ChannelData>, sample_rate: SampleRate) -> Self {
        Self {
            data: Multi(data.into_boxed_slice()),
            sample_rate,
        }
    }

    /// Up/Down-mix to the desired number of channels
    pub fn mix(&self, channels: usize, interpretation: ChannelInterpretation) -> Self {
        // handle silence
        if let Silence(_, len) = &self.data {
            return Self {
                data: Silence(channels, *len),
                sample_rate: self.sample_rate,
            };
        }

        // handle mono
        if let Mono(data, prev_channels) = &self.data {
            if interpretation == ChannelInterpretation::Discrete {
                // discret layout: no mixing required
                if *prev_channels >= channels {
                    return Self {
                        data: Mono(data.clone(), channels),
                        sample_rate: self.sample_rate,
                    };
                } else {
                    let mut new = Vec::with_capacity(channels);
                    for _ in 0..*prev_channels {
                        new.push(data.clone());
                    }
                    let silence = ChannelData::new(self.sample_len());
                    for _ in *prev_channels..channels {
                        new.push(silence.clone());
                    }
                    return Self::from_channels(new, self.sample_rate);
                }
            } else {
                // speaker layout: mixing required
                match channels {
                    1 | 2 => {
                        return Self {
                            data: Mono(data.clone(), channels),
                            sample_rate: self.sample_rate,
                        }
                    }
                    4 => {
                        let silence = ChannelData::new(self.sample_len());
                        return Self::from_channels(
                            vec![data.clone(), data.clone(), silence.clone(), silence],
                            self.sample_rate,
                        );
                    }
                    6 => {
                        let silence = ChannelData::new(self.sample_len());
                        return Self::from_channels(
                            vec![
                                silence.clone(),
                                silence.clone(),
                                data.clone(),
                                silence.clone(),
                                silence,
                            ],
                            self.sample_rate,
                        );
                    }

                    _ => panic!("unknown speaker configuration {}", channels),
                }
            }
        }

        let data = match &self.data {
            Multi(data) => data,
            _ => unreachable!(),
        };

        match (data.len(), channels) {
            (n, m) if n == m => self.clone(),
            (1, c) => Self {
                data: Mono(data[0].clone(), c),
                sample_rate: self.sample_rate,
            },
            (2, 1) => {
                let mut l = data[0].clone();
                l.iter_mut()
                    .zip(data[1].iter())
                    .for_each(|(l, r)| *l = 0.5 * (*l + r));

                Self {
                    data: Mono(l, 1),
                    sample_rate: self.sample_rate,
                }
            }
            _ => todo!(),
        }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        match &self.data {
            Silence(c, _) => *c,
            Mono(_, c) => *c,
            Multi(data) => data.len(),
        }
    }

    /// Number of samples per channel in this AudioBuffer
    pub fn sample_len(&self) -> usize {
        match &self.data {
            Silence(_, len) => *len,
            Mono(data, _) => data.len(),
            Multi(data) => data[0].len(),
        }
    }

    /// Sample rate of this AudioBuffer in Hertz
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Get the samples from this specific channel.
    ///
    /// Returns `None` if this channel is silent or not present
    pub fn channel_data(&self, channel: usize) -> Option<&ChannelData> {
        match &self.data {
            Silence(_, _) => None,
            Mono(data, _) => Some(data),
            Multi(data) => data.get(channel),
        }
    }

    /// Convert this buffer to silence, maintaining the channel and sample counts
    pub fn make_silent(&mut self) {
        match &mut self.data {
            Silence(_, _) => (),
            Mono(data, channels) => self.data = Silence(*channels, data.len()),
            Multi(data) => self.data = Silence(data.len(), data[0].len()),
        }
    }

    /// Convert this buffer to a mono sound, maintaining the channel and sample counts.
    pub fn make_mono(&mut self) {
        let len = self.sample_len();
        let channels = self.number_of_channels();

        match &mut self.data {
            Silence(_, _) => {
                self.data = Mono(ChannelData::new(len), channels);
            }
            Mono(_data, _) => (),
            Multi(data) => {
                self.data = Mono(data[0].clone(), channels);
            }
        }
    }

    /// Convert to Multi type buffer, and return mutable channel data
    fn channel_data_mut(&mut self) -> &mut [ChannelData] {
        let sample_rate = self.sample_rate();
        match &mut self.data {
            Silence(channels, len) => {
                *self = AudioBuffer::from_channels(
                    vec![ChannelData::from(vec![0.; *len]); *channels],
                    sample_rate,
                );
            }
            Mono(data, channels) => {
                *self = AudioBuffer::from_channels(vec![data.clone(); *channels], sample_rate);
            }
            Multi(_) => (),
        };

        match &mut self.data {
            Multi(data) => data,
            _ => unreachable!(),
        }
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        if matches!(&self.data, Silence(_, _)) {
            self.make_mono();
        }

        match &mut self.data {
            Silence(_, _) => unreachable!(),
            Mono(data, _) => (fun)(data),
            Multi(data) => data.iter_mut().for_each(fun),
        }
    }

    /// Sum two AudioBuffers
    ///
    /// This function will panic if the sample_length and sample_rate are not equal
    pub fn add(&self, other: &Self, interpretation: ChannelInterpretation) -> Self {
        assert_eq!(self.sample_rate, other.sample_rate);
        assert_eq!(self.sample_len(), other.sample_len());

        // mix buffers to the max channel count
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let channels = channels_self.max(channels_other);

        if channels_self > channels_other {
            other.mix(channels_self, interpretation);
        }
        if channels_self < channels_other {
            self.mix(channels_other, interpretation);
        }

        // early exit for simple cases, or determine which signal is Multi
        let (mut multi, other) = match (&self.data, &other.data) {
            (Silence(_, _), _) => return other.clone(),
            (_, Silence(_, _)) => return self.clone(),
            (Mono(s, _), Mono(o, _)) => {
                let mut new = s.clone();
                new.add(&o);
                return Self {
                    data: Mono(new, channels),
                    sample_rate: self.sample_rate,
                };
            }
            (Multi(data), _) => (data.clone(), other),
            (_, Multi(data)) => (data.clone(), self),
        };

        // mutate the Multi signal with values from the other
        (0..channels).for_each(|i| {
            if let Some(data) = other.channel_data(i) {
                multi[i].add(data)
            }
        });

        Self {
            data: Multi(multi),
            sample_rate: self.sample_rate,
        }
    }

    /// Extends an AudioBuffer with the contents of another.
    ///
    /// This function will panic if the sample_rate and channel_count are not equal
    pub fn extend(&mut self, other: &Self) {
        assert_eq!(self.sample_rate, other.sample_rate);
        assert_eq!(self.number_of_channels(), other.number_of_channels());

        let data = self.channel_data_mut();
        data.iter_mut()
            .enumerate()
            .for_each(|(channel, channel_data)| {
                let cur_channel_data = Arc::make_mut(&mut channel_data.data);

                if let Some(data) = other.channel_data(channel) {
                    cur_channel_data.extend(data.iter().copied());
                } else {
                    cur_channel_data.extend(std::iter::repeat(0.).take(other.sample_len()));
                }
            })
    }

    /// Split an AudioBuffer in chunks with length `sample_len`.
    ///
    /// The last chunk may be shorter than `sample_len`
    pub fn split(mut self, sample_len: u32) -> Vec<AudioBuffer> {
        let sample_len = sample_len as usize;
        let total_len = self.sample_len();
        let sample_rate = self.sample_rate();

        let mut channels: Vec<_> = self
            .channel_data_mut()
            .iter()
            .map(|channel_data| channel_data.as_slice().chunks(sample_len))
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
            .channel_data_mut()
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
    /// use web_audio_api::SampleRate;
    /// use web_audio_api::buffer::{ChannelData, AudioBuffer};
    ///
    /// let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
    /// let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(48_000));
    ///
    /// // upmix from 48k to 96k Hertz sample rate
    /// buffer.resample(SampleRate(96_000));
    ///
    /// assert_eq!(
    ///     buffer.channel_data(0).unwrap(),
    ///     &ChannelData::from(vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.,])
    /// );
    ///
    /// assert_eq!(buffer.sample_rate().0, 96_000);
    /// ```
    pub fn resample(&mut self, sample_rate: SampleRate) {
        if self.sample_rate() == sample_rate {
            return;
        }

        let rate = sample_rate.0 as f32 / self.sample_rate.0 as f32;
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

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        Arc::make_mut(&mut self.data).iter_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut Arc::make_mut(&mut self.data)[..]
    }

    pub fn add(&mut self, other: &Self) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
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
            None => return AudioBuffer::new(0, 0, SampleRate(0)),
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
/// use web_audio_api::SampleRate;
/// use web_audio_api::buffer::{ChannelData, AudioBuffer, Resampler};
///
/// // construct an input of 3 chunks of 5 samples
/// let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
/// let input_buf = AudioBuffer::from_channels(vec![channel], SampleRate(44_100));
/// let input = vec![input_buf; 3].into_iter().map(|b| Ok(b));
///
/// // resample to chunks of 10 samples
/// let mut resampler = Resampler::new(SampleRate(44_100), 10, input);
///
/// // first chunk contains 10 samples
/// let next = resampler.next().unwrap();
/// assert_eq!(next.sample_len(), 10);
/// assert_eq!(next.channel_data(0).unwrap(), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
///     1., 2., 3., 4., 5.,
/// ]));
///
/// // second chunk contains 5 samples
/// let next = resampler.next().unwrap();
/// assert_eq!(next.sample_len(), 5);
/// assert_eq!(next.channel_data(0).unwrap(), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
/// ]));
///
/// // no further chunks
/// assert!(resampler.next().is_none());
/// ```
pub struct Resampler<I> {
    /// desired sample rate
    sample_rate: SampleRate,
    /// desired sample length
    sample_len: u32,
    /// input stream
    input: I,
    /// internal buffer
    buffer: Option<AudioBuffer>,
}

impl<M: MediaElement> Resampler<M> {
    pub fn new(sample_rate: SampleRate, sample_len: u32, input: M) -> Self {
        Self {
            sample_rate,
            sample_len,
            input,
            buffer: None,
        }
    }
}

impl<M: MediaElement> Iterator for Resampler<M> {
    type Item = AudioBuffer;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = match self.buffer.take() {
            None => match self.input.next() {
                None => return None,
                Some(Err(_e)) => return None, // todo
                Some(Ok(mut data)) => {
                    data.resample(self.sample_rate);
                    data
                }
            },
            Some(data) => data,
        };

        while (buffer.sample_len() as u32) < self.sample_len {
            // buffer is smaller than desired len
            match self.input.next() {
                None => return Some(buffer),
                Some(Err(_e)) => return None, // todo
                Some(Ok(mut data)) => {
                    data.resample(self.sample_rate);
                    buffer.extend(&data)
                }
            }
        }

        if buffer.sample_len() as u32 == self.sample_len {
            return Some(buffer);
        }

        self.buffer = Some(buffer.split_off(self.sample_len));

        Some(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_split() {
        let mut b1 = AudioBuffer::new(2, 5, SampleRate(44_100));
        let b2 = AudioBuffer::new(2, 5, SampleRate(44_100));
        b1.extend(&b2);

        assert_eq!(b1.sample_len(), 10);
        assert_eq!(b1.number_of_channels(), 2);
        assert_eq!(b1.sample_rate().0, 44_100);

        let channel_data = ChannelData::from(vec![1.; 5]);
        let b3 = AudioBuffer::from_channels(vec![channel_data; 2], SampleRate(44_100));

        b1.extend(&b3);

        assert_eq!(b1.sample_len(), 15);
        assert_eq!(b1.number_of_channels(), 2);
        assert_eq!(b1.sample_rate().0, 44_100);
        assert_eq!(
            b1.channel_data(0).unwrap().as_slice(),
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
        );

        let split = b1.split(8);
        assert_eq!(
            split[0].channel_data(0).unwrap().as_slice(),
            &[0., 0., 0., 0., 0., 0., 0., 0.]
        );
        assert_eq!(
            split[1].channel_data(0).unwrap().as_slice(),
            &[0., 0., 1., 1., 1., 1., 1.]
        );
    }

    #[test]
    fn test_resample_upmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(100));
        buffer.resample(SampleRate(200));
        assert_eq!(
            buffer.channel_data(0).unwrap(),
            &ChannelData::from(vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.,])
        );
        assert_eq!(buffer.sample_rate().0, 200);
    }

    #[test]
    fn test_resample_downmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(200));
        buffer.resample(SampleRate(100));
        assert_eq!(
            buffer.channel_data(0).unwrap(),
            &ChannelData::from(vec![2., 4.])
        );
        assert_eq!(buffer.sample_rate().0, 100);
    }

    #[test]
    fn test_resampler_concat() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let input_buf = AudioBuffer::from_channels(vec![channel], SampleRate(44_100));
        let input = vec![input_buf; 3].into_iter().map(|b| Ok(b));
        let mut resampler = Resampler::new(SampleRate(44_100), 10, input);

        let next = resampler.next().unwrap();
        assert_eq!(next.sample_len(), 10);
        assert_eq!(
            next.channel_data(0).unwrap(),
            &ChannelData::from(vec![1., 2., 3., 4., 5., 1., 2., 3., 4., 5.,])
        );

        let next = resampler.next().unwrap();
        assert_eq!(next.sample_len(), 5);
        assert_eq!(
            next.channel_data(0).unwrap(),
            &ChannelData::from(vec![1., 2., 3., 4., 5.,])
        );

        assert!(resampler.next().is_none());
    }

    #[test]
    fn test_resampler_split() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let input_buf = Ok(AudioBuffer::from_channels(
            vec![channel],
            SampleRate(44_100),
        ));
        let input = vec![input_buf].into_iter();
        let mut resampler = Resampler::new(SampleRate(44_100), 5, input);

        let next = resampler.next().unwrap();
        assert_eq!(next.sample_len(), 5);
        assert_eq!(
            next.channel_data(0).unwrap(),
            &ChannelData::from(vec![1., 2., 3., 4., 5.,])
        );

        let next = resampler.next().unwrap();
        assert_eq!(next.sample_len(), 5);
        assert_eq!(
            next.channel_data(0).unwrap(),
            &ChannelData::from(vec![6., 7., 8., 9., 10.])
        );

        assert!(resampler.next().is_none());
    }
}
