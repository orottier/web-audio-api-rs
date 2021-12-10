//! General purpose audio signal data structures
use std::sync::Arc;

use crate::alloc::AudioRenderQuantum;
use crate::media::MediaStream;
use crate::SampleRate;

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    channels: Vec<ChannelData>,
    sample_rate: SampleRate,
}

use std::error::Error;

impl AudioBuffer {
    /// Allocate a silent audiobuffer with given channel and samples count.
    pub fn new(channels: usize, len: usize, sample_rate: SampleRate) -> Self {
        let silence = ChannelData::new(len);

        Self {
            channels: vec![silence; channels],
            sample_rate,
        }
    }

    /// Create a multi-channel audiobuffer.
    pub fn from_channels(channels: Vec<ChannelData>, sample_rate: SampleRate) -> Self {
        Self {
            channels,
            sample_rate,
        }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel in this AudioBuffer
    pub fn sample_len(&self) -> usize {
        self.channels.get(0).map(ChannelData::len).unwrap_or(0)
    }

    /// Sample rate of this AudioBuffer in Hertz
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Duration in seconds of the AudioBuffer
    pub fn duration(&self) -> f64 {
        self.sample_len() as f64 / self.sample_rate.0 as f64
    }

    /// Channel data as slice
    pub fn channels(&self) -> &[ChannelData] {
        &self.channels
    }

    /// Channel data as slice (mutable)
    pub fn channels_mut(&mut self) -> &mut [ChannelData] {
        &mut self.channels
    }

    /// Get the samples from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data(&self, index: usize) -> &ChannelData {
        &self.channels[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data_mut(&mut self, index: usize) -> &mut ChannelData {
        &mut self.channels[index]
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        // todo, optimize for Arcs that are equal
        self.channels.iter_mut().for_each(fun)
    }

    /// Extends an AudioBuffer with the contents of another.
    ///
    /// This function will panic if the sample_rate and channel_count are not equal
    pub fn extend(&mut self, other: &Self) {
        assert_eq!(self.sample_rate, other.sample_rate);
        assert_eq!(self.number_of_channels(), other.number_of_channels());

        let data = self.channels_mut();
        data.iter_mut()
            .zip(other.channels.iter())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend(other_channel.as_slice());
            })
    }

    /// Extends an AudioBuffer with an [`AudioRenderQuantum`]
    ///
    /// This assumes the sample_rate matches. No up/down-mixing is performed
    pub fn extend_alloc(&mut self, other: &AudioRenderQuantum) {
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
        let total_len = self.sample_len();
        let sample_rate = self.sample_rate();

        let mut channels: Vec<_> = self
            .channels_mut()
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
    pub fn split_off(&mut self, index: usize) -> AudioBuffer {
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
    ///     buffer.channel_data(0),
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

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut Arc::make_mut(&mut self.data)[..]
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
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.sample_len(), 10);
/// assert_eq!(next.channel_data(0), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
///     1., 2., 3., 4., 5.,
/// ]));
///
/// // second chunk contains 5 samples of signal, and 5 silent
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.sample_len(), 10);
/// assert_eq!(next.channel_data(0), &ChannelData::from(vec![
///     1., 2., 3., 4., 5.,
///     0., 0., 0., 0., 0.,
/// ]));
///
/// // no further chunks
/// assert!(resampler.next().is_none());
/// ```
pub struct Resampler<I> {
    /// desired sample rate
    sample_rate: SampleRate,
    /// desired sample length
    sample_len: usize,
    /// input stream
    input: I,
    /// internal buffer
    buffer: Option<AudioBuffer>,
}

impl<M: MediaStream> Resampler<M> {
    pub fn new(sample_rate: SampleRate, sample_len: usize, input: M) -> Self {
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

        while buffer.sample_len() < self.sample_len {
            // buffer is smaller than desired len
            match self.input.next() {
                None => {
                    let padding = AudioBuffer::new(
                        buffer.number_of_channels(),
                        self.sample_len - buffer.sample_len(),
                        self.sample_rate,
                    );
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

        if buffer.sample_len() == self.sample_len {
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
        let b = AudioBuffer::new(2, 10, SampleRate(44_100));

        assert_eq!(b.sample_len(), 10);
        assert_eq!(b.number_of_channels(), 2);
        assert_eq!(b.sample_rate().0, 44_100);
        assert_float_eq!(b.channel_data(0).as_slice(), &[0.; 10][..], abs_all <= 0.);
        assert_float_eq!(b.channel_data(1).as_slice(), &[0.; 10][..], abs_all <= 0.);
        assert_eq!(b.channels().get(2), None);
    }

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
        assert_float_eq!(
            b1.channel_data(0).as_slice(),
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
            abs_all <= 0.
        );

        let split = b1.split(8);
        assert_float_eq!(
            split[0].channel_data(0).as_slice(),
            &[0., 0., 0., 0., 0., 0., 0., 0.][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            split[1].channel_data(0).as_slice(),
            &[0., 0., 1., 1., 1., 1., 1.][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_resample_upmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(100));
        buffer.resample(SampleRate(200));
        assert_float_eq!(
            buffer.channel_data(0).as_slice(),
            &[1., 1., 2., 2., 3., 3., 4., 4., 5., 5.,][..],
            abs_all <= 0.
        );
        assert_eq!(buffer.sample_rate().0, 200);
    }

    #[test]
    fn test_resample_downmix() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(200));
        buffer.resample(SampleRate(100));
        assert_float_eq!(
            buffer.channel_data(0).as_slice(),
            &[2., 4.][..],
            abs_all <= 0.
        );
        assert_eq!(buffer.sample_rate().0, 100);
    }

    #[test]
    fn test_resampler_concat() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let input_buf = AudioBuffer::from_channels(vec![channel], SampleRate(44_100));
        let input = vec![input_buf; 3].into_iter().map(Ok);
        let mut resampler = Resampler::new(SampleRate(44_100), 10, input);

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.sample_len(), 10);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[1., 2., 3., 4., 5., 1., 2., 3., 4., 5.,][..],
            abs_all <= 0.
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.sample_len(), 10);
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
        let input_buf = Ok(AudioBuffer::from_channels(
            vec![channel],
            SampleRate(44_100),
        ));
        let input = vec![input_buf].into_iter();
        let mut resampler = Resampler::new(SampleRate(44_100), 5, input);

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.sample_len(), 5);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[1., 2., 3., 4., 5.][..],
            abs_all <= 0.
        );

        let next = resampler.next().unwrap().unwrap();
        assert_eq!(next.sample_len(), 5);
        assert_float_eq!(
            next.channel_data(0).as_slice(),
            &[6., 7., 8., 9., 10.][..],
            abs_all <= 0.
        );

        assert!(resampler.next().is_none());
    }
}
