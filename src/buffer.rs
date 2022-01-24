//! General purpose audio signal data structures
use std::sync::Arc;

use crate::media::MediaStream;
use crate::render::AudioRenderQuantum;
use crate::SampleRate;

/// Options for constructing an [`AudioBuffer`]
#[derive(Copy, Clone, Debug)]
pub struct AudioBufferOptions {
    pub number_of_channels: usize, // defaults to 1
    pub length: usize,             // required
    pub sample_rate: SampleRate,   // required
}

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/AudioBuffer>
/// - specification: <https://webaudio.github.io/web-audio-api/#AudioBuffer>
/// - see also: [`BaseAudioContext::create_buffer`](crate::context::BaseAudioContext::create_buffer)
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    channels: Vec<ChannelData>,
    sample_rate: SampleRate,
}

use std::error::Error;

impl AudioBuffer {
    /// Allocate a silent audiobuffer with [`AudioBufferOptions`]
    ///
    /// # Panics
    ///
    /// This function will panic if the channel count is zero.
    pub fn new(options: AudioBufferOptions) -> Self {
        assert!(options.number_of_channels > 0);

        let silence = ChannelData::new(options.length);

        Self {
            channels: vec![silence; options.number_of_channels],
            sample_rate: options.sample_rate,
        }
    }

    /// Convert raw samples to an AudioBuffer
    ///
    /// The outer Vec determine the channels. The inner Vecs should have the same length.
    ///
    /// # Panics
    ///
    /// This function will panic if `samples` is an empty Vec or any of its items have different
    /// lengths.
    pub fn from(samples: Vec<Vec<f32>>, sample_rate: SampleRate) -> Self {
        assert!(!samples.is_empty());

        let channels: Vec<_> = samples.into_iter().map(ChannelData::from).collect();
        if !channels.iter().all(|c| c.len() == channels[0].len()) {
            panic!("Trying to create AudioBuffer from channel data with unequal length");
        }
        Self {
            channels,
            sample_rate,
        }
    }

    /// Number of channels in this `AudioBuffer`
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel in this `AudioBuffer`
    pub fn length(&self) -> usize {
        self.channels.get(0).map(ChannelData::len).unwrap_or(0)
    }

    /// Sample rate of this `AudioBuffer` in Hertz
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate.0 as f32
    }

    /// The raw sample rate of the `AudioBuffer` (which has more precision than the float
    /// [`sample_rate()`](AudioBuffer::sample_rate) value).
    pub fn sample_rate_raw(&self) -> SampleRate {
        self.sample_rate
    }

    /// Duration in seconds of the `AudioBuffer`
    pub fn duration(&self) -> f64 {
        self.length() as f64 / self.sample_rate.0 as f64
    }

    /// Copy data from a given channel to the given `Vec`
    pub fn copy_from_channel(&self, destination: &mut [f32], channel_number: usize) {
        self.copy_from_channel_with_offset(destination, channel_number, 0);
    }

    /// Copy data from a given channel to the given `Vec` starting at `offset`
    pub fn copy_from_channel_with_offset(
        &self,
        destination: &mut [f32],
        channel_number: usize,
        offset: usize,
    ) {
        let offset = offset.min(self.length());
        // [spec] Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number
        // of elements in the destination array, and 𝑘 be the value of bufferOffset.
        // Then the number of frames copied from buffer to destination is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of destination are not modified.
        let dest_length = destination.len();
        let max_frame = (self.length() - offset).min(dest_length).max(0);
        let channel = self.channel_data(channel_number).as_slice();

        destination[..max_frame].copy_from_slice(&channel[offset..(max_frame + offset)]);
    }

    /// Copy data from a given source to the given channel.
    pub fn copy_to_channel(&mut self, source: &[f32], channel_number: usize) {
        self.copy_to_channel_with_offset(source, channel_number, 0);
    }

    /// Copy data from a given source to the given channel starting at `offset`.
    pub fn copy_to_channel_with_offset(
        &mut self,
        source: &[f32],
        channel_number: usize,
        offset: usize,
    ) {
        let offset = offset.min(self.length());
        // [spec] Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number
        // of elements in the source array, and 𝑘 be the value of bufferOffset. Then
        // the number of frames copied from source to the buffer is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of buffer are not modified.
        let src_len = source.len();
        let max_frame = (self.length() - offset).min(src_len).max(0);
        let channel = self.channel_data_mut(channel_number).as_mut_slice();

        channel[offset..(max_frame + offset)].copy_from_slice(&source[..max_frame]);
    }

    /// Return a read-only copy of the underlying data of the channel
    pub fn get_channel_data(&self, channel_number: usize) -> &[f32] {
        // [spec] According to the rules described in acquire the content either allow writing
        // into or getting a copy of the bytes stored in [[internal data]] in a new Float32Array
        self.channel_data(channel_number).as_slice()
    }

    /// Create a multi-channel audiobuffer directly from `ChannelData`s.
    // @todo - remove in favor of `AudioBuffer::from`
    pub(crate) fn from_channels(channels: Vec<ChannelData>, sample_rate: SampleRate) -> Self {
        Self {
            channels,
            sample_rate,
        }
    }

    /// Channel data as slice
    pub(crate) fn channels(&self) -> &[ChannelData] {
        &self.channels
    }

    /// Channel data as slice (mutable)
    pub(crate) fn channels_mut(&mut self) -> &mut [ChannelData] {
        &mut self.channels
    }

    /// Get the samples from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    // @note - this one is used in
    pub(crate) fn channel_data(&self, index: usize) -> &ChannelData {
        &self.channels[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub(crate) fn channel_data_mut(&mut self, index: usize) -> &mut ChannelData {
        &mut self.channels[index]
    }

    /// Modify every channel in the same way
    pub(crate) fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        // todo, optimize for Arcs that are equal
        self.channels.iter_mut().for_each(fun)
    }

    /// Extends an AudioBuffer with the contents of another.
    ///
    /// This function will panic if the sample_rate and channel_count are not equal
    pub(crate) fn extend(&mut self, other: &Self) {
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
    pub(crate) fn extend_alloc(&mut self, other: &AudioRenderQuantum) {
        self.channels_mut()
            .iter_mut()
            .zip(other.channels())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend_from_slice(&other_channel[..]);
            })
    }

    /// Split an AudioBuffer in two at the given index.
    pub(crate) fn split_off(&mut self, index: usize) -> Self {
        let sample_rate = self.sample_rate_raw();

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
    //
    // ```
    // use web_audio_api::SampleRate;
    // use web_audio_api::buffer::{ChannelData, AudioBuffer};
    //
    // let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
    // let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(48_000));
    //
    // // upmix from 48k to 96k Hertz sample rate
    // buffer.resample(SampleRate(96_000));
    //
    // assert_eq!(
    //     buffer.get_channel_data(0)[..],
    //     vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.][..]
    // );
    //
    // assert_eq!(buffer.sample_rate().0, 96_000);
    // ```
    pub(crate) fn resample(&mut self, sample_rate: SampleRate) {
        if self.sample_rate_raw() == sample_rate {
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
pub(crate) struct ChannelData {
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

    // clippy wants to keep it, so keep it :)
    #[allow(dead_code)]
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

/// Sample rate converter and buffer chunk splitter.
///
/// A `MediaElement` can be wrapped inside a `Resampler` to yield AudioBuffers of the desired sample_rate and length
///
/// ```
/// use web_audio_api::SampleRate;
/// use web_audio_api::buffer::{AudioBuffer, Resampler};
///
/// // construct an input of 3 chunks of 5 samples
/// let samples = vec![vec![1., 2., 3., 4., 5.]];
/// let input_buf = AudioBuffer::from(samples, SampleRate(44_100));
/// let input = vec![input_buf; 3].into_iter().map(|b| Ok(b));
///
/// // resample to chunks of 10 samples
/// let mut resampler = Resampler::new(SampleRate(44_100), 10, input);
///
/// // first chunk contains 10 samples
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.length(), 10);
/// assert_eq!(next.get_channel_data(0)[..], vec![
///     1., 2., 3., 4., 5.,
///     1., 2., 3., 4., 5.,
/// ][..]);
///
/// // second chunk contains 5 samples of signal, and 5 silent
/// let next = resampler.next().unwrap().unwrap();
/// assert_eq!(next.length(), 10);
/// assert_eq!(next.get_channel_data(0)[..], vec![
///     1., 2., 3., 4., 5.,
///     0., 0., 0., 0., 0.,
/// ][..]);
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
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

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

        while buffer.length() < self.sample_len {
            // buffer is smaller than desired len
            match self.input.next() {
                None => {
                    let options = AudioBufferOptions {
                        number_of_channels: buffer.number_of_channels(),
                        length: self.sample_len - buffer.length(),
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

        if buffer.length() == self.sample_len {
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

    // public WebAudio API
    #[test]
    fn test_constructor() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        assert_eq!(audio_buffer.number_of_channels(), 1);
        assert_eq!(audio_buffer.length(), 10);
        assert_float_eq!(audio_buffer.sample_rate(), 1., abs <= 0.);
        assert_eq!(audio_buffer.sample_rate_raw().0, 1);
        assert_float_eq!(audio_buffer.duration(), 10., abs <= 0.);
    }

    #[test]
    #[should_panic]
    fn test_zero_channels() {
        let options = AudioBufferOptions {
            number_of_channels: 0,
            length: 10,
            sample_rate: SampleRate(1),
        };

        AudioBuffer::new(options); // should panic
    }

    #[test]
    #[should_panic]
    fn test_zero_channels_from() {
        let samples = vec![];
        let sample_rate = SampleRate(1);

        AudioBuffer::from(samples, sample_rate); // should panic
    }

    #[test]
    fn test_copy_from_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        // same size
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(dest[..], vec![0.; 10][..], abs_all <= 0.);

        // smaller destination
        let mut dest = vec![1.; 5];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(dest[..], [0., 0., 0., 0., 0.][..], abs_all <= 0.);

        // larger destination
        let mut dest = vec![1.; 11];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(
            dest[..],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.][..],
            abs_all <= 0.
        );

        // with offset
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel_with_offset(&mut dest, 0, 5);
        assert_float_eq!(
            dest[..],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
            abs_all <= 0.
        );

        // w/ offset ouside range
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel_with_offset(&mut dest, 0, usize::MAX);

        assert_float_eq!(dest[..], vec![1.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_copy_to_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        {
            // same size
            let mut audio_buffer = AudioBuffer::new(options);
            let src = vec![1.; 10];
            audio_buffer.copy_to_channel(&src, 0);
            assert_float_eq!(
                audio_buffer.channel_data(0).as_slice()[..],
                [1.; 10][..],
                abs_all <= 0.
            );
        }

        {
            // smaller source
            let mut audio_buffer = AudioBuffer::new(options);
            let src = vec![1.; 5];
            audio_buffer.copy_to_channel(&src, 0);
            assert_float_eq!(
                audio_buffer.channel_data(0).as_slice()[..],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.][..],
                abs_all <= 0.
            );
        }

        {
            // larger source
            let mut audio_buffer = AudioBuffer::new(options);
            let src = vec![1.; 12];
            audio_buffer.copy_to_channel(&src, 0);
            assert_float_eq!(
                audio_buffer.channel_data(0).as_slice()[..],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.][..],
                abs_all <= 0.
            );
        }

        {
            // w/ offset
            let mut audio_buffer = AudioBuffer::new(options);
            let src = vec![1.; 10];
            audio_buffer.copy_to_channel_with_offset(&src, 0, 5);
            assert_float_eq!(
                audio_buffer.channel_data(0).as_slice()[..],
                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
                abs_all <= 0.
            );
        }

        {
            // w/ offset ouside range
            let mut audio_buffer = AudioBuffer::new(options);
            let src = vec![1.; 10];
            audio_buffer.copy_to_channel_with_offset(&src, 0, usize::MAX);
            assert_float_eq!(
                audio_buffer.channel_data(0).as_slice()[..],
                [0.; 10][..],
                abs_all <= 0.
            );
        }
    }

    // internal API
    #[test]
    fn test_silent() {
        let options = AudioBufferOptions {
            number_of_channels: 2,
            length: 10,
            sample_rate: SampleRate(44_100),
        };
        let b = AudioBuffer::new(options);

        assert_eq!(b.length(), 10);
        assert_eq!(b.number_of_channels(), 2);
        assert_eq!(b.sample_rate_raw().0, 44_100);
        assert_float_eq!(b.channel_data(0).as_slice(), &[0.; 10][..], abs_all <= 0.);
        assert_float_eq!(b.channel_data(1).as_slice(), &[0.; 10][..], abs_all <= 0.);
        assert_eq!(b.channels().get(2), None);
    }

    #[test]
    fn test_concat() {
        let options = AudioBufferOptions {
            number_of_channels: 2,
            length: 5,
            sample_rate: SampleRate(44_100),
        };
        let mut b1 = AudioBuffer::new(options);
        let b2 = AudioBuffer::new(options);
        b1.extend(&b2);

        assert_eq!(b1.length(), 10);
        assert_eq!(b1.number_of_channels(), 2);
        assert_eq!(b1.sample_rate_raw().0, 44_100);

        let channel_data = ChannelData::from(vec![1.; 5]);
        let b3 = AudioBuffer::from_channels(vec![channel_data; 2], SampleRate(44_100));

        b1.extend(&b3);

        assert_eq!(b1.length(), 15);
        assert_eq!(b1.number_of_channels(), 2);
        assert_eq!(b1.sample_rate_raw().0, 44_100);
        assert_float_eq!(
            b1.channel_data(0).as_slice(),
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
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
        assert_eq!(buffer.sample_rate_raw().0, 200);
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
        assert_eq!(buffer.sample_rate_raw().0, 100);
    }

    #[test]
    fn test_resampler_concat() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let input_buf = AudioBuffer::from_channels(vec![channel], SampleRate(44_100));
        let input = vec![input_buf; 3].into_iter().map(Ok);
        let mut resampler = Resampler::new(SampleRate(44_100), 10, input);

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
        let input_buf = Ok(AudioBuffer::from_channels(
            vec![channel],
            SampleRate(44_100),
        ));
        let input = vec![input_buf].into_iter();
        let mut resampler = Resampler::new(SampleRate(44_100), 5, input);

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
}
