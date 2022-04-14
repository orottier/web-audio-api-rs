//! General purpose audio signal data structures
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::perf)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::Arc;

use crate::render::AudioRenderQuantum;
use crate::{
    assert_valid_channel_number, assert_valid_number_of_channels, assert_valid_sample_rate,
    SampleRate,
};

/// Options for constructing an [`AudioBuffer`]
// dictionary AudioBufferOptions {
//   unsigned long numberOfChannels = 1;
//   required unsigned long length;
//   required float sampleRate;
// };
#[derive(Clone, Debug)]
pub struct AudioBufferOptions {
    /// The number of channels for the buffer
    pub number_of_channels: usize,
    /// The length in sample frames of the buffer
    pub length: usize,
    /// The sample rate in Hz for the buffer
    pub sample_rate: SampleRate,
}

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An `AudioBuffer` has copy-on-write semantics, so it is cheap to clone.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/AudioBuffer>
/// - specification: <https://webaudio.github.io/web-audio-api/#AudioBuffer>
/// - see also: [`BaseAudioContext::create_buffer`](crate::context::BaseAudioContext::create_buffer)
///
/// # Usage
///
/// ```no_run
/// use std::f32::consts::PI;
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// let length = context.sample_rate() as usize;
/// let sample_rate = context.sample_rate_raw();
/// let mut buffer = context.create_buffer(1, length, sample_rate);
///
/// // fill buffer with a sine wave
/// let mut sine = vec![];
///
/// for i in 0..length {
///     let phase = i as f32 / length as f32 * 2. * PI * 200.;
///     sine.push(phase.sin());
/// }
///
/// buffer.copy_to_channel(&sine, 0);
///
/// // play the buffer in a loop
/// let src = context.create_buffer_source();
/// src.set_buffer(buffer.clone());
/// src.set_loop(true);
/// src.connect(&context.destination());
/// src.start();
/// ```
///
/// # Example
///
/// - `cargo run --release --example audio_buffer`
///
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    channels: Vec<ChannelData>,
    sample_rate: SampleRate,
}

impl AudioBuffer {
    /// Allocate a silent audiobuffer with [`AudioBufferOptions`]
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given sample rate is zero
    /// - the given number of channels is outside the [1, 32] range,
    /// 32 being defined by the `MAX_CHANNELS` constant.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(options: AudioBufferOptions) -> Self {
        assert_valid_sample_rate(options.sample_rate);
        assert_valid_number_of_channels(options.number_of_channels);

        let silence = ChannelData::new(options.length);

        Self {
            channels: vec![silence; options.number_of_channels],
            sample_rate: options.sample_rate,
        }
    }

    /// Convert raw samples to an `AudioBuffer`
    ///
    /// The outer Vec determine the channels. The inner Vecs should have the same length.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given sample rate is zero
    /// - the given number of channels defined by `samples.len()`is outside the
    ///   [1, 32] range, 32 being defined by the `MAX_CHANNELS` constant.
    /// - any of its items have different lengths
    #[must_use]
    pub fn from(samples: Vec<Vec<f32>>, sample_rate: SampleRate) -> Self {
        assert_valid_sample_rate(sample_rate);
        assert_valid_number_of_channels(samples.len());

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
    #[must_use]
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel in this `AudioBuffer`
    #[must_use]
    pub fn length(&self) -> usize {
        self.channels.get(0).map_or(0, ChannelData::len)
    }

    /// Sample rate of this `AudioBuffer` in Hertz
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate.0 as f32
    }

    /// The raw sample rate of the `AudioBuffer` (which has more precision than the float
    /// [`sample_rate()`](AudioBuffer::sample_rate) value).
    #[must_use]
    pub fn sample_rate_raw(&self) -> SampleRate {
        self.sample_rate
    }

    /// Duration in seconds of the `AudioBuffer`
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration(&self) -> f64 {
        self.length() as f64 / f64::from(self.sample_rate.0)
    }

    /// Copy data from a given channel to the given `Vec`
    ///
    /// # Panics
    ///
    /// This function will panic if `channel_number` is greater or equal than
    /// `AudioBuffer::number_of_channels()`
    pub fn copy_from_channel(&self, destination: &mut [f32], channel_number: usize) {
        assert_valid_channel_number(channel_number, self.number_of_channels());

        self.copy_from_channel_with_offset(destination, channel_number, 0);
    }

    /// Copy data from a given channel to the given `Vec` starting at `offset`
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given channel number is greater than or equal to the given number of channels.
    pub fn copy_from_channel_with_offset(
        &self,
        destination: &mut [f32],
        channel_number: usize,
        offset: usize,
    ) {
        assert_valid_channel_number(channel_number, self.number_of_channels());
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
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given channel number is greater than or equal to the given number of channels.
    pub fn copy_to_channel(&mut self, source: &[f32], channel_number: usize) {
        assert_valid_channel_number(channel_number, self.number_of_channels());

        self.copy_to_channel_with_offset(source, channel_number, 0);
    }

    /// Copy data from a given source to the given channel starting at `offset`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given channel number is greater than or equal to the given number of channels.
    pub fn copy_to_channel_with_offset(
        &mut self,
        source: &[f32],
        channel_number: usize,
        offset: usize,
    ) {
        assert_valid_channel_number(channel_number, self.number_of_channels());
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
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given channel number is greater than or equal to the given number of channels.
    #[must_use]
    pub fn get_channel_data(&self, channel_number: usize) -> &[f32] {
        assert_valid_channel_number(channel_number, self.number_of_channels());
        // [spec] According to the rules described in acquire the content either allow writing
        // into or getting a copy of the bytes stored in [[internal data]] in a new Float32Array
        self.channel_data(channel_number).as_slice()
    }

    /// Create a multi-channel audiobuffer directly from `ChannelData`s.
    // @todo - remove in favor of `AudioBuffer::from`
    #[must_use]
    pub(crate) fn from_channels(channels: Vec<ChannelData>, sample_rate: SampleRate) -> Self {
        Self {
            channels,
            sample_rate,
        }
    }

    /// Channel data as slice
    #[must_use]
    pub(crate) fn channels(&self) -> &[ChannelData] {
        &self.channels
    }

    /// Channel data as slice (mutable)
    #[must_use]
    pub(crate) fn channels_mut(&mut self) -> &mut [ChannelData] {
        &mut self.channels
    }

    /// Get the samples from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    // @note - this one is used in
    #[must_use]
    pub(crate) fn channel_data(&self, index: usize) -> &ChannelData {
        &self.channels[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    #[must_use]
    pub(crate) fn channel_data_mut(&mut self, index: usize) -> &mut ChannelData {
        &mut self.channels[index]
    }

    /// Extends an `AudioBuffer` with the contents of another.
    ///
    /// This function will panic if the `sample_rate` and `channel_count` are not equal
    pub(crate) fn extend(&mut self, other: &Self) {
        assert_eq!(self.sample_rate, other.sample_rate);
        assert_eq!(self.number_of_channels(), other.number_of_channels());

        let data = self.channels_mut();
        data.iter_mut()
            .zip(other.channels.iter())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend(other_channel.as_slice());
            });
    }

    /// Extends an `AudioBuffer` with an [`AudioRenderQuantum`]
    ///
    /// This assumes the `sample_rate` matches. No up/down-mixing is performed
    pub(crate) fn extend_alloc(&mut self, other: &AudioRenderQuantum) {
        self.channels_mut()
            .iter_mut()
            .zip(other.channels())
            .for_each(|(channel, other_channel)| {
                let cur_channel_data = Arc::make_mut(&mut channel.data);
                cur_channel_data.extend_from_slice(&other_channel[..]);
            });
    }

    /// Split an `AudioBuffer` in two at the given index.
    #[must_use]
    pub(crate) fn split_off(&mut self, index: usize) -> Self {
        let sample_rate = self.sample_rate_raw();

        let channels: Vec<_> = self
            .channels_mut()
            .iter_mut()
            .map(|channel_data| Arc::make_mut(&mut channel_data.data).split_off(index))
            .map(ChannelData::from)
            .collect();

        Self::from_channels(channels, sample_rate)
    }

    /// Resample to the desired sample rate. The method performs a simple linear
    /// interpolation an keep the first and last sample intacts. The new number
    /// of samples is always ceiled according the ratio defined by old and new
    /// sample rates.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - the given sample rate is zero
    ///
    /// ```ignore
    /// use float_eq::assert_float_eq;
    /// use crate::SampleRate;
    /// use crate::buffer::{ChannelData, AudioBuffer};
    ///
    /// let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
    /// let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(48_000));
    ///
    /// // upmix from 48k to 96k Hertz sample rate
    /// buffer.resample(SampleRate(96_000));
    ///
    /// assert_float_eq!(
    ///     buffer.get_channel_data(0)[..],
    ///     [1.0, 1.4444444, 1.8888888, 2.3333335, 2.7777777, 3.2222223, 3.6666667, 4.111111, 4.5555553, 5.0][..],
    ///     abs_all <= 1e-6
    /// );
    ///
    /// assert_eq!(buffer.sample_rate().0, 96_000);
    /// ```
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub(crate) fn resample(&mut self, sample_rate: SampleRate) {
        if self.sample_rate_raw() == sample_rate {
            return;
        }

        assert_valid_sample_rate(sample_rate);

        // handle zero length case
        if self.length() == 0 {
            self.sample_rate = sample_rate;
            return;
        }

        let source_sr = f64::from(self.sample_rate.0);
        let target_sr = f64::from(sample_rate.0);
        let ratio = target_sr / source_sr;
        let source_length = self.length();
        let target_length = (self.length() as f64 * ratio).ceil() as usize;

        let num_channels = self.number_of_channels();
        let mut resampled = Vec::<Vec<f32>>::with_capacity(num_channels);
        resampled.resize_with(num_channels, || Vec::<f32>::with_capacity(target_length));

        for i in 0..target_length {
            let position = i as f64 / (target_length - 1) as f64; // [0., 1.]
            let playhead = position * (source_length - 1) as f64;
            let playhead_floored = playhead.floor();
            let prev_index = playhead_floored as usize;
            let next_index = (prev_index + 1).min(source_length - 1);

            let k = (playhead - playhead_floored) as f32;
            let k_inv = 1. - k;

            for (channel, resampled_data) in resampled.iter_mut().enumerate() {
                let prev_sample = self.channels[channel].data[prev_index];
                let next_sample = self.channels[channel].data[next_index];

                let value = k_inv.mul_add(prev_sample, k * next_sample);
                resampled_data.push(value);
            }
        }

        self.channels
            .iter_mut()
            .zip(resampled)
            .for_each(|(channel_data, resampled_data)| {
                channel_data.data = Arc::new(resampled_data);
            });

        self.sample_rate = sample_rate;
    }
}

/// Single channel audio samples, basically wraps a `Arc<Vec<f32>>`
///
/// `ChannelData` has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChannelData {
    data: Arc<Vec<f32>>,
}

impl ChannelData {
    #[must_use]
    pub fn new(length: usize) -> Self {
        let buffer = vec![0.; length];
        let data = Arc::new(buffer);

        Self { data }
    }

    #[must_use]
    pub fn from(data: Vec<f32>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    // clippy wants to keep it, so keep it :)
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut Arc::make_mut(&mut self.data)[..]
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::convert::TryFrom;
    use std::f32::consts::PI;

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

        let _buf = AudioBuffer::new(options); // should panic
    }

    #[test]
    #[should_panic]
    fn test_zero_channels_from() {
        let samples = vec![];
        let sample_rate = SampleRate(1);

        let _buf = AudioBuffer::from(samples, sample_rate); // should panic
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(0),
        };

        let _buf = AudioBuffer::new(options); // should panic
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate_from() {
        let samples = vec![vec![0.]];
        let sample_rate = SampleRate(0);

        let _buf = AudioBuffer::from(samples, sample_rate); // should panic
    }

    #[test]
    #[should_panic]
    fn test_invalid_copy_from_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        // same size
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel(&mut dest, 1);
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
    #[should_panic]
    fn test_invalid_copy_to_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let mut audio_buffer = AudioBuffer::new(options);

        // same size
        let src = vec![1.; 10];
        audio_buffer.copy_to_channel(&src, 1);
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
            let mut audio_buffer = AudioBuffer::new(options.clone());
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
            let mut audio_buffer = AudioBuffer::new(options.clone());
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
            let mut audio_buffer = AudioBuffer::new(options.clone());
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
            let mut audio_buffer = AudioBuffer::new(options.clone());
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

    #[test]
    #[should_panic]
    fn test_invalid_get_channel_data() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        let _c = audio_buffer.get_channel_data(1);
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
        let mut b1 = AudioBuffer::new(options.clone());
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
    #[should_panic]
    fn test_resample_to_zero_hertz() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(100));
        buffer.resample(SampleRate(0));
    }

    #[test]
    fn test_resample_from_empty() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 0,
            sample_rate: SampleRate(100),
        };
        let mut buffer = AudioBuffer::new(options);
        buffer.resample(SampleRate(200));

        assert_eq!(buffer.length(), 0);
        assert_eq!(buffer.sample_rate_raw().0, 200);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_upsample() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(100));
        buffer.resample(SampleRate(200));

        let mut expected = [0.; 10];
        let incr: f32 = 4. / 9.; // (5 - 1) / (10 - 1)

        for (i, value) in expected.iter_mut().enumerate() {
            *value = incr.mul_add(i as f32, 1.);
        }

        assert_float_eq!(
            buffer.channel_data(0).as_slice(),
            &expected[..],
            abs_all <= 1e-6
        );

        assert_eq!(buffer.sample_rate_raw().0, 200);
    }

    #[test]
    fn test_downsample() {
        let channel = ChannelData::from(vec![1., 2., 3., 4., 5.]);
        let mut buffer = AudioBuffer::from_channels(vec![channel], SampleRate(200));
        buffer.resample(SampleRate(100));

        assert_float_eq!(
            buffer.channel_data(0).as_slice(),
            &[1., 3., 5.][..],
            abs_all <= 0.
        );

        assert_eq!(buffer.sample_rate_raw().0, 100);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_resample_stereo() {
        for sr in &[22500, 38000, 48000, 96000] {
            let source_sr = *sr;
            let target_sr = 44_100;

            let mut left = Vec::<f32>::with_capacity(source_sr);
            let mut right = Vec::<f32>::with_capacity(source_sr);

            for i in 0..source_sr {
                let phase = i as f32 / source_sr as f32 * 2. * PI;
                left.push(phase.sin());
                right.push(phase.cos());
            }

            let left_chan = ChannelData::from(left);
            let right_chan = ChannelData::from(right);
            let mut buffer = AudioBuffer::from_channels(
                vec![left_chan, right_chan],
                SampleRate(u32::try_from(source_sr).unwrap()),
            );
            buffer.resample(SampleRate(target_sr));

            let mut expected_left = vec![];
            let mut expected_right = vec![];

            for i in 0..target_sr {
                let phase = i as f32 / target_sr as f32 * 2. * PI;
                expected_left.push(phase.sin());
                expected_right.push(phase.cos());
            }

            assert_float_eq!(
                buffer.get_channel_data(0)[..],
                &expected_left[..],
                abs_all <= 1e-3
            );

            assert_float_eq!(
                buffer.get_channel_data(1)[..],
                &expected_right[..],
                abs_all <= 1e-3
            );

            assert_eq!(buffer.sample_rate_raw().0, target_sr);
        }
    }
}
