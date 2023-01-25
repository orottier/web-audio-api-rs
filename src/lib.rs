//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```no_run
//! use std::fs::File;
//! use web_audio_api::context::{BaseAudioContext, AudioContext};
//! use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
//!
//! // set up AudioContext with optimized settings for your hardware
//! let context = AudioContext::default();
//!
//! // create an audio buffer from a given file
//! let file = File::open("samples/sample.wav").unwrap();
//! let buffer = context.decode_audio_data_sync(file).unwrap();
//!
//! // play the buffer at given volume
//! let volume = context.create_gain();
//! volume.connect(&context.destination());
//! volume.gain().set_value(0.5);
//!
//! let buffer_source = context.create_buffer_source();
//! buffer_source.connect(&volume);
//! buffer_source.set_buffer(buffer);
//!
//! // create oscillator branch
//! let osc = context.create_oscillator();
//! osc.connect(&context.destination());
//!
//! // start the sources
//! buffer_source.start();
//! osc.start();
//!
//! // enjoy listening
//! std::thread::sleep(std::time::Duration::from_secs(4));
//! ```

#![warn(clippy::missing_panics_doc)]
#![deny(trivial_numeric_casts)]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Render quantum size, the audio graph is rendered in blocks of RENDER_QUANTUM_SIZE samples
/// see. <https://webaudio.github.io/web-audio-api/#render-quantum>
pub(crate) const RENDER_QUANTUM_SIZE: usize = 128;

/// Maximum number of channels for audio processing
pub const MAX_CHANNELS: usize = 32;

mod buffer;
pub use buffer::*;

mod capacity;
pub use capacity::*;

pub mod context;
pub(crate) mod control;
pub mod media;
pub mod node;

mod events;
pub use events::Event;

mod param;
pub use param::*;

mod periodic_wave;
pub use periodic_wave::*;

pub mod render;

mod sample;
pub(crate) use sample::Sample;

mod spatial;
pub use spatial::AudioListener;

mod io;
pub use io::{enumerate_devices, MediaDeviceInfo, MediaDeviceInfoKind};

mod analysis;
mod message;

#[derive(Debug)]
pub (crate) struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self {
            inner: AtomicU32::new(u32::from_ne_bytes(v.to_ne_bytes())),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_ne_bytes(self.inner.load(ordering).to_ne_bytes())
    }

    pub fn store(&self, v: f32, ordering: Ordering) {
        self.inner
            .store(u32::from_ne_bytes(v.to_ne_bytes()), ordering);
    }
}

/// Atomic float 64, only `load` and `store` are supported, no arithmetics
#[derive(Debug)]
pub(crate) struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    pub fn new(v: f64) -> Self {
        Self {
            inner: AtomicU64::new(u64::from_ne_bytes(v.to_ne_bytes())),
        }
    }

    pub fn load(&self) -> f64 {
        f64::from_ne_bytes(self.inner.load(Ordering::SeqCst).to_ne_bytes())
    }

    pub fn store(&self, v: f64) {
        self.inner
            .store(u64::from_ne_bytes(v.to_ne_bytes()), Ordering::SeqCst)
    }
}

/// Assert that the given sample rate is valid.
///
/// Note that in practice sample rates should stand between 8000Hz (lower bound for
/// voice based applications, e.g. see phone bandwidth) and 96000Hz (for very high
/// quality audio applications and spectrum manipulation).
/// Most common sample rates for musical applications are 44100 and 48000.
/// - see <https://webaudio.github.io/web-audio-api/#dom-baseaudiocontext-createbuffer-samplerate>
///
/// # Panics
///
/// This function will panic if:
/// - the given sample rate is zero
///
#[track_caller]
#[inline(always)]
pub(crate) fn assert_valid_sample_rate(sample_rate: f32) {
    // 1000 Hertz is a just a random cutoff, but it helps a if someone accidentally puts a
    // timestamp in the sample_rate variable
    if sample_rate <= 1000. {
        panic!(
            "NotSupportedError - Invalid sample rate: {:?}, should be greater than 1000",
            sample_rate
        );
    }
}

/// Assert that the given number of channels is valid.
///
/// # Panics
///
/// This function will panic if:
/// - the given number of channels is outside the [1, 32] range,
/// 32 being defined by the MAX_CHANNELS constant.
///
#[track_caller]
#[inline(always)]
pub(crate) fn assert_valid_number_of_channels(number_of_channels: usize) {
    if number_of_channels == 0 || number_of_channels > MAX_CHANNELS {
        panic!(
            "NotSupportedError - Invalid number of channels: {:?} is outside range [1, {:?}]",
            number_of_channels, MAX_CHANNELS
        );
    }
}

/// Assert that the given channel number is valid according the number of channel
/// of an Audio asset (e.g. [`AudioBuffer`](crate::buffer::AudioBuffer))
///
/// # Panics
///
/// This function will panic if:
/// - the given channel number is greater than or equal to the given number of channels.
///
#[track_caller]
#[inline(always)]
pub(crate) fn assert_valid_channel_number(channel_number: usize, number_of_channels: usize) {
    if channel_number >= number_of_channels {
        panic!(
            "IndexSizeError - Invalid channel number {:?} (number of channels: {:?})",
            channel_number, number_of_channels
        );
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_atomic_f64() {
        let f = AtomicF64::new(2.0);
        assert_float_eq!(f.load(), 2.0, abs <= 0.);

        f.store(3.0);
        assert_float_eq!(f.load(), 3.0, abs <= 0.);
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate_zero() {
        assert_valid_sample_rate(0.);
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate_subzero() {
        assert_valid_sample_rate(-48000.);
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate_too_small() {
        assert_valid_sample_rate(100.);
    }

    #[test]
    fn test_valid_sample_rate() {
        assert_valid_sample_rate(48000.);
    }

    #[test]
    #[should_panic]
    fn test_invalid_number_of_channels_min() {
        assert_valid_number_of_channels(0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_number_of_channels_max() {
        assert_valid_number_of_channels(33);
    }

    #[test]
    fn test_valid_number_of_channels() {
        assert_valid_number_of_channels(1);
        assert_valid_number_of_channels(32);
    }
}
