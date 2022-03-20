//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```no_run
//! use std::fs::File;
//! use web_audio_api::context::{BaseAudioContext, AudioContext};
//! use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
//!
//! let context = AudioContext::new(None);
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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Render quantum size, the audio graph is rendered in blocks of RENDER_QUANTUM_SIZE samples
/// see. <https://webaudio.github.io/web-audio-api/#render-quantum>
pub const RENDER_QUANTUM_SIZE: usize = 128;

/// Maximum number of channels for audio processing
pub const MAX_CHANNELS: usize = 32;

pub mod buffer;
pub mod context;
pub(crate) mod control;
pub mod media;
pub mod node;
pub mod param;
pub mod periodic_wave;
pub mod render;

mod spatial;
pub use spatial::AudioListener;

#[cfg(test)]
mod snapshot;

#[cfg(not(test))]
mod io;

mod analysis;
mod message;

/// Number of samples processed per second (Hertz) for a single channel of audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SampleRate(pub u32);

/// Atomic float 32, only `load` and `store` are supported, no arithmetics
#[derive(Debug)]
pub(crate) struct AtomicF32 {
    inner: AtomicU32,
}

// `swap()` is not implemented as `AtomicF32` is only used in `param.rs` for now
impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self {
            inner: AtomicU32::new(u32::from_ne_bytes(v.to_ne_bytes())),
        }
    }

    pub fn load(&self) -> f32 {
        f32::from_ne_bytes(self.inner.load(Ordering::SeqCst).to_ne_bytes())
    }

    pub fn store(&self, v: f32) {
        self.inner
            .store(u32::from_ne_bytes(v.to_ne_bytes()), Ordering::SeqCst)
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
pub(crate) fn assert_valid_sample_rate(sample_rate: SampleRate) {
    if sample_rate.0 == 0 {
        panic!(
            "NotSupportedError - Invalid sample rate: {:?}, should be strictly positive",
            sample_rate.0
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
    fn test_invalid_sample_rate() {
        let sample_rate = SampleRate(0);
        assert_valid_sample_rate(sample_rate);
    }

    #[test]
    fn test_valid_sample_rate() {
        let sample_rate = SampleRate(1);
        assert_valid_sample_rate(sample_rate);
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
