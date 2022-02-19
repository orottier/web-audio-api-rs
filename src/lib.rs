//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```no_run
//! use std::fs::File;
//! use web_audio_api::context::{BaseAudioContext, AudioContext};
//! use web_audio_api::media::{MediaElement, MediaDecoder};
//! use web_audio_api::node::{AudioNode, AudioControllableSourceNode, AudioScheduledSourceNode};
//!
//! let context = AudioContext::new(None);
//!
//! // setup background music:
//! // read from local file
//! let file = File::open("samples/major-scale.ogg").unwrap();
//! // decode file to media stream
//! let stream = MediaDecoder::try_new(file).unwrap();
//! // wrap stream in MediaElement, so we can control it (loop, play/pause)
//! let mut media = MediaElement::new(stream);
//! // register as media element in the audio context
//! let background = context.create_media_element_source(media);
//! // use a gain node to control volume
//! let gain = context.create_gain();
//! // play at low volume
//! gain.gain().set_value(0.5);
//! // connect the media node to the gain node
//! background.connect(&gain);
//! // connect the gain node to the destination node (speakers)
//! gain.connect(&context.destination());
//! // start playback
//! background.set_loop(true);
//! background.start();
//!
//! // mix in an oscillator sound
//! let osc = context.create_oscillator();
//! osc.connect(&context.destination());
//! osc.start();
//!
//! // enjoy listening
//! std::thread::sleep(std::time::Duration::from_secs(4));
//! ```

use std::convert::TryFrom;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Render quantum size, the audio graph is rendered in blocks of RENDER_QUANTUM_SIZE samples
/// see. <https://webaudio.github.io/web-audio-api/#render-quantum>
pub const RENDER_QUANTUM_SIZE: usize = 128;

/// Maximum number of channels for audio processing
pub const MAX_CHANNELS: usize = 32;

pub mod buffer;
pub mod context;
pub mod control;
pub mod media;
pub mod node;
pub mod param;
pub mod periodic_wave;
pub mod render;
pub mod spatial;

#[cfg(test)]
mod snapshot;

#[cfg(not(test))]
mod io;

mod analysis;
mod message;

/// Number of samples processed per second (Hertz) for a single channel of audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SampleRate(pub u32);

/// Media stream buffering lags behind
#[derive(Debug, Clone, Copy)]
pub struct BufferDepletedError {}

impl fmt::Display for BufferDepletedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for BufferDepletedError {}

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

    pub fn swap(&self, v: f64) -> f64 {
        let prev = self
            .inner
            .swap(u64::from_ne_bytes(v.to_ne_bytes()), Ordering::SeqCst);
        f64::from_ne_bytes(prev.to_ne_bytes())
    }
}

/// Utility functions for arguments sanity check
pub(crate) fn assert_is_valid_sample_rate(sample_rate: SampleRate) {
    let sample_rate = sample_rate.0;
    // allow arbitrary sample rates in tests
    if cfg!(test) {
        if sample_rate == 0 {
            panic!(
                "Invalid sample rate: {:?} is negative or zero (test mode)",
                sample_rate
            );
        }
    } else {
        // An implementation MUST support sample rates in at least the range 8000 to 96000
        // @note - `test/media.rs` relies on `RENDER_QUANTUM_SIZE` this should be
        // cleaned together with the MediaElement, see pull #106
        if sample_rate < u32::try_from(RENDER_QUANTUM_SIZE).unwrap() || sample_rate > 96000 {
            panic!(
                "Invalid sample rate: {:?} is outside range [8000, 96000]",
                sample_rate
            );
        }
    }
}

pub(crate) fn assert_is_valid_number_of_channels(number_of_channels: usize) {
    if number_of_channels == 0 || number_of_channels > MAX_CHANNELS {
        panic!(
            "Invalid number of channels: {:?} is outside range [1, 32]",
            number_of_channels
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

        let prev = f.swap(4.0);
        assert_float_eq!(prev, 3.0, abs <= 0.);
        assert_float_eq!(f.load(), 4.0, abs <= 0.);
    }

    #[test]
    #[should_panic]
    fn test_invalid_sample_rate() {
        let sample_rate = SampleRate(0);
        assert_is_valid_sample_rate(sample_rate);
    }

    #[test]
    fn test_valid_sample_rate() {
        let sample_rate = SampleRate(1);
        assert_is_valid_sample_rate(sample_rate);
    }

    #[test]
    #[should_panic]
    fn test_invalid_number_of_channels_min() {
        assert_is_valid_number_of_channels(0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_number_of_channels_max() {
        assert_is_valid_number_of_channels(33);
    }

    #[test]
    fn test_valid_number_of_channels() {
        assert_is_valid_number_of_channels(1);
        assert_is_valid_number_of_channels(32);
    }
}
