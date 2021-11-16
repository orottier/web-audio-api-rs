//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```no_run
//! use std::fs::File;
//! use web_audio_api::context::{AsBaseAudioContext, AudioContext};
//! use web_audio_api::media::{MediaElement, OggVorbisDecoder};
//! use web_audio_api::node::{AudioNode, AudioControllableSourceNode, AudioScheduledSourceNode};
//!
//! let context = AudioContext::new(None);
//!
//! // setup background music:
//! // read from local file
//! let file = File::open("sample.ogg").unwrap();
//! // decode file to media stream
//! let stream = OggVorbisDecoder::try_new(file).unwrap();
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
//! //std::thread::sleep(std::time::Duration::from_secs(4));
//! ```

/// Render quantum size as u32 (audio graph is rendered in blocks of this size)
///
/// derive BUFFER_SIZE from this const to prevent downcasting from `usize` to `u32`
/// (cf. discussion [https://github.com/orottier/web-audio-api-rs/pull/52#issuecomment-969156219])
pub const BUFFER_SIZE_U32: u32 = 128;

/// Render quantum size (audio graph is rendered in blocks of this size)
pub const BUFFER_SIZE: usize = BUFFER_SIZE_U32 as usize;

/// Maximum number of channels for audio processing
pub const MAX_CHANNELS: usize = 32;

pub mod alloc;
pub mod buffer;
pub mod context;
pub mod control;
pub mod media;
pub mod node;
pub mod param;
pub mod process;
pub mod spatial;

#[cfg(test)]
mod snapshot;

#[cfg(not(test))]
mod io;

mod analysis;
mod graph;
mod message;

/// Number of samples processed per second (Hertz) for a single channel of audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SampleRate(pub u32);

/// Input/output with this index does not exist
#[derive(Debug, Clone, Copy)]
pub struct IndexSizeError {}

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

impl fmt::Display for IndexSizeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for IndexSizeError {}

/// Media stream buffering lags behind
#[derive(Debug, Clone, Copy)]
pub struct BufferDepletedError {}

impl fmt::Display for BufferDepletedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for BufferDepletedError {}

/// Atomic float, only `load` and `store` are supported, no arithmetics
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

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_atomic_f64() {
        let f = AtomicF64::new(2.0);
        assert_float_eq!(f.load(), 2.0, ulps <= 0);

        f.store(3.0);
        assert_float_eq!(f.load(), 3.0, ulps <= 0);

        let prev = f.swap(4.0);
        assert_float_eq!(prev, 3.0, ulps <= 0);
        assert_float_eq!(f.load(), 4.0, ulps <= 0);
    }
}
