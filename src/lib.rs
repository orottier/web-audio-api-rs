//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```rust
//! use std::fs::File;
//! use web_audio_api::context::{AsBaseAudioContext, AudioContext};
//! use web_audio_api::media::OggVorbisDecoder;
//! use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
//!
//! let context = AudioContext::new();
//!
//! // play background music
//! let file = File::open("sample.ogg").unwrap();
//! let media = OggVorbisDecoder::try_new(file).unwrap();
//! let background = context.create_media_element_source(media);
//! let gain = context.create_gain();
//! gain.gain().set_value(0.5); // play at low volume
//! background.connect(&gain);
//! gain.connect(&context.destination());
//!
//! // mix in an oscillator sound
//! let osc = context.create_oscillator();
//! osc.connect(&context.destination());
//! osc.start();
//!
//! // enjoy listening
//! std::thread::sleep(std::time::Duration::from_secs(4));
//! ```

/// Render quantum size (audio graph is rendered in blocks of this size)
pub const BUFFER_SIZE: u32 = 512;

pub mod buffer;
pub mod context;
pub mod media;
pub mod node;
pub mod param;

pub(crate) mod control;
pub(crate) mod graph;

/// Number of samples processed per second (Hertz) for a single channel of audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SampleRate(pub u32);

/// Number of channels for an audio buffer of audio node
///
/// 1 = Mono
/// 2 = Stereo
/// 4 = Quad
/// 6 = Surround
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChannelCount(pub u32);

/// Input/output with this index does not exist
#[derive(Debug, Clone, Copy)]
pub struct IndexSizeError {}

use std::fmt;
impl fmt::Display for IndexSizeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for IndexSizeError {}
