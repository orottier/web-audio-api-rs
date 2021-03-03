//! A high-level API for processing and synthesizing audio.
//!
//! # Example
//! ```rust
//! use std::fs::File;
//! use web_audio_api::context::{AsBaseAudioContext, AudioContext};
//! use web_audio_api::media::{MediaElement, OggVorbisDecoder};
//! use web_audio_api::node::{AudioNode, AudioControllableSourceNode, AudioScheduledSourceNode};
//!
//! let context = AudioContext::new();
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
//! background.set_loop(true);
//! // use a gain node to control volume
//! let gain = context.create_gain();
//! // play at low volume
//! gain.gain().set_value(0.5);
//! // connect the media node to the gain node
//! background.connect(&gain);
//! // connect the gain node to the destination node (speakers)
//! gain.connect(&context.destination());
//! // start playback
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

/// Render quantum size (audio graph is rendered in blocks of this size)
pub const BUFFER_SIZE: u32 = 512;

pub mod buffer;
pub mod context;
pub mod control;
pub mod media;
pub mod node;
pub mod param;

pub(crate) mod graph;
pub(crate) mod message;

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
