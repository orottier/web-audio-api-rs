//! Convenience abstractions that are not part of the WebAudio API (media decoding, microphone)

mod decoding;
pub(crate) use decoding::MediaDecoder;

mod element;
pub use element::MediaElement;

#[cfg(any(feature = "cubeb", feature = "cpal"))]
mod mic;
#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub(crate) use mic::Microphone;
#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub(crate) use mic::MicrophoneRender;

mod resampling;
pub(crate) use resampling::Resampler;
