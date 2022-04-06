//! Convenience abstractions that are not part of the WebAudio API (media decoding, microphone)

mod decoding;
pub use decoding::MediaDecoder;
mod mic;
pub use mic::{AudioInputOptions, Microphone};

mod resampling;
pub(crate) use resampling::Resampler;

#[cfg(not(test))]
pub(crate) use mic::MicrophoneRender;

use std::error::Error;

use crate::buffer::AudioBuffer;

/// Interface for media streaming.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator, for example the [`MediaDecoder`] or
/// [`Microphone`].
///
/// Below is an example showing how to play the stream directly in the audio context. However, this
/// is typically not what you should do. The media stream will be polled on the render thread which
/// will have catastrophic effects if the iterator blocks or for another reason takes too much time
/// to yield a new sample frame.
///
// The solution is to wrap the `MediaStream` inside a [`MediaElement`]. This will take care of
// buffering and timely delivery of audio to the render thread. It also allows for media playback
// controls (play/pause, offsets, loops, etc.)
///
/// # Example
///
/// ```no_run
/// use web_audio_api::SampleRate;
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::buffer::{AudioBuffer, AudioBufferOptions};
/// use web_audio_api::node::AudioNode;
///
/// // create a new buffer: 512 samples of silence
/// let options = AudioBufferOptions {
///     number_of_channels: 0,
///     length: 512,
///     sample_rate: SampleRate(44_100),
/// };
/// let silence = AudioBuffer::new(options);
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(5);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream` and can be used in the audio graph
/// let context = AudioContext::default();
/// let node = context.create_media_stream_source(media);
/// node.connect(&context.destination());
/// ```
pub trait MediaStream:
    Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>> + Send + 'static
{
}
impl<M: Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>> + Send + 'static>
    MediaStream for M
{
}
