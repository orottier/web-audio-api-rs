//! Convenience abstractions that are not part of the WebAudio API (media decoding, microphone)

mod decoding;
pub(crate) use decoding::MediaDecoder;

mod element;
pub use element::MediaElement;

#[cfg(any(feature = "cubeb", feature = "cpal"))]
mod mic;
#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub use mic::Microphone;
#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub(crate) use mic::MicrophoneRender;

mod resampling;
pub(crate) use resampling::Resampler;

use std::error::Error;

use crate::buffer::{AudioBuffer, AudioBufferOptions};

pub(crate) trait AudioBufferIter: Iterator<Item = FallibleBuffer> + Send + 'static {}
impl<M: Iterator<Item = FallibleBuffer> + Send + 'static> AudioBufferIter for M {}

type FallibleBuffer = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

/// Interface for media streaming.
///
/// Below is an example showing how to play the stream directly in the audio context. However, this
/// is typically not what you should do. The media stream will be polled on the render thread which
/// will have catastrophic effects if the iterator blocks or for another reason takes too much time
/// to yield a new sample frame.
///
/// The solution is to wrap the `MediaStream` inside a [`MediaElement`]. This will take care of
/// buffering and timely delivery of audio to the render thread. It also allows for media playback
/// controls (play/pause, offsets, loops, etc.)
///
/// # Example
///
/// ```no_run
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::{AudioBuffer, AudioBufferOptions};
/// use web_audio_api::node::AudioNode;
/// use web_audio_api::media::MediaStreamTrack;
///
/// // create a new buffer: 512 samples of silence
/// let options = AudioBufferOptions {
///     number_of_channels: 0,
///     length: 512,
///     sample_rate: 44_100.,
/// };
/// let silence = AudioBuffer::new(options);
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(5);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let sequence = sequence.map(|b| Ok(b));
///
/// // convert to a media track
/// let media = MediaStreamTrack::from(sequence);
///
/// // use in the web audio context
/// let context = AudioContext::default();
/// let node = context.create_media_stream_track_source(media);
/// node.connect(&context.destination());
/// ```
pub struct MediaStreamTrack {
    iter: Box<dyn AudioBufferIter>,
    enabled: bool,
}

impl MediaStreamTrack {
    pub fn from<I: Iterator<Item = FallibleBuffer> + Send + Sync + 'static>(value: I) -> Self {
        MediaStreamTrack {
            iter: Box::new(value),
            enabled: true,
        }
    }
}

impl Iterator for MediaStreamTrack {
    type Item = FallibleBuffer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.enabled {
            self.iter.next()
        } else {
            Some(Ok(AudioBuffer::new(AudioBufferOptions {
                number_of_channels: 1,
                length: 128,
                sample_rate: 48000.,
            })))
        }
    }
}

pub struct MediaStream {
    tracks: Vec<MediaStreamTrack>,
}

impl MediaStream {
    pub(crate) fn from_iter<I: Iterator<Item = FallibleBuffer> + Send + Sync + 'static>(
        value: I,
    ) -> Self {
        let track = MediaStreamTrack::from(value);
        Self {
            tracks: vec![track],
        }
    }

    pub fn first_audio_track(mut self) -> Option<MediaStreamTrack> {
        self.tracks.pop()
    }
}
