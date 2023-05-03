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

use crate::{AudioBufferIter, FallibleBuffer};

use crate::buffer::{AudioBuffer, AudioBufferOptions};

/// Single media track within a [`MediaStream`]
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

/// Stream of media content.
///
/// A stream consists of several tracks, such as video or audio tracks. Each track is specified as
/// an instance of [`MediaStreamTrack`].
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
