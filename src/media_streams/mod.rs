//! Primitives of the Media Capture and Streams API
//!
//! The Media Capture and Streams API, often called the Media Streams API or MediaStream API, is an
//! API related to WebRTC which provides support for streaming audio and video data.
//!
//! It provides the interfaces and methods for working with the streams and their constituent
//! tracks, the constraints associated with data formats, the success and error callbacks when
//! using the data asynchronously, and the events that are fired during the process.
//!
//! <https://developer.mozilla.org/en-US/docs/Web/API/Media_Capture_and_Streams_API>

use crate::{AudioBuffer, FallibleBuffer};
use arc_swap::ArcSwap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ready-state of a [`MediaStreamTrack`]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MediaStreamTrackState {
    /// The track is active (the track's underlying media source is making a best-effort attempt to
    /// provide data in real time).
    Live,
    /// The track has ended (the track's underlying media source is no longer providing data, and
    /// will never provide more data for this track). Once a track enters this state, it never
    /// exits it.
    Ended,
}

/// Single media track within a [`MediaStream`]
#[derive(Clone)]
pub struct MediaStreamTrack {
    inner: Arc<MediaStreamTrackInner>,
}

struct MediaStreamTrackInner {
    data: ArcSwap<FallibleBuffer>,
    position: AtomicU64,
    ended: AtomicBool,
    provider: Mutex<Box<dyn Iterator<Item = FallibleBuffer> + Send + Sync + 'static>>,
}

impl MediaStreamTrack {
    pub fn lazy<T: IntoIterator<Item = FallibleBuffer>>(iter: T) -> Self
    where
        <T as IntoIterator>::IntoIter: Send + Sync + 'static,
    {
        let initial = Ok(AudioBuffer::from(vec![vec![0.]], 48000.));
        let inner = MediaStreamTrackInner {
            data: ArcSwap::from_pointee(initial),
            position: AtomicU64::new(0),
            ended: AtomicBool::new(false),
            provider: Mutex::new(Box::new(iter.into_iter())),
        };
        MediaStreamTrack {
            inner: Arc::new(inner),
        }
    }

    pub fn ready_state(&self) -> MediaStreamTrackState {
        if self.inner.ended.load(Ordering::Relaxed) {
            MediaStreamTrackState::Ended
        } else {
            MediaStreamTrackState::Live
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = FallibleBuffer> {
        MediaStreamTrackIter {
            track: self.clone(),
            position: 0,
        }
    }
}

struct MediaStreamTrackIter {
    track: MediaStreamTrack,
    position: u64,
}

impl Iterator for MediaStreamTrackIter {
    type Item = FallibleBuffer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.track.inner.ended.load(Ordering::Relaxed) {
            return None;
        }

        let mut stream_position = self.track.inner.position.load(Ordering::Relaxed);
        if stream_position == self.position {
            match self.track.inner.provider.lock().unwrap().next() {
                Some(buf) => {
                    let _ = self.track.inner.data.swap(Arc::new(buf));
                }
                None => {
                    self.track.inner.ended.store(true, Ordering::Relaxed);
                    return None;
                }
            }
            stream_position += 1;
            self.track.inner.position.fetch_add(1, Ordering::Relaxed);
        }

        self.position = stream_position;
        Some(match &self.track.inner.data.load().as_ref() {
            Ok(buf) => Ok(buf.clone()),
            Err(e) => Err(e.to_string().into()),
        })
    }
}
/// Stream of media content.
///
/// A stream consists of several tracks, such as video or audio tracks. Each track is specified as
/// an instance of [`MediaStreamTrack`].
#[derive(Clone)]
pub struct MediaStream {
    tracks: Vec<MediaStreamTrack>,
}

impl MediaStream {
    pub fn from_tracks(tracks: Vec<MediaStreamTrack>) -> Self {
        Self { tracks }
    }

    pub fn get_tracks(&self) -> &[MediaStreamTrack] {
        &self.tracks
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_lazy() {
        let iter = vec![
            Ok(AudioBuffer::from(vec![vec![1.]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![2.]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![3.]], 48000.)),
        ];
        let track = MediaStreamTrack::lazy(iter);

        assert_eq!(track.ready_state(), MediaStreamTrackState::Live);

        let mut iter = track.iter();
        assert_float_eq!(
            iter.next().unwrap().unwrap().get_channel_data(0)[..],
            [1.][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            iter.next().unwrap().unwrap().get_channel_data(0)[..],
            &[2.][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            iter.next().unwrap().unwrap().get_channel_data(0)[..],
            &[3.][..],
            abs_all <= 0.
        );
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());

        assert_eq!(track.ready_state(), MediaStreamTrackState::Ended);
    }

    #[test]
    fn test_lazy_multiple_consumers() {
        let iter = vec![
            Ok(AudioBuffer::from(vec![vec![1.]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![2.]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![3.]], 48000.)),
        ];
        let track = MediaStreamTrack::lazy(iter);

        let mut iter1 = track.iter();
        let mut iter2 = track.iter();

        // first poll iter1 once, then iter2 once
        assert_float_eq!(
            iter1.next().unwrap().unwrap().get_channel_data(0)[..],
            [1.][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            iter2.next().unwrap().unwrap().get_channel_data(0)[..],
            &[1.][..],
            abs_all <= 0.
        );

        // then poll iter1 twice
        assert_float_eq!(
            iter1.next().unwrap().unwrap().get_channel_data(0)[..],
            &[2.][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            iter1.next().unwrap().unwrap().get_channel_data(0)[..],
            &[3.][..],
            abs_all <= 0.
        );

        // polling iter2 will now yield the latest buffer
        assert_float_eq!(
            iter2.next().unwrap().unwrap().get_channel_data(0)[..],
            &[3.][..],
            abs_all <= 0.
        );

        assert!(iter1.next().is_none());
        assert!(iter2.next().is_none());
        assert_eq!(track.ready_state(), MediaStreamTrackState::Ended);
    }
}
