use crate::{AudioBuffer, AudioBufferOptions, FallibleBuffer};
use arc_swap::ArcSwap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Single media track within a [`MediaStream`]
#[derive(Clone)]
pub struct MediaStreamTrack {
    inner: Arc<MediaStreamTrackInner>,
}

struct MediaStreamTrackInner {
    data: ArcSwap<FallibleBuffer>,
    position: AtomicU64,
    enabled: AtomicBool,
    muted: AtomicBool,
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
            enabled: AtomicBool::new(true),
            muted: AtomicBool::new(false),
            provider: Mutex::new(Box::new(iter.into_iter())),
        };
        MediaStreamTrack {
            inner: Arc::new(inner),
        }
    }

    pub fn enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, value: bool) {
        self.inner.enabled.store(value, Ordering::Relaxed)
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
        let stream_position = self.track.inner.position.load(Ordering::Relaxed);
        if stream_position == self.position {
            match self.track.inner.provider.lock().unwrap().next() {
                Some(buf) => {
                    let _ = self.track.inner.data.swap(Arc::new(buf));
                }
                None => {
                    self.track.inner.muted.store(true, Ordering::Relaxed);
                    return None;
                }
            }
            self.track.inner.position.fetch_add(1, Ordering::Relaxed);
        }

        if self.track.inner.muted.load(Ordering::Relaxed) {
            return None;
        }

        if !self.track.inner.enabled.load(Ordering::Relaxed) {
            Some(Ok(AudioBuffer::new(AudioBufferOptions {
                number_of_channels: 1,
                length: 128,
                sample_rate: 48000.,
            })))
        } else {
            let buf = match &self.track.inner.data.load().as_ref() {
                Ok(buf) => buf.clone(),
                Err(_) => panic!(),
            };
            self.position = stream_position;
            Some(Ok(buf))
        }
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
