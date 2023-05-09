//! Primitives of the Media Recorder API
//!
//! The MediaRecorder interface of the MediaStream Recording API provides functionality to easily
//! record media.
//!
//! <https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder>

use crate::media_streams::MediaStream;
use crate::{AudioBuffer, ErrorEvent, Event};
use std::error::Error;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type EventCallback = Box<dyn FnOnce(Event) + Send + 'static>;
type BlobEventCallback = Box<dyn FnMut(BlobEvent) + Send + 'static>;
type ErrorEventCallback = Box<dyn FnOnce(ErrorEvent) + Send + 'static>;

struct MediaRecorderInner {
    stream: MediaStream,
    active: AtomicBool,
    data_available_callback: Mutex<Option<BlobEventCallback>>,
    stop_callback: Mutex<Option<EventCallback>>,
    error_callback: Mutex<Option<ErrorEventCallback>>,
}

impl MediaRecorderInner {
    fn handle_error(&self, error: Box<dyn Error + Send + Sync>) {
        self.active.store(false, Ordering::SeqCst);
        if let Some(f) = self.error_callback.lock().unwrap().take() {
            (f)(ErrorEvent {
                message: error.to_string(),
                error: Box::new(error),
                event: Event {
                    type_: "ErrorEvent",
                },
            })
        }
    }

    fn flush(&self, blob: &mut Vec<u8>, start_timecode: Instant, now: &mut Instant) {
        let timecode = now.duration_since(start_timecode).as_secs_f64();

        let send = std::mem::replace(blob, Vec::with_capacity(128 * 1024));
        if let Some(f) = self.data_available_callback.lock().unwrap().as_mut() {
            let event = BlobEvent {
                blob: send,
                timecode,
                event: Event { type_: "BlobEvent" },
            };
            (f)(event)
        }

        *now = Instant::now();
    }
}

/// Record and encode media
///
/// ```no_run
/// use web_audio_api::context::AudioContext;
/// use web_audio_api::media_recorder::MediaRecorder;
///
/// let context = AudioContext::default();
/// let output = context.create_media_stream_destination();
///
/// let recorder = MediaRecorder::new(output.stream());
/// recorder.set_ondataavailable(|event| {
///     println!("Received {} bytes of data", event.blob.len());
/// });
/// recorder.start();
/// ```
///
/// # Examples
///
/// - `cargo run --release --example recorder`
pub struct MediaRecorder {
    inner: Arc<MediaRecorderInner>,
}

impl MediaRecorder {
    /// Creates a new `MediaRecorder` object, given a [`MediaStream`] to record.
    ///
    /// Only supports WAV file format currently.
    pub fn new(stream: &MediaStream) -> Self {
        let inner = MediaRecorderInner {
            stream: stream.clone(),
            active: AtomicBool::new(false),
            data_available_callback: Mutex::new(None),
            stop_callback: Mutex::new(None),
            error_callback: Mutex::new(None),
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_ondataavailable<F: FnMut(BlobEvent) + Send + 'static>(&self, callback: F) {
        *self.inner.data_available_callback.lock().unwrap() = Some(Box::new(callback));
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn clear_ondataavailable(&self) {
        *self.inner.data_available_callback.lock().unwrap() = None;
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_onstop<F: FnOnce(Event) + Send + 'static>(&self, callback: F) {
        *self.inner.stop_callback.lock().unwrap() = Some(Box::new(callback));
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn clear_onstop(&self) {
        *self.inner.stop_callback.lock().unwrap() = None;
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_onerror<F: FnOnce(ErrorEvent) + Send + 'static>(&self, callback: F) {
        *self.inner.error_callback.lock().unwrap() = Some(Box::new(callback));
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn clear_onerror(&self) {
        *self.inner.error_callback.lock().unwrap() = None;
    }

    /// Begin recording media
    ///
    /// # Panics
    ///
    /// Will panic when the recorder has already started
    pub fn start(&self) {
        let prev_active = self.inner.active.swap(true, Ordering::Relaxed);
        if prev_active {
            panic!("InvalidStateError: recorder has already started")
        }

        let inner = self.inner.clone();
        let mut blob = Vec::with_capacity(128 * 1024);

        std::thread::spawn(move || {
            // for now, only record single track
            let mut stream_iter = inner.stream.get_tracks()[0].iter();
            let buf = match stream_iter.next() {
                None => return,
                Some(Err(error)) => {
                    inner.handle_error(error);
                    return;
                }
                Some(Ok(first)) => first,
            };

            let start_timecode = Instant::now();
            let mut now = start_timecode;

            Self::encode_first(&mut blob, buf);

            for item in stream_iter {
                let buf = match item {
                    Ok(buf) => buf,
                    Err(error) => {
                        inner.handle_error(error);
                        if !blob.is_empty() {
                            inner.flush(&mut blob, start_timecode, &mut now);
                        }
                        return;
                    }
                };
                Self::encode_next(&mut blob, buf);

                let active = inner.active.load(Ordering::SeqCst);
                if !active || blob.len() > 128 * 1024 {
                    inner.flush(&mut blob, start_timecode, &mut now);
                }

                if !active {
                    break;
                }
            }

            if let Some(f) = inner.stop_callback.lock().unwrap().take() {
                (f)(Event { type_: "StopEvent" })
            }
        });
    }

    /// Start encoding audio into the blob buffer
    fn encode_first(blob: &mut Vec<u8>, buf: AudioBuffer) {
        let spec = hound::WavSpec {
            channels: buf.number_of_channels() as u16,
            sample_rate: buf.sample_rate() as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let v = spec.into_header_for_infinite_file();
        blob.write_all(&v).unwrap();
        Self::encode_next(blob, buf);
    }

    /// Encode subsequent buffers into the blob buffer
    fn encode_next(blob: &mut Vec<u8>, buf: AudioBuffer) {
        for i in 0..buf.length() {
            for c in 0..buf.number_of_channels() {
                let v = buf.get_channel_data(c)[i];
                hound::Sample::write(v, blob, 32).unwrap();
            }
        }
    }

    pub fn stop(&self) {
        self.inner.active.store(false, Ordering::Relaxed);
    }
}

/// Interface for the `dataavailable` event, containing the recorded data
#[non_exhaustive]
#[derive(Debug)]
pub struct BlobEvent {
    /// The encoded data
    pub blob: Vec<u8>,
    /// The difference between the timestamp of the first chunk in data and the timestamp of the
    /// first chunk in the first BlobEvent produced by this recorder
    pub timecode: f64,
    /// Inherits from this base Event
    pub event: Event,
}

#[cfg(test)]
mod tests {
    use crate::media_streams::MediaStreamTrack;

    use super::*;

    #[test]
    fn test_start_stop() {
        let buffers = vec![
            Ok(AudioBuffer::from(vec![vec![1.; 1024]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![2.; 1024]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![3.; 1024]], 48000.)),
        ];
        let track = MediaStreamTrack::from_iter(buffers);
        let stream = MediaStream::from_tracks(vec![track]);
        let recorder = MediaRecorder::new(&stream);
        recorder.start();
        recorder.stop();
    }
}
