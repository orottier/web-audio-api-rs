//! Primitives of the Media Recorder API
//!
//! The MediaRecorder interface of the MediaStream Recording API provides functionality to easily
//! record media.
//!
//! <https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder>

use crate::media_streams::MediaStream;
use crate::Event;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct MediaRecorderInner {
    stream: MediaStream,
    active: AtomicBool,
    data_available_callback: Mutex<Option<Box<dyn FnMut(BlobEvent) + Send + 'static>>>,
    stop_callback: Mutex<Option<Box<dyn FnMut() + Send + 'static>>>,
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
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn set_ondataavailable<F: Fn(BlobEvent) + Send + 'static>(&self, callback: F) {
        *self.inner.data_available_callback.lock().unwrap() = Some(Box::new(callback));
    }

    pub fn clear_ondataavailable(&self) {
        *self.inner.data_available_callback.lock().unwrap() = None;
    }

    pub fn set_onstop<F: Fn() + Send + 'static>(&self, callback: F) {
        *self.inner.stop_callback.lock().unwrap() = Some(Box::new(callback));
    }

    pub fn clear_onstop(&self) {
        *self.inner.stop_callback.lock().unwrap() = None;
    }

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
                Some(Err(e)) => Err(e).unwrap(),
                Some(Ok(first)) => first,
            };

            let start_timecode = Instant::now();
            let mut now = start_timecode;

            let spec = hound::WavSpec {
                channels: buf.number_of_channels() as u16,
                sample_rate: buf.sample_rate() as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let v = spec.into_header_for_infinite_file();
            blob.write_all(&v).unwrap();
            for i in 0..buf.length() {
                for c in 0..buf.number_of_channels() {
                    let v = buf.get_channel_data(c)[i];
                    hound::Sample::write(v, &mut blob, 32).unwrap();
                }
            }

            for item in stream_iter {
                let buf = item.unwrap();
                for i in 0..buf.length() {
                    for c in 0..buf.number_of_channels() {
                        let v = buf.get_channel_data(c)[i];
                        hound::Sample::write(v, &mut blob, 32).unwrap();
                    }
                }

                let active = inner.active.load(Ordering::Relaxed);
                if !active || blob.len() > 128 * 1024 {
                    let timecode = now.duration_since(start_timecode).as_secs_f64();

                    let send = std::mem::replace(&mut blob, Vec::with_capacity(128 * 1024));
                    if let Some(f) = inner.data_available_callback.lock().unwrap().as_mut() {
                        let event = BlobEvent {
                            blob: send,
                            timecode,
                            event: Event { type_: "BlobEvent" },
                        };
                        (f)(event)
                    }

                    now = Instant::now();
                }

                if !active {
                    break;
                }
            }

            if let Some(f) = inner.stop_callback.lock().unwrap().as_mut() {
                (f)()
            }
        });
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
