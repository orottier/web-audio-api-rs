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

struct RecordedData {
    blob: Vec<u8>,
    start_timecode: Instant,
    current_timecode: Instant,
}

impl RecordedData {
    fn new(blob: Vec<u8>) -> Self {
        let now = Instant::now();

        Self {
            blob,
            start_timecode: now,
            current_timecode: now,
        }
    }

    /// Start encoding audio into the blob buffer
    fn encode_first(&mut self, buf: AudioBuffer) {
        let spec = hound::WavSpec {
            channels: buf.number_of_channels() as u16,
            sample_rate: buf.sample_rate() as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let v = spec.into_header_for_infinite_file();
        self.blob.write_all(&v).unwrap();
        self.encode_next(buf);
    }

    /// Encode subsequent buffers into the blob buffer
    fn encode_next(&mut self, buf: AudioBuffer) {
        for i in 0..buf.length() {
            for c in 0..buf.number_of_channels() {
                let v = buf.get_channel_data(c)[i];
                hound::Sample::write(v, &mut self.blob, 32).unwrap();
            }
        }
    }
}

struct MediaRecorderInner {
    stream: MediaStream,
    active: AtomicBool,
    recorded_data: Mutex<RecordedData>,
    data_available_callback: Mutex<Option<BlobEventCallback>>,
    stop_callback: Mutex<Option<EventCallback>>,
    error_callback: Mutex<Option<ErrorEventCallback>>,
}

impl MediaRecorderInner {
    fn record(&self, buf: AudioBuffer) {
        let mut recorded_data = self.recorded_data.lock().unwrap();

        recorded_data.encode_next(buf);

        if recorded_data.blob.len() > 128 * 1024 {
            drop(recorded_data);
            self.flush();
        }
    }

    fn handle_error(&self, error: Box<dyn Error + Send + Sync>) {
        self.flush();

        if let Some(f) = self.error_callback.lock().unwrap().take() {
            (f)(ErrorEvent {
                message: error.to_string(),
                error: Box::new(error),
                event: Event {
                    type_: "ErrorEvent",
                },
            })
        }

        self.stop();
    }

    fn flush(&self) {
        let mut recorded_data = self.recorded_data.lock().unwrap();

        let timecode = recorded_data
            .current_timecode
            .duration_since(recorded_data.start_timecode)
            .as_secs_f64();

        let send = std::mem::replace(&mut recorded_data.blob, Vec::with_capacity(128 * 1024));
        if let Some(f) = self.data_available_callback.lock().unwrap().as_mut() {
            let event = BlobEvent {
                blob: send,
                timecode,
                event: Event { type_: "BlobEvent" },
            };
            (f)(event)
        }

        recorded_data.current_timecode = Instant::now();
    }

    fn stop(&self) {
        self.active.store(false, Ordering::SeqCst);

        if let Some(f) = self.stop_callback.lock().unwrap().take() {
            (f)(Event { type_: "StopEvent" })
        }
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

impl std::fmt::Debug for MediaRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaRecorder")
            .field("stream", &self.inner.stream)
            .field("active", &self.inner.active)
            .finish_non_exhaustive()
    }
}

impl MediaRecorder {
    /// Creates a new `MediaRecorder` object, given a [`MediaStream`] to record.
    ///
    /// Only supports WAV file format currently.
    pub fn new(stream: &MediaStream) -> Self {
        let inner = MediaRecorderInner {
            stream: stream.clone(),
            active: AtomicBool::new(false),
            recorded_data: Mutex::new(RecordedData::new(vec![])),
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
        assert!(
            !prev_active,
            "InvalidStateError - recorder has already started"
        );

        let inner = Arc::clone(&self.inner);
        let blob = Vec::with_capacity(128 * 1024);

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

            let mut recorded_data = RecordedData::new(blob);
            recorded_data.encode_first(buf);
            *inner.recorded_data.lock().unwrap() = recorded_data;

            for item in stream_iter {
                if !inner.active.load(Ordering::Relaxed) {
                    return; // recording has stopped
                }

                let buf = match item {
                    Ok(buf) => buf,
                    Err(error) => {
                        inner.handle_error(error);
                        return;
                    }
                };

                inner.record(buf);
            }

            inner.flush();
            inner.stop();
        });
    }

    pub fn stop(&self) {
        self.inner.flush();
        self.inner.stop();
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
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::media_streams::MediaStreamTrack;
    use float_eq::assert_float_eq;
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_record() {
        let data_received = Arc::new(AtomicBool::new(false));

        let buffers = vec![
            Ok(AudioBuffer::from(vec![vec![1.; 1024]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![2.; 1024]], 48000.)),
            Ok(AudioBuffer::from(vec![vec![3.; 1024]], 48000.)),
        ];
        let track = MediaStreamTrack::from_iter(buffers);
        let stream = MediaStream::from_tracks(vec![track]);
        let recorder = MediaRecorder::new(&stream);

        {
            let data_received = Arc::clone(&data_received);
            recorder.set_ondataavailable(move |_| data_received.store(true, Ordering::Relaxed));
        }

        // setup channel to await recorder completion
        let (send, recv) = crossbeam_channel::bounded(1);
        recorder.set_onstop(move |_| {
            let _ = send.send(());
        });

        recorder.start();

        let _ = recv.recv();
        assert!(data_received.load(Ordering::Relaxed));
    }

    #[test]
    fn test_error() {
        let data_received = Arc::new(AtomicBool::new(false));
        let error_received = Arc::new(AtomicBool::new(false));

        let buffers = vec![
            Ok(AudioBuffer::from(vec![vec![1.; 1024]], 48000.)),
            Err(String::from("error").into()),
            Ok(AudioBuffer::from(vec![vec![3.; 1024]], 48000.)),
        ];
        let track = MediaStreamTrack::from_iter(buffers);
        let stream = MediaStream::from_tracks(vec![track]);
        let recorder = MediaRecorder::new(&stream);

        {
            let data_received = Arc::clone(&data_received);
            recorder.set_ondataavailable(move |_| data_received.store(true, Ordering::Relaxed));
        }
        {
            let error_received = Arc::clone(&error_received);
            recorder.set_onerror(move |_| error_received.store(true, Ordering::Relaxed));
        }

        // setup channel to await recorder completion
        let (send, recv) = crossbeam_channel::bounded(1);
        recorder.set_onstop(move |_| {
            let _ = send.send(());
        });

        recorder.start();

        let _ = recv.recv();
        assert!(data_received.load(Ordering::Relaxed));
        assert!(error_received.load(Ordering::Relaxed));
    }

    #[test]
    fn test_encode_decode() {
        let buffers = vec![Ok(AudioBuffer::from(
            vec![vec![1.; 1024], vec![-1.; 1024]],
            48000.,
        ))];
        let track = MediaStreamTrack::from_iter(buffers);
        let stream = MediaStream::from_tracks(vec![track]);
        let recorder = MediaRecorder::new(&stream);

        let samples: Arc<Mutex<Vec<u8>>> = Default::default();
        {
            let samples = Arc::clone(&samples);
            recorder.set_ondataavailable(move |e| {
                samples.lock().unwrap().extend_from_slice(&e.blob);
            });
        }

        // setup channel to await recorder completion
        let (send, recv) = crossbeam_channel::bounded(1);
        recorder.set_onstop(move |_| {
            let _ = send.send(());
        });

        recorder.start();
        let _ = recv.recv();

        let samples = samples.lock().unwrap().clone();

        let ctx = OfflineAudioContext::new(1, 128, 48000.);
        let buf = ctx.decode_audio_data_sync(Cursor::new(samples)).unwrap();
        assert_eq!(buf.number_of_channels(), 2);
        assert_eq!(buf.length(), 1024);
        assert_float_eq!(buf.get_channel_data(0), &[1.; 1024][..], abs_all <= 0.);
        assert_float_eq!(buf.get_channel_data(1), &[-1.; 1024][..], abs_all <= 0.);
    }
}
