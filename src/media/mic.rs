use std::error::Error;

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::media_streams::{MediaStream, MediaStreamTrack};
use crate::RENDER_QUANTUM_SIZE;

use crate::context::AudioContextOptions;

use crossbeam_channel::Sender;

use crate::buffer::ChannelData;
use crate::io::{self, AudioBackendManager};

use crossbeam_channel::{Receiver, TryRecvError};

/// Microphone input stream
///
/// The Microphone can set up a [`MediaStream`] value which can be used
/// inside a [`MediaStreamAudioSourceNode`](crate::node::MediaStreamAudioSourceNode).
///
/// It is okay for the Microphone struct to go out of scope, any corresponding stream will still be
/// kept alive and emit audio buffers. Call the `close()` method if you want to stop the microphone
/// input and release all system resources.
///
/// # Warning
///
/// This abstraction is not part of the Web Audio API and does not aim at implementing
/// the full MediaDevices API. It is only provided for convenience reasons.
///
/// # Example
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::context::{AudioContextLatencyCategory, AudioContextOptions};
/// use web_audio_api::media::Microphone;
/// use web_audio_api::node::AudioNode;
///
/// let context = AudioContext::default();
///
/// // Request an input sample rate of 44.1 kHz and default latency (buffer size 128, if available)
/// let opts = AudioContextOptions {
///     sample_rate: Some(44100.),
///     ..AudioContextOptions::default()
/// };
/// let mic = Microphone::new(opts);
/// // or you can create Microphone with default options
/// // let stream = Microphone::default();
///
/// // register as media element in the audio context
/// let background = context.create_media_stream_source(mic.stream());
/// // connect the node directly to the destination node (speakers)
/// background.connect(&context.destination());
///
/// // enjoy listening
/// std::thread::sleep(std::time::Duration::from_secs(4));
/// ```
pub struct Microphone {
    backend: Box<dyn AudioBackendManager>,
    stream: MediaStream,
}

impl Microphone {
    /// Setup the default microphone input stream
    ///
    /// Note: the specified `latency_hint` is currently ignored, follow our progress at
    /// <https://github.com/orottier/web-audio-api-rs/issues/51>
    pub fn new(options: AudioContextOptions) -> Self {
        // select backend based on cargo features
        let (backend, receiver) = io::build_input(options);

        let media_iter = MicrophoneStream {
            receiver,
            number_of_channels: backend.number_of_channels(),
            sample_rate: backend.sample_rate(),
            _stream: backend.boxed_clone(),
        };
        let track = MediaStreamTrack::lazy(media_iter);
        let stream = MediaStream::from_tracks(vec![track]);

        Self { backend, stream }
    }

    /// Suspends the input stream, temporarily halting audio hardware access and reducing
    /// CPU/battery usage in the process.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The input device is not available
    /// * For a `BackendSpecificError`
    pub fn suspend(&self) {
        self.backend.suspend();
    }

    /// Resumes the input stream that has previously been suspended/paused.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The input device is not available
    /// * For a `BackendSpecificError`
    pub fn resume(&self) {
        self.backend.resume();
    }

    /// Closes the microphone input stream, releasing the system resources being used.
    #[allow(clippy::missing_panics_doc)]
    pub fn close(self) {
        self.backend.close()
    }

    /// A [`MediaStream`] producing audio buffers from the microphone input
    pub fn stream(&self) -> &MediaStream {
        &self.stream
    }
}

impl Default for Microphone {
    fn default() -> Self {
        Self::new(AudioContextOptions::default())
    }
}

struct MicrophoneStream {
    receiver: Receiver<AudioBuffer>,
    number_of_channels: usize,
    sample_rate: f32,

    _stream: Box<dyn AudioBackendManager>,
}

impl Iterator for MicrophoneStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.receiver.try_recv() {
            Ok(buffer) => {
                // new frame was ready
                buffer
            }
            Err(TryRecvError::Empty) => {
                // frame not received in time, emit silence
                // log::debug!("input frame delayed");

                let options = AudioBufferOptions {
                    number_of_channels: self.number_of_channels,
                    length: RENDER_QUANTUM_SIZE,
                    sample_rate: self.sample_rate,
                };

                AudioBuffer::new(options)
            }
            Err(TryRecvError::Disconnected) => {
                // MicrophoneRender has stopped, close stream
                return None;
            }
        };

        Some(Ok(next))
    }
}

pub(crate) struct MicrophoneRender {
    number_of_channels: usize,
    sample_rate: f32,
    sender: Sender<AudioBuffer>,
}

impl MicrophoneRender {
    pub fn new(number_of_channels: usize, sample_rate: f32, sender: Sender<AudioBuffer>) -> Self {
        Self {
            number_of_channels,
            sample_rate,
            sender,
        }
    }

    pub fn render<S: dasp_sample::ToSample<f32> + Copy>(&self, data: &[S]) {
        let mut channels = Vec::with_capacity(self.number_of_channels);

        // copy rendered audio into output slice
        for i in 0..self.number_of_channels {
            channels.push(ChannelData::from(
                data.iter()
                    .skip(i)
                    .step_by(self.number_of_channels)
                    .map(|v| v.to_sample_())
                    .collect(),
            ));
        }

        let buffer = AudioBuffer::from_channels(channels, self.sample_rate);
        let result = self.sender.try_send(buffer); // can fail (frame dropped)
        if result.is_err() {
            log::debug!("input frame dropped");
        }
    }
}

impl Drop for MicrophoneRender {
    fn drop(&mut self) {
        log::debug!("Microphone input has been dropped");
    }
}
