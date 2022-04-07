use std::error::Error;

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::media::MediaStream;
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

#[cfg(not(test))]
use crossbeam_channel::Sender;
#[cfg(not(test))]
use std::sync::Arc;

#[cfg(not(test))]
use crate::buffer::ChannelData;
#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Sample, Stream};

use crossbeam_channel::{Receiver, TryRecvError};

#[cfg(not(test))]
mod private {
    use super::*;

    pub struct StreamHolder(Arc<Stream>);

    impl StreamHolder {
        pub fn new(stream: Arc<Stream>) -> Self {
            StreamHolder(stream)
        }
    }

    // SAFETY: Stream is not Send, but we do want to have shared ownership via an Arc.
    // The StreamHolder is simply keeping the Stream alive, it does not allow any mutation to it.
    #[allow(clippy::non_send_fields_in_send_ty)]
    unsafe impl Send for StreamHolder {}
}

#[cfg(not(test))]
use private::StreamHolder;

/// Microphone input stream
///
/// It implements the [`MediaStream`](crate::media::MediaStream) trait so can be used inside a
/// [`MediaStreamAudioSourceNode`](crate::node::MediaStreamAudioSourceNode)
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
/// use web_audio_api::media::Microphone;
/// use web_audio_api::node::AudioNode;
///
/// let context = AudioContext::default();
///
/// let mic = Microphone::new();
/// // register as media element in the audio context
/// let background = context.create_media_stream_source(mic.stream());
/// // connect the node directly to the destination node (speakers)
/// background.connect(&context.destination());
///
/// // enjoy listening
/// std::thread::sleep(std::time::Duration::from_secs(4));
/// ```
pub struct Microphone {
    receiver: Receiver<AudioBuffer>,
    number_of_channels: usize,
    sample_rate: SampleRate,

    #[cfg(not(test))]
    stream: Arc<Stream>,
}

impl Microphone {
    /// Setup the default microphone input stream
    #[cfg(not(test))]
    pub fn new() -> Self {
        let (stream, config, receiver) = io::build_input();
        log::debug!("Input {:?}", config);

        let sample_rate = SampleRate(config.sample_rate.0);
        let number_of_channels = config.channels as usize;

        // shared ownership for the stream, because the Microphone is allowed to go out of scope
        // but all corresponding streams should still yield output
        let stream = Arc::new(stream);

        Self {
            receiver,
            number_of_channels,
            sample_rate,
            stream,
        }
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
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.pause().unwrap()
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
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.play().unwrap()
    }

    /// A [`MediaStream`] iterator producing audio buffers from the microphone input
    ///
    /// Note that while you can call this function multiple times and poll all iterators
    /// concurrently, this could lead to unexpected behavior as the buffers will only be offered
    /// once.
    pub fn stream(&self) -> impl MediaStream {
        MicrophoneStream {
            receiver: self.receiver.clone(),
            number_of_channels: self.number_of_channels,
            sample_rate: self.sample_rate,
            #[cfg(not(test))]
            _stream: StreamHolder::new(self.stream.clone()),
        }
    }
}

#[cfg(not(test))]
impl Default for Microphone {
    fn default() -> Self {
        Self::new()
    }
}

// no need for public documentation because the concrete type is never returned (an impl
// MediaStream is returned instead)
#[doc(hidden)]
pub struct MicrophoneStream {
    receiver: Receiver<AudioBuffer>,
    number_of_channels: usize,
    sample_rate: SampleRate,

    #[cfg(not(test))]
    _stream: StreamHolder,
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

#[cfg(not(test))]
pub(crate) struct MicrophoneRender {
    number_of_channels: usize,
    sample_rate: SampleRate,
    sender: Sender<AudioBuffer>,
}

#[cfg(not(test))]
impl MicrophoneRender {
    pub fn new(
        number_of_channels: usize,
        sample_rate: SampleRate,
        sender: Sender<AudioBuffer>,
    ) -> Self {
        Self {
            number_of_channels,
            sample_rate,
            sender,
        }
    }

    pub fn render<S: Sample>(&self, data: &[S]) {
        let mut channels = Vec::with_capacity(self.number_of_channels);

        // copy rendered audio into output slice
        for i in 0..self.number_of_channels {
            channels.push(ChannelData::from(
                data.iter()
                    .skip(i)
                    .step_by(self.number_of_channels)
                    .map(|v| v.to_f32())
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

#[cfg(not(test))]
impl Drop for MicrophoneRender {
    fn drop(&mut self) {
        log::debug!("Microphone input has been dropped");
    }
}
