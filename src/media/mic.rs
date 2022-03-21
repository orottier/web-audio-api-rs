use std::error::Error;

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

#[cfg(not(test))]
use crossbeam_channel::Sender;

#[cfg(not(test))]
use crate::buffer::ChannelData;
#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Sample, Stream};

use crossbeam_channel::{Receiver, TryRecvError};

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
/// let context = AudioContext::new(None);
///
/// let stream = Microphone::new();
/// // register as media element in the audio context
/// let background = context.create_media_stream_source(stream);
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
    stream: Stream,
}

// Todo, the Microphone struct is shipped to the render thread
// but it contains a Stream which is not Send.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Microphone {}

impl Microphone {
    /// Setup the default microphone input stream
    #[cfg(not(test))]
    pub fn new() -> Self {
        let (stream, config, receiver) = io::build_input();
        log::debug!("Input {:?}", config);

        let sample_rate = SampleRate(config.sample_rate.0);
        let number_of_channels = config.channels as usize;

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
}

#[cfg(not(test))]
impl Default for Microphone {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for Microphone {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.receiver.try_recv() {
            Ok(buffer) => {
                // new frame was ready
                buffer
            }
            Err(TryRecvError::Empty) => {
                // frame not received in time, emit silence
                log::debug!("input frame delayed");

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
