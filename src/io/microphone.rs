use std::error::Error;
use std::sync::{Arc, Mutex};

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::io::AudioBackendManager;
use crate::RENDER_QUANTUM_SIZE;

use crossbeam_channel::{Receiver, Sender, TryRecvError};

use super::echo_cancellation::EchoCanceller;

pub(crate) struct MicrophoneStream {
    receiver: Receiver<AudioBuffer>,
    number_of_channels: usize,
    sample_rate: f32,
    stream: Box<dyn AudioBackendManager>,
    echo_canceller: Option<Arc<Mutex<EchoCanceller>>>,
}

impl MicrophoneStream {
    pub(crate) fn new(
        receiver: Receiver<AudioBuffer>,
        backend: Box<dyn AudioBackendManager>,
    ) -> Self {
        Self {
            receiver,
            number_of_channels: backend.number_of_channels(),
            sample_rate: backend.sample_rate(),
            stream: backend,
            echo_canceller: None,
        }
    }
    
    pub(crate) fn with_echo_canceller(
        receiver: Receiver<AudioBuffer>,
        backend: Box<dyn AudioBackendManager>,
        echo_canceller: Arc<Mutex<EchoCanceller>>,
    ) -> Self {
        Self {
            receiver,
            number_of_channels: backend.number_of_channels(),
            sample_rate: backend.sample_rate(),
            stream: backend,
            echo_canceller: Some(echo_canceller),
        }
    }
    
    pub(crate) fn echo_canceller(&self) -> Option<&Arc<Mutex<EchoCanceller>>> {
        self.echo_canceller.as_ref()
    }
}

impl Drop for MicrophoneStream {
    fn drop(&mut self) {
        log::debug!("Microphone stream has been dropped");
        self.stream.close()
    }
}

impl Iterator for MicrophoneStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = match self.receiver.try_recv() {
            Ok(buffer) => {
                // new frame was ready
                buffer
            }
            Err(TryRecvError::Empty) => {
                // frame not received in time, emit silence
                log::debug!("empty channel: input frame delayed");

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
        
        // Apply echo cancellation if enabled
        if let Some(echo_canceller) = &self.echo_canceller {
            let canceller = echo_canceller.lock().unwrap();
            
            // Process each channel through echo cancellation
            let mut processed_channels = Vec::with_capacity(self.number_of_channels);
            
            for ch in 0..self.number_of_channels {
                let input_data = next.get_channel_data(ch);
                let processed = canceller.process(input_data);
                processed_channels.push(processed);
            }
            
            // Create new buffer with processed audio
            next = AudioBuffer::from(processed_channels, self.sample_rate);
        }

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
            channels.push(
                data.iter()
                    .skip(i)
                    .step_by(self.number_of_channels)
                    .map(|v| v.to_sample_())
                    .collect(),
            );
        }

        let buffer = AudioBuffer::from(channels, self.sample_rate);
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
