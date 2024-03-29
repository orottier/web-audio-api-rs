use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::events::{AudioProcessingEvent, EventHandler, EventPayload, EventType};
use crate::node::{ChannelCountMode, ChannelInterpretation};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::{AudioBuffer, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

use std::any::Any;

/// An AudioNode which can generate, process, or analyse audio directly using a script (deprecated)
#[derive(Debug)]
pub struct ScriptProcessorNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    //  bufferSize MUST be one of the following values: 256, 512, 1024, 2048, 4096, 8192, 16384
    buffer_size: usize,
}

impl AudioNode for ScriptProcessorNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }

    // TODO channel config constraints
}

impl ScriptProcessorNode {
    pub(crate) fn new<C: BaseAudioContext>(
        context: &C,
        buffer_size: usize,
        number_of_input_channels: usize,
        number_of_output_channels: usize,
    ) -> Self {
        // TODO assert valid arguments

        context.base().register(move |registration| {
            let render = ScriptProcessorRenderer {
                input_buffer: Vec::with_capacity(buffer_size / RENDER_QUANTUM_SIZE),
                output_buffer: Vec::with_capacity(buffer_size / RENDER_QUANTUM_SIZE),
                next_output_buffer: Vec::with_capacity(buffer_size / RENDER_QUANTUM_SIZE),
                buffer_size,
                number_of_output_channels,
            };

            let channel_config = ChannelConfigOptions {
                count: number_of_input_channels,
                count_mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            };

            let node = ScriptProcessorNode {
                registration,
                channel_config: channel_config.into(),
                buffer_size,
            };

            (node, Box::new(render))
        })
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Register callback to run when the AudioProcessingEvent is dispatched
    ///
    /// The event handler processes audio from the input (if any) by accessing the audio data from
    /// the inputBuffer attribute. The audio data which is the result of the processing (or the
    /// synthesized data if there are no inputs) is then placed into the outputBuffer.
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    pub fn set_onaudioprocess<F: FnMut(&mut AudioProcessingEvent) + Send + 'static>(
        &self,
        mut callback: F,
    ) {
        let registration = self.registration.clone();
        let callback = move |v| {
            let mut payload = match v {
                EventPayload::AudioProcessing(v) => v,
                _ => unreachable!(),
            };
            callback(&mut payload);
            registration.post_message(payload.output_buffer);
        };

        self.context().set_event_handler(
            EventType::AudioProcessing(self.registration().id()),
            EventHandler::Multiple(Box::new(callback)),
        );
    }

    /// Unset the callback to run when the AudioProcessingEvent is dispatched
    pub fn clear_onaudioprocess(&self) {
        self.context()
            .clear_event_handler(EventType::AudioProcessing(self.registration().id()));
    }
}

struct ScriptProcessorRenderer {
    input_buffer: Vec<AudioRenderQuantum>,
    output_buffer: Vec<AudioRenderQuantum>,
    next_output_buffer: Vec<AudioRenderQuantum>,
    buffer_size: usize,
    number_of_output_channels: usize,
}

// SAFETY:
// AudioRenderQuantums are not Send but we promise the `buffer` VecDeque is empty before we ship it
// to the render thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ScriptProcessorRenderer {}

impl AudioProcessor for ScriptProcessorRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        output.make_silent();

        let number_of_quanta = self.input_buffer.capacity();

        self.input_buffer.push(input.clone());
        if self.input_buffer.len() == number_of_quanta {
            // convert self.input_buffer to an AudioBuffer
            let number_of_input_channels = self
                .input_buffer
                .iter()
                .map(|i| i.number_of_channels())
                .max()
                .unwrap();
            let mut input_samples = vec![vec![0.; self.buffer_size]; number_of_input_channels];
            self.input_buffer.iter().enumerate().for_each(|(i, b)| {
                let offset = RENDER_QUANTUM_SIZE * i;
                b.channels()
                    .iter()
                    .zip(input_samples.iter_mut())
                    .for_each(|(c, o)| {
                        o[offset..(offset + RENDER_QUANTUM_SIZE)].copy_from_slice(c);
                    });
            });
            let input_buffer = AudioBuffer::from(input_samples, scope.sample_rate);

            // create a suitable output AudioBuffer
            let output_samples = vec![vec![0.; self.buffer_size]; self.number_of_output_channels];
            let output_buffer = AudioBuffer::from(output_samples, scope.sample_rate);

            // emit event to control thread
            let playback_time =
                scope.current_time + self.buffer_size as f64 / scope.sample_rate as f64;
            scope.send_audio_processing_event(input_buffer, output_buffer, playback_time);

            // clear existing input buffer
            self.input_buffer.clear();

            // move next output buffer into current output buffer
            std::mem::swap(&mut self.output_buffer, &mut self.next_output_buffer);
            // fill next output buffer with silence
            self.next_output_buffer.clear();
            let silence = output.clone();
            self.next_output_buffer.resize(number_of_quanta, silence);
        }

        if !self.output_buffer.is_empty() {
            *output = self.output_buffer.remove(0);
        }

        true // TODO - spec says false but that seems weird
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(buffer) = msg.downcast_mut::<AudioBuffer>() {
            buffer.channels().iter().enumerate().for_each(|(i, c)| {
                c.as_slice()
                    .chunks(RENDER_QUANTUM_SIZE)
                    .zip(self.next_output_buffer.iter_mut())
                    .for_each(|(s, o)| o.channel_data_mut(i).copy_from_slice(s))
            });
            return;
        };

        log::warn!("ScriptProcessorRenderer: Dropping incoming message {msg:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::OfflineAudioContext;
    use float_eq::assert_float_eq;

    #[test]
    fn test() {
        // TODO how to test?
    }
}
