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
                buffer: None,
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
    buffer: Option<AudioRenderQuantum>, // TODO buffer_size
    buffer_size: usize,
    number_of_output_channels: usize,
}

// SAFETY:
// AudioRenderQuantums are not Send but we promise the `buffer` is None before we ship it to the
// render thread.
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

        let mut silence = input.clone();
        silence.make_silent();
        if let Some(buffer) = self.buffer.replace(silence) {
            *output = buffer;
        }

        // TODO buffer_size
        let input_samples = input.channels().iter().map(|c| c.to_vec()).collect();
        let input_buffer = AudioBuffer::from(input_samples, scope.sample_rate);
        let output_samples = vec![vec![0.; RENDER_QUANTUM_SIZE]; self.number_of_output_channels];
        let output_buffer = AudioBuffer::from(output_samples, scope.sample_rate);

        let playback_time =
            scope.current_time + (RENDER_QUANTUM_SIZE as f32 / scope.sample_rate) as f64; // TODO
        scope.send_audio_processing_event(input_buffer, output_buffer, playback_time);

        true // TODO - spec says false but that seems weird
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(buffer) = msg.downcast_mut::<AudioBuffer>() {
            if let Some(render_quantum) = &mut self.buffer {
                buffer
                    .channels()
                    .iter()
                    .zip(render_quantum.channels_mut())
                    .for_each(|(i, o)| o.copy_from_slice(i.as_slice())); // TODO bounds check
            }
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
