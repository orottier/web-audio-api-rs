use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{AudioNode, ChannelConfig};

use std::ops::DerefMut;

/// A user-defined AudioNode
pub struct AudioWorkletNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for AudioWorkletNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1 // todo, can be any number via AudioWorkletNodeOptions
    }

    fn number_of_outputs(&self) -> usize {
        1 // todo, can be any number via AudioWorkletNodeOptions
    }
}

impl AudioWorkletNode {
    pub fn new<C: BaseAudioContext>(
        context: &C,
        callback: impl FnMut(&[&[f32]], &[&mut [f32]]) -> bool + Send + 'static,
        // todo AudioWorkletNodeOptions
    ) -> Self {
        context.register(move |registration| {
            let node = AudioWorkletNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            let render = AudioWorkletProcessor {
                callback: Box::new(callback),
            };

            (node, Box::new(render))
        })
    }
}

struct AudioWorkletProcessor {
    callback: Box<dyn FnMut(&[&[f32]], &[&mut [f32]]) -> bool + Send>,
}

impl AudioProcessor for AudioWorkletProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // only single input/output is supported now

        // todo prevent allocation (collect) per call
        let input: Vec<_> = inputs[0].channels().iter().map(|c| c.as_ref()).collect();
        let output: Vec<_> = outputs[0]
            .channels_mut()
            .iter_mut()
            .map(|c| c.deref_mut())
            .collect();

        (self.callback)(&input[..], &output[..])
    }
}
