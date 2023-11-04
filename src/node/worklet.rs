use super::{AudioNode, ChannelConfig};
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::MAX_CHANNELS;

use std::ops::DerefMut;

use arrayvec::ArrayVec;

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
                inputs: ArrayVec::new(),
                outputs: ArrayVec::new(),
            };

            (node, Box::new(render))
        })
    }
}

type AudioWorkletProcessCallback = dyn FnMut(&[&[f32]], &[&mut [f32]]) -> bool + Send;

struct AudioWorkletProcessor {
    callback: Box<AudioWorkletProcessCallback>,
    inputs: ArrayVec<&'static [f32], MAX_CHANNELS>,
    outputs: ArrayVec<&'static mut [f32], MAX_CHANNELS>,
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

        inputs[0].channels().iter().for_each(|c| {
            let slice: &[f32] = c.as_ref();
            // SAFETY
            // We're upgrading the lifetime of the inputs to `static`. This is okay because
            // `self.callback` is a HRTB (for <'a> FnMut &'a ...) so the references cannot
            // escape. The inputs are dropped at the end of the `process` method.
            let static_slice: &'static [f32] = unsafe { core::mem::transmute(slice) };
            self.inputs.push(static_slice)
        });

        outputs[0].channels_mut().iter_mut().for_each(|c| {
            let slice: &mut [f32] = c.deref_mut();
            // SAFETY
            // We're upgrading the lifetime of the outputs to `static`. This is okay because
            // `self.callback` is a HRTB (for <'a> FnMut &'a ...) so the references cannot
            // escape. The outputs are dropped at the end of the `process` method.
            let static_slice: &'static mut [f32] = unsafe { core::mem::transmute(slice) };
            self.outputs.push(static_slice)
        });

        let tail_time = (self.callback)(&self.inputs[..], &self.outputs[..]);

        self.inputs.clear();
        self.outputs.clear();

        tail_time
    }
}
