use super::{AudioNode, ChannelConfig, ChannelConfigOptions};
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::MAX_CHANNELS;

use std::collections::HashMap;
use std::ops::DerefMut;

use arrayvec::ArrayVec;

pub trait AudioWorkletProcessor: Send {
    fn parameter_descriptors() -> &'static [AudioParamDescriptor]
    where
        Self: Sized,
    {
        &[] // empty by default
    }

    fn process<'a, 'b>(
        &mut self,
        inputs: &'b [&'a [f32]],
        outputs: &'b mut [&'a mut [f32]],
    ) -> bool;
}

/// Options for constructing an [`AudioWorkletNode`]
// dictionary AudioWorkletNodeOptions : AudioNodeOptions {
//     unsigned long numberOfInputs = 1;
//     unsigned long numberOfOutputs = 1;
//     sequence<unsigned long> outputChannelCount;
//     record<DOMString, double> parameterData;
//     object processorOptions;
// };
#[derive(Clone, Debug)]
pub struct AudioWorkletNodeOptions {
    /// This is used to initialize the value of the AudioNode numberOfInputs attribute.
    pub number_of_inputs: usize,
    /// This is used to initialize the value of the AudioNode numberOfOutputs attribute.
    pub number_of_outputs: usize,
    /// This array is used to configure the number of channels in each output.
    pub output_channel_count: Vec<usize>,
    /// This is a list of user-defined key-value pairs that are used to set the initial value of an
    /// AudioParam with the matched name in the AudioWorkletNode.
    pub parameter_data: HashMap<String, f64>,
    // processorOptions - ignored for now since rust allows to move data into the processor closure
    pub channel_config: ChannelConfigOptions,
}

impl Default for AudioWorkletNodeOptions {
    fn default() -> Self {
        Self {
            number_of_inputs: 1,
            number_of_outputs: 1,
            output_channel_count: Vec::new(),
            parameter_data: HashMap::new(),
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// A user-defined AudioNode
pub struct AudioWorkletNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    number_of_inputs: usize,
    number_of_outputs: usize,
    #[allow(dead_code)]
    audio_param_map: HashMap<String, AudioParam>,
}

impl AudioNode for AudioWorkletNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        self.number_of_inputs
    }

    fn number_of_outputs(&self) -> usize {
        self.number_of_outputs
    }
}

impl AudioWorkletNode {
    /// # Panics
    ///
    /// This function panics when the number of inputs and the number of outputs of the supplied
    /// options are both equal to zero.
    pub fn new<C: BaseAudioContext>(
        context: &C,
        audio_worklet_processor: impl AudioWorkletProcessor + 'static,
        options: AudioWorkletNodeOptions,
    ) -> Self {
        context.register(move |registration| {
            let AudioWorkletNodeOptions {
                number_of_inputs,
                number_of_outputs,
                output_channel_count: _,
                parameter_data: _,
                channel_config,
            } = options;

            if number_of_inputs == 0 && number_of_outputs == 0 {
                panic!("NotSupportedError: number of inputs and outputs cannot both be zero")
            }

            // todo handle output_channel_count

            // todo setup audio params, set initial values when supplied via parameter_data
            let audio_param_map = HashMap::new();

            let node = AudioWorkletNode {
                registration,
                channel_config: channel_config.into(),
                number_of_inputs,
                number_of_outputs,
                audio_param_map,
            };

            let render = AudioWorkletRenderer {
                processor: Box::new(audio_worklet_processor),
            };

            (node, Box::new(render))
        })
    }
}

struct AudioWorkletRenderer {
    processor: Box<dyn AudioWorkletProcessor>,
}

impl AudioProcessor for AudioWorkletRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // only single input/output is supported now

        let inputs_cast: ArrayVec<&[f32], MAX_CHANNELS> =
            inputs[0].channels().iter().map(|c| c.as_ref()).collect();

        let mut outputs_cast: ArrayVec<&mut [f32], MAX_CHANNELS> = outputs[0]
            .channels_mut()
            .iter_mut()
            .map(|c| c.deref_mut())
            .collect();

        let tail_time = self
            .processor
            .process(&inputs_cast[..], &mut outputs_cast[..]);

        tail_time
    }
}
