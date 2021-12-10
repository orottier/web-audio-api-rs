use std::fmt::Debug;

use crate::context::{AsBaseAudioContext, AudioContextRegistration};
use crate::renderer::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// Options for constructing a ChannelMergerNode
pub struct ChannelMergerOptions {
    pub number_of_inputs: u32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelMergerOptions {
    fn default() -> Self {
        Self {
            number_of_inputs: 6,
            channel_config: ChannelConfigOptions {
                count: 1,
                mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// AudioNode for combining channels from multiple audio streams into a single audio stream.
pub struct ChannelMergerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for ChannelMergerNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn set_channel_count(&self, _v: usize) {
        panic!("Cannot edit channel count of ChannelMergerNode")
    }
    fn set_channel_count_mode(&self, _v: ChannelCountMode) {
        panic!("Cannot edit channel count mode of ChannelMergerNode")
    }
    fn set_channel_interpretation(&self, _v: ChannelInterpretation) {
        panic!("Cannot edit channel interpretation of ChannelMergerNode")
    }

    fn number_of_inputs(&self) -> u32 {
        self.channel_count() as _
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl ChannelMergerNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, mut options: ChannelMergerOptions) -> Self {
        context.base().register(move |registration| {
            options.channel_config.count = options.number_of_inputs as _;

            let node = ChannelMergerNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let render = ChannelMergerRenderer {};

            (node, Box::new(render))
        })
    }
}

#[derive(Debug)]
struct ChannelMergerRenderer {}

impl AudioProcessor for ChannelMergerRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];
        output.set_number_of_channels(inputs.len());

        inputs.iter().enumerate().for_each(|(i, input)| {
            *output.channel_data_mut(i) = input.channel_data(0).clone();
        });

        false
    }
}
