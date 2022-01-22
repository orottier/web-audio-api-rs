use std::fmt::Debug;

use crate::context::{AudioContextRegistration, Context};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// Options for constructing a ChannelSplitterNode
pub struct ChannelSplitterOptions {
    pub number_of_outputs: u32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelSplitterOptions {
    fn default() -> Self {
        Self {
            number_of_outputs: 6,
            channel_config: ChannelConfigOptions {
                count: 6, // must be same as number_of_outputs
                mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Discrete,
            },
        }
    }
}

/// AudioNode for accessing the individual channels of an audio stream in the routing graph
pub struct ChannelSplitterNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for ChannelSplitterNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn set_channel_count(&self, _v: usize) {
        panic!("Cannot edit channel count of ChannelSplitterNode")
    }
    fn set_channel_count_mode(&self, _v: ChannelCountMode) {
        panic!("Cannot edit channel count mode of ChannelSplitterNode")
    }
    fn set_channel_interpretation(&self, _v: ChannelInterpretation) {
        panic!("Cannot edit channel interpretation of ChannelSplitterNode")
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        self.channel_count() as _
    }
}

impl ChannelSplitterNode {
    pub fn new<C: Context>(context: &C, mut options: ChannelSplitterOptions) -> Self {
        context.register(move |registration| {
            options.channel_config.count = options.number_of_outputs as _;

            let node = ChannelSplitterNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let render = ChannelSplitterRenderer {
                number_of_outputs: node.channel_count() as _,
            };

            (node, Box::new(render))
        })
    }
}

#[derive(Debug)]
struct ChannelSplitterRenderer {
    pub number_of_outputs: usize,
}

impl AudioProcessor for ChannelSplitterRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input node
        let input = &inputs[0];

        // assert number of outputs was correctly set by renderer
        assert_eq!(self.number_of_outputs, outputs.len());

        for (i, output) in outputs.iter_mut().enumerate() {
            output.force_mono();
            if i < input.number_of_channels() {
                *output.channel_data_mut(0) = input.channel_data(i).clone();
            } else {
                // input does not have this channel filled, emit silence
                output.make_silent();
            }
        }

        false
    }
}
