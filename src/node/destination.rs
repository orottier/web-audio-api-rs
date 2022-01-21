use crate::context::{AudioContextRegistration, Context};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct AudioDestinationNode {
    pub(crate) registration: AudioContextRegistration,
    pub(crate) channel_count: usize,
}

struct DestinationRenderer {}

impl AudioProcessor for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // todo, actually fill cpal buffer here
        *output = input.clone();

        true
    }
}

impl AudioNode for AudioDestinationNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        unreachable!()
    }

    fn channel_config_cloned(&self) -> ChannelConfig {
        ChannelConfigOptions {
            count: self.channel_count,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Speakers,
        }
        .into()
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::Explicit
    }

    fn channel_interpretation(&self) -> ChannelInterpretation {
        ChannelInterpretation::Speakers
    }

    fn channel_count(&self) -> usize {
        self.channel_count
    }
}

impl AudioDestinationNode {
    pub fn new<C: Context>(context: &C, channel_count: usize) -> Self {
        context.register(move |registration| {
            let node = Self {
                registration,
                channel_count,
            };
            let proc = DestinationRenderer {};

            (node, Box::new(proc))
        })
    }

    /// The maximum number of channels that the channelCount attribute can be set to
    /// This is the limit number that audio hardware can support.
    pub fn max_channels_count(&self) -> u32 {
        self.registration.channels()
    }
}
