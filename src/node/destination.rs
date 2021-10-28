use crate::buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::context::{AsBaseAudioContext, AudioContextRegistration};
use crate::process::{AudioParamValues, AudioProcessor};
use crate::SampleRate;

use super::AudioNode;

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode {
    pub(crate) registration: AudioContextRegistration,
    pub(crate) channel_count: usize,
}

struct DestinationRenderer {}

impl AudioProcessor for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // todo, actually fill cpal buffer here
        *output = input.clone();
    }

    fn tail_time(&self) -> bool {
        unreachable!() // will never drop in control thread
    }
}

impl AudioNode for DestinationNode {
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

impl DestinationNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, channel_count: usize) -> Self {
        context.base().register(move |registration| {
            let node = Self {
                registration,
                channel_count,
            };
            let proc = DestinationRenderer {};

            (node, Box::new(proc))
        })
    }
}
