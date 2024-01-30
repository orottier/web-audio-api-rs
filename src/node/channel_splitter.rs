use std::fmt::Debug;

use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// Assert that the channel count mode is valid for the ChannelSplitterNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcountmode-constraints>
///
/// # Panics
///
/// This function panics if the mode is not equal to Explicit
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count_mode(mode: ChannelCountMode) {
    assert!(
        mode == ChannelCountMode::Explicit,
        "InvalidStateError - channel count of ChannelSplitterNode must be set to Explicit"
    );
}

/// Assert that the channel interpretation is valid for the ChannelSplitterNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelinterpretation-constraints>
///
/// # Panics
///
/// This function panics if the mode is not equal to Explicit
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_interpretation(interpretation: ChannelInterpretation) {
    assert!(
        interpretation == ChannelInterpretation::Discrete,
        "InvalidStateError - channel interpretation of ChannelSplitterNode must be set to Discrete"
    );
}

/// Options for constructing a [`ChannelSplitterNode`]
// dictionary ChannelSplitterOptions : AudioNodeOptions {
//   unsigned long numberOfOutputs = 6;
// };
#[derive(Clone, Debug)]
pub struct ChannelSplitterOptions {
    pub number_of_outputs: usize,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelSplitterOptions {
    fn default() -> Self {
        Self {
            number_of_outputs: 6,
            channel_config: ChannelConfigOptions {
                count: 6, // must be same as number_of_outputs
                count_mode: ChannelCountMode::Explicit,
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

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn set_channel_count(&self, count: usize) {
        assert_eq!(
            count,
            self.channel_count(),
            "InvalidStateError - Cannot edit channel count of ChannelSplitterNode"
        );
    }

    fn set_channel_count_mode(&self, mode: ChannelCountMode) {
        assert_valid_channel_count_mode(mode);
        self.channel_config.set_count_mode(mode);
    }

    fn set_channel_interpretation(&self, interpretation: ChannelInterpretation) {
        assert_valid_channel_interpretation(interpretation);
        self.channel_config.set_interpretation(interpretation);
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        self.channel_count()
    }
}

impl ChannelSplitterNode {
    pub fn new<C: BaseAudioContext>(context: &C, mut options: ChannelSplitterOptions) -> Self {
        context.register(move |registration| {
            crate::assert_valid_number_of_channels(options.number_of_outputs);
            options.channel_config.count = options.number_of_outputs;

            assert_valid_channel_count_mode(options.channel_config.count_mode);
            assert_valid_channel_interpretation(options.channel_config.interpretation);

            let node = ChannelSplitterNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let render = ChannelSplitterRenderer {
                number_of_outputs: node.channel_count(),
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
        _params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // single input node
        let input = &inputs[0];

        // assert number of outputs was correctly set by renderer
        assert_eq!(self.number_of_outputs, outputs.len());

        for (i, output) in outputs.iter_mut().enumerate() {
            output.set_number_of_channels(1);

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

#[cfg(test)]
mod tests {
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode};
    use crate::AudioBuffer;

    use float_eq::assert_float_eq;

    #[test]
    fn test_splitter() {
        let sample_rate = 48000.;
        let mut context = OfflineAudioContext::new(1, 128, sample_rate);

        let splitter = context.create_channel_splitter(2);
        // connect the 2nd output to the destination
        splitter.connect_at(&context.destination(), 1, 0);

        // create buffer with sample value 1. left, value -1. right
        let audio_buffer = AudioBuffer::from(vec![vec![1.], vec![-1.]], 48000.);
        let mut src = context.create_buffer_source();
        src.set_buffer(audio_buffer);
        src.set_loop(true);
        src.start();
        src.connect(&splitter);

        let buffer = context.start_rendering_sync();

        let mono = buffer.get_channel_data(0);
        assert_float_eq!(&mono[..], &[-1.; 128][..], abs_all <= 0.);
    }
}
