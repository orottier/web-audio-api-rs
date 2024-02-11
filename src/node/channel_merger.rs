use std::fmt::Debug;

use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// Assert that the channel count is valid for the ChannelMergerNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcount-constraints>
///
/// # Panics
///
/// This function panics if given count is greater than 2
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count(count: usize) {
    assert!(
        count == 1,
        "InvalidStateError - channel count of ChannelMergerNode must be equal to 1"
    );
}

/// Assert that the channel count mode is valid for the ChannelMergerNode
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
        "InvalidStateError - channel count of ChannelMergerNode must be set to Explicit"
    );
}

/// Options for constructing a [`ChannelMergerNode`]
// dictionary ChannelMergerOptions : AudioNodeOptions {
//   unsigned long numberOfInputs = 6;
// };
#[derive(Clone, Debug)]
pub struct ChannelMergerOptions {
    pub number_of_inputs: usize,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelMergerOptions {
    fn default() -> Self {
        Self {
            number_of_inputs: 6,
            channel_config: ChannelConfigOptions {
                count: 1,
                count_mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// AudioNode for combining channels from multiple audio streams into a single audio stream.
#[derive(Debug)]
pub struct ChannelMergerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    number_of_inputs: usize,
}

impl AudioNode for ChannelMergerNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn set_channel_count(&self, count: usize) {
        assert_valid_channel_count(count);
        self.channel_config.set_count(count, self.registration());
    }

    fn set_channel_count_mode(&self, mode: ChannelCountMode) {
        assert_valid_channel_count_mode(mode);
        self.channel_config
            .set_count_mode(mode, self.registration());
    }

    fn number_of_inputs(&self) -> usize {
        self.number_of_inputs
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl ChannelMergerNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: ChannelMergerOptions) -> Self {
        context.register(move |registration| {
            crate::assert_valid_number_of_channels(options.number_of_inputs);

            assert_valid_channel_count(options.channel_config.count);
            assert_valid_channel_count_mode(options.channel_config.count_mode);

            let node = ChannelMergerNode {
                registration,
                channel_config: options.channel_config.into(),
                number_of_inputs: options.number_of_inputs,
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
        _params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // [spec] There is a single output whose audio stream has a number of
        // channels equal to the number of inputs when any of the inputs is actively
        // processing. If none of the inputs are actively processing, then output
        // is a single channel of silence.
        if inputs.iter().any(|input| !input.is_silent()) {
            output.set_number_of_channels(inputs.len());

            inputs.iter().enumerate().for_each(|(i, input)| {
                *output.channel_data_mut(i) = input.channel_data(0).clone();
            });
        } else {
            output.make_silent();
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode};

    use float_eq::assert_float_eq;

    #[test]
    fn test_merge() {
        let sample_rate = 48000.;
        let mut context = OfflineAudioContext::new(2, 128, sample_rate);

        let merger = context.create_channel_merger(2);
        merger.connect(&context.destination());

        let mut src1 = context.create_constant_source();
        src1.offset().set_value(2.);
        src1.connect_at(&merger, 0, 0);
        src1.start();

        let mut src2 = context.create_constant_source();
        src2.offset().set_value(3.);
        src2.connect_at(&merger, 0, 1);
        src2.start();

        let buffer = context.start_rendering_sync();

        let left = buffer.get_channel_data(0);
        assert_float_eq!(left, &[2.; 128][..], abs_all <= 0.);

        let right = buffer.get_channel_data(1);
        assert_float_eq!(right, &[3.; 128][..], abs_all <= 0.);
    }

    #[test]
    fn test_merge_disconnect() {
        let sample_rate = 48000.;
        let length = 4 * 128;
        let disconnect_at = length as f64 / sample_rate as f64 / 2.;
        let mut context = OfflineAudioContext::new(2, length, sample_rate);

        let merger = context.create_channel_merger(2);
        merger.connect(&context.destination());

        let mut src1 = context.create_constant_source();
        src1.offset().set_value(2.);
        src1.connect_at(&merger, 0, 0);
        src1.start();

        let mut src2 = context.create_constant_source();
        src2.offset().set_value(3.);
        src2.connect_at(&merger, 0, 1);
        src2.start();

        context.suspend_sync(disconnect_at, move |_| src2.disconnect());

        let buffer = context.start_rendering_sync();

        let left = buffer.get_channel_data(0);
        assert_float_eq!(left, &vec![2.; length][..], abs_all <= 0.);

        let right = buffer.get_channel_data(1);
        assert_float_eq!(
            &right[0..length / 2],
            &vec![3.; length / 2][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            &right[length / 2..],
            &vec![0.; length / 2][..],
            abs_all <= 0.
        );
    }
}
