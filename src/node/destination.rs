use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

/// The AudioDestinationNode interface represents the terminal node of an audio
/// graph in a given context. usually the speakers of your device, or the node that
/// will "record" the audio data with an OfflineAudioContext.
///
/// The output of a AudioDestinationNode is produced by summing its input, allowing to capture
/// the output of an AudioContext into, for example, a MediaStreamAudioDestinationNode, or a
/// MediaRecorder.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/AudioDestinationNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#AudioDestinationNode>
/// - see also: [`BaseAudioContext::destination`]
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// let mut osc = context.create_oscillator();
/// osc.connect(&context.destination());
/// osc.start();
/// ```
///
#[derive(Debug)]
pub struct AudioDestinationNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for AudioDestinationNode {
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

    fn set_channel_count(&self, v: usize) {
        assert!(
            !self.registration.context().offline() || v == self.max_channel_count(),
            "NotSupportedError - not allowed to change OfflineAudioContext destination channel count"
        );

        assert!(
            v <= self.max_channel_count(),
            "IndexSizeError - channel count cannot be greater than maxChannelCount ({})",
            self.max_channel_count()
        );

        self.channel_config.set_count(v, self.registration());
    }

    fn set_channel_count_mode(&self, v: ChannelCountMode) {
        // [spec] If the AudioDestinationNode is the destination node of an
        // OfflineAudioContext, then the channel count mode cannot be changed.
        // An InvalidStateError exception MUST be thrown for any attempt to change the value.
        assert!(
            !self.registration.context().offline(),
            "InvalidStateError - AudioDestinationNode has channel count mode constraints",
        );

        self.channel_config.set_count_mode(v, self.registration());
    }
}

impl AudioDestinationNode {
    pub(crate) fn new<C: BaseAudioContext>(context: &C, channel_count: usize) -> Self {
        context.base().register(move |registration| {
            let channel_config = ChannelConfigOptions {
                count: channel_count,
                count_mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            }
            .into();
            let node = Self {
                registration,
                channel_config,
            };
            let proc = DestinationRenderer {};

            (node, Box::new(proc))
        })
    }

    pub(crate) fn into_channel_config(self) -> ChannelConfig {
        self.channel_config
    }

    pub(crate) fn from_raw_parts(
        registration: AudioContextRegistration,
        channel_config: ChannelConfig,
    ) -> Self {
        Self {
            registration,
            channel_config,
        }
    }
    /// The maximum number of channels that the channelCount attribute can be set to (the max
    /// number of channels that the hardware is capable of supporting).
    /// <https://www.w3.org/TR/webaudio/#dom-audiodestinationnode-maxchannelcount>
    pub fn max_channel_count(&self) -> usize {
        self.registration.context().base().max_channel_count()
    }
}

struct DestinationRenderer {}

impl AudioProcessor for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // just move input to output
        *output = input.clone();

        true
    }
}
