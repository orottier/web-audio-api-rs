use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};

use super::{AudioNode, AudioNodeOptions, ChannelConfig};

/// Options for constructing a [`GainNode`]
// dictionary GainOptions : AudioNodeOptions {
//   float gain = 1.0;
// };
#[derive(Clone, Debug)]
pub struct GainOptions {
    pub gain: f32,
    pub audio_node_options: AudioNodeOptions,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self {
            gain: 1.,
            audio_node_options: AudioNodeOptions::default(),
        }
    }
}

/// AudioNode for volume control
#[derive(Debug)]
pub struct GainNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    gain: AudioParam,
}

impl AudioNode for GainNode {
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
}

impl GainNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: GainOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamDescriptor {
                name: String::new(),
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_opts, &registration);

            param.set_value(options.gain);

            let render = GainRenderer { gain: proc };

            let node = GainNode {
                registration,
                channel_config: options.audio_node_options.into(),
                gain: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }
}

struct GainRenderer {
    gain: AudioParamId,
}

impl AudioProcessor for GainRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        if input.is_silent() {
            output.make_silent();
            return false;
        }

        let gain = params.get(&self.gain);

        // very fast track for mute or pass-through
        if gain.len() == 1 {
            // 1e-6 is -120 dB when close to 0 and ±8.283506e-6 dB when close to 1
            // very probably small enough to not be audible
            let threshold = 1e-6;

            let diff_to_zero = gain[0].abs();
            if diff_to_zero <= threshold {
                output.make_silent();
                return false;
            }

            let diff_to_one = (1. - gain[0]).abs();
            if diff_to_one <= threshold {
                *output = input.clone();
                return false;
            }
        }

        *output = input.clone();

        if gain.len() == 1 {
            let g = gain[0];

            output.channels_mut().iter_mut().for_each(|channel| {
                channel.iter_mut().for_each(|o| *o *= g);
            });
        } else {
            output.channels_mut().iter_mut().for_each(|channel| {
                channel
                    .iter_mut()
                    .zip(gain.iter().cycle())
                    .for_each(|(o, g)| *o *= g);
            });
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::OfflineAudioContext;
    use float_eq::assert_float_eq;

    #[test]
    fn test_audioparam_value_applies_immediately() {
        let context = OfflineAudioContext::new(1, 128, 48000.);
        let options = GainOptions {
            gain: 0.12,
            ..Default::default()
        };
        let src = GainNode::new(&context, options);
        assert_float_eq!(src.gain.value(), 0.12, abs_all <= 0.);
    }
}
