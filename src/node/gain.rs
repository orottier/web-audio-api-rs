use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Options for constructing a [`GainNode`]
// dictionary GainOptions : AudioNodeOptions {
//   float gain = 1.0;
// };
#[derive(Clone, Debug)]
pub struct GainOptions {
    pub gain: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self {
            gain: 1.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// AudioNode for volume control
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
        context.register(move |registration| {
            let param_opts = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_opts, &registration);

            param.set_value_at_time(options.gain, 0.);

            let render = GainRenderer { gain: proc };

            let node = GainNode {
                registration,
                channel_config: options.channel_config.into(),
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
        _scope: &RenderScope,
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
