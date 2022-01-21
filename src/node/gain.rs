use crate::context::{Context,AudioContextRegistration, AudioParamId};
use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Options for constructing a GainNode
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

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl GainNode {
    pub fn new<C: Context>(context: &C, options: GainOptions) -> Self {
        context.register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                
                .create_audio_param(param_opts, registration.id());

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
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        let gain_values = params.get(&self.gain);

        *output = input.clone();

        output.modify_channels(|channel| {
            channel
                .iter_mut()
                .zip(gain_values.iter())
                .for_each(|(value, g)| *value *= g)
        });

        false
    }
}
