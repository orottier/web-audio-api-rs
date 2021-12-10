use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::param::{AudioParam, AudioParamOptions};
use crate::renderer::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Options for constructing an ConstantSourceNode
pub struct ConstantSourceOptions {
    pub offset: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ConstantSourceOptions {
    fn default() -> Self {
        Self {
            offset: 1.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Audio source whose output is nominally a constant value
pub struct ConstantSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    offset: AudioParam,
}

impl AudioNode for ConstantSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl ConstantSourceNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: ConstantSourceOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());
            param.set_value(options.offset);

            let render = ConstantSourceRenderer { offset: proc };
            let node = ConstantSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                offset: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn offset(&self) -> &AudioParam {
        &self.offset
    }
}

struct ConstantSourceRenderer {
    pub offset: AudioParamId,
}

impl AudioProcessor for ConstantSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        let offset_values = params.get(&self.offset);

        output.force_mono();
        output.channel_data_mut(0).copy_from_slice(offset_values);

        true
    }
}
