use std::f32::consts::PI;

use crate::{
    buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::AudioParam,
    process::{AudioParamValues, AudioProcessor},
};

use super::AudioNode;

/// Options for constructing a PannerNode
#[derive(Default)]
pub struct PannerOptions {
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub forward_x: f32,
    pub forward_y: f32,
    pub forward_z: f32,
    pub up_x: f32,
    pub up_y: f32,
    pub up_z: f32,
}

/// Positions / spatializes an incoming audio stream in three-dimensional space.
pub struct PannerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    position_x: AudioParam,
    position_y: AudioParam,
    position_z: AudioParam,
}

impl AudioNode for PannerNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1 + 9 // todo, user should not be able to see these ports
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl PannerNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: PannerOptions) -> Self {
        context.base().register(move |registration| {
            use crate::spatial::PARAM_OPTS;
            let id = registration.id();
            let (position_x, render_px) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_y, render_py) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_z, render_pz) = context.base().create_audio_param(PARAM_OPTS, id);

            position_x.set_value_at_time(options.position_x, 0.);
            position_y.set_value_at_time(options.position_y, 0.);
            position_z.set_value_at_time(options.position_z, 0.);

            let render = PannerRenderer {
                position_x: render_px,
                position_y: render_py,
                position_z: render_pz,
            };

            let node = PannerNode {
                registration,
                channel_config: ChannelConfigOptions {
                    count: 2,
                    mode: ChannelCountMode::ClampedMax,
                    interpretation: ChannelInterpretation::Speakers,
                }
                .into(),
                position_x,
                position_y,
                position_z,
            };

            context.base().connect_listener_to_panner(node.id());

            (node, Box::new(render))
        })
    }

    pub fn position_x(&self) -> &AudioParam {
        &self.position_x
    }

    pub fn position_y(&self) -> &AudioParam {
        &self.position_y
    }

    pub fn position_z(&self) -> &AudioParam {
        &self.position_z
    }
}

struct PannerRenderer {
    position_x: AudioParamId,
    position_y: AudioParamId,
    position_z: AudioParamId,
}

impl AudioProcessor for PannerRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: f32,
    ) {
        // single input node, assume mono, not silent
        let input = inputs[0].channel_data(0);
        // single output node
        let output = &mut outputs[0];

        // a-rate processing for now

        // source parameters (Panner)
        let source_position_x = params.get(&self.position_x)[0];
        let source_position_y = params.get(&self.position_y)[0];
        let source_position_z = params.get(&self.position_z)[0];

        // listener parameters (AudioListener)
        let l_position_x = inputs[1].channel_data(0)[0];
        let l_position_y = inputs[2].channel_data(0)[0];
        let l_position_z = inputs[3].channel_data(0)[0];
        let l_forward_x = inputs[4].channel_data(0)[0];
        let l_forward_y = inputs[5].channel_data(0)[0];
        let l_forward_z = inputs[6].channel_data(0)[0];
        let l_up_x = inputs[7].channel_data(0)[0];
        let l_up_y = inputs[8].channel_data(0)[0];
        let l_up_z = inputs[9].channel_data(0)[0];

        let (mut azimuth, _elevation) = crate::spatial::azimuth_and_elevation(
            [source_position_x, source_position_y, source_position_z],
            [l_position_x, l_position_y, l_position_z],
            [l_forward_x, l_forward_y, l_forward_z],
            [l_up_x, l_up_y, l_up_z],
        );

        // First, clamp azimuth to allowed range of [-180, 180].
        azimuth = azimuth.max(-180.);
        azimuth = azimuth.min(180.);
        // Then wrap to range [-90, 90].
        if azimuth < -90. {
            azimuth = -180. - azimuth;
        } else if azimuth > 90. {
            azimuth = 180. - azimuth;
        }

        let x = (azimuth + 90.) / 180.;
        let gain_l = (x * PI / 2.).cos();
        let gain_r = (x * PI / 2.).sin();

        let distance = crate::spatial::distance(
            [source_position_x, source_position_y, source_position_z],
            [l_position_x, l_position_y, l_position_z],
        );
        let dist_gain = 1. / distance;

        let left = input.iter().map(|&v| v * gain_l * dist_gain);
        let right = input.iter().map(|&v| v * gain_r * dist_gain);

        output.set_number_of_channels(2);
        output
            .channel_data_mut(0)
            .iter_mut()
            .zip(left)
            .for_each(|(o, i)| *o = i);
        output
            .channel_data_mut(1)
            .iter_mut()
            .zip(right)
            .for_each(|(o, i)| *o = i);
    }

    fn tail_time(&self) -> bool {
        false // only for panning model HRTF
    }
}
