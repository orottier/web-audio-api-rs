use std::f32::consts::PI;
use std::sync::Arc;

use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{AtomicF64, SampleRate};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};

#[derive(Copy, Clone, Debug)]
pub enum PanningModelType {
    EqualPower,
    HRTF,
}

#[derive(Copy, Clone, Debug)]
pub enum DistanceModelType {
    Linear,
    Inverse,
    Exponential,
}

/// Options for constructing a [`PannerNode`]
// dictionary PannerOptions : AudioNodeOptions {
//   PanningModelType panningModel = "equalpower";
//   DistanceModelType distanceModel = "inverse";
//   float positionX = 0;
//   float positionY = 0;
//   float positionZ = 0;
//   float orientationX = 1;
//   float orientationY = 0;
//   float orientationZ = 0;
//   double refDistance = 1;
//   double maxDistance = 10000;
//   double rolloffFactor = 1;
//   double coneInnerAngle = 360;
//   double coneOuterAngle = 360;
//   double coneOuterGain = 0;
// };
#[derive(Clone, Debug)]
pub struct PannerOptions {
    #[allow(dead_code)]
    pub panning_model: PanningModelType,
    #[allow(dead_code)]
    pub distance_model: DistanceModelType,
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub orientation_x: f32,
    pub orientation_y: f32,
    pub orientation_z: f32,
    #[allow(dead_code)]
    pub ref_distance: f64,
    #[allow(dead_code)]
    pub max_distance: f64,
    #[allow(dead_code)]
    pub rolloff_factor: f64,
    pub cone_inner_angle: f64,
    pub cone_outer_angle: f64,
    pub cone_outer_gain: f64,
}

impl Default for PannerOptions {
    fn default() -> Self {
        PannerOptions {
            panning_model: PanningModelType::EqualPower,
            distance_model: DistanceModelType::Inverse,
            position_x: 0.,
            position_y: 0.,
            position_z: 0.,
            orientation_x: 1.,
            orientation_y: 0.,
            orientation_z: 0.,
            ref_distance: 1.,
            max_distance: 10000.,
            rolloff_factor: 1.,
            cone_inner_angle: 360.,
            cone_outer_angle: 360.,
            cone_outer_gain: 0.,
        }
    }
}

/// Node that positions / spatializes an incoming audio stream in three-dimensional space.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/PannerNode>
/// - specification: <https://www.w3.org/TR/webaudio/#pannernode> and
/// <https://www.w3.org/TR/webaudio/#Spatialization>
/// - see also:
/// [`BaseAudioContext::create_panner`](crate::context::BaseAudioContext::create_panner)
///
/// # Usage
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::AudioNode;
/// use web_audio_api::node::AudioScheduledSourceNode;
///
/// // Setup a new audio context
/// let context = AudioContext::new(None);
///
/// // Create a friendly tone
/// let tone = context.create_oscillator();
/// tone.frequency().set_value_at_time(300.0f32, 0.);
/// tone.start();
///
/// // Connect tone > panner node > destination node
/// let panner = context.create_panner();
/// tone.connect(&panner);
/// panner.connect(&context.destination());
///
/// // The panner node is 1 unit in front of listener
/// panner.position_z().set_value_at_time(1., 0.);
///
/// // And sweeps 10 units left to right, every second
/// let moving = context.create_oscillator();
/// moving.start();
/// moving.frequency().set_value_at_time(1., 0.);
/// let gain = context.create_gain();
/// gain.gain().set_value_at_time(10., 0.);
/// moving.connect(&gain);
/// gain.connect(panner.position_x());
///
/// // enjoy listening
/// std::thread::sleep(std::time::Duration::from_secs(4));
/// ```
///
/// # Examples
///
/// - `cargo run --release --example spatial`
/// - `cargo run --release --example panner_cone`
pub struct PannerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    position_x: AudioParam,
    position_y: AudioParam,
    position_z: AudioParam,
    orientation_x: AudioParam,
    orientation_y: AudioParam,
    orientation_z: AudioParam,
    cone_inner_angle: Arc<AtomicF64>,
    cone_outer_angle: Arc<AtomicF64>,
    cone_outer_gain: Arc<AtomicF64>,
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
    pub fn new<C: BaseAudioContext>(context: &C, options: PannerOptions) -> Self {
        let node = context.base().register(move |registration| {
            let id = registration.id();

            use crate::spatial::PARAM_OPTS;
            // position params
            let (position_x, render_px) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_y, render_py) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_z, render_pz) = context.base().create_audio_param(PARAM_OPTS, id);
            position_x.set_value_at_time(options.position_x, 0.);
            position_y.set_value_at_time(options.position_y, 0.);
            position_z.set_value_at_time(options.position_z, 0.);

            // orientation params
            let orientation_x_opts = AudioParamDescriptor {
                default_value: 1.0,
                ..PARAM_OPTS
            };
            let (orientation_x, render_ox) =
                context.base().create_audio_param(orientation_x_opts, id);
            let (orientation_y, render_oy) = context.base().create_audio_param(PARAM_OPTS, id);
            let (orientation_z, render_oz) = context.base().create_audio_param(PARAM_OPTS, id);
            orientation_x.set_value_at_time(options.orientation_x, 0.);
            orientation_y.set_value_at_time(options.orientation_y, 0.);
            orientation_z.set_value_at_time(options.orientation_z, 0.);

            // cone attributes
            let cone_inner_angle = Arc::new(AtomicF64::new(options.cone_inner_angle));
            let cone_outer_angle = Arc::new(AtomicF64::new(options.cone_outer_angle));
            let cone_outer_gain = Arc::new(AtomicF64::new(options.cone_outer_gain));

            let render = PannerRenderer {
                position_x: render_px,
                position_y: render_py,
                position_z: render_pz,
                orientation_x: render_ox,
                orientation_y: render_oy,
                orientation_z: render_oz,
                cone_inner_angle: cone_inner_angle.clone(),
                cone_outer_angle: cone_outer_angle.clone(),
                cone_outer_gain: cone_outer_gain.clone(),
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
                orientation_x,
                orientation_y,
                orientation_z,
                cone_inner_angle,
                cone_outer_angle,
                cone_outer_gain,
            };

            (node, Box::new(render))
        });

        // after the node is registered, connect the AudioListener
        context.base().connect_listener_to_panner(node.id());

        node
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

    pub fn orientation_x(&self) -> &AudioParam {
        &self.orientation_x
    }

    pub fn orientation_y(&self) -> &AudioParam {
        &self.orientation_y
    }

    pub fn orientation_z(&self) -> &AudioParam {
        &self.orientation_z
    }

    pub fn cone_inner_angle(&self) -> f64 {
        self.cone_inner_angle.load()
    }

    pub fn set_cone_inner_angle(&self, value: f64) {
        self.cone_inner_angle.store(value);
    }

    pub fn cone_outer_angle(&self) -> f64 {
        self.cone_outer_angle.load()
    }

    pub fn set_cone_outer_angle(&self, value: f64) {
        self.cone_outer_angle.store(value);
    }

    pub fn cone_outer_gain(&self) -> f64 {
        self.cone_outer_gain.load()
    }

    pub fn set_cone_outer_gain(&self, value: f64) {
        self.cone_outer_gain.store(value);
    }
}

struct PannerRenderer {
    position_x: AudioParamId,
    position_y: AudioParamId,
    position_z: AudioParamId,
    orientation_x: AudioParamId,
    orientation_y: AudioParamId,
    orientation_z: AudioParamId,
    cone_inner_angle: Arc<AtomicF64>,
    cone_outer_angle: Arc<AtomicF64>,
    cone_outer_gain: Arc<AtomicF64>,
}

impl AudioProcessor for PannerRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // Single input node
        // assume mono (todo issue #44)
        let input = inputs[0].channel_data(0);

        // Single output node
        let output = &mut outputs[0];

        // K-rate processing for now (todo issue #44)

        // source parameters (Panner)
        let source_position_x = params.get(&self.position_x)[0];
        let source_position_y = params.get(&self.position_y)[0];
        let source_position_z = params.get(&self.position_z)[0];
        let source_orientation_x = params.get(&self.orientation_x)[0];
        let source_orientation_y = params.get(&self.orientation_y)[0];
        let source_orientation_z = params.get(&self.orientation_z)[0];

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

        // define base vectors in 3D
        let source_position = [source_position_x, source_position_y, source_position_z];
        let source_orientation = [
            source_orientation_x,
            source_orientation_y,
            source_orientation_z,
        ];
        let listener_position = [l_position_x, l_position_y, l_position_z];
        let listener_forward = [l_forward_x, l_forward_y, l_forward_z];
        let listener_up = [l_up_x, l_up_y, l_up_z];

        // azimuth and elevation of listener <> panner.
        // elevation is not used in the equal power panningModel (todo issue #44)
        let (mut azimuth, _elevation) = crate::spatial::azimuth_and_elevation(
            source_position,
            listener_position,
            listener_forward,
            listener_up,
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

        // determine left/right ear gain
        let x = (azimuth + 90.) / 180.;
        let gain_l = (x * PI / 2.).cos();
        let gain_r = (x * PI / 2.).sin();

        // determine distance gain
        let distance = crate::spatial::distance(source_position, listener_position);
        let dist_gain = 1. / distance; // inverse distance model is assumed (todo issue #44)

        // determine cone effect gain
        let abs_inner_angle = self.cone_inner_angle.load().abs() as f32 / 2.;
        let abs_outer_angle = self.cone_outer_angle.load().abs() as f32 / 2.;
        let cone_gain = if abs_inner_angle >= 180. && abs_outer_angle >= 180. {
            1. // no cone specified
        } else {
            let cone_outer_gain = self.cone_outer_gain.load() as f32;

            let abs_angle =
                crate::spatial::angle(source_position, source_orientation, listener_position);

            if abs_angle < abs_inner_angle {
                1. // No attenuation
            } else if abs_angle >= abs_outer_angle {
                cone_outer_gain // Max attenuation
            } else {
                // Between inner and outer cones: inner -> outer, x goes from 0 -> 1
                let x = (abs_angle - abs_inner_angle) / (abs_outer_angle - abs_inner_angle);
                (1. - x) + cone_outer_gain * x
            }
        };

        // multiply signal with gain per ear
        let left = input.iter().map(|&v| v * gain_l * dist_gain * cone_gain);
        let right = input.iter().map(|&v| v * gain_r * dist_gain * cone_gain);

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

        false // only true for panning model HRTF
    }
}
