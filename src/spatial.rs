//! Spatialization/Panning primitives
//!
//! Required for panning algorithm, distance and cone effects of [`crate::node::PannerNode`]s

use crate::alloc::AudioBuffer;
use crate::buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::node::AudioNode;
use crate::param::{AudioParam, AudioParamOptions, AutomationEvent, AutomationRate};
use crate::process::{AudioParamValues, AudioProcessor};
use crate::AtomicF64;
use crate::SampleRate;

use std::f32::consts::PI;
use std::sync::Arc;

use crossbeam_channel::Sender;

/// AudioParam settings for the carthesian coordinates
pub(crate) const PARAM_OPTS: AudioParamOptions = AudioParamOptions {
    min_value: f32::MIN,
    max_value: f32::MAX,
    default_value: 0.,
    automation_rate: AutomationRate::A,
};

/// Represents the position and orientation of the person listening to the audio scene
///
/// All PannerNode objects spatialize in relation to the [`crate::context::BaseAudioContext`]'s listener.
pub struct AudioListener {
    pub(crate) position_x: AudioParam,
    pub(crate) position_y: AudioParam,
    pub(crate) position_z: AudioParam,
    pub(crate) forward_x: AudioParam,
    pub(crate) forward_y: AudioParam,
    pub(crate) forward_z: AudioParam,
    pub(crate) up_x: AudioParam,
    pub(crate) up_y: AudioParam,
    pub(crate) up_z: AudioParam,
}

impl AudioListener {
    pub fn position_x(&self) -> &AudioParam {
        &self.position_x
    }
    pub fn position_y(&self) -> &AudioParam {
        &self.position_y
    }
    pub fn position_z(&self) -> &AudioParam {
        &self.position_z
    }
    pub fn forward_x(&self) -> &AudioParam {
        &self.forward_x
    }
    pub fn forward_y(&self) -> &AudioParam {
        &self.forward_y
    }
    pub fn forward_z(&self) -> &AudioParam {
        &self.forward_z
    }
    pub fn up_x(&self) -> &AudioParam {
        &self.up_x
    }
    pub fn up_y(&self) -> &AudioParam {
        &self.up_y
    }
    pub fn up_z(&self) -> &AudioParam {
        &self.up_z
    }
}

/// Wrapper for the [`AudioListener`] so it can be placed in the audio graph.
///
/// This node has no input, but takes the position/orientation AudioParams and copies them into the
/// 9 outputs. The outputs are connected to the PannerNodes (via an AudioParam).
///
/// The AudioListener is always connected to the DestinationNode so each render quantum its
/// positions are recalculated.
pub(crate) struct AudioListenerNode {
    registration: AudioContextRegistration,
    fields: AudioListener,
}

impl AudioNode for AudioListenerNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        unreachable!()
    }

    fn channel_config_cloned(&self) -> ChannelConfig {
        ChannelConfigOptions {
            count: 1,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Discrete,
        }
        .into()
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }

    fn number_of_outputs(&self) -> u32 {
        9 // return all audio params as output
    }

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::Explicit
    }

    fn channel_interpretation(&self) -> ChannelInterpretation {
        ChannelInterpretation::Discrete
    }

    fn channel_count(&self) -> usize {
        1
    }
}

impl AudioListenerNode {
    pub fn new<C: AsBaseAudioContext>(context: &C) -> Self {
        context.base().register(move |registration| {
            let reg_id = registration.id();
            let base = context.base();

            let forward_z_opts = AudioParamOptions {
                default_value: -1.,
                ..PARAM_OPTS
            };
            let up_y_opts = AudioParamOptions {
                default_value: 1.,
                ..PARAM_OPTS
            };

            let (p1, v1) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p2, v2) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p3, v3) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p4, v4) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p5, v5) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p6, v6) = base.create_audio_param(forward_z_opts, reg_id);
            let (p7, v7) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p8, v8) = base.create_audio_param(up_y_opts, reg_id);
            let (p9, v9) = base.create_audio_param(PARAM_OPTS, reg_id);

            let node = Self {
                registration,
                fields: AudioListener {
                    position_x: p1,
                    position_y: p2,
                    position_z: p3,
                    forward_x: p4,
                    forward_y: p5,
                    forward_z: p6,
                    up_x: p7,
                    up_y: p8,
                    up_z: p9,
                },
            };
            let proc = ListenerRenderer {
                position_x: v1,
                position_y: v2,
                position_z: v3,
                forward_x: v4,
                forward_y: v5,
                forward_z: v6,
                up_x: v7,
                up_y: v8,
                up_z: v9,
            };

            (node, Box::new(proc))
        })
    }

    pub fn into_fields(self) -> AudioListener {
        self.fields
    }
}

struct ListenerRenderer {
    position_x: AudioParamId,
    position_y: AudioParamId,
    position_z: AudioParamId,
    forward_x: AudioParamId,
    forward_y: AudioParamId,
    forward_z: AudioParamId,
    up_x: AudioParamId,
    up_y: AudioParamId,
    up_z: AudioParamId,
}

impl AudioProcessor for ListenerRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioBuffer],
        outputs: &mut [AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // for now: persist param values in output, so PannerNodes have access
        outputs[0] = params.get_raw(&self.position_x).clone();
        outputs[1] = params.get_raw(&self.position_y).clone();
        outputs[2] = params.get_raw(&self.position_z).clone();
        outputs[3] = params.get_raw(&self.forward_x).clone();
        outputs[4] = params.get_raw(&self.forward_y).clone();
        outputs[5] = params.get_raw(&self.forward_z).clone();
        outputs[6] = params.get_raw(&self.up_x).clone();
        outputs[7] = params.get_raw(&self.up_y).clone();
        outputs[8] = params.get_raw(&self.up_z).clone();
    }

    fn tail_time(&self) -> bool {
        unreachable!() // will never drop in control thread
    }
}

/// Data holder for the BaseAudioContext so it can reconstruct the AudioListener on request
pub(crate) struct AudioListenerParams {
    pub position_x: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub position_y: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub position_z: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub forward_x: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub forward_y: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub forward_z: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub up_x: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub up_y: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
    pub up_z: (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ),
}

use vecmath::{
    vec3_cross, vec3_dot, vec3_len, vec3_normalized, vec3_scale, vec3_square_len, vec3_sub, Vector3,
};

/// Direction to source position measured from listener in 3D
pub fn azimuth_and_elevation(
    source_position: Vector3<f32>,
    listener_position: Vector3<f32>,
    listener_forward: Vector3<f32>,
    listener_up: Vector3<f32>,
) -> (f32, f32) {
    let relative_pos = vec3_sub(source_position, listener_position);

    // Handle degenerate case if source and listener are at the same point.
    if vec3_square_len(relative_pos) <= f32::MIN_POSITIVE {
        return (0., 0.);
    }

    // Calculate the source-listener vector.
    let source_listener = vec3_normalized(relative_pos);

    // Align axes.
    let listener_right = vec3_cross(listener_forward, listener_up);

    if vec3_square_len(listener_right) == 0. {
        // Handle the case where listener’s 'up' and 'forward' vectors are linearly dependent, in
        // which case 'right' cannot be determined
        return (0., 0.);
    }

    // Determine a unit vector orthogonal to listener’s right, forward
    let listener_right_norm = vec3_normalized(listener_right);
    let listener_forward_norm = vec3_normalized(listener_forward);
    let up = vec3_cross(listener_right_norm, listener_forward_norm);

    // Determine elevation first
    let mut elevation = 90. - 180. * vec3_dot(source_listener, up).acos() / PI;
    if elevation > 90. {
        elevation = 180. - elevation;
    } else if elevation < -90. {
        elevation = -180. - elevation;
    }

    let up_projection = vec3_dot(source_listener, up);
    let projected_source = vec3_sub(source_listener, vec3_scale(up, up_projection));

    // this case is not handled by the spec, so I stole the solution from
    // https://hg.mozilla.org/mozilla-central/rev/1100a5bc013b541c635bc42bd753531e95c952e4
    if vec3_square_len(projected_source) == 0. {
        return (0., elevation);
    }
    let projected_source = vec3_normalized(projected_source);

    let mut azimuth = 180. * vec3_dot(projected_source, listener_right_norm).acos() / PI;

    // Source in front or behind the listener.
    let front_back = vec3_dot(projected_source, listener_forward_norm);
    if front_back < 0. {
        azimuth = 360. - azimuth;
    }

    // Make azimuth relative to "forward" and not "right" listener vector.
    let max270 = std::ops::RangeInclusive::new(0., 270.);
    if max270.contains(&azimuth) {
        azimuth = 90. - azimuth;
    } else {
        azimuth = 450. - azimuth;
    }

    (azimuth, elevation)
}

/// Distance between two points in 3D
pub fn distance(source_position: Vector3<f32>, listener_position: Vector3<f32>) -> f32 {
    vec3_len(vec3_sub(source_position, listener_position))
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    // listener coordinates/directions
    const LP: [f32; 3] = [0., 0., 0.];
    const LF: [f32; 3] = [0., 0., -1.];
    const LU: [f32; 3] = [0., 1., 0.];

    #[test]
    fn azimuth_elevation_equal_pos() {
        let pos = [0., 0., 0.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);

        assert_float_eq!(azimuth, 0., ulps <= 0);
        assert_float_eq!(elevation, 0., ulps <= 0);
    }

    #[test]
    fn azimuth_elevation_horizontal_plane() {
        // horizontal plane is spanned by x-z axes

        let pos = [10., 0., 0.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, 90., abs <= 0.001);
        assert_float_eq!(elevation, 0., ulps <= 0);

        let pos = [-10., 0., 0.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, -90., abs <= 0.001);
        assert_float_eq!(elevation, 0., ulps <= 0);

        let pos = [10., 0., -10.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, 45., abs <= 0.001);
        assert_float_eq!(elevation, 0., ulps <= 0);

        let pos = [-10., 0., -10.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, -45., abs <= 0.001);
        assert_float_eq!(elevation, 0., ulps <= 0);
    }

    #[test]
    fn azimuth_elevation_vertical() {
        let pos = [0., -10., 0.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, 0., ulps <= 1);
        assert_float_eq!(elevation, -90., abs <= 0.001);

        let pos = [0., 10., 0.];
        let (azimuth, elevation) = azimuth_and_elevation(pos, LP, LF, LU);
        assert_float_eq!(azimuth, 0., ulps <= 1);
        assert_float_eq!(elevation, 90., abs <= 0.001);
    }
}
