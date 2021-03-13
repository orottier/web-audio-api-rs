//! Spatialization/Panning primitives

use crate::buffer::{
    AudioBuffer, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};
use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::node::AudioNode;
use crate::param::AutomationEvent;
use crate::param::{AudioParam, AudioParamOptions, AutomationRate};
use crate::process::{AudioParamValues, AudioProcessor};
use crate::AtomicF64;
use crate::SampleRate;

use std::sync::mpsc::Sender;
use std::sync::Arc;

/// AudioParam settings for the carthesian coordinates
const PARAM_OPTS: AudioParamOptions = AudioParamOptions {
    min_value: f32::MIN,
    max_value: f32::MAX,
    default_value: 0.,
    automation_rate: AutomationRate::A,
};

/// Represents the position and orientation of the person listening to the audio scene
///
/// All PannerNode objects spatialize in relation to the [`crate::context::BaseAudioContext`]'s listener.
pub struct AudioListener<'a> {
    pub(crate) position_x: AudioParam<'a>,
    pub(crate) position_y: AudioParam<'a>,
    pub(crate) position_z: AudioParam<'a>,
    pub(crate) forward_x: AudioParam<'a>,
    pub(crate) forward_y: AudioParam<'a>,
    pub(crate) forward_z: AudioParam<'a>,
    pub(crate) up_x: AudioParam<'a>,
    pub(crate) up_y: AudioParam<'a>,
    pub(crate) up_z: AudioParam<'a>,
}

impl<'a> AudioListener<'a> {
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
pub(crate) struct AudioListenerNode<'a> {
    registration: AudioContextRegistration<'a>,
    fields: AudioListener<'a>,
}

impl<'a> AudioNode for AudioListenerNode<'a> {
    fn registration(&self) -> &AudioContextRegistration<'a> {
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

impl<'a> AudioListenerNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C) -> Self {
        context.base().register(move |registration| {
            let reg_id = registration.id();
            let base = context.base();
            let (p1, v1) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p2, v2) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p3, v3) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p4, v4) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p5, v5) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p6, v6) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p7, v7) = base.create_audio_param(PARAM_OPTS, reg_id);
            let (p8, v8) = base.create_audio_param(PARAM_OPTS, reg_id);
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

    pub fn to_fields(self) -> AudioListener<'a> {
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
        _inputs: &[&AudioBuffer],
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
    pub position_x: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub position_y: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub position_z: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub forward_x: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub forward_y: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub forward_z: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub up_x: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub up_y: (Arc<AtomicF64>, Sender<AutomationEvent>),
    pub up_z: (Arc<AtomicF64>, Sender<AutomationEvent>),
}
