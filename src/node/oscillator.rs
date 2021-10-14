use std::{
    f32::consts::PI,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use crate::{
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    control::Scheduler,
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};

use super::{AudioNode, AudioScheduledSourceNode};

/// Options for constructing an OscillatorNode
pub struct OscillatorOptions {
    pub type_: OscillatorType,
    pub frequency: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Waveform of an oscillator
#[derive(Copy, Clone)]
pub enum OscillatorType {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Custom,
}

impl Default for OscillatorType {
    fn default() -> Self {
        OscillatorType::Sine
    }
}

impl From<u32> for OscillatorType {
    fn from(i: u32) -> Self {
        use OscillatorType::*;

        match i {
            0 => Sine,
            1 => Square,
            2 => Sawtooth,
            3 => Triangle,
            4 => Custom,
            _ => unreachable!(),
        }
    }
}

/// Audio source generating a periodic waveform
pub struct OscillatorNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    frequency: AudioParam,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
}

impl AudioScheduledSourceNode for OscillatorNode {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl AudioNode for OscillatorNode {
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

impl OscillatorNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: OscillatorOptions) -> Self {
        context.base().register(move |registration| {
            let nyquist = context.base().sample_rate().0 as f32 / 2.;
            let param_opts = AudioParamOptions {
                min_value: -nyquist,
                max_value: nyquist,
                default_value: 440.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());
            f_param.set_value(options.frequency);

            let type_ = Arc::new(AtomicU32::new(options.type_ as u32));
            let scheduler = Scheduler::new();

            let render = OscillatorRenderer {
                frequency: f_proc,
                type_: type_.clone(),
                scheduler: scheduler.clone(),
            };
            let node = OscillatorNode {
                registration,
                channel_config: options.channel_config.into(),
                frequency: f_param,
                type_,
                scheduler,
            };

            (node, Box::new(render))
        })
    }

    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    pub fn type_(&self) -> OscillatorType {
        self.type_.load(Ordering::SeqCst).into()
    }

    pub fn set_type(&self, type_: OscillatorType) {
        self.type_.store(type_ as u32, Ordering::SeqCst);
    }
}

struct OscillatorRenderer {
    frequency: AudioParamId,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
}

impl AudioProcessor for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];

        // re-use previous buffer
        output.force_mono();

        // todo, sub-quantum start/stop
        if !self.scheduler.is_active(timestamp) {
            output.make_silent();
            return;
        }

        let freq_values = params.get(&self.frequency);
        let freq = freq_values[0]; // force a-rate processing

        let type_ = self.type_.load(Ordering::SeqCst).into();

        let buffer = output.channel_data_mut(0);
        let io = buffer
            .iter_mut()
            .enumerate()
            .map(move |(i, v)| (timestamp as f32 + i as f32 / sample_rate.0 as f32, v));

        use OscillatorType::*;

        match type_ {
            Sine => io.for_each(|(t, o)| *o = (2. * PI * freq * t).sin()),
            Square => io.for_each(|(t, o)| *o = if (freq * t).fract() < 0.5 { 1. } else { -1. }),
            Sawtooth => io.for_each(|(t, o)| *o = 2. * ((freq * t).fract() - 0.5)),
            _ => todo!(),
        }
    }

    fn tail_time(&self) -> bool {
        true
    }
}
