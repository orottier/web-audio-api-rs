use std::sync::{atomic::AtomicU32, Arc};

use crate::{
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};

use super::AudioNode;

pub enum BiquadFilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Lowshelf,
    Highshelf,
    Peaking,
    Notch,
    Allpass,
}

impl Default for BiquadFilterType {
    fn default() -> Self {
        BiquadFilterType::Lowpass
    }
}

impl From<u32> for BiquadFilterType {
    fn from(i: u32) -> Self {
        use BiquadFilterType::*;

        match i {
            0 => Lowpass,
            1 => Highpass,
            2 => Bandpass,
            3 => Lowshelf,
            4 => Highshelf,
            5 => Peaking,
            6 => Notch,
            7 => Allpass,
            _ => unreachable!(),
        }
    }
}

pub struct BiquadFilterOptions {
    /// The desired initial value for Q, if None default to 1.
    pub q: Option<f32>,
    /// The desired initial value for detune, if None default to 0.
    pub detune: Option<f32>,
    /// The desired initial value for frequency, if None default to 350.
    pub frequency: Option<f32>,
    /// The desired initial value for gain, if None default to 0.
    pub gain: Option<f32>,
    /// The desired initial value for type, if None default to Lowpass.
    pub type_: Option<BiquadFilterType>,
    /// audio node options
    pub channel_config: ChannelConfigOptions,
}

impl Default for BiquadFilterOptions {
    fn default() -> Self {
        Self {
            q: Some(1.),
            detune: Some(0.),
            frequency: Some(350.),
            gain: Some(0.),
            type_: Some(BiquadFilterType::default()),
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// AudioNode for volume control
pub struct BiquadFilterNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    q: AudioParam,
    detune: AudioParam,
    frequency: AudioParam,
    gain: AudioParam,
    type_: Arc<AtomicU32>,
}

impl AudioNode for BiquadFilterNode {
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

impl BiquadFilterNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<BiquadFilterOptions>) -> Self {
        context.base().register(move |registration| {
            let options = options.unwrap_or_default();

            let default_freq = 350.;
            let default_gain = 0.;
            let default_det = 0.;
            let default_q = 1.;

            let q_param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: default_q,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (q_param, q_proc) = context
                .base()
                .create_audio_param(q_param_opts, registration.id());

            q_param.set_value(options.detune.unwrap_or(default_det));

            let d_param_opts = AudioParamOptions {
                min_value: -153600.,
                max_value: 153600.,
                default_value: default_det,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (d_param, d_proc) = context
                .base()
                .create_audio_param(d_param_opts, registration.id());

            d_param.set_value(options.detune.unwrap_or(default_det));

            let niquyst = context.base().sample_rate().0 / 2;
            let f_param_opts = AudioParamOptions {
                min_value: 0.,
                max_value: niquyst as f32,
                default_value: default_freq,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context
                .base()
                .create_audio_param(f_param_opts, registration.id());

            f_param.set_value(options.frequency.unwrap_or(default_freq));

            let g_param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: default_gain,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (g_param, g_proc) = context
                .base()
                .create_audio_param(g_param_opts, registration.id());

            g_param.set_value(options.gain.unwrap_or(default_gain));

            let render = BiquadFilterRenderer { biquad: g_proc };

            let node = BiquadFilterNode {
                registration,
                channel_config: options.channel_config.into(),
                type_: Arc::new(AtomicU32::new(
                    options.type_.unwrap_or(BiquadFilterType::Lowpass) as u32,
                )),
                q: q_param,
                detune: d_param,
                frequency: f_param,
                gain: g_param,
            };

            (node, Box::new(render))
        })
    }

    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }

    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }
}

struct BiquadFilterRenderer {
    biquad: AudioParamId,
}

impl AudioProcessor for BiquadFilterRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        let biquad_values = params.get(&self.biquad);

        *output = input.clone();

        output.modify_channels(|channel| {
            channel
                .iter_mut()
                .zip(biquad_values.iter())
                .for_each(|(value, g)| *value *= g)
        });
    }

    fn tail_time(&self) -> bool {
        false
    }
}
