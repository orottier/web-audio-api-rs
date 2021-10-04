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
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};

use super::AudioNode;

#[derive(Debug, Clone, Copy)]
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

            let sample_rate = context.base().sample_rate().0 as f32;

            let default_freq = 350.;
            let default_gain = 0.;
            let default_det = 0.;
            let default_q = 1.;

            let q_value = options.detune.unwrap_or(default_det);
            let d_value = options.detune.unwrap_or(default_det);
            let f_value = options.frequency.unwrap_or(default_freq);
            let g_value = options.gain.unwrap_or(default_gain);
            let t_value = options.type_.unwrap_or(BiquadFilterType::Lowpass);

            let q_param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: default_q,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (q_param, q_proc) = context
                .base()
                .create_audio_param(q_param_opts, registration.id());

            q_param.set_value(q_value);

            let d_param_opts = AudioParamOptions {
                min_value: -153600.,
                max_value: 153600.,
                default_value: default_det,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (d_param, d_proc) = context
                .base()
                .create_audio_param(d_param_opts, registration.id());

            d_param.set_value(d_value);

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

            f_param.set_value(f_value);

            let g_param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: default_gain,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (g_param, g_proc) = context
                .base()
                .create_audio_param(g_param_opts, registration.id());

            g_param.set_value(g_value);

            let type_ = Arc::new(AtomicU32::new(t_value as u32));

            let inits = Params {
                q: q_value,
                detune: d_value,
                frequency: f_value,
                gain: g_value,
                type_: t_value,
            };

            let config = RendererConfig {
                sample_rate,
                gain: g_proc,
                detune: d_proc,
                frequency: f_proc,
                q: q_proc,
                type_: type_.clone(),
                params: inits,
            };

            let render = BiquadFilterRenderer::new(config);
            let node = BiquadFilterNode {
                registration,
                channel_config: options.channel_config.into(),
                type_,
                q: q_param,
                detune: d_param,
                frequency: f_param,
                gain: g_param,
            };

            (node, Box::new(render))
        })
    }

    /// Returns the gain audio paramter
    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }

    /// Returns the frequency audio paramter
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the detune audio paramter
    pub fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Returns the Q audio paramter
    pub fn q(&self) -> &AudioParam {
        &self.q
    }

    /// Returns the biquad filter type
    pub fn type_(&self) -> BiquadFilterType {
        self.type_.load(Ordering::SeqCst).into()
    }

    /// biquad filter type setter
    ///
    /// # Arguments
    ///
    /// * `type_` - the biquad filter type (lowpass, highpass,...)
    pub fn set_type(&mut self, type_: BiquadFilterType) {
        self.type_.store(type_ as u32, Ordering::SeqCst);
    }

    /// Returns the frequency response for the specified frequencies
    ///
    /// # Arguments
    ///
    /// * `frequency_hz` - frequencies for which frequency response of the filter should be calculated
    /// * `mag_response` - magnitude of the frequency response of the filter
    /// * `phase_response` - phase of the frequency response of the filter
    pub fn get_frequency_response(
        &self,
        frequency_hz: Vec<f32>,
        mag_response: Vec<f32>,
        phase_response: Vec<f32>,
    ) {
        todo!()
    }
}

struct Params {
    q: f32,
    detune: f32,
    frequency: f32,
    gain: f32,
    type_: BiquadFilterType,
}

struct RendererConfig {
    sample_rate: f32,
    q: AudioParamId,
    detune: AudioParamId,
    frequency: AudioParamId,
    gain: AudioParamId,
    type_: Arc<AtomicU32>,
    params: Params,
}

/// Biquad filter coefficients
#[derive(Clone, Copy, Debug)]
struct Coefficients {
    // Denominator coefficients
    a0: f32,
    a1: f32,
    a2: f32,

    // Nominator coefficients
    b0: f32,
    b1: f32,
    b2: f32,
}

struct BiquadFilterRenderer {
    sample_rate: f32,
    q: AudioParamId,
    detune: AudioParamId,
    frequency: AudioParamId,
    gain: AudioParamId,
    type_: Arc<AtomicU32>,
    s1: f32,
    s2: f32,
    coeffs: Coefficients,
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

        let g_values = params.get(&self.gain);
        let det_values = params.get(&self.detune);
        let freq_values = params.get(&self.frequency);
        let q_values = params.get(&self.q);
    }

    fn tail_time(&self) -> bool {
        false
    }
}

impl BiquadFilterRenderer {
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            sample_rate,
            q,
            detune,
            frequency,
            gain,
            type_,
            params,
        } = config;

        let coeffs = Self::init_coeffs(params);

        Self {
            sample_rate,
            gain,
            detune,
            frequency,
            q,
            type_,
            s1: 0.,
            s2: 0.,
            coeffs,
        }
    }

    fn tick(&mut self, input: f32) -> f32 {
        self.update_coeffs();
        let out = self.s1 + self.coeffs.b0 * input;
        self.s1 = self.s2 + self.coeffs.b1 * input - self.coeffs.a1 * out;
        self.s2 = self.coeffs.b2 * input - self.coeffs.a2 * out;

        out
    }

    fn init_coeffs(params: Params) -> Coefficients {
        todo!()
    }

    fn a(gain: f32) -> f32 {
        10f32.powf(gain / 40.)
    }

    fn w0(sample_rate: f32, computed_freq: f32) -> f32 {
        2.0 * PI * computed_freq / sample_rate
    }

    fn alpha_q(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * q)
    }

    fn alpha_q_db(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * 10f32.powf(q / 20.))
    }

    fn s() -> f32 {
        1.0
    }

    fn alpha_s(sample_rate: f32, computed_freq: f32, gain: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        let a = Self::a(gain);
        let s = Self::s();

        (w0.sin() / 2.0) * ((a + (1. / a)) * ((1. / s) - 1.0) + 2.0)
    }

    fn update_coeffs(&mut self) {
        todo!()
    }

    fn b0(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::b0_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b0_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::b0_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::b0_notch(),
            BiquadFilterType::Allpass => Self::b0_allpass(sample_rate, computed_freq, q),
            _ => todo!(),
        }
    }

    fn b0_lowpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    fn b0_highpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    fn b0_bandpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        Self::alpha_q(sample_rate, computed_freq, q)
    }

    fn b0_notch() -> f32 {
        1.0
    }

    fn b0_allpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 - alpha_q
    }

    fn b1(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::b1_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b1_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::b1_bandpass(),
            BiquadFilterType::Notch => Self::b1_notch(sample_rate, computed_freq),
            BiquadFilterType::Allpass => Self::b1_allpass(sample_rate, computed_freq),
            _ => todo!(),
        }
    }

    fn b1_lowpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        1.0 - w0.cos()
    }

    fn b1_highpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -(1.0 + w0.cos())
    }

    fn b1_bandpass() -> f32 {
        0.0
    }

    fn b1_notch(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn b1_allpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn b2(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::b2_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b2_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::b2_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::b2_notch(),
            BiquadFilterType::Allpass => Self::b2_allpass(sample_rate, computed_freq, q),
            _ => todo!(),
        }
    }

    fn b2_lowpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    fn b2_highpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    fn b2_bandpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        -Self::alpha_q(sample_rate, computed_freq, q)
    }

    fn b2_notch() -> f32 {
        1.0
    }

    fn b2_allpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    fn a0(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::a0_lowpass(sample_rate, computed_freq, q),
            BiquadFilterType::Highpass => Self::a0_highpass(sample_rate, computed_freq, q),
            BiquadFilterType::Bandpass => Self::a0_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::a0_notch(sample_rate, computed_freq, q),
            BiquadFilterType::Allpass => Self::a0_allpass(sample_rate, computed_freq, q),
            _ => todo!(),
        }
    }

    fn a0_lowpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 + alpha_q_db
    }

    fn a0_highpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 + alpha_q_db
    }

    fn a0_bandpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    fn a0_notch(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    fn a0_allpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    fn a1(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::a1_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::a1_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::a1_bandpass(sample_rate, computed_freq),
            BiquadFilterType::Notch => Self::a1_notch(sample_rate, computed_freq),
            BiquadFilterType::Allpass => Self::a1_allpass(sample_rate, computed_freq),
            _ => todo!(),
        }
    }

    fn a1_lowpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn a1_highpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn a1_bandpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn a1_notch(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn a1_allpass(sample_rate: f32, computed_freq: f32) -> f32 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    fn a2(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        match type_ {
            BiquadFilterType::Lowpass => Self::a2_lowpass(sample_rate, computed_freq, q),
            BiquadFilterType::Highpass => Self::a2_lowpass(sample_rate, computed_freq, q),
            BiquadFilterType::Bandpass => Self::a2_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::a2_notch(sample_rate, computed_freq, q),
            BiquadFilterType::Allpass => Self::a2_allpass(sample_rate, computed_freq, q),
            _ => todo!(),
        }
    }

    fn a2_lowpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    fn a2_highpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    fn a2_bandpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    fn a2_notch(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    fn a2_allpass(sample_rate: f32, computed_freq: f32, q: f32) -> f32 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }
}
