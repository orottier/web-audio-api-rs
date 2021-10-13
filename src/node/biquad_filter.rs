use std::{
    f64::consts::{PI, SQRT_2},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use crossbeam_channel::{Receiver, Sender};
use num_complex::Complex;

use crate::{
    alloc::AudioBuffer,
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate, MAX_CHANNELS,
};

use super::AudioNode;

struct CoeffsReq(Sender<[f64; 5]>);

#[derive(Debug, Clone, Copy, PartialEq)]
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
    sample_rate: f32,
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    q: AudioParam,
    detune: AudioParam,
    frequency: AudioParam,
    gain: AudioParam,
    type_: Arc<AtomicU32>,
    sender: Sender<CoeffsReq>,
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

            let q_value = options.q.unwrap_or(default_det);
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

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let config = RendererConfig {
                sample_rate,
                gain: g_proc,
                detune: d_proc,
                frequency: f_proc,
                q: q_proc,
                type_: type_.clone(),
                params: inits,
                receiver,
            };

            let render = BiquadFilterRenderer::new(config);
            let node = BiquadFilterNode {
                sample_rate,
                registration,
                channel_config: options.channel_config.into(),
                type_,
                q: q_param,
                detune: d_param,
                frequency: f_param,
                gain: g_param,
                sender,
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
        frequency_hz: &mut [f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        self.validate_inputs(frequency_hz, mag_response, phase_response);
        let (sender, receiver) = crossbeam_channel::bounded(0);
        self.sender
            .send(CoeffsReq(sender))
            .expect("Sending CoeffsReq failed");

        loop {
            match receiver.try_recv() {
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    println!("Receiver Error: disconnected type");
                    continue;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    println!("Receiver Error: empty type");
                    continue;
                }
                Ok([b0, b1, b2, a1, a2]) => {
                    println!("received...");
                    for (i, &f) in frequency_hz.iter().enumerate() {
                        let f = f as f64;
                        let sample_rate = self.sample_rate as f64;
                        let num = b0
                            + Complex::from_polar(b1, -1.0 * 2.0 * PI * f / sample_rate)
                            + Complex::from_polar(b2, -2.0 * 2.0 * PI * f / sample_rate);
                        let denom = 1.0
                            + Complex::from_polar(a1, -1.0 * 2.0 * PI * f / sample_rate)
                            + Complex::from_polar(a2, -2.0 * 2.0 * PI * f / sample_rate);
                        let h_f = num / denom;

                        mag_response[i] = h_f.norm() as f32;
                        phase_response[i] = h_f.arg() as f32
                    }
                    break;
                }
            }
        }
    }

    #[inline]
    fn validate_inputs(
        &self,
        frequency_hz: &mut [f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        assert_eq!(
            frequency_hz.len(),
            mag_response.len(),
            " InvalidAccessError: All paramaters should be the same length"
        );
        assert_eq!(
            mag_response.len(),
            phase_response.len(),
            " InvalidAccessError: All paramaters should be the same length"
        );

        // Ensures that given frequencies are in the correct range
        let min = 0.;
        let max = self.sample_rate / 2.;
        for f in frequency_hz.iter_mut() {
            *f = f.clamp(min, max);
        }
    }

    /// Mock of `get_frequency_response`
    /// This function is the same as `get_frequency_response` except it never send the `CoeffsReq`.
    /// In tests, we use OfflineAudioContext and in this context the CoeffsReq is not sendable.
    ///
    /// # Arguments
    ///
    /// * `frequency_hz` - frequencies for which frequency response of the filter should be calculated
    /// * `mag_response` - magnitude of the frequency response of the filter
    /// * `phase_response` - phase of the frequency response of the filter
    #[cfg(test)]
    fn get_frequency_response_mock(
        &self,
        frequency_hz: &mut [f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        self.validate_inputs(frequency_hz, mag_response, phase_response);
    }
}

#[derive(Debug)]
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
    receiver: Receiver<CoeffsReq>,
}

/// Biquad filter coefficients
#[derive(Clone, Copy, Debug)]
struct Coefficients {
    // Denominator coefficients
    a1: f64,
    a2: f64,

    // Nominator coefficients
    b0: f64,
    b1: f64,
    b2: f64,
}

struct BiquadFilterRenderer {
    sample_rate: f32,
    q: AudioParamId,
    detune: AudioParamId,
    frequency: AudioParamId,
    gain: AudioParamId,
    type_: Arc<AtomicU32>,
    ss1: [f64; MAX_CHANNELS],
    ss2: [f64; MAX_CHANNELS],
    coeffs: Coefficients,
    receiver: Receiver<CoeffsReq>,
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

        self.filter(input, output, g_values, det_values, freq_values, q_values);
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
            receiver,
        } = config;

        let coeffs = Self::init_coeffs(sample_rate, params);

        let s1 = [0.; MAX_CHANNELS];
        let s2 = [0.; MAX_CHANNELS];

        Self {
            sample_rate,
            gain,
            detune,
            frequency,
            q,
            type_,
            ss1: s1,
            ss2: s2,
            coeffs,
            receiver,
        }
    }

    /// Generate an output by filtering the input following the params values
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `output` - Audiobuffer output
    /// * `params` - biquadfilter params which resolves into biquad coeffs
    #[inline]
    fn filter(
        &mut self,
        input: &AudioBuffer,
        output: &mut AudioBuffer,
        g_values: &[f32],
        det_values: &[f32],
        freq_values: &[f32],
        q_values: &[f32],
    ) {
        let Coefficients { b0, b1, b2, a1, a2 } = self.coeffs;

        let coeffs_resp = [b0, b1, b2, a1, a2];

        // Respond to request at K-rate following the specs
        if let Ok(msg) = self.receiver.try_recv() {
            let sender = msg.0;
            sender.send(coeffs_resp).unwrap();
        }

        for (idx, (i_data, o_data)) in input
            .channels()
            .iter()
            .zip(output.channels_mut())
            .enumerate()
        {
            let p = Params {
                q: q_values[idx],
                detune: det_values[idx],
                frequency: freq_values[idx],
                gain: g_values[idx],
                type_: BiquadFilterType::from(self.type_.load(Ordering::SeqCst)),
            };

            // A-rate params
            self.update_coeffs(p);

            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i, idx);
            }
        }
    }

    /// Generate an output sample by filtering an input sample
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `idx` - channel index mapping to the filter state index
    #[inline]
    fn tick(&mut self, input: f32, idx: usize) -> f32 {
        let input = input as f64;
        let out = self.ss1[idx] + self.coeffs.b0 * input;
        self.ss1[idx] = self.ss2[idx] + self.coeffs.b1 * input - self.coeffs.a1 * out;
        self.ss2[idx] = self.coeffs.b2 * input - self.coeffs.a2 * out;

        out as f32
    }

    /// initializes biquad filter coefficients
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio context sample rate
    /// * `params` - params resolving into biquad coeffs
    #[inline]
    fn init_coeffs(sample_rate: f32, params: Params) -> Coefficients {
        let Params {
            q,
            detune,
            frequency,
            gain,
            type_,
        } = params;

        let computed_freq = frequency * 10f32.powf(detune / 1200.);

        // compute a0 first to normalize others coeffs by a0
        let a0 = Self::a0(type_, sample_rate, computed_freq, q, gain);

        let b0 = Self::b0(type_, sample_rate, computed_freq, q, gain) / a0;
        let b1 = Self::b1(type_, sample_rate, computed_freq, gain) / a0;
        let b2 = Self::b2(type_, sample_rate, computed_freq, q, gain) / a0;

        let a1 = Self::a1(type_, sample_rate, computed_freq, gain) / a0;
        let a2 = Self::a2(type_, sample_rate, computed_freq, q, gain) / a0;

        Coefficients { b0, b1, b2, a1, a2 }
    }

    /// updates biquad filter coefficients when params are modified
    ///
    /// # Arguments
    ///
    /// * `params` - params resolving into biquad coeffs
    #[inline]
    fn update_coeffs(&mut self, params: Params) {
        let Params {
            q,
            detune,
            frequency,
            gain,
            type_,
        } = params;

        let computed_freq = frequency * 10f32.powf(detune / 1200.);

        // compute a0 first to normalize others coeffs by a0
        let a0 = Self::a0(type_, self.sample_rate, computed_freq, q, gain);

        self.coeffs.b0 = Self::b0(type_, self.sample_rate, computed_freq, q, gain) / a0;
        self.coeffs.b1 = Self::b1(type_, self.sample_rate, computed_freq, gain) / a0;
        self.coeffs.b2 = Self::b2(type_, self.sample_rate, computed_freq, q, gain) / a0;
        self.coeffs.a1 = Self::a1(type_, self.sample_rate, computed_freq, gain) / a0;
        self.coeffs.a2 = Self::a2(type_, self.sample_rate, computed_freq, q, gain) / a0;
    }

    /// calculates b_0 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn b0(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let q = q as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => Self::b0_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b0_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::b0_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => 1.0,
            BiquadFilterType::Allpass => Self::b0_allpass(sample_rate, computed_freq, q),
            BiquadFilterType::Peaking => Self::b0_peaking(sample_rate, computed_freq, q, gain),
            BiquadFilterType::Lowshelf => Self::b0_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::b0_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn b0_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    #[inline]
    fn b0_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    #[inline]
    fn b0_bandpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::alpha_q(sample_rate, computed_freq, q)
    }

    #[inline]
    fn b0_allpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 - alpha_q
    }

    #[inline]
    fn b0_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 + alpha_q * a
    }

    #[inline]
    fn b0_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a + 1.0) - (a - 1.0) * w0.cos() + 2.0 * alpha_s * a.sqrt())
    }

    #[inline]
    fn b0_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a + 1.0) + (a - 1.0) * w0.cos() + 2.0 * alpha_s * a.sqrt())
    }

    /// calculates b_1 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `gain` - filter gain
    #[inline]
    fn b1(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => Self::b1_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b1_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => 0.0,
            BiquadFilterType::Notch => Self::b1_notch_all_peak(sample_rate, computed_freq),
            BiquadFilterType::Allpass => Self::b1_notch_all_peak(sample_rate, computed_freq),
            BiquadFilterType::Peaking => Self::b1_notch_all_peak(sample_rate, computed_freq),
            BiquadFilterType::Lowshelf => Self::b1_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::b1_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn b1_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        1.0 - w0.cos()
    }

    #[inline]
    fn b1_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -(1.0 + w0.cos())
    }

    #[inline]
    fn b1_notch_all_peak(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    #[inline]
    fn b1_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        2.0 * a * ((a - 1.0) - (a + 1.0) * w0.cos())
    }

    #[inline]
    fn b1_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * a * ((a - 1.0) + (a + 1.0) * w0.cos())
    }

    /// calculates b_2 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn b2(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let q = q as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => Self::b2_lowpass(sample_rate, computed_freq),
            BiquadFilterType::Highpass => Self::b2_highpass(sample_rate, computed_freq),
            BiquadFilterType::Bandpass => Self::b2_bandpass(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => 1.0,
            BiquadFilterType::Allpass => Self::b2_allpass(sample_rate, computed_freq, q),
            BiquadFilterType::Peaking => Self::b2_peaking(sample_rate, computed_freq, q, gain),
            BiquadFilterType::Lowshelf => Self::b2_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::b2_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn b2_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    #[inline]
    fn b2_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    #[inline]
    fn b2_bandpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        -Self::alpha_q(sample_rate, computed_freq, q)
    }

    #[inline]
    fn b2_allpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    #[inline]
    fn b2_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 - alpha_q * a
    }

    #[inline]
    fn b2_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * alpha_s * a.sqrt())
    }

    #[inline]
    fn b2_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a + 1.0) + (a - 1.0) * w0.cos() - 2.0 * alpha_s * a.sqrt())
    }

    /// calculates a_0 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn a0(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let q = q as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => Self::a0_lp_hp(sample_rate, computed_freq, q),
            BiquadFilterType::Highpass => Self::a0_lp_hp(sample_rate, computed_freq, q),
            BiquadFilterType::Bandpass => Self::a0_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::a0_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Allpass => Self::a0_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Peaking => Self::a0_peaking(sample_rate, computed_freq, q, gain),
            BiquadFilterType::Lowshelf => Self::a0_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::a0_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn a0_lp_hp(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 + alpha_q_db
    }

    #[inline]
    fn a0_bp_notch_all(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    #[inline]
    fn a0_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 + (alpha_q / a)
    }

    #[inline]
    fn a0_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (a + 1.0) + (a - 1.0) * w0.cos() + 2.0 * alpha_s * a.sqrt()
    }

    #[inline]
    fn a0_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (a + 1.0) - (a - 1.0) * w0.cos() + 2.0 * alpha_s * a.sqrt()
    }

    /// calculates a_1 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `gain` - filter gain
    #[inline]
    fn a1(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            BiquadFilterType::Highpass => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            BiquadFilterType::Bandpass => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            BiquadFilterType::Notch => Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq),
            BiquadFilterType::Allpass => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            BiquadFilterType::Peaking => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            BiquadFilterType::Lowshelf => Self::a1_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::a1_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn a1_lp_hp_bp_notch_all_peak(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    #[inline]
    fn a1_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);

        -2.0 * ((a - 1.0) + (a + 1.0) * w0.cos())
    }

    #[inline]
    fn a1_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);

        2.0 * ((a - 1.0) - (a + 1.0) * w0.cos())
    }

    /// calculates a_2 coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - computedOscFreq
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn a2(type_: BiquadFilterType, sample_rate: f32, computed_freq: f32, q: f32, gain: f32) -> f64 {
        let sample_rate = sample_rate as f64;
        let computed_freq = computed_freq as f64;
        let q = q as f64;
        let gain = gain as f64;
        match type_ {
            BiquadFilterType::Lowpass => Self::a2_lp_hp(sample_rate, computed_freq, q),
            BiquadFilterType::Highpass => Self::a2_lp_hp(sample_rate, computed_freq, q),
            BiquadFilterType::Bandpass => Self::a2_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Notch => Self::a2_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Allpass => Self::a2_bp_notch_all(sample_rate, computed_freq, q),
            BiquadFilterType::Peaking => Self::a2_peaking(sample_rate, computed_freq, q, gain),
            BiquadFilterType::Lowshelf => Self::a2_lowshelf(sample_rate, computed_freq, gain),
            BiquadFilterType::Highshelf => Self::a2_highshelf(sample_rate, computed_freq, gain),
        }
    }

    #[inline]
    fn a2_lp_hp(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    #[inline]
    fn a2_bp_notch_all(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 - alpha_q
    }

    #[inline]
    fn a2_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 - (alpha_q / a)
    }

    #[inline]
    fn a2_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (a + 1.0) + (a - 1.0) * w0.cos() - 2.0 * alpha_s * a.sqrt()
    }

    #[inline]
    fn a2_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * alpha_s * a.sqrt()
    }

    /// Returns A parameter used to calculate biquad coeffs
    #[inline]
    fn a(gain: f64) -> f64 {
        10f64.powf(gain / 40.)
    }

    /// Returns w0 (omega 0) parameter used to calculate biquad coeffs
    #[inline]
    fn w0(sample_rate: f64, computed_freq: f64) -> f64 {
        2.0 * PI * computed_freq / sample_rate
    }

    /// Returns alpha_q parameter used to calculate biquad coeffs
    #[inline]
    fn alpha_q(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * q)
    }

    /// Returns alpha_q_db parameter used to calculate biquad coeffs
    #[inline]
    fn alpha_q_db(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * 10f64.powf(q / 20.))
    }

    /// Returns alpha_S parameter used to calculate biquad coeffs
    #[inline]
    fn alpha_s(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);

        (w0.sin() / 2.0) * SQRT_2
    }
}

#[cfg(test)]
mod test {
    use float_eq::assert_float_eq;

    use crate::{
        context::{AsBaseAudioContext, OfflineAudioContext},
        node::{BiquadFilterOptions, BiquadFilterType},
        SampleRate,
    };

    use super::BiquadFilterNode;

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _ = BiquadFilterNode::new(&context, None);
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _ = context.create_biquad_filter();
    }

    #[test]
    fn default_audio_params_are_correct_with_no_options() {
        let default_q = 1.0;
        let default_detune = 0.;
        let default_gain = 0.;
        let default_freq = 350.;
        let default_type = BiquadFilterType::Lowpass;
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);

        assert_float_eq!(biquad.q().value(), default_q, ulps <= 0);
        assert_float_eq!(biquad.detune().value(), default_detune, ulps <= 0);
        assert_float_eq!(biquad.gain().value(), default_gain, ulps <= 0);
        assert_float_eq!(biquad.frequency().value(), default_freq, ulps <= 0);
        assert_eq!(biquad.type_(), default_type);
    }

    #[test]
    fn default_audio_params_are_correct_with_default_options() {
        let default_q = 1.0;
        let default_detune = 0.;
        let default_gain = 0.;
        let default_freq = 350.;
        let default_type = BiquadFilterType::Lowpass;
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = BiquadFilterOptions::default();

        let biquad = BiquadFilterNode::new(&context, Some(options));

        assert_float_eq!(biquad.q().value(), default_q, ulps <= 0);
        assert_float_eq!(biquad.detune().value(), default_detune, ulps <= 0);
        assert_float_eq!(biquad.gain().value(), default_gain, ulps <= 0);
        assert_float_eq!(biquad.frequency().value(), default_freq, ulps <= 0);
        assert_eq!(biquad.type_(), default_type);
    }

    #[test]
    fn options_sets_audio_params() {
        let q = 2.0;
        let detune = 100.;
        let gain = 1.;
        let frequency = 3050.;
        let type_ = BiquadFilterType::Highpass;
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = BiquadFilterOptions {
            q: Some(q),
            detune: Some(detune),
            gain: Some(gain),
            frequency: Some(frequency),
            type_: Some(type_),
            ..BiquadFilterOptions::default()
        };

        let biquad = BiquadFilterNode::new(&context, Some(options));

        context.start_rendering();

        assert_float_eq!(biquad.q().value(), q, ulps <= 0);
        assert_float_eq!(biquad.detune().value(), detune, ulps <= 0);
        assert_float_eq!(biquad.gain().value(), gain, ulps <= 0);
        assert_float_eq!(biquad.frequency().value(), frequency, ulps <= 0);
        assert_eq!(biquad.type_(), type_);
    }

    #[test]
    fn change_audio_params_after_build() {
        let q = 2.0;
        let detune = 100.;
        let gain = 1.;
        let frequency = 3050.;
        let type_ = BiquadFilterType::Highpass;
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let mut biquad = BiquadFilterNode::new(&context, None);

        biquad.q().set_value(q);
        biquad.detune().set_value(detune);
        biquad.gain().set_value(gain);
        biquad.frequency().set_value(frequency);
        biquad.set_type(type_);

        context.start_rendering();

        assert_float_eq!(biquad.q().value(), q, ulps <= 0);
        assert_float_eq!(biquad.detune().value(), detune, ulps <= 0);
        assert_float_eq!(biquad.gain().value(), gain, ulps <= 0);
        assert_float_eq!(biquad.frequency().value(), frequency, ulps <= 0);
        assert_eq!(biquad.type_(), type_);
    }

    #[test]
    #[should_panic]
    fn panics_when_not_the_same_length() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);

        let mut frequency_hz = [0.];
        let mut mag_response = [0., 1.0];
        let mut phase_response = [0.];

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response)
    }

    #[test]
    #[should_panic]
    fn panics_when_not_the_same_length_2() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);

        let mut frequency_hz = [0.];
        let mut mag_response = [0.];
        let mut phase_response = [0., 1.0];

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response)
    }

    #[test]
    fn frequencies_are_clamped() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);
        let niquyst = context.sample_rate().0 as f32 / 2.0;

        let mut frequency_hz = [-100., 1_000_000.];
        let mut mag_response = [0., 0.];
        let mut phase_response = [0., 0.];

        biquad.get_frequency_response_mock(
            &mut frequency_hz,
            &mut mag_response,
            &mut phase_response,
        );

        let ref_arr = [0., niquyst];
        assert_float_eq!(frequency_hz, ref_arr, ulps_all <= 0);
    }
}
