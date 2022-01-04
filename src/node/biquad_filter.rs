//! The biquad filter control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]
use std::{
    f64::consts::{PI, SQRT_2},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender};
use num_complex::Complex;

use crate::{
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum},
    SampleRate, MAX_CHANNELS,
};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Coefficients request
/// This request is send by the control thread and send back by the rendering thread with
/// current coefficients array
struct CoeffsReq(Sender<[f64; 5]>);

/// enumerates all the biquad filter types
#[derive(Debug, Clone, Copy, PartialEq)]
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub enum BiquadFilterType {
    /// Allows frequencies below the cutoff frequency to pass through and attenuates frequencies above the cutoff. (12dB/oct rolloff)
    Lowpass,
    /// Frequencies above the cutoff frequency are passed through, but frequencies below the cutoff are attenuated. (12dB/oct rolloff)
    Highpass,
    /// Allows a range of frequencies to pass through and attenuates the frequencies below and above this frequency range.
    Bandpass,
    /// Allows all frequencies through, but adds a boost (or attenuation) to the lower frequencies.
    Lowshelf,
    /// Allows all frequencies through, but adds a boost to the higher frequencies.
    Highshelf,
    /// Allows all frequencies through, but adds a boost (or attenuation) to a range of frequencies.
    Peaking,
    /// Allows all frequencies through, except for a set of frequencies.
    Notch,
    /// Allows all frequencies through, but changes the phase relationship between the various frequencies.
    Allpass,
}

impl Default for BiquadFilterType {
    fn default() -> Self {
        Self::Lowpass
    }
}

impl From<u32> for BiquadFilterType {
    fn from(i: u32) -> Self {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };

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

/// `BiquadFilterOptions` is used to pass options
/// during the construction of `BiquadFilterNode` using its
/// constructor method `new`
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
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

/// `BiquadFilterNode` is a second order IIR filter
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct BiquadFilterNode {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// quality factor - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    q: AudioParam,
    /// A detune value, in cents, for the frequency.
    /// It forms a compound parameter with frequency to form the computedFrequency.
    detune: AudioParam,
    /// frequency where the filter is applied - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    frequency: AudioParam,
    /// boost/attenuation (dB) - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    gain: AudioParam,
    /// `BiquadFilterType` repesented as u32
    type_: Arc<AtomicU32>,
    /// sender used to send message to the rendering part of the node
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
    /// returns a `BiquadFilterNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - biquad filter options
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<BiquadFilterOptions>) -> Self {
        context.base().register(move |registration| {
            let options = options.unwrap_or_default();
            // cannot guarantee that the cast will be without loss of precision for all fs
            // but for usual sample rate (44.1kHz, 48kHz, 96kHz) it is
            #[allow(clippy::cast_precision_loss)]
            let sample_rate = context.base().sample_rate().0 as f32;

            let default_freq = 350.;
            let default_gain = 0.;
            let default_det = 0.;
            let default_q = 1.;

            let q_value = options.q.unwrap_or(default_q);
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
                min_value: -153_600.,
                max_value: 153_600.,
                default_value: default_det,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (d_param, d_proc) = context
                .base()
                .create_audio_param(d_param_opts, registration.id());

            d_param.set_value(d_value);

            let niquyst = context.base().sample_rate().0 / 2;
            // It should be fine for usual fs
            #[allow(clippy::cast_precision_loss)]
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

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let config = RendererConfig {
                sample_rate,
                gain: g_proc,
                detune: d_proc,
                frequency: f_proc,
                q: q_proc,
                type_: type_.clone(),
                receiver,
            };

            let renderer = BiquadFilterRenderer::new(config);
            let node = Self {
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

            (node, Box::new(renderer))
        })
    }

    /// Returns the gain audio paramter
    #[must_use]
    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }

    /// Returns the frequency audio paramter
    #[must_use]
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the detune audio paramter
    #[must_use]
    pub fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Returns the Q audio paramter
    #[must_use]
    pub fn q(&self) -> &AudioParam {
        &self.q
    }

    /// Returns the biquad filter type
    #[must_use]
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
    #[allow(clippy::cast_possible_truncation)]
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

        match receiver.recv_timeout(Duration::from_millis(10)) {
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                println!("Receiver Error: disconnected type ");
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                println!("Receiver Error: timeout type");
            }
            Ok([b0, b1, b2, a1, a2]) => {
                for (i, &f) in frequency_hz.iter().enumerate() {
                    let f = f64::from(f);
                    let sample_rate = f64::from(self.sample_rate);
                    let num = b0
                        + Complex::from_polar(b1, -1.0 * 2.0 * PI * f / sample_rate)
                        + Complex::from_polar(b2, -2.0 * 2.0 * PI * f / sample_rate);
                    let denom = 1.0
                        + Complex::from_polar(a1, -1.0 * 2.0 * PI * f / sample_rate)
                        + Complex::from_polar(a2, -2.0 * 2.0 * PI * f / sample_rate);
                    let h_f = num / denom;

                    // Possible truncation is fine. f32 precision should be sufficients
                    // And it is required by the specs
                    mag_response[i] = h_f.norm() as f32;
                    phase_response[i] = h_f.arg() as f32;
                }
            }
        }
    }

    /// validates that the params given to `get_frequency_response` method
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
            " InvalidAccessError: All parameters should be the same length"
        );
        assert_eq!(
            mag_response.len(),
            phase_response.len(),
            " InvalidAccessError: All parameters should be the same length"
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
    /// In tests, we use `OfflineAudioContext` and in this context the `CoeffsReq` is not sendable.
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

/// Represents the audio parameters values required to compute
/// the biquad coefficients
#[derive(Debug)]
struct CoeffsConfig {
    /// quality factor - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    q: f32,
    /// A detune value, in cents, for the frequency.
    /// It forms a compound parameter with frequency to form the computedFrequency.
    detune: f32,
    /// frequency where the filter is applied - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    frequency: f32,
    /// boost/attenuation (dB) - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    gain: f32,
    /// represents the biquad filter type
    type_: BiquadFilterType,
}

/// Helper struct which regroups all parameters
/// required to build `BiquadFilterRenderer`
struct RendererConfig {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// quality factor - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    q: AudioParamId,
    /// A detune value, in cents, for the frequency.
    /// It forms a compound parameter with frequency to form the computedFrequency.
    detune: AudioParamId,
    /// frequency where the filter is applied - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    frequency: AudioParamId,
    /// boost/attenuation (dB) - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    gain: AudioParamId,
    /// `BiquadFilterType` repesented as u32
    type_: Arc<AtomicU32>,
    /// receiver used to receive message from the control node part
    receiver: Receiver<CoeffsReq>,
}

/// Biquad filter coefficients
#[derive(Clone, Copy, Debug)]
struct Coefficients {
    /// Denominator coefficient
    a1: f64,
    /// Denominator coefficient
    a2: f64,
    /// Numerator coefficient
    b0: f64,
    /// Numerator coefficient
    b1: f64,
    /// Numerator coefficient
    b2: f64,
}

/// `BiquadFilterRenderer` represents the rendering part of `BiquadFilterNode`
struct BiquadFilterRenderer {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// quality factor - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    q: AudioParamId,
    /// A detune value, in cents, for the frequency.
    /// It forms a compound parameter with frequency to form the computedFrequency.
    detune: AudioParamId,
    /// frequency where the filter is applied - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    frequency: AudioParamId,
    /// boost/attenuation (dB) - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    gain: AudioParamId,
    /// `BiquadFilterType` repesented as u32
    type_: Arc<AtomicU32>,
    /// First level of the biquad filter state
    ss1: [f64; MAX_CHANNELS],
    /// Second level of the biquad filter state
    ss2: [f64; MAX_CHANNELS],
    /// Biquad filter coefficients computed from freq, q, gain,...
    coeffs: Coefficients,
    /// receiver used to receive message from the control node part
    receiver: Receiver<CoeffsReq>,
}

impl AudioProcessor for BiquadFilterRenderer {
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

        let g_values = params.get(&self.gain);
        let det_values = params.get(&self.detune);
        let freq_values = params.get(&self.frequency);
        let q_values = params.get(&self.q);

        self.filter(input, output, g_values, det_values, freq_values, q_values);

        true // todo tail time - issue #34
    }
}

impl BiquadFilterRenderer {
    /// returns an `BiquadFilterRenderer` instance
    // new cannot be qualified as const, since constant functions cannot evaluate destructors
    // and config param need this evaluation
    #[allow(clippy::missing_const_for_fn)]
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            sample_rate,
            q,
            detune,
            frequency,
            gain,
            type_,
            receiver,
        } = config;

        let coeffs = Coefficients {
            a1: 0.,
            a2: 0.,
            b0: 0.,
            b1: 0.,
            b2: 0.,
        };

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
        input: &AudioRenderQuantum,
        output: &mut AudioRenderQuantum,
        g_values: &[f32],
        det_values: &[f32],
        freq_values: &[f32],
        q_values: &[f32],
    ) {
        for (channel_idx, (i_data, o_data)) in input
            .channels()
            .iter()
            .zip(output.channels_mut())
            .enumerate()
        {
            for (sample_idx, (&i, o)) in i_data.iter().zip(o_data.iter_mut()).enumerate() {
                let p = CoeffsConfig {
                    q: q_values[sample_idx],
                    detune: det_values[sample_idx],
                    frequency: freq_values[sample_idx],
                    gain: g_values[sample_idx],
                    type_: BiquadFilterType::from(self.type_.load(Ordering::SeqCst)),
                };

                // A-rate params
                self.update_coeffs(&p);
                *o = self.tick(i, channel_idx);
            }
        }

        let Coefficients { b0, b1, b2, a1, a2 } = self.coeffs;

        let coeffs_resp = [b0, b1, b2, a1, a2];

        // Respond to request at K-rate following the specs
        if let Ok(msg) = self.receiver.try_recv() {
            let sender = msg.0;
            sender.send(coeffs_resp).unwrap();
        }
    }

    /// Generate an output sample by filtering an input sample
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `idx` - channel index mapping to the filter state index
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn tick(&mut self, input: f32, idx: usize) -> f32 {
        let input = f64::from(input);
        let out = self.coeffs.b0.mul_add(input, self.ss1[idx]);
        self.ss1[idx] = self.coeffs.b1.mul_add(input, self.ss2[idx]) - self.coeffs.a1 * out;
        self.ss2[idx] = self.coeffs.b2 * input - self.coeffs.a2 * out;

        // Value truncation will not be hearable
        out as f32
    }

    /// updates biquad filter coefficients when params are modified
    ///
    /// # Arguments
    ///
    /// * `params` - params resolving into biquad coeffs
    #[inline]
    fn update_coeffs(&mut self, params: &CoeffsConfig) {
        let CoeffsConfig {
            q,
            detune,
            frequency,
            gain,
            type_,
        } = params;

        let computed_freq = frequency * 10_f32.powf(detune / 1200.);

        let sample_rate = f64::from(self.sample_rate);
        let computed_freq = f64::from(computed_freq);
        let q = f64::from(*q);
        let gain = f64::from(*gain);

        // compute a0 first to normalize others coeffs by a0
        let a0 = Self::a0(*type_, sample_rate, computed_freq, q, gain);

        self.coeffs.b0 = Self::b0(*type_, sample_rate, computed_freq, q, gain) / a0;
        self.coeffs.b1 = Self::b1(*type_, sample_rate, computed_freq, gain) / a0;
        self.coeffs.b2 = Self::b2(*type_, sample_rate, computed_freq, q, gain) / a0;
        self.coeffs.a1 = Self::a1(*type_, sample_rate, computed_freq, gain) / a0;
        self.coeffs.a2 = Self::a2(*type_, sample_rate, computed_freq, q, gain) / a0;
    }

    /// calculates `b_0` numerator coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn b0(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
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

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Lowpass`
    #[inline]
    fn b0_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Highpass`
    #[inline]
    fn b0_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Bandpass`
    #[inline]
    fn b0_bandpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::alpha_q(sample_rate, computed_freq, q)
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Allpass`
    #[inline]
    fn b0_allpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 - alpha_q
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Peaking`
    #[inline]
    fn b0_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        alpha_q.mul_add(a, 1.0)
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn b0_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * (2.0 * alpha_s).mul_add(a.sqrt(), (a + 1.0) - (a - 1.0) * w0.cos())
    }

    /// returns the computed `b0` numerator coefficient for `BiquadFilterType::Highshelf`
    #[inline]
    fn b0_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * (2.0 * alpha_s).mul_add(a.sqrt(), (a - 1.0).mul_add(w0.cos(), a + 1.0))
    }

    /// calculates `b_1` coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `gain` - filter gain
    #[inline]
    fn b1(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };
        match type_ {
            Lowpass => Self::b1_lowpass(sample_rate, computed_freq),
            Highpass => Self::b1_highpass(sample_rate, computed_freq),
            Bandpass => 0.0,
            Notch | Allpass | Peaking => Self::b1_notch_all_peak(sample_rate, computed_freq),
            Lowshelf => Self::b1_lowshelf(sample_rate, computed_freq, gain),
            Highshelf => Self::b1_highshelf(sample_rate, computed_freq, gain),
        }
    }

    /// returns the computed `b1` numerator coefficient for `BiquadFilterType::Lowpass`
    #[inline]
    fn b1_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        1.0 - w0.cos()
    }

    /// returns the computed `b1` numerator coefficient for `BiquadFilterType::Highpass`
    #[inline]
    fn b1_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -(1.0 + w0.cos())
    }

    /// returns the computed `b1` numerator coefficient for:
    /// * `BiquadFilterType::Notch`
    /// * `BiquadFilterType::Allpass`
    /// * `BiquadFilterType::Peaking`
    #[inline]
    fn b1_notch_all_peak(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    /// returns the computed `b1` numerator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn b1_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        2.0 * a * ((a - 1.0) - (a + 1.0) * w0.cos())
    }

    /// returns the computed `b1` numerator coefficient for `BiquadFilterType::Highshelf`
    #[inline]
    fn b1_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * a * (a + 1.0).mul_add(w0.cos(), a - 1.0)
    }

    /// calculates `b_2` numerator coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn b2(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };
        match type_ {
            Lowpass => Self::b2_lowpass(sample_rate, computed_freq),
            Highpass => Self::b2_highpass(sample_rate, computed_freq),
            Bandpass => Self::b2_bandpass(sample_rate, computed_freq, q),
            Notch => 1.0,
            Allpass => Self::b2_allpass(sample_rate, computed_freq, q),
            Peaking => Self::b2_peaking(sample_rate, computed_freq, q, gain),
            Lowshelf => Self::b2_lowshelf(sample_rate, computed_freq, gain),
            Highshelf => Self::b2_highshelf(sample_rate, computed_freq, gain),
        }
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Lowpass`
    #[inline]
    fn b2_lowpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 - w0.cos()) / 2.0
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Highpass`
    #[inline]
    fn b2_highpass(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        (1.0 + w0.cos()) / 2.0
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Bandpass`
    #[inline]
    fn b2_bandpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        -Self::alpha_q(sample_rate, computed_freq, q)
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Allpass`
    #[inline]
    fn b2_allpass(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Peaking`
    #[inline]
    fn b2_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 - alpha_q * a
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn b2_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * alpha_s * a.sqrt())
    }

    /// returns the computed `b2` numerator coefficient for `BiquadFilterType::Highshelf`
    #[inline]
    fn b2_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        a * ((a - 1.0).mul_add(w0.cos(), a + 1.0) - 2.0 * alpha_s * a.sqrt())
    }

    /// calculates `a_0` coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn a0(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };
        match type_ {
            Lowpass | Highpass => Self::a0_lp_hp(sample_rate, computed_freq, q),
            Bandpass | Notch | Allpass => Self::a0_bp_notch_all(sample_rate, computed_freq, q),
            Peaking => Self::a0_peaking(sample_rate, computed_freq, q, gain),
            Lowshelf => Self::a0_lowshelf(sample_rate, computed_freq, gain),
            Highshelf => Self::a0_highshelf(sample_rate, computed_freq, gain),
        }
    }

    /// returns the computed `a0` denominator coefficient for:
    /// * `BiquadFilterType::Lowpass`
    /// * `BiquadFilterType::Highpass`
    #[inline]
    fn a0_lp_hp(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 + alpha_q_db
    }

    /// returns the computed `a0` denominator coefficient for:
    /// * `BiquadFilterType::Bandpass`
    /// * `BiquadFilterType::Notch`
    /// * `BiquadFilterType::Allpass`
    #[inline]
    fn a0_bp_notch_all(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 + alpha_q
    }

    /// returns the computed `a0` denominator coefficient for `BiquadFilterType::Peaking`
    #[inline]
    fn a0_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 + (alpha_q / a)
    }

    /// returns the computed `a0` denominator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn a0_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (2.0 * alpha_s).mul_add(a.sqrt(), (a - 1.0).mul_add(w0.cos(), a + 1.0))
    }

    /// returns the computed `a0` denominator coefficient for `BiquadFilterType::Highshelf`
    #[inline]
    fn a0_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (2.0 * alpha_s).mul_add(a.sqrt(), (a + 1.0) - (a - 1.0) * w0.cos())
    }

    /// calculates `a_1` coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `gain` - filter gain
    #[inline]
    fn a1(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };
        match type_ {
            Lowpass | Highpass | Bandpass | Notch | Allpass | Peaking => {
                Self::a1_lp_hp_bp_notch_all_peak(sample_rate, computed_freq)
            }
            Lowshelf => Self::a1_lowshelf(sample_rate, computed_freq, gain),
            Highshelf => Self::a1_highshelf(sample_rate, computed_freq, gain),
        }
    }

    /// returns the computed `a1` denominator coefficient for:
    /// * `BiquadFilterType::Lowpass`
    /// * `BiquadFilterType::Highpass`
    /// * `BiquadFilterType::Bandpass`
    /// * `BiquadFilterType::Notch`
    /// * `BiquadFilterType::Allpass`
    /// * `BiquadFilterType::Peaking`
    #[inline]
    fn a1_lp_hp_bp_notch_all_peak(sample_rate: f64, computed_freq: f64) -> f64 {
        let w0 = Self::w0(sample_rate, computed_freq);
        -2.0 * w0.cos()
    }

    /// returns the computed `a1` denominator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn a1_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);

        -2.0 * (a + 1.0).mul_add(w0.cos(), a - 1.0)
    }

    /// returns the computed `a1` denominator coefficient for `BiquadFilterType::Highshelf`
    #[inline]
    fn a1_highshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);

        2.0 * ((a - 1.0) - (a + 1.0) * w0.cos())
    }

    /// calculates `a_2` coefficient
    ///
    /// # Arguments
    ///
    /// * `type_` - biquadfilter type
    /// * `sample_rate` - audio context sample rate
    /// * `computed_freq` - `computedOscFreq`
    /// * `q` - Q factor
    /// * `gain` - filter gain
    #[inline]
    fn a2(type_: BiquadFilterType, sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        use BiquadFilterType::{
            Allpass, Bandpass, Highpass, Highshelf, Lowpass, Lowshelf, Notch, Peaking,
        };
        match type_ {
            Lowpass | Highpass => Self::a2_lp_hp(sample_rate, computed_freq, q),
            Bandpass | Notch | Allpass => Self::a2_bp_notch_all(sample_rate, computed_freq, q),
            Peaking => Self::a2_peaking(sample_rate, computed_freq, q, gain),
            Lowshelf => Self::a2_lowshelf(sample_rate, computed_freq, gain),
            Highshelf => Self::a2_highshelf(sample_rate, computed_freq, gain),
        }
    }

    /// returns the computed `a2` denominator coefficient for:
    /// * `BiquadFilterType::Lowpass`
    /// * `BiquadFilterType::Highpass`
    #[inline]
    fn a2_lp_hp(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q_db = Self::alpha_q_db(sample_rate, computed_freq, q);
        1.0 - alpha_q_db
    }

    /// returns the computed `a2` denominator coefficient for:
    /// * `BiquadFilterType::Bandpass`
    /// * `BiquadFilterType::Notch`
    /// * `BiquadFilterType::Allpass`
    #[inline]
    fn a2_bp_notch_all(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        1.0 - alpha_q
    }

    /// returns the computed `a2` denominator coefficient for `BiquadFilterType::Peaking`
    #[inline]
    fn a2_peaking(sample_rate: f64, computed_freq: f64, q: f64, gain: f64) -> f64 {
        let alpha_q = Self::alpha_q(sample_rate, computed_freq, q);
        let a = Self::a(gain);
        1.0 - (alpha_q / a)
    }

    /// returns the computed `a2` denominator coefficient for `BiquadFilterType::Lowshelf`
    #[inline]
    fn a2_lowshelf(sample_rate: f64, computed_freq: f64, gain: f64) -> f64 {
        let a = Self::a(gain);
        let w0 = Self::w0(sample_rate, computed_freq);
        let alpha_s = Self::alpha_s(sample_rate, computed_freq);

        (a - 1.0).mul_add(w0.cos(), a + 1.0) - 2.0 * alpha_s * a.sqrt()
    }

    /// returns the computed `a2` denominator coefficient for `BiquadFilterType::Highshelf`
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
        10_f64.powf(gain / 40.)
    }

    /// Returns w0 (omega 0) parameter used to calculate biquad coeffs
    #[inline]
    fn w0(sample_rate: f64, computed_freq: f64) -> f64 {
        2.0 * PI * computed_freq / sample_rate
    }

    /// Returns `alpha_q` parameter used to calculate biquad coeffs
    #[inline]
    fn alpha_q(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * q)
    }

    /// Returns `alpha_q_db` parameter used to calculate biquad coeffs
    #[inline]
    fn alpha_q_db(sample_rate: f64, computed_freq: f64, q: f64) -> f64 {
        Self::w0(sample_rate, computed_freq).sin() / (2. * 10_f64.powf(q / 20.))
    }

    /// Returns `alpha_s` parameter used to calculate biquad coeffs
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
        SampleRate,
    };

    use super::{BiquadFilterNode, BiquadFilterOptions, BiquadFilterType, ChannelConfigOptions};

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _biquad = BiquadFilterNode::new(&context, None);
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _biquad = context.create_biquad_filter();
    }

    #[test]
    fn default_audio_params_are_correct_with_no_options() {
        let default_q = 1.0;
        let default_detune = 0.;
        let default_gain = 0.;
        let default_freq = 350.;
        let default_type = BiquadFilterType::Lowpass;
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);

        context.start_rendering();

        assert_float_eq!(biquad.q().value(), default_q, abs <= 0.);
        assert_float_eq!(biquad.detune().value(), default_detune, abs <= 0.);
        assert_float_eq!(biquad.gain().value(), default_gain, abs <= 0.);
        assert_float_eq!(biquad.frequency().value(), default_freq, abs <= 0.);
        assert_eq!(biquad.type_(), default_type);
    }

    #[test]
    fn default_audio_params_are_correct_with_no_options_in_options() {
        let default_q = 1.0;
        let default_detune = 0.;
        let default_gain = 0.;
        let default_freq = 350.;
        let default_type = BiquadFilterType::Lowpass;

        let options = BiquadFilterOptions {
            q: None,
            detune: None,
            frequency: None,
            gain: None,
            type_: None,
            channel_config: ChannelConfigOptions::default(),
        };
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, Some(options));

        context.start_rendering();

        assert_float_eq!(biquad.q().value(), default_q, abs <= 0.);
        assert_float_eq!(biquad.detune().value(), default_detune, abs <= 0.);
        assert_float_eq!(biquad.gain().value(), default_gain, abs <= 0.);
        assert_float_eq!(biquad.frequency().value(), default_freq, abs <= 0.);
        assert_eq!(biquad.type_(), default_type);
    }

    #[test]
    fn default_audio_params_are_correct_with_default_options() {
        let default_q = 1.0;
        let default_detune = 0.;
        let default_gain = 0.;
        let default_freq = 350.;
        let default_type = BiquadFilterType::Lowpass;
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = BiquadFilterOptions::default();

        let biquad = BiquadFilterNode::new(&context, Some(options));

        context.start_rendering();

        assert_float_eq!(biquad.q().value(), default_q, abs <= 0.);
        assert_float_eq!(biquad.detune().value(), default_detune, abs <= 0.);
        assert_float_eq!(biquad.gain().value(), default_gain, abs <= 0.);
        assert_float_eq!(biquad.frequency().value(), default_freq, abs <= 0.);
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

        assert_float_eq!(biquad.q().value(), q, abs <= 0.);
        assert_float_eq!(biquad.detune().value(), detune, abs <= 0.);
        assert_float_eq!(biquad.gain().value(), gain, abs <= 0.);
        assert_float_eq!(biquad.frequency().value(), frequency, abs <= 0.);
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

        assert_float_eq!(biquad.q().value(), q, abs <= 0.);
        assert_float_eq!(biquad.detune().value(), detune, abs <= 0.);
        assert_float_eq!(biquad.gain().value(), gain, abs <= 0.);
        assert_float_eq!(biquad.frequency().value(), frequency, abs <= 0.);
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

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    #[should_panic]
    fn panics_when_not_the_same_length_2() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);

        let mut frequency_hz = [0.];
        let mut mag_response = [0.];
        let mut phase_response = [0., 1.0];

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    fn frequencies_are_clamped() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let biquad = BiquadFilterNode::new(&context, None);
        // It will be fine for the usual fs
        #[allow(clippy::cast_precision_loss)]
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
        assert_float_eq!(frequency_hz, ref_arr, abs_all <= 0.);
    }
}
