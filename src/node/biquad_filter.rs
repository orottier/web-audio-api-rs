//! The biquad filter control and renderer parts
use std::any::Any;
use std::f64::consts::{PI, SQRT_2};

use num_complex::Complex;

use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::{MAX_CHANNELS, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

fn get_computed_freq(freq: f32, detune: f32) -> f32 {
    freq * (detune / 1200.).exp2()
}

/// Biquad filter coefficients normalized against a0
#[derive(Clone, Copy, Debug, Default)]
struct Coefficients {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

// allow non snake to better the variable names in the spec
#[allow(non_snake_case)]
fn calculate_coefs(
    filter_type: BiquadFilterType,
    sample_rate: f64,
    f0: f64,
    gain: f64,
    q: f64,
) -> Coefficients {
    let b0: f64;
    let b1: f64;
    let b2: f64;
    let a0: f64;
    let a1: f64;
    let a2: f64;

    match filter_type {
        BiquadFilterType::Lowpass => {
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q_db = sin_w0 / (2. * 10_f64.powf(q / 20.));

            b0 = (1. - cos_w0) / 2.;
            b1 = 1. - cos_w0;
            b2 = (1. - cos_w0) / 2.;
            a0 = 1. + alpha_q_db;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q_db;
        }
        BiquadFilterType::Highpass => {
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q_db = sin_w0 / (2. * 10_f64.powf(q / 20.));

            b0 = (1. + cos_w0) / 2.;
            b1 = -1. * (1. + cos_w0);
            b2 = (1. + cos_w0) / 2.;
            a0 = 1. + alpha_q_db;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q_db;
        }
        BiquadFilterType::Bandpass => {
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q = sin_w0 / (2. * q);

            b0 = alpha_q;
            b1 = 0.;
            b2 = -1. * alpha_q;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Notch => {
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q = sin_w0 / (2. * q);

            b0 = 1.;
            b1 = -2. * cos_w0;
            b2 = 1.;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Allpass => {
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q = sin_w0 / (2. * q);

            b0 = 1. - alpha_q;
            b1 = -2. * cos_w0;
            b2 = 1. + alpha_q;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Peaking => {
            let A = 10_f64.powf(gain / 40.);
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_q = sin_w0 / (2. * q);

            b0 = 1. + alpha_q * A;
            b1 = -2. * cos_w0;
            b2 = 1. - alpha_q * A;
            a0 = 1. + alpha_q / A;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q / A;
        }
        BiquadFilterType::Lowshelf => {
            let A = 10_f64.powf(gain / 40.);
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_s = sin_w0 / 2. * SQRT_2; // formula simplified as S is 0
            let two_alpha_s_A_squared = 2. * alpha_s * A.sqrt();
            let A_plus_one = A + 1.;
            let A_minus_one = A - 1.;

            b0 = A * (A_plus_one - A_minus_one * cos_w0 + two_alpha_s_A_squared);
            b1 = 2. * A * (A_minus_one - A_plus_one * cos_w0);
            b2 = A * (A_plus_one - A_minus_one * cos_w0 - two_alpha_s_A_squared);
            a0 = A_plus_one + A_minus_one * cos_w0 + two_alpha_s_A_squared;
            a1 = -2. * (A_minus_one + A_plus_one * cos_w0);
            a2 = A_plus_one + A_minus_one * cos_w0 - two_alpha_s_A_squared;
        }
        BiquadFilterType::Highshelf => {
            let A = 10_f64.powf(gain / 40.);
            let w0 = 2. * PI * f0 / sample_rate;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha_s = sin_w0 / 2. * SQRT_2; // formula simplified as S is 0
            let two_alpha_s_A_squared = 2. * alpha_s * A.sqrt();
            let A_plus_one = A + 1.;
            let A_minus_one = A - 1.;

            b0 = A * (A_plus_one + A_minus_one * cos_w0 + two_alpha_s_A_squared);
            b1 = -2. * A * (A_minus_one + A_plus_one * cos_w0);
            b2 = A * (A_plus_one + A_minus_one * cos_w0 - two_alpha_s_A_squared);
            a0 = A_plus_one - A_minus_one * cos_w0 + two_alpha_s_A_squared;
            a1 = 2. * (A_minus_one - A_plus_one * cos_w0);
            a2 = A_plus_one - A_minus_one * cos_w0 - two_alpha_s_A_squared;
        }
    }

    Coefficients {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// Biquad filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiquadFilterType {
    /// Allows frequencies below the cutoff frequency to pass through and
    /// attenuates frequencies above the cutoff. (12dB/oct rolloff)
    Lowpass,
    /// Frequencies above the cutoff frequency are passed through, but
    /// frequencies below the cutoff are attenuated. (12dB/oct rolloff)
    Highpass,
    /// Allows a range of frequencies to pass through and attenuates the
    /// frequencies below and above this frequency range.
    Bandpass,
    /// Allows all frequencies through, except for a set of frequencies.
    Notch,
    /// Allows all frequencies through, but changes the phase relationship
    /// between the various frequencies.
    Allpass,
    /// Allows all frequencies through, but adds a boost (or attenuation) to
    /// a range of frequencies.
    Peaking,
    /// Allows all frequencies through, but adds a boost (or attenuation) to
    /// the lower frequencies.
    Lowshelf,
    /// Allows all frequencies through, but adds a boost (or attenuation) to
    /// the higher frequencies.
    Highshelf,
}

impl Default for BiquadFilterType {
    fn default() -> Self {
        Self::Lowpass
    }
}

impl From<u32> for BiquadFilterType {
    fn from(i: u32) -> Self {
        // @note - must be in same order as the struct declaration
        match i {
            0 => BiquadFilterType::Lowpass,
            1 => BiquadFilterType::Highpass,
            2 => BiquadFilterType::Bandpass,
            3 => BiquadFilterType::Notch,
            4 => BiquadFilterType::Allpass,
            5 => BiquadFilterType::Peaking,
            6 => BiquadFilterType::Lowshelf,
            7 => BiquadFilterType::Highshelf,
            _ => unreachable!(),
        }
    }
}

/// Options for constructing a [`BiquadFilterNode`]
// dictionary BiquadFilterOptions : AudioNodeOptions {
//   BiquadFilterType type = "lowpass";
//   float Q = 1;
//   float detune = 0;
//   float frequency = 350;
//   float gain = 0;
// };
#[derive(Clone, Debug)]
pub struct BiquadFilterOptions {
    pub q: f32,
    pub detune: f32,
    pub frequency: f32,
    pub gain: f32,
    pub type_: BiquadFilterType,
    pub channel_config: ChannelConfigOptions,
}

impl Default for BiquadFilterOptions {
    fn default() -> Self {
        Self {
            q: 1.,
            detune: 0.,
            frequency: 350.,
            gain: 0.,
            type_: BiquadFilterType::default(),
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// BiquadFilterNode is an AudioNode processor implementing very common low-order
/// IIR filters.
///
/// Low-order filters are the building blocks of basic tone controls
/// (bass, mid, treble), graphic equalizers, and more advanced filters. Multiple
/// BiquadFilterNode filters can be combined to form more complex filters.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/BiquadFilterNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#BiquadFilterNode>
/// - see also: [`BaseAudioContext::create_biquad_filter`]
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// let file = File::open("samples/think-stereo-48000.wav").unwrap();
/// let buffer = context.decode_audio_data_sync(file).unwrap();
///
/// // create a lowpass filter (default) and open frequency parameter over time
/// let biquad = context.create_biquad_filter();
/// biquad.connect(&context.destination());
/// biquad.frequency().set_value(10.);
/// biquad
///     .frequency()
///     .exponential_ramp_to_value_at_time(10000., context.current_time() + 10.);
///
/// // pipe the audio buffer source into the lowpass filter
/// let mut src = context.create_buffer_source();
/// src.connect(&biquad);
/// src.set_buffer(buffer);
/// src.set_loop(true);
/// src.start();
/// ```
///
/// # Examples
///
/// - `cargo run --release --example biquad`
///
pub struct BiquadFilterNode {
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
    /// frequency where the filter is applied - its impact on the frequency
    /// response of the filter, depends on the `BiquadFilterType`
    frequency: AudioParam,
    /// boost/attenuation (dB) - its impact on the frequency response of the
    /// filter, depends on the `BiquadFilterType`
    gain: AudioParam,
    /// Current biquad filter type
    type_: BiquadFilterType,
}

impl AudioNode for BiquadFilterNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
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
    pub fn new<C: BaseAudioContext>(context: &C, options: BiquadFilterOptions) -> Self {
        context.register(move |registration| {
            let sample_rate = context.sample_rate();

            let BiquadFilterOptions {
                q,
                detune,
                frequency,
                gain,
                type_,
                channel_config,
            } = options;

            let q_param_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (q_param, q_proc) = context.create_audio_param(q_param_options, &registration);
            q_param.set_value(q);

            let detune_param_options = AudioParamDescriptor {
                min_value: -153_600.,
                max_value: 153_600.,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (d_param, d_proc) = context.create_audio_param(detune_param_options, &registration);
            d_param.set_value(detune);

            let freq_options = AudioParamDescriptor {
                min_value: 0.,
                max_value: sample_rate / 2.,
                default_value: 350.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context.create_audio_param(freq_options, &registration);
            f_param.set_value(frequency);

            let gain_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (g_param, g_proc) = context.create_audio_param(gain_options, &registration);
            g_param.set_value(gain);

            let renderer = BiquadFilterRenderer {
                gain: g_proc,
                detune: d_proc,
                frequency: f_proc,
                q: q_proc,
                type_,
                x1: Vec::with_capacity(MAX_CHANNELS),
                x2: Vec::with_capacity(MAX_CHANNELS),
                y1: Vec::with_capacity(MAX_CHANNELS),
                y2: Vec::with_capacity(MAX_CHANNELS),
            };

            let node = Self {
                registration,
                channel_config: channel_config.into(),
                type_,
                q: q_param,
                detune: d_param,
                frequency: f_param,
                gain: g_param,
            };

            (node, Box::new(renderer))
        })
    }

    /// Returns the gain audio parameter
    #[must_use]
    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }

    /// Returns the frequency audio parameter
    #[must_use]
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the detune audio parameter
    #[must_use]
    pub fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Returns the Q audio parameter
    #[must_use]
    pub fn q(&self) -> &AudioParam {
        &self.q
    }

    /// Returns the biquad filter type
    #[must_use]
    pub fn type_(&self) -> BiquadFilterType {
        self.type_
    }

    /// biquad filter type setter
    ///
    /// # Arguments
    ///
    /// * `type_` - the biquad filter type (lowpass, highpass,...)
    pub fn set_type(&mut self, type_: BiquadFilterType) {
        self.type_ = type_;
        self.registration.post_message(type_);
    }

    /// Returns the frequency response for the specified frequencies
    ///
    /// # Arguments
    ///
    /// * `frequency_hz` - frequencies for which frequency response of the filter should be calculated
    /// * `mag_response` - magnitude of the frequency response of the filter
    /// * `phase_response` - phase of the frequency response of the filter
    ///
    /// # Panics
    ///
    /// This function will panic if arguments' lengths don't match
    ///
    pub fn get_frequency_response(
        &self,
        frequency_hz: &[f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        if frequency_hz.len() != mag_response.len() || mag_response.len() != phase_response.len() {
            panic!("InvalidAccessError - Parameter lengths must match");
        }

        let sample_rate = self.context().sample_rate();
        let n_quist = sample_rate / 2.;

        let type_ = self.type_();
        let frequency = self.frequency().value();
        let detune = self.detune().value();
        let gain = self.gain().value();
        let q = self.q().value();

        // get coefs
        let computed_freq = get_computed_freq(frequency, detune);
        let Coefficients { b0, b1, b2, a1, a2 } = calculate_coefs(
            type_,
            sample_rate as f64,
            computed_freq as f64,
            gain as f64,
            q as f64,
        );

        // @note - comment from Firefox source code, blink/Biquad.cpp
        //
        // Evaluate the Z-transform of the filter at given normalized
        // frequency from 0 to 1.  (1 corresponds to the Nyquist
        // frequency.)
        //
        // The z-transform of the filter is
        //
        // H(z) = (b0 + b1*z^(-1) + b2*z^(-2))/(1 + a1*z^(-1) + a2*z^(-2))
        //
        // Evaluate as
        //
        // b0 + (b1 + b2*z1)*z1
        // --------------------
        // 1 + (a1 + a2*z1)*z1
        //
        // with z1 = 1/z and z = exp(j*pi*frequency). Hence z1 = exp(-j*pi*frequency)
        for (i, freq) in frequency_hz.iter().enumerate() {
            // clamp and normalize frequency
            let f = freq.clamp(0., n_quist) / n_quist;

            let omega = -1. * PI * f64::from(f);
            let z = Complex::new(omega.cos(), omega.sin());
            let numerator = b0 + (b1 + b2 * z) * z;
            let denominator = Complex::new(1., 0.) + (a1 + a2 * z) * z;
            let response = numerator / denominator;

            let (mag, phase) = response.to_polar();
            mag_response[i] = mag as f32;
            phase_response[i] = phase as f32;
        }
    }
}

/// `BiquadFilterRenderer` represents the rendering part of `BiquadFilterNode`
struct BiquadFilterRenderer {
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
    /// `BiquadFilterType`
    type_: BiquadFilterType,
    // keep filter state for each channel
    x1: Vec<f64>,
    x2: Vec<f64>,
    y1: Vec<f64>,
    y2: Vec<f64>,
}

impl AudioProcessor for BiquadFilterRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];
        let sample_rate = scope.sample_rate;

        // handle tail time
        if input.is_silent() {
            let mut ended = true;

            if self.x1.iter().any(|&v| v.is_normal())
                || self.x2.iter().any(|&v| v.is_normal())
                || self.y1.iter().any(|&v| v.is_normal())
                || self.y2.iter().any(|&v| v.is_normal())
            {
                ended = false;
            }

            // input is silent and filter history is clean
            if ended {
                output.make_silent();
                return false;
            }
        }

        // eventually resize state according to input number of channels
        // if in tail time, we should continue with previous number of channels
        if !input.is_silent() {
            // @todo - handle channel change cleanly, could cause discontinuities
            // see https://github.com/WebAudio/web-audio-api/issues/1719
            // see https://webaudio.github.io/web-audio-api/#channels-tail-time
            let num_channels = input.number_of_channels();

            if num_channels != self.x1.len() {
                self.x1.resize(num_channels, 0.);
                self.x2.resize(num_channels, 0.);
                self.y1.resize(num_channels, 0.);
                self.y2.resize(num_channels, 0.);
            }

            output.set_number_of_channels(num_channels);
        } else {
            let num_channels = self.x1.len();
            output.set_number_of_channels(num_channels);
        }

        // get a-rate parameters
        let type_ = self.type_;
        let frequency = params.get(&self.frequency);
        let detune = params.get(&self.detune);
        let q = params.get(&self.q);
        let gain = params.get(&self.gain);
        let sample_rate_f64 = f64::from(sample_rate);
        // compute first coef and fill the coef list with this value
        let computed_freq = get_computed_freq(frequency[0], detune[0]);
        let coef = calculate_coefs(
            type_,
            sample_rate_f64,
            f64::from(computed_freq),
            f64::from(gain[0]),
            f64::from(q[0]),
        );

        let mut coefs_list = [coef; RENDER_QUANTUM_SIZE];
        // if one of the params has a length of RENDER_QUANTUM_SIZE, we need
        // to compute the coefs for each frame
        if frequency.len() != 1 || detune.len() != 1 || q.len() != 1 || gain.len() != 1 {
            coefs_list
                .iter_mut()
                .zip(frequency.iter().cycle())
                .zip(detune.iter().cycle())
                .zip(q.iter().cycle())
                .zip(gain.iter().cycle())
                .skip(1)
                .for_each(|((((coefs, &f), &d), &q), &g)| {
                    let computed_freq = get_computed_freq(f, d);
                    *coefs = calculate_coefs(
                        type_,
                        sample_rate_f64,
                        f64::from(computed_freq),
                        f64::from(g),
                        f64::from(q),
                    );
                });
        };

        for (channel_number, output_channel) in output.channels_mut().iter_mut().enumerate() {
            let input_channel = input.channel_data(channel_number);
            // retrieve state from previous block
            let mut x1 = self.x1[channel_number];
            let mut x2 = self.x2[channel_number];
            let mut y1 = self.y1[channel_number];
            let mut y2 = self.y2[channel_number];

            output_channel
                .iter_mut()
                .zip(input_channel.iter())
                .zip(coefs_list.iter())
                .for_each(|((o, &i), c)| {
                    // 𝑎0𝑦(𝑛)+𝑎1𝑦(𝑛−1)+𝑎2𝑦(𝑛−2)=𝑏0𝑥(𝑛)+𝑏1𝑥(𝑛−1)+𝑏2𝑥(𝑛−2)
                    // as all coefs are normalized against 𝑎0, we get
                    // 𝑦(𝑛) = 𝑏0𝑥(𝑛) + 𝑏1𝑥(𝑛−1) + 𝑏2𝑥(𝑛−2) - 𝑎1𝑦(𝑛−1) - 𝑎2𝑦(𝑛−2)
                    let x = f64::from(i);
                    let y = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
                    // update state
                    x2 = x1;
                    x1 = x;
                    y2 = y1;
                    y1 = y;
                    // cast output value as f32
                    *o = y as f32;
                });

            // store channel state for next block
            self.x1[channel_number] = x1;
            self.x2[channel_number] = x2;
            self.y1[channel_number] = y1;
            self.y2[channel_number] = y2;
        }

        true
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(&type_) = msg.downcast_ref::<BiquadFilterType>() {
            self.type_ = type_;
            return;
        }

        log::warn!("BiquadFilterRenderer: Dropping incoming message {msg:?}");
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{BaseAudioContext, OfflineAudioContext};

    use super::*;

    #[test]
    fn test_computed_freq() {
        let g_sharp = 415.3;
        let a = 440.;
        let b_flat = 466.16;

        // 100 cents is 1 semi tone up
        let res = get_computed_freq(a, 100.);
        assert_float_eq!(res, b_flat, abs <= 0.01);
        // -100 cents is 1 semi tone below
        let res = get_computed_freq(a, -100.);
        assert_float_eq!(res, g_sharp, abs <= 0.01);
    }

    #[test]
    fn test_constructor() {
        {
            let default_q = 1.0;
            let default_detune = 0.;
            let default_gain = 0.;
            let default_freq = 350.;
            let default_type = BiquadFilterType::Lowpass;

            let context = OfflineAudioContext::new(2, 1, 44_100.);
            let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

            assert_float_eq!(biquad.q().value(), default_q, abs <= 0.);
            assert_float_eq!(biquad.detune().value(), default_detune, abs <= 0.);
            assert_float_eq!(biquad.gain().value(), default_gain, abs <= 0.);
            assert_float_eq!(biquad.frequency().value(), default_freq, abs <= 0.);
            assert_eq!(biquad.type_(), default_type);
        }

        {
            let options = BiquadFilterOptions {
                q: 2.0,
                detune: 100.,
                gain: 1.,
                frequency: 3050.,
                type_: BiquadFilterType::Highpass,
                ..BiquadFilterOptions::default()
            };
            let clone = options.clone();

            let context = OfflineAudioContext::new(2, 1, 44_100.);
            let biquad = BiquadFilterNode::new(&context, options);

            assert_float_eq!(biquad.q().value(), clone.q, abs <= 0.);
            assert_float_eq!(biquad.detune().value(), clone.detune, abs <= 0.);
            assert_float_eq!(biquad.gain().value(), clone.gain, abs <= 0.);
            assert_float_eq!(biquad.frequency().value(), clone.frequency, abs <= 0.);
            assert_eq!(biquad.type_(), clone.type_);
        }
    }

    #[test]
    #[should_panic]
    fn test_frequency_response_arguments() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);
        let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

        let frequency_hz = [0.];
        let mut mag_response = [0., 1.0];
        let mut phase_response = [0.];

        biquad.get_frequency_response(&frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    #[should_panic]
    fn test_frequency_response_arguments_2() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);
        let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

        let frequency_hz = [0.];
        let mut mag_response = [0.];
        let mut phase_response = [0., 1.0];

        biquad.get_frequency_response(&frequency_hz, &mut mag_response, &mut phase_response);
    }

    // @note: expected values retrieved from chrome and firefox, both being coherent
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_frequency_responses() {
        let context = OfflineAudioContext::new(1, 128, 44_100.);

        let frequency = 2000.;
        let q = 1.;
        let gain = 3.;
        let freqs = [
            400., 800., 1200., 1600., 2000., 2400., 2800., 3200., 3600., 4000.,
        ];

        {
            let type_ = BiquadFilterType::Lowpass;
            let expected_mags = [
                1.023848056793213,
                1.0948060750961304,
                1.19772469997406,
                1.2522060871124268,
                1.1220184564590454,
                0.8600019216537476,
                0.6262584328651428,
                0.46187180280685425,
                0.3505324125289917,
                0.27358654141426086,
            ];
            let expected_phases = [
                -0.18232205510139465,
                -0.3985414505004883,
                -0.691506564617157,
                -1.0987391471862793,
                -1.5707963705062866,
                -1.9669616222381592,
                -2.236342191696167,
                -2.4131083488464355,
                -2.533737897872925,
                -2.6204006671905518,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Highpass;
            let expected_mags = [
                0.0404227040708065,
                0.17317812144756317,
                0.42743849754333496,
                0.7974866628646851,
                1.1220184564590454,
                1.2458853721618652,
                1.2437469959259033,
                1.208056092262268,
                1.1714074611663818,
                1.1408127546310425,
            ];
            let expected_phases = [
                2.959270715713501,
                2.743051290512085,
                2.4500861167907715,
                2.042853593826294,
                1.570796251296997,
                1.1746309995651245,
                0.9052504897117615,
                0.7284843325614929,
                0.6078547239303589,
                0.5211920142173767,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Bandpass;
            let expected_mags = [
                0.2025768756866455,
                0.4271776080131531,
                0.6805755496025085,
                0.9101988673210144,
                1.,
                0.9370073676109314,
                0.8193633556365967,
                0.7074796557426453,
                0.6153367757797241,
                0.5415573716163635,
            ];
            let expected_phases = [
                1.3668076992034912,
                1.129427433013916,
                0.8222484588623047,
                0.42703235149383545,
                -6.948182118549084e-8,
                -0.3568341135978699,
                -0.6104966998100281,
                -0.7848706841468811,
                -0.9079831838607788,
                -0.9985077977180481,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Notch;
            let expected_mags = [
                0.979266345500946,
                0.9041677117347717,
                0.7326779365539551,
                0.4141714870929718,
                6.948182118549084e-8,
                0.3493095338344574,
                0.573274552822113,
                0.7067337036132812,
                0.7882643342018127,
                0.8406637907028198,
            ];
            let expected_phases = [
                -0.20398865640163422,
                -0.4413689076900482,
                -0.7485478520393372,
                -1.1437640190124512,
                1.570796251296997,
                1.213962197303772,
                0.9602996110916138,
                0.7859256267547607,
                0.662813127040863,
                0.5722885727882385,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Allpass;
            let expected_mags = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
            let expected_phases = [
                -0.40797731280326843,
                -0.8827378153800964,
                -1.4970957040786743,
                -2.2875280380249023,
                3.141592502593994,
                2.427924394607544,
                1.9205992221832275,
                1.5718512535095215,
                1.325626254081726,
                1.144577145576477,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Peaking;
            let expected_mags = [
                1.0145272016525269,
                1.0657449960708618,
                1.1736305952072144,
                1.330430030822754,
                1.4125374555587769,
                1.3534939289093018,
                1.2603179216384888,
                1.1887166500091553,
                1.1401562690734863,
                1.107250690460205,
            ];
            let expected_phases = [
                0.06874943524599075,
                0.13327200710773468,
                0.17138442397117615,
                0.13011260330677032,
                -2.411762878296031e-8,
                -0.1131250336766243,
                -0.16162104904651642,
                -0.17184172570705414,
                -0.16679927706718445,
                -0.1567305326461792,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Lowshelf;
            let expected_mags = [
                1.411763310432434,
                1.4004594087600708,
                1.3577604293823242,
                1.2777900695800781,
                1.1885021924972534,
                1.1184993982315063,
                1.07305908203125,
                1.045626163482666,
                1.029238224029541,
                1.0192826986312866,
            ];
            let expected_phases = [
                -0.050444066524505615,
                -0.10995279997587204,
                -0.17566977441310883,
                -0.22642207145690918,
                -0.24332194030284882,
                -0.23164276778697968,
                -0.2076151967048645,
                -0.18214666843414307,
                -0.15946431457996368,
                -0.1404205560684204,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }

        {
            let type_ = BiquadFilterType::Highshelf;
            let expected_mags = [
                1.0005483627319336,
                1.0086243152618408,
                1.0403436422348022,
                1.1054534912109375,
                1.1885021924972534,
                1.2628861665725708,
                1.3163650035858154,
                1.3509010076522827,
                1.3724106550216675,
                1.385815143585205,
            ];
            let expected_phases = [
                0.050444066524505615,
                0.10995279997587204,
                0.17566977441310883,
                0.22642207145690918,
                0.24332194030284882,
                0.23164276778697968,
                0.2076151967048645,
                0.18214666843414307,
                0.15946431457996368,
                0.1404205560684204,
            ];

            let mut filter = context.create_biquad_filter();
            filter.set_type(type_);
            filter.frequency().set_value(frequency);
            filter.q().set_value(q);
            filter.gain().set_value(gain);

            let mut mags = [0.; 10];
            let mut phases = [0.; 10];

            filter.get_frequency_response(&freqs, &mut mags, &mut phases);

            assert_float_eq!(mags, expected_mags, abs_all <= 1e-6);
            assert_float_eq!(phases, expected_phases, abs_all <= 1e-6);
        }
    }
}
