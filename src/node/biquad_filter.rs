//! The biquad filter control and renderer parts
use std::f32::consts::{PI, SQRT_2};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
// use std::time::Duration;
// use num_complex::Complex;

use crate::{MAX_CHANNELS, RENDER_QUANTUM_SIZE};
use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

fn get_computed_freq(freq: f32, detune: f32) -> f32 {
    freq * (detune / 1200.).exp2()
}

/// Biquad filter coefficients
#[derive(Clone, Copy, Debug, Default)]
struct Coefficients {
    a0: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
}

// allow non snake to better the variable names in the spec
#[allow(non_snake_case)]
fn calculate_coefs(
    filter_type: BiquadFilterType,
    sample_rate: f32,
    f0: f32,
    gain: f32,
    q: f32,
) -> Coefficients {
    let b0: f32;
    let b1: f32;
    let b2: f32;
    let a0: f32;
    let a1: f32;
    let a2: f32;

    let A = 10_f32.powf(gain / 40.);
    let w0 = 2. * PI * f0 / sample_rate;
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha_q = sin_w0 / (2. * q);
    let alpha_q_db = sin_w0 / (2. * 10_f32.powf(q / 20.));
    let alpha_s = sin_w0 / 2. * SQRT_2; // formula simplified as S is 0

    match filter_type {
        BiquadFilterType::Lowpass => {
            b0 = (1. - cos_w0) / 2.;
            b1 = 1. - cos_w0;
            b2 = (1. - cos_w0) / 2.;
            a0 = 1. + alpha_q_db;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q_db;
        }
        BiquadFilterType::Highpass => {
            b0 = (1. + cos_w0) / 2.;
            b1 = -1. * (1. + cos_w0);
            b2 = (1. + cos_w0) / 2.;
            a0 = 1. + alpha_q_db;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q_db;
        }
        BiquadFilterType::Bandpass => {
            b0 = alpha_q;
            b1 = 0.;
            b2 = -1. * alpha_q;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Notch => {
            b0 = 1.;
            b1 = -2. * cos_w0;
            b2 = 1.;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Allpass => {
            b0 = 1. - alpha_q;
            b1 = -2. * cos_w0;
            b2 = 1. + alpha_q;
            a0 = 1. + alpha_q;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q;
        }
        BiquadFilterType::Peaking => {
            b0 = 1. + alpha_q * A;
            b1 = -2. * cos_w0;
            b2 = 1. - alpha_q * A;
            a0 = 1. + alpha_q / A;
            a1 = -2. * cos_w0;
            a2 = 1. - alpha_q / A;
        }
        BiquadFilterType::Lowshelf => {
            let two_alpha_s_A_squared = 2. * alpha_s * A.sqrt();

            b0 = A * ((A + 1.) - (A - 1.) * cos_w0 + two_alpha_s_A_squared);
            b1 = 2. * A * ((A - 1.) - (A + 1.) * cos_w0);
            b2 = A * ((A + 1.) - (A - 1.) * cos_w0 - two_alpha_s_A_squared);
            a0 = (A + 1.) + (A - 1.) * cos_w0 + two_alpha_s_A_squared;
            a1 = -2. * A * ((A - 1.) + (A + 1.) * cos_w0);
            a2 = (A + 1.) + (A - 1.) * cos_w0 - two_alpha_s_A_squared;
        }
        BiquadFilterType::Highshelf => {
            let two_alpha_s_A_squared = 2. * alpha_s * A.sqrt();

            b0 = A * ((A + 1.) + (A - 1.) * cos_w0 + two_alpha_s_A_squared);
            b1 = -2. * A * ((A - 1.) + (A + 1.) * cos_w0);
            b2 = A * ((A + 1.) + (A - 1.) * cos_w0 - two_alpha_s_A_squared);
            a0 = (A + 1.) - (A - 1.) * cos_w0 + two_alpha_s_A_squared;
            a1 = -2. * A * ((A - 1.) - (A + 1.) * cos_w0);
            a2 = (A + 1.) - (A - 1.) * cos_w0 - two_alpha_s_A_squared;
        }
    }

    Coefficients {
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
    }
}

/// Biquad filter types
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// `BiquadFilterNode` is a second order IIR filter
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
    /// frequency where the filter is applied - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    frequency: AudioParam,
    /// boost/attenuation (dB) - its impact on the frequency response of the filter
    /// depends on the `BiquadFilterType`
    gain: AudioParam,
    /// `BiquadFilterType` repesented as u32
    type_: Arc<AtomicU32>,
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

            let q_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (q_param, q_proc) = context.create_audio_param(q_options, &registration);
            q_param.set_value(options.q);

            let detune = AudioParamDescriptor {
                min_value: -153_600.,
                max_value: 153_600.,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (d_param, d_proc) = context.create_audio_param(detune, &registration);
            d_param.set_value(options.detune);

            let freq_options = AudioParamDescriptor {
                min_value: 0.,
                max_value: sample_rate / 2.,
                default_value: 350.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context.create_audio_param(freq_options, &registration);
            f_param.set_value(options.frequency);

            let gain_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (g_param, g_proc) = context.create_audio_param(gain_options, &registration);
            g_param.set_value(options.gain);

            let type_ = Arc::new(AtomicU32::new(options.type_ as u32));

            let renderer = BiquadFilterRenderer {
                gain: g_proc,
                detune: d_proc,
                frequency: f_proc,
                q: q_proc,
                type_: type_.clone(),
                x1: Vec::with_capacity(MAX_CHANNELS),
                x2: Vec::with_capacity(MAX_CHANNELS),
                y1: Vec::with_capacity(MAX_CHANNELS),
                y2: Vec::with_capacity(MAX_CHANNELS),
            };

            let node = Self {
                registration,
                channel_config: options.channel_config.into(),
                type_,
                q: q_param,
                detune: d_param,
                frequency: f_param,
                gain: g_param,
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
    pub fn set_type(&self, type_: BiquadFilterType) {
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

        let sample_rate = self.context().sample_rate();

        // Ensures that given frequencies are in the correct range
        let min = 0.;
        let max = sample_rate / 2.;
        for f in frequency_hz.iter_mut() {
            *f = f.clamp(min, max);
        }

        // let type_ = self.type_();
        // let frequency = self.frequency().value();
        // let detune = self.detune().value();
        // let gain = self.gain().value();
        // let q = self.q().value();

        // // get coefs
        // let computed_freq = get_computed_freq(frequency, detune);
        // let Coefficients {
        //     a0,
        //     a1,
        //     a2,
        //     b0,
        //     b1,
        //     b2,
        // } = calculate_coefs(type_, sample_rate, computed_freq, gain, q);

        // @todo - confirm this is correct, this does not use a0
        // for (i, &f) in frequency_hz.iter().enumerate() {
        //     let f = f64::from(f);
        //     let sample_rate = f64::from(self.context().sample_rate());
        //     let num = b0
        //         + Complex::from_polar(b1, -1.0 * 2.0 * PI * f / sample_rate)
        //         + Complex::from_polar(b2, -2.0 * 2.0 * PI * f / sample_rate);
        //     let denom = 1.0
        //         + Complex::from_polar(a1, -1.0 * 2.0 * PI * f / sample_rate)
        //         + Complex::from_polar(a2, -2.0 * 2.0 * PI * f / sample_rate);
        //     let h_f = num / denom;

        //     // Possible truncation is fine. f32 precision should be sufficients
        //     // And it is required by the specs
        //     mag_response[i] = h_f.norm() as f32;
        //     phase_response[i] = h_f.arg() as f32;
        // }
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
    /// `BiquadFilterType` repesented as u32
    type_: Arc<AtomicU32>,
    // keep filter state for each channel
    x1: Vec<f32>,
    x2: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
}

impl AudioProcessor for BiquadFilterRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];
        let sample_rate = scope.sample_rate;

        // handle tail time
        if input.is_silent() {
            let mut ended = true;

            self.x1.iter()
                .zip(self.x2.iter())
                .zip(self.y1.iter())
                .zip(self.y2.iter())
                .for_each(|(((x1, x2), y1), y2)| {
                    if *x1 != 0. || *x2 != 0. || *y1 != 0. || *y2 != 0. {
                        ended = false;
                    }
                });

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
        let type_: BiquadFilterType = self.type_.load(Ordering::SeqCst).into();
        let frequency = params.get(&self.frequency);
        let detune = params.get(&self.detune);
        let q = params.get(&self.q);
        let gain = params.get(&self.gain);

        let mut coefs_list = [Coefficients::default(); RENDER_QUANTUM_SIZE];
        let mut current_frequency = frequency[0];
        let mut current_detune = detune[0];
        let mut current_q = q[0];
        let mut current_gain = gain[0];
        let mut current_coefs = Coefficients::default();

        coefs_list.iter_mut()
            .zip(frequency.iter())
            .zip(detune.iter())
            .zip(q.iter())
            .zip(gain.iter())
            .enumerate()
            .for_each(|(index, ((((coefs, f), d), q), g))| {
                // recompute coefs only if param change, done at least once per block
                if index == 0
                    || current_frequency != *f
                    || current_detune != *d
                    || current_q != *q
                    || current_gain != *g
                {
                    let computed_freq = get_computed_freq(*f, *d);
                    current_coefs = calculate_coefs(type_, sample_rate, computed_freq, *g, *q);

                    current_frequency = *f;
                    current_detune = *d;
                    current_q = *q;
                    current_gain = *g;
                }

                *coefs = current_coefs;
            });

        for (channel_number, output_channel) in output.channels_mut().iter_mut().enumerate() {
            let input_channel = input.channel_data(channel_number);
            // retrieve state from previous block
            let mut x1 = self.x1[channel_number];
            let mut x2 = self.x2[channel_number];
            let mut y1 = self.y1[channel_number];
            let mut y2 = self.y2[channel_number];

            input_channel
                .iter()
                .zip(output_channel.iter_mut())
                .zip(coefs_list.iter())
                .for_each(|((i, o), c)| {
                    // 𝑎0𝑦(𝑛)+𝑎1𝑦(𝑛−1)+𝑎2𝑦(𝑛−2)=𝑏0𝑥(𝑛)+𝑏1𝑥(𝑛−1)+𝑏2𝑥(𝑛−2), then:
                    // 𝑦(𝑛) = [𝑏0𝑥(𝑛)+𝑏1𝑥(𝑛−1)+𝑏2𝑥(𝑛−2) - 𝑎1𝑦(𝑛−1)+𝑎2𝑦(𝑛−2)] / 𝑎0
                    *o = (c.b0 * *i + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2) / c.a0;

                    // fush subnormal to zero
                    if o.is_subnormal() {
                        *o = 0.;
                    }
                    // update state
                    x2 = x1;
                    x1 = *i;
                    y2 = y1;
                    y1 = *o;
                });

            // store channel state for next block
            self.x1[channel_number] = x1;
            self.x2[channel_number] = x2;
            self.y1[channel_number] = y1;
            self.y2[channel_number] = y2;
        }

        true
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
    fn test_dummy() {
        let q = 2.0;
        let detune = 100.;
        let gain = 1.;
        let frequency = 3050.;
        let type_ = BiquadFilterType::Highpass;
        let context = OfflineAudioContext::new(1, 128, 44_100.);

        let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

        biquad.connect(&context.destination());
        biquad.q().set_value(q);
        biquad.detune().set_value(detune);
        biquad.gain().set_value(gain);
        biquad.frequency().set_value(frequency);
        biquad.set_type(type_);

        context.start_rendering_sync();
    }

    // #[test]
    // #[should_panic]
    // fn panics_when_not_the_same_length() {
    //     let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
    //     let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

    //     let mut frequency_hz = [0.];
    //     let mut mag_response = [0., 1.0];
    //     let mut phase_response = [0.];

    //     biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    // }

    // #[test]
    // #[should_panic]
    // fn panics_when_not_the_same_length_2() {
    //     let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
    //     let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());

    //     let mut frequency_hz = [0.];
    //     let mut mag_response = [0.];
    //     let mut phase_response = [0., 1.0];

    //     biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    // }

    // #[test]
    // fn frequencies_are_clamped() {
    //     let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
    //     let biquad = BiquadFilterNode::new(&context, BiquadFilterOptions::default());
    //     let niquyst = context.sample_rate() / 2.0;

    //     let mut frequency_hz = [-100., 1_000_000.];
    //     let mut mag_response = [0., 0.];
    //     let mut phase_response = [0., 0.];

    //     biquad.get_frequency_response_mock(
    //         &mut frequency_hz,
    //         &mut mag_response,
    //         &mut phase_response,
    //     );

    //     let ref_arr = [0., niquyst];
    //     assert_float_eq!(frequency_hz, ref_arr, abs_all <= 0.);
    // }
}
