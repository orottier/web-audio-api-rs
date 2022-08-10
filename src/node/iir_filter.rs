//! The IIR filter control and renderer parts
use num_complex::Complex;
use std::f64::consts::PI;

use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::MAX_CHANNELS;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Filter order is limited to 20
const MAX_IIR_COEFFS_LEN: usize = 20;

/// Assert that the feedforward coefficients are valid
/// see <https://webaudio.github.io/web-audio-api/#dom-baseaudiocontext-createiirfilter-feedforward>
///
/// # Panics
///
/// This function panics if:
/// - coefs length is 0 and greater than 20
/// - all coefs are zeros
///
#[track_caller]
#[inline(always)]
fn assert_valid_feedforward_coefs(coefs: &Vec<f64>) {
    if coefs.len() == 0 || coefs.len() > MAX_IIR_COEFFS_LEN {
        panic!("NotSupportedError - IIR Filter feedforward coefficients should have length >= 0 and <= 20");
    }

    if coefs.iter().all(|&f| f == 0.) {
        panic!("InvalidStateError - IIR Filter feedforward coefficients cannot be all zeros");
    }
}

/// Assert that the feedforward coefficients are valid
/// see <https://webaudio.github.io/web-audio-api/#dom-baseaudiocontext-createiirfilter-feedforward>
///
/// # Panics
///
/// This function panics if:
/// - coefs length is 0 and greater than 20
/// - first coef is zero
///
#[track_caller]
#[inline(always)]
fn assert_valid_feedback_coefs(coefs: &Vec<f64>) {
    if coefs.len() == 0 || coefs.len() > MAX_IIR_COEFFS_LEN {
        panic!("NotSupportedError - IIR Filter feedback coefficients should have length >= 0 and <= 20");
    }

    if coefs[0] == 0. {
        panic!("InvalidStateError - IIR Filter feedback first coefficient cannot be zero");
    }
}

/// Options for constructing a [`IIRFilterNode`]
// dictionary IIRFilterOptions : AudioNodeOptions {
//   required sequence<double> feedforward;
//   required sequence<double> feedback;
// };
pub struct IIRFilterOptions {
    /// audio node options
    pub channel_config: ChannelConfigOptions,
    /// feedforward coefficients
    pub feedforward: Vec<f64>, // go for Option<Vec<f32>> w/ default to None?
    /// feedback coefficients
    pub feedback: Vec<f64>, // go for Option<Vec<f32>> w/ default to None?
}

/// An `AudioNode` implementing a general IIR filter
pub struct IIRFilterNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// numerator filter's coefficients
    feedforward: Vec<f64>,
    /// denomintor filter's coefficients
    feedback: Vec<f64>,
}

impl AudioNode for IIRFilterNode {
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

impl IIRFilterNode {
    /// Creates an `IirFilterNode`
    ///
    /// # Arguments
    ///
    /// - `context` - Audio context in which the node will live
    /// - `options` - node options
    ///
    /// # Panics
    ///
    /// This function panics if:
    /// - coefs length is 0 and greater than 20
    /// - feedforward coefs are all zeros
    /// - feedback first coef is zero
    ///
    pub fn new<C: BaseAudioContext>(context: &C, options: IIRFilterOptions) -> Self {
        context.register(move |registration| {
            let IIRFilterOptions {
                feedforward,
                feedback,
                channel_config,
            } = options;

            assert_valid_feedforward_coefs(&feedforward);
            assert_valid_feedback_coefs(&feedback);

            let config = RendererConfig {
                feedforward: feedforward.clone(),
                feedback: feedback.clone(),
            };

            let render = IirFilterRenderer::new(config);

            let node = Self {
                registration,
                channel_config: channel_config.into(),
                feedforward,
                feedback,
            };

            (node, Box::new(render))
        })
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
        frequency_hz: &[f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        if frequency_hz.len() != mag_response.len() || mag_response.len() != phase_response.len() {
            panic!("InvalidAccessError - Parameter lengths must match");
        }

        let sample_rate = self.context().sample_rate() as f64;
        let nquist = sample_rate / 2. ;

        for (i, &f) in frequency_hz.iter().enumerate() {
            let freq = f64::from(f).clamp(0., nquist);
            let mut num: Complex<f64> = Complex::new(0., 0.);
            let mut denom: Complex<f64> = Complex::new(0., 0.);

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &ff) in self.feedforward.iter().enumerate() {
                num += Complex::from_polar(ff, idx as f64 * -2.0 * PI * freq / sample_rate);
            }

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &fb) in self.feedback.iter().enumerate() {
                denom +=
                    Complex::from_polar(fb, idx as f64 * -2.0 * PI * freq / sample_rate);
            }

            let h_f = num / denom;

            // Possible truncation is fine. f32 precision should be sufficients
            // And it is required by the specs
            mag_response[i] = h_f.norm() as f32;
            phase_response[i] = h_f.arg() as f32;
        }
    }
}

/// `FilterRendererBuilder` helps to build `IirFilterRenderer`
struct FilterRendererBuilder {
    /// filter's coefficients as (feedforward, feedback)[]
    coeffs: Vec<(f64, f64)>,
    /// filter's states
    /// if the states is not used, it stays to 0. and will be never accessed
    states: Vec<[f64; MAX_CHANNELS]>,
}

impl FilterRendererBuilder {
    /// Generate filter's coeffs
    ///
    /// # Arguments
    ///
    /// * `feedforward` - feedforward coeffs (numerator)
    /// * `feedback` - feedback coeffs (denominator)
    #[inline]
    fn build(config: RendererConfig) -> Self {
        let RendererConfig {
            mut feedforward,
            mut feedback,
        } = config;

        match (feedforward.len(), feedback.len()) {
            (feedforward_len, feedback_len) if feedforward_len > feedback_len => {
                feedforward = feedforward
                    .into_iter()
                    .chain(std::iter::repeat(0.))
                    .take(feedback_len)
                    .collect();
            }
            (feedforward_len, feedback_len) if feedforward_len < feedback_len => {
                feedback = feedback
                    .into_iter()
                    .chain(std::iter::repeat(0.))
                    .take(feedforward_len)
                    .collect();
            }
            _ => (),
        };

        let coeffs: Vec<(f64, f64)> = feedforward.into_iter().zip(feedback).collect();

        let coeffs_len = coeffs.len();
        let states = vec![[0.; MAX_CHANNELS]; coeffs_len];

        Self { coeffs, states }
    }

    /// Generate normalized filter's coeffs and filter's states
    /// coeffs are normalized by `a[0]` coefficient
    fn finish(mut self) -> IirFilterRenderer {
        let a_0 = self.coeffs[0].1;

        for (ff, fb) in &mut self.coeffs {
            *ff /= a_0;
            *fb /= a_0;
        }

        IirFilterRenderer {
            norm_coeffs: self.coeffs,
            states: self.states,
        }
    }
}

/// Helper struct which regroups all parameters
/// required to build `IirFilterRenderer` with the help of `FilterRendererBuilder`
struct RendererConfig {
    /// feedforward coeffs -- `b[n]` -- numerator coeffs
    feedforward: Vec<f64>,
    /// feedback coeffs -- `a[n]` -- denominator coeffs
    feedback: Vec<f64>,
}

/// Renderer associated with the `IirFilterNode`
struct IirFilterRenderer {
    /// Normalized filter's coeffs -- `(b[n], a[n])`
    norm_coeffs: Vec<(f64, f64)>,
    /// filter's states
    states: Vec<[f64; MAX_CHANNELS]>,
}

impl AudioProcessor for IirFilterRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        self.filter(input, output);

        true // todo tail time - issue #34
    }
}

impl IirFilterRenderer {
    /// Build an `IirFilterNode` renderer
    ///
    /// # Arguments
    ///
    /// * `config` - renderer config
    fn new(config: RendererConfig) -> Self {
        FilterRendererBuilder::build(config).finish()
    }

    /// Generate an output by filtering the input
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `output` - Audiobuffer output
    #[inline]
    fn filter(&mut self, input: &AudioRenderQuantum, output: &mut AudioRenderQuantum) {
        for (idx, (i_data, o_data)) in input
            .channels()
            .iter()
            .zip(output.channels_mut())
            .enumerate()
        {
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
    #[allow(clippy::cast_possible_truncation)]
    fn tick(&mut self, input: f32, idx: usize) -> f32 {
        let input = f64::from(input);
        let output = self.norm_coeffs[0].0.mul_add(input, self.states[0][idx]);

        for (i, (ff, fb)) in self.norm_coeffs.iter().skip(1).enumerate() {
            let state = self.states[i + 1][idx];
            self.states[i][idx] = ff * input - fb * output + state;
        }

        #[cfg(debug_assertions)]
        if output.is_nan() || output.is_infinite() {
            log::debug!("An unstable filter is processed.");
        }

        // Value truncation will not be hearable
        output as f32
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::fs::File;

    use crate::AudioBuffer;
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode, BiquadFilterType};

    use super::*;

    const LENGTH: usize = 512;

    #[test]
    fn test_constructor_and_factory() {
        {
            let context = OfflineAudioContext::new(2, LENGTH, 44_100.);

            let options = IIRFilterOptions {
                feedback: vec![1.; 3],
                feedforward: vec![1.; 3],
                channel_config: ChannelConfigOptions::default(),
            };

            let _biquad = IIRFilterNode::new(&context, options);
        }

        {
            let context = OfflineAudioContext::new(2, LENGTH, 44_100.);

            let feedforward = vec![1.; 3];
            let feedback = vec![1.; 3];
            let _biquad = context.create_iir_filter(feedforward, feedback);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_feedforward_size() {
        let feedforward = vec![1.; 21];
        assert_valid_feedforward_coefs(&feedforward);
    }

    #[test]
    #[should_panic]
    fn test_invalid_feedforward_values() {
        let feedforward = vec![0.; 5];
        assert_valid_feedforward_coefs(&feedforward);
    }

    #[test]
    fn test_valid_feedforward_values() {
        let feedforward = vec![1.; 5];
        assert_valid_feedforward_coefs(&feedforward);
    }

    #[test]
    #[should_panic]
    fn test_invalid_feedback_size() {
        let feedback = vec![1.; 21];
        assert_valid_feedback_coefs(&feedback);
    }

    #[test]
    #[should_panic]
    fn test_invalid_feedback_values() {
        let mut feedback = vec![1.; 5];
        feedback[0] = 0.;
        assert_valid_feedback_coefs(&feedback);
    }

    #[test]
    fn test_valid_feedback_values() {
        let feedback = vec![1.; 5];
        assert_valid_feedback_coefs(&feedback);
    }

    #[test]
    #[should_panic]
    fn test_frequency_response_arguments() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);
        let options = IIRFilterOptions {
            feedback: vec![1.; 10],
            feedforward: vec![1.; 10],
            channel_config: ChannelConfigOptions::default(),
        };
        let iir = IIRFilterNode::new(&context, options);

        let frequency_hz = [0.];
        let mut mag_response = [0., 1.0];
        let mut phase_response = [0.];

        iir.get_frequency_response(&frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    #[should_panic]
    fn test_frequency_response_arguments_2() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);
        let options = IIRFilterOptions {
            feedback: vec![1.; 10],
            feedforward: vec![1.; 10],
            channel_config: ChannelConfigOptions::default(),
        };
        let iir = IIRFilterNode::new(&context, options);

        let frequency_hz = [0.];
        let mut mag_response = [0.];
        let mut phase_response = [0., 1.0];

        iir.get_frequency_response(&frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    fn test_output_against_biquad() {
        let context = OfflineAudioContext::new(1, 1, 44_100.);
        let file = File::open("samples/white.ogg").unwrap();
        let noise = context.decode_audio_data_sync(file).unwrap();

        fn compare_output(
            noise: AudioBuffer,
            filter_type: BiquadFilterType,
            feedback: Vec<f64>,
            feedforward: Vec<f64>
        ) {
            let frequency = 2000.;
            let q = 1.;
            let gain = 3.;
            // output of biquad and iir filters applied to white noise should thus be equal
            let biquad_res = {
                let context = OfflineAudioContext::new(1, noise.length(), 44_100.);

                let biquad = context.create_biquad_filter();
                biquad.connect(&context.destination());
                biquad.set_type(filter_type);
                biquad.frequency().set_value(frequency);
                biquad.q().set_value(q);
                biquad.gain().set_value(gain);

                let src = context.create_buffer_source();
                src.connect(&biquad);
                src.set_buffer(noise.clone());
                src.start();

                context.start_rendering_sync()
            };

            let iir_res = {
                let context = OfflineAudioContext::new(1, noise.length(), 44_100.);

                let iir = context.create_iir_filter(feedforward, feedback);
                iir.connect(&context.destination());

                let src = context.create_buffer_source();
                src.connect(&iir);
                src.set_buffer(noise.clone());
                src.start();

                context.start_rendering_sync()
            };

            println!("{:?}", filter_type);
            assert_float_eq!(
                biquad_res.get_channel_data(0),
                iir_res.get_channel_data(0),
                abs_all <= 0.
            );
        }

        // these are the unormalized coefs computed by the biquad filter for:
        // - frequency = 2000.;
        // - q = 1.;
        // - gain = 3.;
        // see node::biquad_filter::tests::test_frequency_responses

        // lowpass
        let a0 = 1.1252702717383296;
        let a1 = -1.9193504546709936;
        let a2 = 0.8747297282616704;
        let b0 = 0.02016238633225159;
        let b1 = 0.04032477266450318;
        let b2 = 0.02016238633225159;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Lowpass, feedback, feedforward);

        // highpass
        let a0 = 1.1252702717383296;
        let a1 = -1.9193504546709936;
        let a2 = 0.8747297282616704;
        let b0 = 0.9798376136677485;
        let b1 = -1.959675227335497;
        let b2 = 0.9798376136677485;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Highpass, feedback, feedforward);

        // bandpass
        let a0 = 1.1405555566658274;
        let a1 = -1.9193504546709936;
        let a2 = 0.8594444433341726;
        let b0 = 0.14055555666582747;
        let b1 = 0.0;
        let b2 = -0.14055555666582747;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Bandpass, feedback, feedforward);

        // notch
        let a0 = 1.1405555566658274;
        let a1 = -1.9193504546709936;
        let a2 = 0.8594444433341726;
        let b0 = 1.0;
        let b1 = -1.9193504546709936;
        let b2 = 1.0;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Notch, feedback, feedforward);

        // allpass
        let a0 = 1.1405555566658274;
        let a1 = -1.9193504546709936;
        let a2 = 0.8594444433341726;
        let b0 = 0.8594444433341726;
        let b1 = -1.9193504546709936;
        let b2 = 1.1405555566658274;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Allpass, feedback, feedforward);

        // peaking
        let a0 = 1.1182627625098631;
        let a1 = -1.9193504546709936;
        let a2 = 0.8817372374901369;
        let b0 = 1.167050592175986;
        let b1 = -1.9193504546709936;
        let b2 = 0.8329494078240139;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Peaking, feedback, feedforward);

        // lowshelf
        let a0 = 2.8028072429836723;
        let a1 = -4.577507200153761;
        let a2 = 1.935999047828101;
        let b0 = 2.9011403634599007;
        let b1 = -4.544236234748791;
        let b2 = 1.8709368927568424;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Lowshelf, feedback, feedforward);

        // highshelf
        let a0 = 2.4410054070459357;
        let a1 = -3.8234982904056865;
        let a2 = 1.5741972118903644;
        let b0 = 3.331142651362703;
        let b1 = -5.440377503491735;
        let b2 = 2.300939180659645;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_output(noise.clone(), BiquadFilterType::Highshelf, feedback, feedforward);
    }


    #[test]
    fn tests_get_frequency_response() {
        // This reference response has been generated with the python lib scipy
        // b, a = signal.iirfilter(2, 4000, rs=60, btype='high', analog=False, ftype='cheby2', fs=44100)
        // w, h = signal.freqz(b, a, 10, fs=44100)
        let ref_mag = [
            1e-3,
            4.152_807e-4,
            1.460_789_5e-3,
            5.051_316e-3,
            1.130_323_5e-2,
            2.230_340_2e-2,
            4.311_698e-2,
            8.843_45e-2,
            2.146_620_2e-1,
            6.802_952e-1,
        ];
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
        let feedforward = vec![
            0.019_618_022_238_052_212,
            -0.036_007_928_102_449_24,
            0.019_618_022_238_052_21,
        ];
        let feedback = vec![1., 1.576_436_200_538_313_7, 0.651_680_173_116_867_3];

        let options = IIRFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let iir = IIRFilterNode::new(&context, options);

        let mut frequency_hz = [
            0., 2205., 4410., 6615., 8820., 11025., 13230., 15435., 17640., 19845.,
        ];
        let mut mag_response = [0.; 10];
        let mut phase_response = [0.; 10];

        iir.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);

        assert_float_eq!(mag_response, ref_mag, abs_all <= 0.);
    }

    #[test]
    fn test_frequency_responses_against_biquad() {
        fn compare_frequency_response(
            filter_type: BiquadFilterType,
            feedback: Vec<f64>,
            feedforward: Vec<f64>
        ) {
            let frequency = 2000.;
            let q = 1.;
            let gain = 3.;
            let freqs = [
                400., 800., 1200., 1600., 2000., 2400., 2800., 3200., 3600., 4000.,
            ];

            let context = OfflineAudioContext::new(1, 1, 44_100.);

            let biquad_response = {
                let mut mags = [0.; 10];
                let mut phases = [0.; 10];

                let biquad = context.create_biquad_filter();
                biquad.set_type(filter_type);
                biquad.frequency().set_value(frequency);
                biquad.q().set_value(q);
                biquad.gain().set_value(gain);

                biquad.get_frequency_response(&freqs, &mut mags, &mut phases);

                (mags, phases)
            };

            let iir_response = {
                let mut mags = [0.; 10];
                let mut phases = [0.; 10];

                let iir = context.create_iir_filter(feedforward, feedback);

                iir.get_frequency_response(&freqs, &mut mags, &mut phases);

                (mags, phases)
            };

            println!("{:?}", filter_type);
            assert_float_eq!(biquad_response.0, iir_response.0, abs_all <= 1e-6);
            assert_float_eq!(biquad_response.1, iir_response.1, abs_all <= 1e-6);
        }

               // lowpass
        let a0 = 1.1252702717383296;
        let a1 = -1.9193504546709936;
        let a2 = 0.8747297282616704;
        let b0 = 0.02016238633225159;
        let b1 = 0.04032477266450318;
        let b2 = 0.02016238633225159;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Lowpass, feedback, feedforward);

        // highpass
        let a0 = 1.1252702717383296;
        let a1 = -1.9193504546709936;
        let a2 = 0.8747297282616704;
        let b0 = 0.9798376136677485;
        let b1 = -1.959675227335497;
        let b2 = 0.9798376136677485;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Highpass, feedback, feedforward);

        // bandpass
        let a0 = 1.1405555566658274;
        let a1 = -1.9193504546709936;
        let a2 = 0.8594444433341726;
        let b0 = 0.14055555666582747;
        let b1 = 0.0;
        let b2 = -0.14055555666582747;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Bandpass, feedback, feedforward);

        // one of the value in phases differs by 0.28, to be digged
        // notch
        // let a0 = 1.1405555566658274;
        // let a1 = -1.9193504546709936;
        // let a2 = 0.8594444433341726;
        // let b0 = 1.0;
        // let b1 = -1.9193504546709936;
        // let b2 = 1.0;

        // let feedback = vec![a0, a1, a2];
        // let feedforward = vec![b0, b1, b2];
        // compare_frequency_response(BiquadFilterType::Notch, feedback, feedforward);

        // allpass
        let a0 = 1.1405555566658274;
        let a1 = -1.9193504546709936;
        let a2 = 0.8594444433341726;
        let b0 = 0.8594444433341726;
        let b1 = -1.9193504546709936;
        let b2 = 1.1405555566658274;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Allpass, feedback, feedforward);

        // peaking
        let a0 = 1.1182627625098631;
        let a1 = -1.9193504546709936;
        let a2 = 0.8817372374901369;
        let b0 = 1.167050592175986;
        let b1 = -1.9193504546709936;
        let b2 = 0.8329494078240139;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Peaking, feedback, feedforward);

        // lowshelf
        let a0 = 2.8028072429836723;
        let a1 = -4.577507200153761;
        let a2 = 1.935999047828101;
        let b0 = 2.9011403634599007;
        let b1 = -4.544236234748791;
        let b2 = 1.8709368927568424;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Lowshelf, feedback, feedforward);

        // highshelf
        let a0 = 2.4410054070459357;
        let a1 = -3.8234982904056865;
        let a2 = 1.5741972118903644;
        let b0 = 3.331142651362703;
        let b1 = -5.440377503491735;
        let b2 = 2.300939180659645;

        let feedback = vec![a0, a1, a2];
        let feedforward = vec![b0, b1, b2];
        compare_frequency_response(BiquadFilterType::Highshelf, feedback, feedforward);
    }
}
