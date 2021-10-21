#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::perf)]
use super::AudioNode;
use crate::{
    alloc::AudioBuffer,
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration},
    process::{AudioParamValues, AudioProcessor},
    SampleRate, MAX_CHANNELS,
};
use num_complex::Complex;
use std::f64::consts::PI;

const MAX_IIR_COEFFS_LEN: usize = 20;

/// The `IirFilterOptions` is used to specify the filter coefficients
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct IirFilterOptions {
    /// audio node options
    pub channel_config: ChannelConfigOptions,
    /// feedforward coefficients
    pub feedforward: Vec<f64>,
    /// feedback coefficients
    pub feedback: Vec<f64>,
}

/// An `AudioNode` implementing a general IIR filter
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct IirFilterNode {
    sample_rate: f32,
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    feedforward: Vec<f64>,
    feedback: Vec<f64>,
}

impl AudioNode for IirFilterNode {
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

impl IirFilterNode {
    /// Creates an `IirFilterNode`
    ///
    /// # Arguments
    ///
    /// * `context` - Audio context in which the node will live
    /// * `options` - node options
    ///
    /// # Panics
    ///
    /// will panic if:
    /// * coefficients length is more than 20
    /// * `feedforward` or/and `feedback` is an empty vector
    /// * all `feedforward` element or/and all `feedback` element are eqaul to 0.
    /// *
    pub fn new<C: AsBaseAudioContext>(context: &C, options: IirFilterOptions) -> Self {
        context.base().register(move |registration| {
            let IirFilterOptions {
                feedforward,
                feedback,
                channel_config,
            } = options;

            assert!(feedforward.len() <= MAX_IIR_COEFFS_LEN, "NotSupportedError");
            assert!(!feedforward.is_empty(), "NotSupportedError");
            assert!(!feedforward.iter().all(|&ff| ff == 0.), "InvalidStateError");
            assert!(feedback.len() <= MAX_IIR_COEFFS_LEN, "NotSupportedError");
            assert!(!feedback.is_empty(), "NotSupportedError");
            assert!(!feedback.iter().all(|&ff| ff == 0.), "InvalidStateError");

            // cast will be without loss of precission for usual fs
            #[allow(clippy::cast_precision_loss)]
            let sample_rate = context.base().sample_rate().0 as f32;

            let config = RendererConfig {
                feedforward: feedforward.clone(),
                feedback: feedback.clone(),
            };

            let render = IirFilterRenderer::new(config);

            let node = Self {
                sample_rate,
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
        frequency_hz: &mut [f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        self.validate_inputs(frequency_hz, mag_response, phase_response);

        for (i, &f) in frequency_hz.iter().enumerate() {
            let mut num: Complex<f64> = Complex::new(0., 0.);
            let mut denom: Complex<f64> = Complex::new(0., 0.);

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &ff) in self.feedforward.iter().enumerate() {
                num += Complex::from_polar(
                    ff,
                    idx as f64 * -2.0 * PI * f64::from(f) / f64::from(self.sample_rate),
                );
            }

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &fb) in self.feedback.iter().enumerate() {
                denom += Complex::from_polar(
                    fb,
                    idx as f64 * -2.0 * PI * f64::from(f) / f64::from(self.sample_rate),
                );
            }

            let h_f = num / denom;

            // Possible truncation is fine. f32 precision should be sufficients
            // And it is required by the specs
            mag_response[i] = h_f.norm() as f32;
            phase_response[i] = h_f.arg() as f32;
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
}

struct RendererConfig {
    // feedforward coeffs -- b[n] -- numerator coeffs
    feedforward: Vec<f64>,
    // feedback coeffs -- a[n] -- denominator coeffs
    feedback: Vec<f64>,
}

/// Renderer associated with the `IirFilterNode`
struct IirFilterRenderer {
    // Normalized filter's coeffs -- (b[n],a[n])
    norm_coeffs: Vec<(f64, f64)>,
    // filter's states
    states: Vec<[f64; MAX_CHANNELS]>,
}

impl AudioProcessor for IirFilterRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        self.filter(input, output);
    }

    fn tail_time(&self) -> bool {
        true
    }
}

impl IirFilterRenderer {
    /// Build an `IirFilterNode` renderer
    ///
    /// # Arguments
    ///
    /// * `config` - renderer config
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            feedforward,
            feedback,
        } = config;

        let coeffs = Self::build_coeffs(feedforward, feedback);
        let norm_coeffs = Self::normalize_coeffs(&coeffs);
        let states = Self::build_filter_states(&norm_coeffs);

        Self {
            norm_coeffs,
            states,
        }
    }

    /// Generate filter's coeffs
    ///
    /// # Arguments
    ///
    /// * `feedforward` - feedforward coeffs (numerator)
    /// * `feedback` - feedback coeffs (denominator)
    #[inline]
    fn build_coeffs(mut feedforward: Vec<f64>, mut feedback: Vec<f64>) -> Vec<(f64, f64)> {
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

        coeffs
    }

    /// Generate normalized filter's coeffs
    /// coeffs are normalized by a[0] coefficient
    ///
    /// # Arguments
    ///
    /// * `coeffs` - unormalized filter's coeffs (numerator)
    #[inline]
    fn normalize_coeffs(coeffs: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let a_0 = coeffs[0].1;

        coeffs.iter().map(|(ff, fb)| (ff / a_0, fb / a_0)).collect()
    }

    /// initialize filter states
    ///
    /// # Arguments
    ///
    /// * `coeffs` - filter's coeffs
    #[inline]
    fn build_filter_states(coeffs: &[(f64, f64)]) -> Vec<[f64; MAX_CHANNELS]> {
        let coeffs_len = coeffs.len();
        vec![[0.; MAX_CHANNELS]; coeffs_len - 1]
    }

    /// Generate an output by filtering the input
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `output` - Audiobuffer output
    #[inline]
    fn filter(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
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
            let state = self.states.get(i + 1).unwrap_or(&[0.; MAX_CHANNELS])[idx];
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
mod test {
    use float_eq::assert_float_eq;

    use crate::{
        buffer::ChannelConfigOptions,
        context::{AsBaseAudioContext, OfflineAudioContext},
        SampleRate,
    };

    use super::{IirFilterNode, IirFilterOptions};

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let options = IirFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };

        let _biquad = IirFilterNode::new(&context, options);
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_ffs_is_above_max_len() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_fbs_is_above_max_len() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedback = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        let feedforward = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_fbs_is_empty() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedback = vec![];
        let feedforward = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_ffs_is_empty() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_ffs_are_zeros() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![0., 0.];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_fbs_are_zeros() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedback = vec![0., 0.];
        let feedforward = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let _biquad = context.create_iir_filter(feedforward, feedback);
    }

    #[test]
    #[should_panic]
    fn panics_when_not_the_same_length() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let options = IirFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let biquad = IirFilterNode::new(&context, options);

        let mut frequency_hz = [0.];
        let mut mag_response = [0., 1.0];
        let mut phase_response = [0.];

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    #[should_panic]
    fn panics_when_not_the_same_length_2() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let options = IirFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let biquad = IirFilterNode::new(&context, options);

        let mut frequency_hz = [0.];
        let mut mag_response = [0.];
        let mut phase_response = [0., 1.0];

        biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    }

    #[test]
    fn frequencies_are_clamped() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let options = IirFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let iir = IirFilterNode::new(&context, options);
        // It will be fine for the usual fs
        #[allow(clippy::cast_precision_loss)]
        let niquyst = context.sample_rate().0 as f32 / 2.0;

        let mut frequency_hz = [-100., 1_000_000.];
        let mut mag_response = [0., 0.];
        let mut phase_response = [0., 0.];

        iir.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);

        let ref_arr = [0., niquyst];
        assert_float_eq!(frequency_hz, ref_arr, ulps_all <= 0);
    }
}
