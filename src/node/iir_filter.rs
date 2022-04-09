//! The IIR filter control and renderer parts
// #![warn(
//     clippy::all,
//     clippy::pedantic,
//     clippy::nursery,
//     clippy::perf,
//     clippy::missing_docs_in_private_items
// )]
use num_complex::Complex;
use std::f64::consts::PI;

use crate::{
    context::{AudioContextRegistration, BaseAudioContext},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum},
    SampleRate, MAX_CHANNELS,
};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Filter order is limited to 20
const MAX_IIR_COEFFS_LEN: usize = 20;

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

    fn number_of_inputs(&self) -> u32 {
        1
    }

    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl IIRFilterNode {
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
    pub fn new<C: BaseAudioContext>(context: &C, options: IIRFilterOptions) -> Self {
        context.base().register(move |registration| {
            let IIRFilterOptions {
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
        frequency_hz: &mut [f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        self.validate_inputs(frequency_hz, mag_response, phase_response);
        let sample_rate = f64::from(self.context().sample_rate_raw().0);

        for (i, &f) in frequency_hz.iter().enumerate() {
            let mut num: Complex<f64> = Complex::new(0., 0.);
            let mut denom: Complex<f64> = Complex::new(0., 0.);

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &ff) in self.feedforward.iter().enumerate() {
                num += Complex::from_polar(ff, idx as f64 * -2.0 * PI * f64::from(f) / sample_rate);
            }

            // 0 through 20 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            for (idx, &fb) in self.feedback.iter().enumerate() {
                denom +=
                    Complex::from_polar(fb, idx as f64 * -2.0 * PI * f64::from(f) / sample_rate);
            }

            let h_f = num / denom;

            // Possible truncation is fine. f32 precision should be sufficients
            // And it is required by the specs
            mag_response[i] = h_f.norm() as f32;
            phase_response[i] = h_f.arg() as f32;
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
            " InvalidAccessError: All paramaters should be the same length"
        );
        assert_eq!(
            mag_response.len(),
            phase_response.len(),
            " InvalidAccessError: All paramaters should be the same length"
        );

        // Ensures that given frequencies are in the correct range
        let min = 0.;
        let max = self.context().sample_rate() / 2.;
        for f in frequency_hz.iter_mut() {
            *f = f.clamp(min, max);
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
        _timestamp: f64,
        _sample_rate: SampleRate,
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
mod test {
    use float_eq::assert_float_eq;
    use realfft::num_traits::Zero;
    use std::{
        cmp::min,
        fs::File,
        io::{BufRead, BufReader},
    };

    use crate::{
        context::{BaseAudioContext, OfflineAudioContext},
        node::{AudioNode, AudioScheduledSourceNode},
        SampleRate,
    };

    use super::{ChannelConfigOptions, IIRFilterNode, IIRFilterOptions};

    const LENGTH: usize = 512;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let feedforward = vec![
            0.000_016_636_797_512_844_526,
            0.000_033_273_595_025_689_05,
            0.000_016_636_797_512_844_526,
        ];
        let feedback = vec![1.0, -1.988_430_010_622_553_9, 0.988_496_557_812_605_4];

        let options = IIRFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };

        let _biquad = IIRFilterNode::new(&context, options);
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

        let options = IIRFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let biquad = IIRFilterNode::new(&context, options);

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

        let options = IIRFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let biquad = IIRFilterNode::new(&context, options);

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

        let options = IIRFilterOptions {
            feedback,
            feedforward,
            channel_config: ChannelConfigOptions::default(),
        };
        let iir = IIRFilterNode::new(&context, options);
        // It will be fine for the usual fs
        #[allow(clippy::cast_precision_loss)]
        let niquyst = context.sample_rate() / 2.0;

        let mut frequency_hz = [-100., 1_000_000.];
        let mut mag_response = [0., 0.];
        let mut phase_response = [0., 0.];

        iir.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);

        let ref_arr = [0., niquyst];
        assert_float_eq!(frequency_hz, ref_arr, abs_all <= 0.);
    }

    #[test]
    fn check_get_frequency_response() {
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
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
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
    fn default_periodic_wave_rendering_should_match_snapshot() {
        // the snapshot data has been verified by fft to make sure that the frequency
        // response correspond to a HP filter with Fc 4000 Hz
        let file = File::open("./snapshots/white_hp.json").expect("Reading snapshot file failed");
        let reader = BufReader::new(file);
        let ref_filtered: Vec<f32> = reader
            .lines()
            .map(|l| l.unwrap().parse::<f32>().unwrap())
            .collect();

        let mut context = OfflineAudioContext::new(1, LENGTH, SampleRate(44_100));

        let file = File::open("samples/white.ogg").unwrap();
        let audio_buffer = context.decode_audio_data_sync(file).unwrap();

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
        iir.connect(&context.destination());

        let src = context.create_buffer_source();
        src.set_buffer(audio_buffer);
        src.connect(&iir);
        src.start();

        let output = context.start_rendering_sync();

        // review the following, this should be fixed using an AudioBufferSourceNode

        // retrieve processed data by removing silence chunk
        // These silence slices are inserted inconsistently from an test execution to another
        // Without these processing the test would be brittle
        let data_ch: Vec<f32> = output
            .channel_data(0)
            .as_slice()
            .iter()
            .filter(|x| !x.is_zero())
            .copied()
            .collect();

        let min_len = min(data_ch.len(), ref_filtered.len());

        // todo instable test
        assert_float_eq!(
            data_ch[0..min_len],
            ref_filtered[0..min_len],
            abs_all <= 1.0
        );
    }
}
