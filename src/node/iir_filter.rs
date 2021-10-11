use super::AudioNode;
use crate::{
    alloc::AudioBuffer,
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};
use num_complex::Complex;
use std::{collections::VecDeque, f64::consts::PI};

const MAX_IIR_COEFFS_LEN: usize = 20;

/// The IirFilterOptions is used to specify the filter coefficients
pub struct IirFilterOptions {
    /// audio node options
    pub channel_config: ChannelConfigOptions,
    /// feedforward coefficients
    pub feedforward: Vec<f64>,
    /// feedback coefficients
    pub feedback: Vec<f64>,
}

/// An AudioNode implementing a general IIR filter
pub struct IirFilterNode {
    sample_rate: f64,
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
    /// Creates an IirFilterNode
    ///
    /// # Arguments
    ///
    /// * `context` - Audio context in which the node will live
    /// * `options` - node options
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

            let sample_rate = context.base().sample_rate().0 as f64;

            let config = RendererConfig {
                feedforward: feedforward.clone(),
                feedback: feedback.clone(),
            };

            let render = IirFilterRenderer::new(config);

            let node = IirFilterNode {
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
    pub fn get_frequency_response(
        &self,
        frequency_hz: &[f32],
        mag_response: &mut [f32],
        phase_response: &mut [f32],
    ) {
        for (i, &f) in frequency_hz.iter().enumerate() {
            let mut num: Complex<f64> = Complex::new(0., 0.);
            let mut denom: Complex<f64> = Complex::new(0., 0.);

            for (idx, &ff) in self.feedforward.iter().enumerate() {
                num +=
                    Complex::from_polar(ff, idx as f64 * -2.0 * PI * f as f64 / self.sample_rate);
            }

            for (idx, &fb) in self.feedback.iter().enumerate() {
                denom +=
                    Complex::from_polar(fb, idx as f64 * -2.0 * PI * f as f64 / self.sample_rate);
            }

            let h_f = num / denom;

            mag_response[i] = h_f.norm() as f32;
            phase_response[i] = h_f.arg() as f32;
        }
    }
}

struct RendererConfig {
    feedforward: Vec<f64>,
    feedback: Vec<f64>,
}

/// Renderer associated with the IirFilterNode
struct IirFilterRenderer {
    /// Numerator coefficients
    feedforward: Vec<f64>,
    /// Denominator coefficients
    feedback: Vec<f64>,
    /// input states -- x[n-k] from k=0
    x_n: VecDeque<f64>,
    /// output states -- y[n-k] from k=0
    y_n: VecDeque<f64>,
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
        false
    }
}

impl IirFilterRenderer {
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            feedforward,
            feedback,
        } = config;

        let ffs_len = feedforward.len();
        let fbs_len = feedback.len();

        Self {
            feedforward,
            feedback,
            x_n: VecDeque::from(vec![0.; ffs_len]),
            y_n: VecDeque::from(vec![0.; fbs_len - 1]),
        }
    }

    /// Generate an output by filtering the input
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    /// * `output` - Audiobuffer output
    fn filter(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        for (i_data, o_data) in input.channels().iter().zip(output.channels_mut()) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i);
            }
        }
    }

    /// Generate an output sample by filtering an input sample
    ///
    /// # Arguments
    ///
    /// * `input` - Audiobuffer input
    fn tick(&mut self, input: f32) -> f32 {
        let mut output = 0.;
        let a0 = self.feedback[0];
        self.x_n.push_front(input as f64);
        self.x_n.pop_back();
        for (b, x) in self.feedforward.iter().zip(&self.x_n) {
            output += b / a0 * x;
        }

        for (a, y) in self.feedback.iter().skip(1).zip(&self.y_n) {
            output -= a / a0 * y;
        }
        self.y_n.push_front(output);
        self.y_n.pop_back();

        output as f32
    }
}
