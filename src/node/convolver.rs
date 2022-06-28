use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};
use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use realfft::{num_complex::Complex, RealFftPlanner};

/// `ConvolverNode` options
//dictionary ConvolverOptions : AudioNodeOptions {
//  AudioBuffer? buffer;
//  boolean disableNormalization = false;
//};
#[derive(Clone, Debug, Default)]
pub struct ConvolverOptions {
    /// The desired buffer for the ConvolverNode
    pub buffer: Option<AudioBuffer>,
    /// The opposite of the desired initial value for the normalize attribute
    pub disable_normalization: bool,
    /// AudioNode options
    pub channel_config: ChannelConfigOptions,
}

pub struct ConvolverNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Info about audio node channel configuration
    channel_config: ChannelConfig,
    /// Perform equal power normalization on response buffer
    normalize: bool,
}

impl AudioNode for ConvolverNode {
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

impl ConvolverNode {
    /// returns a `ConvolverNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - convolver options
    pub fn new<C: BaseAudioContext>(context: &C, options: ConvolverOptions) -> Self {
        context.base().register(move |registration| {
            let ConvolverOptions {
                buffer,
                disable_normalization,
                channel_config,
            } = options;

            if disable_normalization {
                panic!("unimplemented");
            }

            let buffer = buffer.expect("optional buffer not yet supported");
            let length = buffer.length();

            // Pad the response buffer with zeroes so its size is a power of 2
            let padded_length = length.next_power_of_two();
            let sample_rate = buffer.sample_rate();
            let mut samples = vec![0.; padded_length];
            samples[..length].copy_from_slice(&buffer.get_channel_data(0));
            let buffer = AudioBuffer::from(vec![samples], sample_rate);

            let renderer = ConvolverRenderer::new(buffer);

            let node = Self {
                registration,
                channel_config: channel_config.into(),
                normalize: !disable_normalization,
            };

            (node, Box::new(renderer))
        })
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }

    pub fn set_normalize(&mut self, value: bool) {
        // TODO, use AtomicBool to prevent &mut self?
        self.normalize = value;
    }
}

struct ConvolverRenderer {
    length: usize,
    response_fft: Vec<Complex<f32>>,
    sample_buffer: Vec<f32>,
    fft_planner: RealFftPlanner<f32>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,
}

impl ConvolverRenderer {
    fn new(response: AudioBuffer) -> Self {
        let length = response.length();
        let sample_buffer = vec![0.; length];

        let mut fft_planner = RealFftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(length);
        let mut fft_input = fft.make_input_vec();
        let mut fft_scratch = fft.make_scratch_vec();
        let mut fft_output = fft.make_output_vec();

        fft_input.copy_from_slice(response.get_channel_data(0));
        fft.process_with_scratch(&mut fft_input, &mut fft_output, &mut fft_scratch)
            .unwrap();
        let response_fft = fft_output.clone();

        Self {
            length,
            response_fft,
            sample_buffer,
            fft_planner,
            fft_input,
            fft_scratch,
            fft_output,
        }
    }
}

impl AudioProcessor for ConvolverRenderer {
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
        output.force_mono();

        // shift previous samples to the right
        self.sample_buffer.copy_within(128.., 0);

        // add current input to sample buffer
        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);
        self.sample_buffer[self.length - 128..].copy_from_slice(mono.channel_data(0).as_slice());

        // FFT the entire sample buffer
        let fft = self.fft_planner.plan_fft_forward(self.length);
        fft.process_with_scratch(
            &mut self.sample_buffer,
            &mut self.fft_output,
            &mut self.fft_scratch,
        )
        .unwrap();

        // Multiply frequency data with the response
        self.fft_output
            .iter_mut()
            .zip(self.response_fft.iter())
            .for_each(|(o, r)| *o *= r);

        // inverse FFT
        let fft = self.fft_planner.plan_fft_inverse(self.length);
        let fft_result = fft.process_with_scratch(
            &mut self.fft_output,
            &mut self.fft_input,
            &mut self.fft_scratch,
        );
        fft_result.unwrap();

        // write samples back to output, take them somewhere from the middle to prevent boundary
        // artefacts - hence, add a delay of 512 samples
        self.fft_input[self.length - 512..self.length - 512 + 128]
            .iter()
            .zip(output.channel_data_mut(0).iter_mut())
            .for_each(|(f, o)| *o = *f / self.length as f32);

        true
    }
}
