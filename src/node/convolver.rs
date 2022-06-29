use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};
use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use crossbeam_channel::{Receiver, Sender};

use realfft::{num_complex::Complex, RealFftPlanner};

/// Scale buffer by an equal-power normalization
fn normalize_buffer(buffer: &mut AudioBuffer) {
    let gain_calibration = 0.00125;
    let gain_calibration_sample_rate = 44100.;
    let min_power = 0.000125;

    // Normalize by RMS power.
    let number_of_channels = buffer.number_of_channels();
    let length = buffer.length();

    let mut power: f32 = buffer
        .channels()
        .iter()
        .map(|c| c.as_slice().iter().map(|&s| s * s).sum::<f32>())
        .sum();

    power = (power / (number_of_channels * length) as f32).sqrt();

    // Protect against accidental overload.
    if !power.is_finite() || power.is_nan() || power < min_power {
        power = min_power;
    }

    let mut scale = 1. / power;

    // Calibrate to make perceived volume same as unprocessed.
    scale *= gain_calibration;

    // Scale depends on sample-rate.
    scale *= gain_calibration_sample_rate / buffer.sample_rate();

    // True-stereo compensation.
    if number_of_channels == 4 {
        scale *= 0.5;
    }

    buffer
        .channels_mut()
        .iter_mut()
        .for_each(|c| c.as_mut_slice().iter_mut().for_each(|s| *s *= scale))
}

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

/// Processing node which applies a linear convolution effect given an impulse response.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/ConvolverNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#ConvolverNode>
/// - see also:
/// [`BaseAudioContext::create_convolver`](crate::context::BaseAudioContext::create_convolver)
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
///
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ConvolverNode, ConvolverOptions};
///
/// let context = AudioContext::default();
/// let file = File::open("samples/vocals-dry.wav").unwrap();
/// let audio_buffer = context.decode_audio_data_sync(file).unwrap();
///
/// let impulse_file = File::open("samples/small-room-response.wav").unwrap();
/// let impulse_buffer = context.decode_audio_data_sync(impulse_file).unwrap();
///
/// let src = context.create_buffer_source();
/// src.set_buffer(audio_buffer);
///
/// let mut convolve = ConvolverNode::new(&context, ConvolverOptions::default());
/// convolve.set_buffer(Some(impulse_buffer));
///
/// src.connect(&convolve);
/// convolve.connect(&context.destination());
/// src.start();
/// std::thread::sleep(std::time::Duration::from_millis(4_000));
/// ```
///
/// # Examples
///
/// - `cargo run --release --example convolution`
///
pub struct ConvolverNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Info about audio node channel configuration
    channel_config: ChannelConfig,
    /// Perform equal power normalization on response buffer
    normalize: bool,
    /// The response buffer, nullable
    buffer: Option<AudioBuffer>,
    /// Message bus to the renderer
    sender: Sender<ConvolverRendererInner>,
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
    ///
    /// # Panics
    ///
    /// Panics when an AudioBuffer is provided via the `ConvolverOptions` with a sample rate
    /// different from the audio context sample rate.
    pub fn new<C: BaseAudioContext>(context: &C, options: ConvolverOptions) -> Self {
        context.base().register(move |registration| {
            let ConvolverOptions {
                buffer,
                disable_normalization,
                channel_config,
            } = options;

            // Channel to send buffer channels references to the renderer.  A capacity of 1
            // suffices, it will simply block the control thread when used concurrently
            let (sender, receiver) = crossbeam_channel::bounded(1);

            let renderer = ConvolverRenderer::new(receiver);

            let mut node = Self {
                registration,
                channel_config: channel_config.into(),
                normalize: !disable_normalization,
                sender,
                buffer: None,
            };

            node.set_buffer(buffer);

            (node, Box::new(renderer))
        })
    }

    /// Get the current impulse response buffer
    pub fn buffer(&self) -> Option<&AudioBuffer> {
        self.buffer.as_ref()
    }

    /// Set or update the impulse response buffer
    ///
    /// # Panics
    ///
    /// Panics when the sample rate of the provided AudioBuffer differs from the audio context
    /// sample rate.
    pub fn set_buffer(&mut self, buffer: Option<AudioBuffer>) {
        if let Some(buffer) = &buffer {
            // todo, resample when this happens?
            assert_eq!(buffer.sample_rate(), self.context().sample_rate());

            let length = buffer.length();

            // Pad the response buffer with zeroes so its size is a power of 2
            let padded_length = length.next_power_of_two();
            let sample_rate = buffer.sample_rate();
            let mut samples = vec![0.; padded_length];
            samples[..length].copy_from_slice(buffer.get_channel_data(0));
            let mut padded_buffer = AudioBuffer::from(vec![samples], sample_rate);

            if self.normalize {
                normalize_buffer(&mut padded_buffer);
            }

            let convolve = ConvolverRendererInner::new(padded_buffer);
            let _ = self.sender.send(convolve); // can fail when render thread shut down
        }

        self.buffer = buffer;
    }

    /// Denotes if the response buffer will be scaled with an equal-power normalization
    pub fn normalize(&self) -> bool {
        self.normalize
    }

    /// Update the `normalize` setting. This will only have an effect when `set_buffer` is called.
    pub fn set_normalize(&mut self, value: bool) {
        // TODO, use AtomicBool to prevent &mut self?
        self.normalize = value;
    }
}

struct ConvolverRenderer {
    receiver: Receiver<ConvolverRendererInner>,
    inner: Option<ConvolverRendererInner>,
}

impl ConvolverRenderer {
    fn new(receiver: Receiver<ConvolverRendererInner>) -> Self {
        Self {
            receiver,
            inner: None,
        }
    }
}

struct ConvolverRendererInner {
    length: usize,
    response_fft: Vec<Complex<f32>>,
    sample_buffer: Vec<f32>,
    fft_planner: RealFftPlanner<f32>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,
}

impl ConvolverRendererInner {
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

        if let Ok(mut msg) = self.receiver.try_recv() {
            // Copy over previous sample buffer, if any
            if let Some(inner) = &mut self.inner {
                let prev_len = inner.sample_buffer.len();
                let new_len = msg.sample_buffer.len();
                let shared_size = prev_len.min(new_len);
                msg.sample_buffer[..shared_size]
                    .copy_from_slice(&inner.sample_buffer[..shared_size]);
            }
            self.inner = Some(msg);
        }

        let convolver = match &mut self.inner {
            None => {
                // no convolution buffer set, passthrough
                *output = input.clone();
                return true;
            }
            Some(convolver) => convolver,
        };

        // shift previous samples to the right
        convolver.sample_buffer.copy_within(128.., 0);

        // add current input to sample buffer
        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);
        convolver.sample_buffer[convolver.length - 128..]
            .copy_from_slice(mono.channel_data(0).as_slice());

        // FFT the entire sample buffer
        let fft = convolver.fft_planner.plan_fft_forward(convolver.length);
        fft.process_with_scratch(
            &mut convolver.sample_buffer,
            &mut convolver.fft_output,
            &mut convolver.fft_scratch,
        )
        .unwrap();

        // Multiply frequency data with the response
        convolver
            .fft_output
            .iter_mut()
            .zip(convolver.response_fft.iter())
            .for_each(|(o, r)| *o *= r);

        // inverse FFT
        let fft = convolver.fft_planner.plan_fft_inverse(convolver.length);
        let fft_result = fft.process_with_scratch(
            &mut convolver.fft_output,
            &mut convolver.fft_input,
            &mut convolver.fft_scratch,
        );
        fft_result.unwrap();

        // write samples back to output, take them somewhere from the middle to prevent boundary
        // artefacts - hence, add a delay of 512 samples
        convolver.fft_input[convolver.length - 512..convolver.length - 512 + 128]
            .iter()
            .zip(output.channel_data_mut(0).iter_mut())
            .for_each(|(f, o)| *o = *f / convolver.length as f32);

        true
    }
}
