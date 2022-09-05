use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};
use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::RENDER_QUANTUM_SIZE;

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

fn roll_zero<T: Default + Copy>(signal: &mut [T], n: usize) {
    // roll array by n elements
    // zero out the last n elements
    let len = signal.len();
    signal.copy_within(n.., 0);
    signal[len - n..].fill(T::default());
}

struct FFT {
    fft_planner: RealFftPlanner<f32>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,
}

impl FFT {
    fn new(length: usize) -> Self {
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(length);
        let fft_input = fft.make_input_vec();
        let fft_scratch = fft.make_scratch_vec();
        let fft_output = fft.make_output_vec();

        Self {
            fft_planner,
            fft_input,
            fft_scratch,
            fft_output,
        }
    }

    fn real(&mut self) -> &mut [f32] {
        &mut self.fft_input[..]
    }

    fn complex(&mut self) -> &mut [Complex<f32>] {
        &mut self.fft_output[..]
    }

    fn process(&mut self) -> &[Complex<f32>] {
        let fft = self.fft_planner.plan_fft_forward(self.fft_input.len());
        fft.process_with_scratch(
            &mut self.fft_input,
            &mut self.fft_output,
            &mut self.fft_scratch,
        )
        .unwrap();
        &self.fft_output[..]
    }

    fn inverse(&mut self) -> &[f32] {
        let fft = self.fft_planner.plan_fft_inverse(self.fft_input.len());
        fft.process_with_scratch(
            &mut self.fft_output,
            &mut self.fft_input,
            &mut self.fft_scratch,
        )
        .unwrap();
        &self.fft_input[..]
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
    num_ir_blocks: usize,
    h: Vec<Complex<f32>>,
    fdl: Vec<Complex<f32>>,
    out: Vec<f32>,
    fft2: FFT,
}

impl ConvolverRendererInner {
    fn new(response: AudioBuffer) -> Self {
        // mono processing only for now
        let response = response.channel_data(0).as_slice();

        let mut fft2 = FFT::new(2 * RENDER_QUANTUM_SIZE);
        let p = response.len();

        let num_ir_blocks = p / RENDER_QUANTUM_SIZE;

        let mut h = vec![Complex::default(); num_ir_blocks * 2 * RENDER_QUANTUM_SIZE];
        for (resp_fft, resp) in h
            .chunks_mut(2 * RENDER_QUANTUM_SIZE)
            .zip(response.chunks(RENDER_QUANTUM_SIZE))
        {
            // fill resp_fft with FFT of resp.zero_pad(RENDER_QUANTUM_SIZE)
            fft2.real()[..RENDER_QUANTUM_SIZE].copy_from_slice(resp);
            fft2.real()[RENDER_QUANTUM_SIZE..].fill(0.);
            resp_fft[..fft2.complex().len()].copy_from_slice(fft2.process());
        }

        let fdl = vec![Complex::default(); 2 * RENDER_QUANTUM_SIZE * num_ir_blocks];
        let out = vec![0.; 2 * RENDER_QUANTUM_SIZE - 1];

        Self {
            num_ir_blocks,
            h,
            fdl,
            out,
            fft2,
        }
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> bool {
        self.fft2.real()[..RENDER_QUANTUM_SIZE].copy_from_slice(input);
        self.fft2.real()[RENDER_QUANTUM_SIZE..].fill(0.);
        let spectrum = self.fft2.process();

        self.fdl
            .chunks_mut(2 * RENDER_QUANTUM_SIZE)
            .zip(self.h.chunks(2 * RENDER_QUANTUM_SIZE))
            .for_each(|(fdl_c, h_c)| {
                fdl_c
                    .iter_mut()
                    .zip(h_c)
                    .zip(spectrum)
                    .for_each(|((f, h), s)| *f += h * s)
            });

        let c_len = self.fft2.complex().len();
        self.fft2.complex().copy_from_slice(&self.fdl[..c_len]);
        let inverse = self.fft2.inverse();
        self.out.iter_mut().zip(inverse).for_each(|(o, i)| {
            *o += i / RENDER_QUANTUM_SIZE as f32;
        });

        output.copy_from_slice(&self.out[..RENDER_QUANTUM_SIZE]);

        roll_zero(&mut self.fdl[..], 2 * RENDER_QUANTUM_SIZE);
        roll_zero(&mut self.out[..], RENDER_QUANTUM_SIZE);

        true
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

        // handle new impulse response buffer, if any
        if let Ok(msg) = self.receiver.try_recv() {
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

        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);
        let input = &mono.channel_data(0)[..];
        let output = &mut output.channel_data_mut(0)[..];

        convolver.process(input, output)
    }
}
