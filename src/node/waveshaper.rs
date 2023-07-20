use std::any::Any;
use std::cell::OnceCell;
use std::sync::atomic::{AtomicU32, Ordering};

use rubato::{FftFixedInOut, Resampler as _};

use crate::{
    context::{AudioContextRegistration, BaseAudioContext},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope},
    RENDER_QUANTUM_SIZE,
};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// enumerates the oversampling rate available for `WaveShaperNode`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// the naming comes from the web audio specification
pub enum OverSampleType {
    /// No oversampling is applied
    None,
    /// Oversampled by a factor of 2
    X2,
    /// Oversampled by a factor of 4
    X4,
}

impl Default for OverSampleType {
    fn default() -> Self {
        Self::None
    }
}

impl From<u32> for OverSampleType {
    fn from(i: u32) -> Self {
        match i {
            0 => OverSampleType::None,
            1 => OverSampleType::X2,
            2 => OverSampleType::X4,
            _ => unreachable!(),
        }
    }
}

/// `WaveShaperNode` options
// dictionary WaveShaperOptions : AudioNodeOptions {
//   sequence<float> curve;
//   OverSampleType oversample = "none";
// };
#[derive(Clone, Debug)]
pub struct WaveShaperOptions {
    /// The distortion curve
    pub curve: Option<Vec<f32>>,
    /// Oversampling rate - default to `None`
    pub oversample: OverSampleType,
    /// audio node options
    pub channel_config: ChannelConfigOptions,
}

impl Default for WaveShaperOptions {
    fn default() -> Self {
        Self {
            oversample: OverSampleType::None,
            curve: None,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// `WaveShaperNode` allows to apply non-linear distortion effect on a audio
/// signal. Arbitrary non-linear shaping curves may be specified.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/WaveShaperNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#WaveShaperNode>
/// - see also: [`BaseAudioContext::create_wave_shaper`](crate::context::BaseAudioContext::create_wave_shaper)
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// # use std::f32::consts::PI;
/// # fn make_distortion_curve(size: usize) -> Vec<f32> {
/// #     let mut curve = vec![0.; size];
/// #     let mut phase = 0.;
/// #     let phase_incr = PI / (size - 1) as f32;
/// #     for i in 0..size {
/// #         curve[i] = (PI + phase).cos();
/// #         phase += phase_incr;
/// #     }
/// #     curve
/// # }
/// let context = AudioContext::default();
///
/// let file = File::open("sample.wav").unwrap();
/// let buffer = context.decode_audio_data_sync(file).unwrap();
/// let curve = make_distortion_curve(2048);
/// let drive = 4.;
///
/// let post_gain = context.create_gain();
/// post_gain.connect(&context.destination());
/// post_gain.gain().set_value(1. / drive);
///
/// let shaper = context.create_wave_shaper();
/// shaper.connect(&post_gain);
/// shaper.set_curve(curve);
///
/// let pre_gain = context.create_gain();
/// pre_gain.connect(&shaper);
/// pre_gain.gain().set_value(drive);
///
/// let src = context.create_buffer_source();
/// src.connect(&pre_gain);
/// src.set_buffer(buffer);
///
/// src.start();
/// ```
///
/// # Example
///
/// - `cargo run --release --example waveshaper`
pub struct WaveShaperNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// distortion curve
    curve: OnceCell<Vec<f32>>,
    /// oversample type
    oversample: AtomicU32,
}

impl AudioNode for WaveShaperNode {
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

impl WaveShaperNode {
    /// returns a `WaveShaperNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - waveshaper options
    pub fn new<C: BaseAudioContext>(context: &C, options: WaveShaperOptions) -> Self {
        context.register(move |registration| {
            let WaveShaperOptions {
                oversample,
                curve,
                channel_config,
            } = options;

            let sample_rate = context.sample_rate() as usize;
            let channel_config = channel_config.into();

            let config = RendererConfig {
                oversample,
                curve: curve.clone(),
                sample_rate,
            };

            let renderer = WaveShaperRenderer::new(config);
            let node = Self {
                registration,
                channel_config,
                curve: OnceCell::new(),
                oversample: AtomicU32::new(oversample as u32),
            };

            if let Some(c) = curve {
                // we are sure the OnceCell is empty, cannot fail
                let _ = node.curve.set(c);
            }

            (node, Box::new(renderer))
        })
    }

    /// Returns the distortion curve
    #[must_use]
    pub fn curve(&self) -> Option<&[f32]> {
        self.curve.get().map(Vec::as_slice)
    }

    /// Set the distortion `curve` of this node
    ///
    /// # Arguments
    ///
    /// * `curve` - the desired distortion `curve`
    ///
    /// # Panics
    ///
    /// Panics if a curve has already been given to the source (though `new` or through
    /// `set_curve`)
    pub fn set_curve(&self, curve: Vec<f32>) {
        let clone = curve.clone();

        if self.curve.set(curve).is_err() {
            panic!("InvalidStateError - cannot assign curve twice");
        }

        self.registration.post_message(clone);
    }

    /// Returns the `oversample` faactor of this node
    #[must_use]
    pub fn oversample(&self) -> OverSampleType {
        self.oversample.load(Ordering::Acquire).into()
    }

    /// set the `oversample` factor of this node
    ///
    /// # Arguments
    ///
    /// * `oversample` - the desired `OversampleType` variant
    pub fn set_oversample(&self, oversample: OverSampleType) {
        self.oversample.store(oversample as u32, Ordering::Release);
        self.registration.post_message(oversample);
    }
}

/// Helper struct which regroups all parameters
/// required to build `WaveShaperRenderer`
struct RendererConfig {
    /// oversample factor
    oversample: OverSampleType,
    /// oversample factor
    curve: Option<Vec<f32>>,
    /// Sample rate (equals to audio context sample rate)
    sample_rate: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResamplerConfig {
    channels: usize,
    chunk_size_in: usize,
    sample_rate_in: usize,
    sample_rate_out: usize,
}

impl ResamplerConfig {
    fn upsample_x2(channels: usize, sample_rate: usize) -> Self {
        let chunk_size_in = RENDER_QUANTUM_SIZE * 2;
        let sample_rate_in = sample_rate;
        let sample_rate_out = sample_rate * 2;
        Self {
            channels,
            chunk_size_in,
            sample_rate_in,
            sample_rate_out,
        }
    }

    fn upsample_x4(channels: usize, sample_rate: usize) -> Self {
        let chunk_size_in = RENDER_QUANTUM_SIZE * 4;
        let sample_rate_in = sample_rate;
        let sample_rate_out = sample_rate * 4;
        Self {
            channels,
            chunk_size_in,
            sample_rate_in,
            sample_rate_out,
        }
    }

    fn downsample_x2(channels: usize, sample_rate: usize) -> Self {
        let chunk_size_in = RENDER_QUANTUM_SIZE;
        let sample_rate_in = sample_rate * 2;
        let sample_rate_out = sample_rate;
        Self {
            channels,
            chunk_size_in,
            sample_rate_in,
            sample_rate_out,
        }
    }

    fn downsample_x4(channels: usize, sample_rate: usize) -> Self {
        let chunk_size_in = RENDER_QUANTUM_SIZE;
        let sample_rate_in = sample_rate * 4;
        let sample_rate_out = sample_rate;
        Self {
            channels,
            chunk_size_in,
            sample_rate_in,
            sample_rate_out,
        }
    }
}

struct Resampler {
    config: ResamplerConfig,
    processor: FftFixedInOut<f32>,
    samples_out: Vec<Vec<f32>>,
}

impl Resampler {
    fn new(config: ResamplerConfig) -> Self {
        let ResamplerConfig {
            channels,
            chunk_size_in,
            sample_rate_in,
            sample_rate_out,
        } = &config;

        let processor =
            FftFixedInOut::new(*sample_rate_in, *sample_rate_out, *chunk_size_in, *channels)
                .unwrap();

        let samples_out = processor.output_buffer_allocate(true);

        Self {
            config,
            processor,
            samples_out,
        }
    }

    fn process<T>(&mut self, samples_in: &[T])
    where
        T: AsRef<[f32]>,
    {
        debug_assert_eq!(self.config.channels, samples_in.len());
        // Processing the output from another resampler directly as input requires this assumption.
        debug_assert!(samples_in
            .iter()
            .all(|channel| channel.as_ref().len() == self.processor.input_frames_next()));
        let (in_len, out_len) = self
            .processor
            .process_into_buffer(samples_in, &mut self.samples_out[..], None)
            .unwrap();
        // All input samples must have been consumed.
        debug_assert_eq!(in_len, samples_in[0].as_ref().len());
        // All output samples must have been initialized.
        debug_assert!(self
            .samples_out
            .iter()
            .all(|channel| channel.len() == out_len));
    }

    fn samples_out(&self) -> &[Vec<f32>] {
        &self.samples_out[..]
    }

    fn samples_out_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.samples_out[..]
    }
}

/// `WaveShaperRenderer` represents the rendering part of `WaveShaperNode`
struct WaveShaperRenderer {
    /// oversample factor
    oversample: OverSampleType,
    /// distortion curve
    curve: Vec<f32>,
    /// whether the distortion curve has been set
    curve_set: bool,
    /// Sample rate (equals to audio context sample rate)
    sample_rate: usize,
    /// Number of channels used to build the up/down sampler X2
    channels_x2: usize,
    /// Number of channels used to build the up/down sampler X4
    channels_x4: usize,
    // up sampler configured to multiply by 2 the input signal
    upsampler_x2: Resampler,
    // up sampler configured to multiply by 4 the input signal
    upsampler_x4: Resampler,
    // down sampler configured to divide by 4 the upsampled signal
    downsampler_x2: Resampler,
    // down sampler configured to divide by 4 the upsampled signal
    downsampler_x4: Resampler,
}

impl AudioProcessor for WaveShaperRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        if input.is_silent() {
            output.make_silent();
            return false;
        }

        *output = input.clone();

        if self.curve_set {
            let curve = &self.curve;

            match self.oversample {
                OverSampleType::None => {
                    output.modify_channels(|channel| {
                        channel.iter_mut().for_each(|o| *o = apply_curve(curve, *o));
                    });
                }
                OverSampleType::X2 => {
                    let channels = output.channels();

                    // recreate up/down sampler if number of channels changed
                    if channels.len() != self.channels_x2 {
                        self.channels_x2 = channels.len();

                        self.upsampler_x2 = Resampler::new(ResamplerConfig::upsample_x2(
                            self.channels_x2,
                            self.sample_rate,
                        ));

                        self.downsampler_x2 = Resampler::new(ResamplerConfig::downsample_x2(
                            self.channels_x2,
                            self.sample_rate,
                        ));
                    }

                    self.upsampler_x2.process(channels);
                    for channel in self.upsampler_x2.samples_out_mut().iter_mut() {
                        for s in channel.iter_mut() {
                            *s = apply_curve(curve, *s);
                        }
                    }

                    self.downsampler_x2.process(self.upsampler_x2.samples_out());

                    for (processed, output) in self
                        .downsampler_x2
                        .samples_out()
                        .iter()
                        .zip(output.channels_mut())
                    {
                        output.copy_from_slice(&processed[..]);
                    }
                }
                OverSampleType::X4 => {
                    let channels = output.channels();

                    // recreate up/down sampler if number of channels changed
                    if channels.len() != self.channels_x4 {
                        self.channels_x4 = channels.len();

                        self.upsampler_x4 = Resampler::new(ResamplerConfig::upsample_x4(
                            self.channels_x4,
                            self.sample_rate,
                        ));

                        self.downsampler_x4 = Resampler::new(ResamplerConfig::downsample_x4(
                            self.channels_x4,
                            self.sample_rate,
                        ));
                    }

                    self.upsampler_x4.process(channels);

                    for channel in self.upsampler_x4.samples_out_mut().iter_mut() {
                        for s in channel.iter_mut() {
                            *s = apply_curve(curve, *s);
                        }
                    }

                    self.downsampler_x4.process(self.upsampler_x4.samples_out());

                    for (processed, output) in self
                        .downsampler_x4
                        .samples_out()
                        .iter()
                        .zip(output.channels_mut())
                    {
                        output.copy_from_slice(&processed[..]);
                    }
                }
            }
        }

        // @tbc - rubato::FftFixedInOut doesn't seem to introduce any latency
        false
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(&oversample) = msg.downcast_ref::<OverSampleType>() {
            self.oversample = oversample;
            return;
        }

        if let Some(curve) = msg.downcast_mut::<Vec<f32>>() {
            std::mem::swap(&mut self.curve, curve);
            self.curve_set = true;

            return;
        }

        log::warn!("WaveShaperRenderer: Dropping incoming message {msg:?}");
    }
}

impl WaveShaperRenderer {
    /// returns an `WaveShaperRenderer` instance
    #[allow(clippy::missing_const_for_fn)]
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            sample_rate,
            oversample,
            curve,
        } = config;

        let curve_set = curve.is_some();
        let curve = curve.unwrap_or(Vec::with_capacity(0));

        let channels_x2 = 1;
        let channels_x4 = 1;

        let upsampler_x2 = Resampler::new(ResamplerConfig::upsample_x2(channels_x2, sample_rate));

        let downsampler_x2 =
            Resampler::new(ResamplerConfig::downsample_x2(channels_x2, sample_rate));

        let upsampler_x4 = Resampler::new(ResamplerConfig::upsample_x4(channels_x2, sample_rate));

        let downsampler_x4 =
            Resampler::new(ResamplerConfig::downsample_x4(channels_x2, sample_rate));

        Self {
            oversample,
            curve,
            curve_set,
            sample_rate,
            channels_x2,
            channels_x4,
            upsampler_x2,
            upsampler_x4,
            downsampler_x2,
            downsampler_x4,
        }
    }
}

#[inline]
fn apply_curve(curve: &[f32], input: f32) -> f32 {
    if curve.is_empty() {
        return 0.;
    }

    let n = curve.len() as f32;
    let v = (n - 1.) / 2.0 * (input + 1.);

    if v <= 0. {
        curve[0]
    } else if v >= n - 1. {
        curve[(n - 1.) as usize]
    } else {
        let k = v.floor();
        let f = v - k;
        (1. - f) * curve[k as usize] + f * curve[(k + 1.) as usize]
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::OfflineAudioContext;
    use crate::node::AudioScheduledSourceNode;

    use super::*;

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
        let _shaper = WaveShaperNode::new(&context, WaveShaperOptions::default());
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
        let _shaper = context.create_wave_shaper();
    }

    #[test]
    fn test_default_options() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);
        let shaper = WaveShaperNode::new(&context, WaveShaperOptions::default());

        assert_eq!(shaper.curve(), None);
        assert_eq!(shaper.oversample(), OverSampleType::None);
    }

    #[test]
    fn test_user_defined_options() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);

        let options = WaveShaperOptions {
            curve: Some(vec![1.0]),
            oversample: OverSampleType::X2,
            ..Default::default()
        };

        let shaper = WaveShaperNode::new(&context, options);

        context.start_rendering_sync();

        assert_eq!(shaper.curve(), Some(&[1.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X2);
    }

    #[test]
    #[should_panic]
    fn change_a_curve_for_another_curve_should_panic() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);

        let options = WaveShaperOptions {
            curve: Some(vec![1.0]),
            oversample: OverSampleType::X2,
            ..Default::default()
        };

        let shaper = WaveShaperNode::new(&context, options);
        assert_eq!(shaper.curve(), Some(&[1.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X2);

        shaper.set_curve(vec![2.0]);
        shaper.set_oversample(OverSampleType::X4);

        context.start_rendering_sync();

        assert_eq!(shaper.curve(), Some(&[2.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X4);
    }

    #[test]
    fn change_none_for_curve_after_build() {
        let context = OfflineAudioContext::new(2, LENGTH, 44_100.);

        let options = WaveShaperOptions {
            curve: None,
            oversample: OverSampleType::X2,
            ..Default::default()
        };

        let shaper = WaveShaperNode::new(&context, options);
        assert_eq!(shaper.curve(), None);
        assert_eq!(shaper.oversample(), OverSampleType::X2);

        shaper.set_curve(vec![2.0]);
        shaper.set_oversample(OverSampleType::X4);

        context.start_rendering_sync();

        assert_eq!(shaper.curve(), Some(&[2.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X4);
    }

    #[test]
    fn test_shape_boundaries() {
        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, 3 * RENDER_QUANTUM_SIZE, sample_rate);

        let shaper = context.create_wave_shaper();
        let curve = vec![-0.5, 0., 0.5];
        shaper.set_curve(curve);
        shaper.connect(&context.destination());

        let mut data = vec![0.; 3 * RENDER_QUANTUM_SIZE];
        let mut expected = vec![0.; 3 * RENDER_QUANTUM_SIZE];
        for i in 0..(3 * RENDER_QUANTUM_SIZE) {
            if i < RENDER_QUANTUM_SIZE {
                data[i] = -1.;
                expected[i] = -0.5;
            } else if i < 2 * RENDER_QUANTUM_SIZE {
                data[i] = 0.;
                expected[i] = 0.;
            } else {
                data[i] = 1.;
                expected[i] = 0.5;
            }
        }
        let mut buffer = context.create_buffer(1, 3 * RENDER_QUANTUM_SIZE, sample_rate);
        buffer.copy_to_channel(&data, 0);

        let src = context.create_buffer_source();
        src.connect(&shaper);
        src.set_buffer(buffer);
        src.start_at(0.);

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_shape_interpolation() {
        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

        let shaper = context.create_wave_shaper();
        let curve = vec![-0.5, 0., 0.5];
        shaper.set_curve(curve);
        shaper.connect(&context.destination());

        let mut data = vec![0.; RENDER_QUANTUM_SIZE];
        let mut expected = vec![0.; RENDER_QUANTUM_SIZE];

        for i in 0..RENDER_QUANTUM_SIZE {
            let sample = i as f32 / (RENDER_QUANTUM_SIZE as f32) * 2. - 1.;
            data[i] = sample;
            expected[i] = sample / 2.;
        }

        let mut buffer = context.create_buffer(1, 3 * RENDER_QUANTUM_SIZE, sample_rate);
        buffer.copy_to_channel(&data, 0);

        let src = context.create_buffer_source();
        src.connect(&shaper);
        src.set_buffer(buffer);
        src.start_at(0.);

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }
}
