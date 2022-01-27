use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use crossbeam_channel::{Receiver, Sender};
use once_cell::sync::OnceCell;
use rubato::{FftFixedInOut, Resampler};

use crate::{
    context::{AudioContextRegistration, BaseAudioContext},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum},
    SampleRate,
};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

struct CurveMessage(Vec<f32>);

/// enumerates the oversampling rate available for `WaveShaperNode`
#[derive(Debug, Clone, Copy, PartialEq)]
// the naming comes from the web audio specfication
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
            curve: None,
            oversample: OverSampleType::None,
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
/// use web_audio_api::node::AudioNode;
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
/// let context = AudioContext::new(None);
///
/// let file = File::open("sample.wav").unwrap();
/// let buffer = context.decode_audio_data(file).unwrap();
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
    oversample: Arc<AtomicU32>,
    /// Channel between node and renderer (sender part)
    sender: Sender<CurveMessage>,
}

impl AudioNode for WaveShaperNode {
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

impl WaveShaperNode {
    /// returns a `WaveShaperNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - waveshaper options
    pub fn new<C: BaseAudioContext>(context: &C, options: WaveShaperOptions) -> Self {
        context.base().register(move |registration| {
            let WaveShaperOptions {
                curve,
                oversample,
                channel_config,
            } = options;

            let sample_rate = context.sample_rate_raw().0 as usize;
            let channel_config = channel_config.into();
            let oversample = Arc::new(AtomicU32::new(oversample as u32));

            // Channel to send the `curve` to the renderer
            // A capacity of 1 suffices since it is not allowed to set the value multiple times
            let (sender, receiver) = crossbeam_channel::bounded(1);

            let config = RendererConfig {
                sample_rate,
                oversample: oversample.clone(),
                receiver,
            };

            let renderer = WaveShaperRenderer::new(config);
            let node = Self {
                registration,
                channel_config,
                curve: OnceCell::new(),
                oversample,
                sender,
            };

            if let Some(c) = curve {
                node.set_curve(c);
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

        self.sender
            .send(CurveMessage(clone))
            .expect("Sending CurveMessage failed");
    }

    /// Returns the `oversample` faactor of this node
    #[must_use]
    pub fn oversample(&self) -> OverSampleType {
        self.oversample.load(Ordering::SeqCst).into()
    }

    /// set the `oversample` factor of this node
    ///
    /// # Arguments
    ///
    /// * `oversample` - the desired `OversampleType` variant
    pub fn set_oversample(&self, oversample: OverSampleType) {
        self.oversample.store(oversample as u32, Ordering::SeqCst);
    }
}

/// Helper struct which regroups all parameters
/// required to build `WaveShaperRenderer`
struct RendererConfig {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: usize,
    /// oversample factor
    oversample: Arc<AtomicU32>,
    /// Channel between node and renderer (receiver part)
    receiver: Receiver<CurveMessage>,
}

/// `WaveShaperRenderer` represents the rendering part of `WaveShaperNode`
struct WaveShaperRenderer {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: usize,
    /// oversample factor
    oversample: Arc<AtomicU32>,
    /// Number of channels used to build the up/down sampler X2
    channels_x2: usize,
    /// Number of channels used to build the up/down sampler X4
    channels_x4: usize,
    // up sampler configured to multiply by 2 the input signal
    upsampler_x2: FftFixedInOut<f32>,
    // up sampler configured to multiply by 4 the input signal
    upsampler_x4: FftFixedInOut<f32>,
    // down sampler configured to divide by 4 the upsampled signal
    downsampler_x2: FftFixedInOut<f32>,
    // down sampler configured to divide by 4 the upsampled signal
    downsampler_x4: FftFixedInOut<f32>,
    /// distortion curve
    curve: Option<Vec<f32>>,
    /// Channel between node and renderer (receiver part)
    receiver: Receiver<CurveMessage>,
}

impl AudioProcessor for WaveShaperRenderer {
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

        // Check if a curve have been set at k-rate
        if let Ok(msg) = self.receiver.try_recv() {
            self.curve = Some(msg.0);
        }

        *output = input.clone();

        if self.curve.is_some() {
            match self.oversample.load(Ordering::SeqCst).into() {
                OverSampleType::None => {
                    output.modify_channels(|channel| {
                        channel.iter_mut().for_each(|o| *o = self.apply_curve(*o));
                    });
                }
                OverSampleType::X2 => {
                    let channels = output.channels();

                    // recreate up/down sampler if number of channels changed
                    if channels.len() != self.channels_x2 {
                        self.channels_x2 = channels.len();

                        self.upsampler_x2 = FftFixedInOut::<f32>::new(
                            self.sample_rate,
                            self.sample_rate * 2,
                            256,
                            self.channels_x2,
                        );

                        self.downsampler_x2 = FftFixedInOut::<f32>::new(
                            self.sample_rate * 2,
                            self.sample_rate,
                            128,
                            self.channels_x2,
                        );
                    }

                    let mut up_channels = self.upsampler_x2.process(channels).unwrap();

                    for channel in up_channels.iter_mut() {
                        for s in channel.iter_mut() {
                            *s = self.apply_curve(*s);
                        }
                    }

                    let down_channels = self.downsampler_x2.process(&up_channels).unwrap();

                    for (processed, output) in down_channels.iter().zip(output.channels_mut()) {
                        output.copy_from_slice(&processed[..]);
                    }
                }
                OverSampleType::X4 => {
                    let channels = output.channels();

                    // recreate up/down sampler if number of channels changed
                    if channels.len() != self.channels_x4 {
                        self.channels_x4 = channels.len();

                        self.upsampler_x4 = FftFixedInOut::<f32>::new(
                            self.sample_rate,
                            self.sample_rate * 4,
                            512,
                            self.channels_x4,
                        );

                        self.downsampler_x4 = FftFixedInOut::<f32>::new(
                            self.sample_rate * 4,
                            self.sample_rate,
                            128,
                            self.channels_x4,
                        );
                    }

                    let mut up_channels = self.upsampler_x4.process(channels).unwrap();

                    for channel in up_channels.iter_mut() {
                        for s in channel.iter_mut() {
                            *s = self.apply_curve(*s);
                        }
                    }

                    let down_channels = self.downsampler_x4.process(&up_channels).unwrap();

                    for (processed, output) in down_channels.iter().zip(output.channels_mut()) {
                        output.copy_from_slice(&processed[..]);
                    }
                }
            }
        }

        // @tbc - rubato::FftFixedInOut doesn't seem to introduce any latency
        false
    }
}

impl WaveShaperRenderer {
    /// returns an `WaveShaperRenderer` instance
    #[allow(clippy::missing_const_for_fn)]
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            sample_rate,
            oversample,
            receiver,
        } = config;

        let channels_x2 = 1;
        let channels_x4 = 1;

        let upsampler_x2 = FftFixedInOut::<f32>::new(
            sample_rate as usize,
            sample_rate as usize * 2,
            256,
            channels_x2,
        );

        let downsampler_x2 = FftFixedInOut::<f32>::new(
            sample_rate as usize * 2,
            sample_rate as usize,
            128,
            channels_x2,
        );

        let upsampler_x4 = FftFixedInOut::<f32>::new(
            sample_rate as usize,
            sample_rate as usize * 4,
            512,
            channels_x4,
        );

        let downsampler_x4 = FftFixedInOut::<f32>::new(
            sample_rate as usize * 4,
            sample_rate as usize,
            128,
            channels_x4,
        );

        Self {
            sample_rate,
            oversample,
            channels_x2,
            channels_x4,
            upsampler_x2,
            upsampler_x4,
            downsampler_x2,
            downsampler_x4,
            curve: None,
            receiver,
        }
    }

    #[inline]
    fn apply_curve(&self, input: f32) -> f32 {
        // curve is always set at this point
        let curve = self.curve.as_deref().unwrap();

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
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::OfflineAudioContext;
    use crate::SampleRate;

    use super::*;

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _shaper = WaveShaperNode::new(&context, WaveShaperOptions::default());
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _shaper = context.create_wave_shaper();
    }

    #[test]
    fn test_default_options() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let shaper = WaveShaperNode::new(&context, WaveShaperOptions::default());

        assert_eq!(shaper.curve(), None);
        assert_eq!(shaper.oversample(), OverSampleType::None);
    }

    #[test]
    fn test_user_defined_options() {
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

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
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

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
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

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
        let sample_rate = SampleRate(128);
        let mut context = OfflineAudioContext::new(1, 3 * 128, sample_rate);

        let shaper = context.create_wave_shaper();
        let curve = vec![-0.5, 0., 0.5];
        shaper.set_curve(curve);
        shaper.connect(&context.destination());

        let mut data = vec![0.; 3 * 128];
        let mut expected = vec![0.; 3 * 128];
        for i in 0..(3 * 128) {
            if i < 128 {
                data[i] = -1.;
                expected[i] = -0.5;
            } else if i < 2 * 128 {
                data[i] = 0.;
                expected[i] = 0.;
            } else {
                data[i] = 1.;
                expected[i] = 0.5;
            }
        }
        let mut buffer = context.create_buffer(1, 3 * 128, sample_rate);
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
        let sample_rate = SampleRate(128);
        let mut context = OfflineAudioContext::new(1, 128, sample_rate);

        let shaper = context.create_wave_shaper();
        let curve = vec![-0.5, 0., 0.5];
        shaper.set_curve(curve);
        shaper.connect(&context.destination());

        let mut data = vec![0.; 128];
        let mut expected = vec![0.; 128];

        for i in 0..128 {
            let sample = i as f32 / 128. * 2. - 1.;
            data[i] = sample;
            expected[i] = sample / 2.;
        }

        let mut buffer = context.create_buffer(1, 3 * 128, sample_rate);
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
