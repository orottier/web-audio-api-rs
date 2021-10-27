//! The wave shaper control and renderer parts
// #![warn(
//     clippy::all,
//     clippy::pedantic,
//     clippy::nursery,
//     clippy::perf,
//     clippy::missing_docs_in_private_items
// )]
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use rubato::{FftFixedInOut, Resampler};

use crate::{
    alloc::AudioBuffer,
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};

use super::AudioNode;

/// enumerates the oversampling rate available for `WaveShaperNode`
#[derive(Debug, Clone, Copy, PartialEq)]
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
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
        use OverSampleType::{None, X2, X4};

        match i {
            0 => None,
            1 => X2,
            2 => X4,
            _ => unreachable!(),
        }
    }
}

/// `WaveShaperOptions` is used to pass options
/// during the construction of `WaveShaperNode` using its
/// constructor method `new`
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct WaveShaperOptions {
    /// The distortion curve
    pub curve: Option<Vec<f32>>,
    /// Oversampling rate - default to `None`
    pub oversample: Option<OverSampleType>,
    /// audio node options
    pub channel_config: Option<ChannelConfigOptions>,
}

impl Default for WaveShaperOptions {
    fn default() -> Self {
        Self {
            curve: Default::default(),
            oversample: Some(OverSampleType::None),
            channel_config: Default::default(),
        }
    }
}

/// `WaveShaperNode` implemnets non-linear distortion effects
/// Arbitrary non-linear shaping curves may be specified.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct WaveShaperNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// distortion curve
    curve: Option<Vec<f32>>,
    /// ovesample type
    oversample: Arc<AtomicU32>,
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
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<WaveShaperOptions>) -> Self {
        context.base().register(move |registration| {
            let WaveShaperOptions {
                curve,
                oversample,
                channel_config,
            } = options.unwrap_or_default();
            // cannot guarantee that the cast will be without loss of precision for all fs
            // but for usual sample rate (44.1kHz, 48kHz, 96kHz) it is
            #[allow(clippy::cast_precision_loss)]
            let sample_rate = context.base().sample_rate().0 as usize;
            let channel_config = channel_config.unwrap_or_default().into();
            let oversample = Arc::new(AtomicU32::new(
                oversample.expect("oversample should be OversampleType variant") as u32,
            ));

            let config = RendererConfig {
                sample_rate,
                curve: curve.clone(),
                oversample: oversample.clone(),
            };

            let renderer = WaveShaperRenderer::new(config);
            let node = Self {
                registration,
                channel_config,
                curve,
                oversample,
            };

            (node, Box::new(renderer))
        })
    }

    /// Returns the distortion curve
    #[must_use]
    pub fn curve(&self) -> Option<&[f32]> {
        self.curve.as_deref()
    }

    /// set the distortion `curve` of this node
    ///
    /// # Arguments
    ///
    /// * `curve` - the desired distortion `curve`
    pub fn set_curve(&mut self, curve: Vec<f32>) {
        self.curve = Some(curve);
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
    pub fn set_oversample(&mut self, oversample: OverSampleType) {
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
    /// distortion curve
    curve: Option<Vec<f32>>,
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
    // up sampler configured to multiply by 2 the input fs
    upsampler_x2: FftFixedInOut<f32>,
    // up sampler configured to multiply by 4 the input fs
    upsampler_x4: FftFixedInOut<f32>,
    // down sampler configured to divide by 4 the input fs
    downsampler_x2: FftFixedInOut<f32>,
    // down sampler configured to divide by 4 the input fs
    downsampler_x4: FftFixedInOut<f32>,
    /// distortion curve
    curve: Vec<f32>,
    /// set to true if curve is not None
    curve_set: bool,
}

impl AudioProcessor for WaveShaperRenderer {
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

        use OverSampleType::*;

        match self.oversample.load(Ordering::SeqCst).into() {
            None => self.process_none(input, output),
            X2 => {
                if input.channels().len() != self.channels_x2 {
                    self.update_2x(input.channels().len());
                }
                self.process_2x(input, output)
            }
            X4 => {
                if input.channels().len() != self.channels_x4 {
                    self.update_4x(input.channels().len());
                }
                self.process_4x(input, output)
            }
        }
    }

    fn tail_time(&self) -> bool {
        true
    }
}

impl WaveShaperRenderer {
    /// returns an `WaveShaperRenderer` instance
    // new cannot be qualified as const, since constant functions cannot evaluate destructors
    // and config param need this evaluation
    #[allow(clippy::missing_const_for_fn)]
    fn new(config: RendererConfig) -> Self {
        let RendererConfig {
            sample_rate,
            oversample,
            curve,
        } = config;

        let (curve, curve_set) = match curve {
            Some(c) => (c, true),
            None => (Vec::new(), false),
        };

        let channels_x2 = 1;
        let channels_x4 = 1;

        let upsampler_x2 = FftFixedInOut::<f32>::new(
            sample_rate as usize,
            sample_rate as usize * 2,
            256,
            channels_x2,
        );

        let downsampler_x2 = FftFixedInOut::<f32>::new(
            sample_rate as usize,
            sample_rate as usize / 2,
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
            sample_rate as usize,
            sample_rate as usize / 4,
            128,
            channels_x4,
        );

        Self {
            sample_rate,
            oversample,
            curve,
            curve_set,
            upsampler_x2,
            upsampler_x4,
            downsampler_x2,
            downsampler_x4,
            channels_x2,
            channels_x4,
        }
    }

    fn process_none(&self, input: &AudioBuffer, output: &mut AudioBuffer) {
        for (i_data, o_data) in input.channels().iter().zip(output.channels_mut()) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i);
            }
        }
    }

    fn process_2x(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let wave_in = input.channels();

        let up_wave_in = self.upsampler_x2.process(wave_in).unwrap();
        let mut up_wave_out = up_wave_in.clone();

        for (i_data, o_data) in up_wave_in.iter().zip(&mut up_wave_out) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i);
            }
        }

        let wave_out = self.downsampler_x2.process(&up_wave_out).unwrap();

        for (i_data, o_data) in wave_out.iter().zip(output.channels_mut()) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = i;
            }
        }
    }

    fn process_4x(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let wave_in = input.channels();

        let up_wave_in = self.upsampler_x4.process(wave_in).unwrap();
        let mut up_wave_out = up_wave_in.clone();

        for (i_data, o_data) in up_wave_in.iter().zip(&mut up_wave_out) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i);
            }
        }

        let wave_out = self.downsampler_x4.process(&up_wave_out).unwrap();

        for (i_data, o_data) in wave_out.iter().zip(output.channels_mut()) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = i;
            }
        }
    }

    fn tick(&self, input: f32) -> f32 {
        if !self.curve_set {
            input
        } else {
            let n = self.curve.len() as f32;
            let v = (n - 1.) / 2.0 * (input + 1.);
            let k = v.floor();
            let f = v - k;

            match v {
                v if v <= 0. => self.curve[0],
                v if v > n - 1. => self.curve[(n - 1.) as usize],
                _ => (1. - f) * self.curve[k as usize] + f * self.curve[(k + 1.) as usize],
            }
        }
    }

    fn update_2x(&mut self, channels_x2: usize) {
        self.channels_x2 = channels_x2;

        self.upsampler_x2 =
            FftFixedInOut::<f32>::new(self.sample_rate, self.sample_rate * 2, 256, channels_x2);

        self.downsampler_x2 =
            FftFixedInOut::<f32>::new(self.sample_rate, self.sample_rate / 2, 128, channels_x2);
    }

    fn update_4x(&mut self, channels_x4: usize) {
        self.channels_x4 = channels_x4;

        self.upsampler_x4 =
            FftFixedInOut::<f32>::new(self.sample_rate, self.sample_rate * 4, 512, channels_x4);

        self.downsampler_x4 =
            FftFixedInOut::<f32>::new(self.sample_rate, self.sample_rate / 4, 128, channels_x4);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        context::{AsBaseAudioContext, OfflineAudioContext},
        node::WaveShaperOptions,
        SampleRate,
    };

    use super::{OverSampleType, WaveShaperNode};

    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _shaper = WaveShaperNode::new(&context, None);
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _shaper = context.create_wave_shaper();
    }

    #[test]
    fn default_audio_params_are_correct_with_no_options() {
        let default_oversample = OverSampleType::None;
        let default_curve = None;
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let shaper = WaveShaperNode::new(&context, None);

        assert_eq!(shaper.curve(), default_curve);
        assert_eq!(shaper.oversample(), default_oversample);
    }

    #[test]
    fn default_audio_params_are_correct_with_default_options() {
        let default_oversample = OverSampleType::None;
        let default_curve = None;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = WaveShaperOptions::default();
        let shaper = WaveShaperNode::new(&context, Some(options));

        assert_eq!(shaper.curve(), default_curve);
        assert_eq!(shaper.oversample(), default_oversample);
    }

    #[test]
    fn options_sets_audio_params() {
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = WaveShaperOptions {
            curve: Some(vec![1.0]),
            oversample: Some(OverSampleType::X2),
            ..Default::default()
        };

        let shaper = WaveShaperNode::new(&context, Some(options));

        context.start_rendering();

        assert_eq!(shaper.curve(), Some(&[1.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X2);
    }

    #[test]
    fn change_audio_params_after_build() {
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let options = WaveShaperOptions {
            curve: Some(vec![1.0]),
            oversample: Some(OverSampleType::X2),
            ..Default::default()
        };

        let mut shaper = WaveShaperNode::new(&context, Some(options));
        assert_eq!(shaper.curve(), Some(&[1.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X2);

        shaper.set_curve(vec![2.0]);
        shaper.set_oversample(OverSampleType::X4);

        context.start_rendering();

        assert_eq!(shaper.curve(), Some(&[2.0][..]));
        assert_eq!(shaper.oversample(), OverSampleType::X4);
    }
}
