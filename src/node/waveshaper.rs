// //! The stereo panner control and renderer parts
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

use crate::{
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
    /// The distorsion curve
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

/// `BiquadFilterNode` is a second order IIR filter
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct WaveShaperNode {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// Distorsion curve
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
            let sample_rate = context.base().sample_rate().0 as f32;
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
                sample_rate,
                registration,
                channel_config,
                curve,
                oversample,
            };

            (node, Box::new(renderer))
        })
    }

    /// Returns the distorsion curve
    #[must_use]
    pub fn curve(&self) -> Option<&[f32]> {
        self.curve.as_deref()
    }

    /// set the distorsion `curve` of this node
    ///
    /// # Arguments
    ///
    /// * `curve` - the desired distorsion `curve`
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
    sample_rate: f32,
    /// oversample factor
    oversample: Arc<AtomicU32>,
    /// distorsion curve
    curve: Option<Vec<f32>>,
}

/// `BiquadFilterRenderer` represents the rendering part of `BiquadFilterNode`
struct WaveShaperRenderer {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// oversample factor
    oversample: Arc<AtomicU32>,
    /// distorsion curve
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

        for (i_data, o_data) in input.channels().iter().zip(output.channels_mut()) {
            for (&i, o) in i_data.iter().zip(o_data.iter_mut()) {
                *o = self.tick(i);
            }
        }
    }

    fn tail_time(&self) -> bool {
        true
    }
}

impl WaveShaperRenderer {
    /// returns an `BiquadFilterRenderer` instance
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

        Self {
            sample_rate,
            oversample,
            curve,
            curve_set,
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
}

#[cfg(test)]
mod test {

    const LENGTH: usize = 555;

    #[test]
    fn testing_testing() {}
}
