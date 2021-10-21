//! The biquad filter control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use crate::{
    buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate,
};

use super::AudioNode;

/// `StereoPannerOptions` is used to pass options
/// during the construction of `StereoPannerNode` using its
/// constructor method `new`
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
#[derive(Default)]
pub struct StereoPannerOptions {
    /// initial value for the pan parameter
    pan: Option<f32>,
    /// audio node options
    pub channel_config: ChannelConfigOptions,
}

/// `StereoPannerNode` positions an incoming audio stream in a stereo image
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct StereoPannerNode {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// The position of the input in the output’s stereo image. -1 represents full left, +1 represents full right.
    pan: AudioParam,
}

impl AudioNode for StereoPannerNode {
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

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::ClampedMax
    }

    fn set_channel_count_mode(&self, v: ChannelCountMode) {
        assert!(v != ChannelCountMode::Max, "NotSupportedError");
        self.channel_config.set_count_mode(v);
    }

    fn set_channel_count(&self, v: usize) {
        assert!(v <= 2, "NotSupportedError");
        self.channel_config.set_count(v);
    }
}

impl StereoPannerNode {
    /// returns a `StereoPannerNode` instance
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * `options.channel_config.count` is more than 2
    /// * `options.channel_config.mode` is `ChannelCountMode::Max`
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - stereo panner options
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<StereoPannerOptions>) -> Self {
        context.base().register(move |registration| {
            let options = options.unwrap_or_default();

            assert!(options.channel_config.count <= 2, "NotSupportedError");
            assert!(
                options.channel_config.mode != ChannelCountMode::Max,
                "NotSupportedError"
            );

            // cannot guarantee that the cast will be without loss of precision for all fs
            // but for usual sample rate (44.1kHz, 48kHz, 96kHz) it is
            #[allow(clippy::cast_precision_loss)]
            let sample_rate = context.base().sample_rate().0 as f32;

            let default_pan = 0.;

            let pan_value = options.pan.unwrap_or(default_pan);

            let pan_param_opts = AudioParamOptions {
                min_value: -1.,
                max_value: 1.,
                default_value: default_pan,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (pan_param, pan_proc) = context
                .base()
                .create_audio_param(pan_param_opts, registration.id());

            pan_param.set_value(pan_value);

            let renderer = StereoPannerRenderer::new(sample_rate, pan_proc);
            let node = Self {
                sample_rate,
                registration,
                channel_config: options.channel_config.into(),
                pan: pan_param,
            };

            (node, Box::new(renderer))
        })
    }

    /// Returns the pan audio paramter
    #[must_use]
    pub const fn pan(&self) -> &AudioParam {
        &self.pan
    }
}

/// `StereoPannerRenderer` represents the rendering part of `StereoPannerNode`
struct StereoPannerRenderer {
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
    /// The position of the input in the output’s stereo image.
    /// -1 represents full left, +1 represents full right.
    pan: AudioParamId,
}

impl AudioProcessor for StereoPannerRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        let pan_values = params.get(&self.pan);
    }

    fn tail_time(&self) -> bool {
        false
    }
}

impl StereoPannerRenderer {
    /// returns an `StereoPannerRenderer` instance
    // new cannot be qualified as const, since constant functions cannot evaluate destructors
    // and config param need this evaluation
    #[allow(clippy::missing_const_for_fn)]
    fn new(sample_rate: f32, pan: AudioParamId) -> Self {
        Self { sample_rate, pan }
    }
}

#[cfg(test)]
mod test {
    use super::StereoPannerNode;
    const LENGTH: usize = 555;

    #[test]
    fn testing_testing() {}
}
