//! The biquad filter control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use std::f32::consts::PI;

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

            let renderer = StereoPannerRenderer::new(pan_proc);
            let node = Self {
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

        match input.number_of_channels() {
            0 => (),
            1 => {
                let in_data = input.channels();
                let out_data = output.channels_mut();

                for (sample_idx, &input) in in_data[0].iter().enumerate() {
                    // A-rate params
                    let pan = pan_values[sample_idx];
                    let (left, right) = Self::mono_tick(input, pan);
                    out_data[0][sample_idx] = left;
                    out_data[1][sample_idx] = right;
                }
            }
            2 => {
                let in_data = input.channels();
                let out_data = output.channels_mut();

                for sample_idx in 0..in_data[0].len() {
                    // A-rate params
                    let pan = pan_values[sample_idx];
                    let (left, right) =
                        Self::stereo_tick((in_data[0][sample_idx], in_data[1][sample_idx]), pan);
                    out_data[0][sample_idx] = left;
                    out_data[1][sample_idx] = right;
                }
            }
            _ => panic!("StereoPannerNode should not have more than 2 channels to process"),
        }
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
    fn new(pan: AudioParamId) -> Self {
        Self { pan }
    }

    /// Generates the output samples for a mono input
    fn mono_tick(input: f32, pan: f32) -> (f32, f32) {
        let x = (pan + 1.) / 2.0;
        let (g_l, g_r) = Self::stereo_gains(x);

        (input * g_l, input * g_r)
    }

    /// Generates the output samples for a stereo input
    fn stereo_tick(inputs: (f32, f32), pan: f32) -> (f32, f32) {
        let x = if pan <= 0. { pan + 1. } else { pan };

        let (g_l, g_r) = Self::stereo_gains(x);

        (inputs.0 * g_l, inputs.1 * g_r)
    }

    /// Generates the stereo gains for a specific x derived from pan
    fn stereo_gains(x: f32) -> (f32, f32) {
        let gain_l = (x * PI / 2.0).cos();
        let gain_r = (x * PI / 2.0).sin();
        (gain_l, gain_r)
    }
}

#[cfg(test)]
mod test {
    use super::StereoPannerNode;
    const LENGTH: usize = 555;

    #[test]
    fn testing_testing() {}
}
