//! The stereo panner control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use std::f32::consts::PI;

use float_eq::debug_assert_float_eq;

use crate::{
    context::{Context, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum},
    SampleRate,
};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
    SINETABLE, TABLE_LENGTH_BY_4_F32, TABLE_LENGTH_BY_4_USIZE,
};

/// `StereoPannerOptions` is used to pass options
/// during the construction of `StereoPannerNode` using its
/// constructor method `new`
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct StereoPannerOptions {
    /// initial value for the pan parameter
    pan: Option<f32>,
    /// audio node options
    pub channel_config: ChannelConfigOptions,
}

impl Default for StereoPannerOptions {
    fn default() -> Self {
        Self {
            pan: Some(0.),
            channel_config: ChannelConfigOptions {
                count: 2,
                mode: ChannelCountMode::ClampedMax,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
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
    pub fn new<C: Context+?Sized>(context: &C, options: Option<StereoPannerOptions>) -> Self {
        context.register(move |registration| {
            let options = options.unwrap_or_default();

            assert!(
                options.channel_config.count <= 2,
                "NotSupportedError: channel count"
            );
            assert!(
                options.channel_config.mode != ChannelCountMode::Max,
                "NotSupportedError: count mode"
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
    pub fn pan(&self) -> &AudioParam {
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
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];
        output.set_number_of_channels(2);

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

                for (sample_idx, &p) in pan_values.iter().enumerate() {
                    // A-rate params
                    let pan = p;
                    let (left, right) =
                        Self::stereo_tick((in_data[0][sample_idx], in_data[1][sample_idx]), pan);
                    out_data[0][sample_idx] = left;
                    out_data[1][sample_idx] = right;
                }
            }
            _ => panic!("StereoPannerNode should not have more than 2 channels to process"),
        }

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
    #[inline]
    fn mono_tick(input: f32, pan: f32) -> (f32, f32) {
        let x = (pan + 1.) * 0.5;
        let (g_l, g_r) = Self::stereo_gains(x);

        (input * g_l, input * g_r)
    }

    /// Generates the output samples for a stereo input
    #[inline]
    fn stereo_tick(inputs: (f32, f32), pan: f32) -> (f32, f32) {
        match pan {
            p if p <= 0. => {
                let x = p + 1.;
                let (g_l, g_r) = Self::stereo_gains(x);
                (inputs.1.mul_add(g_l, inputs.0), inputs.1 * g_r)
            }
            x => {
                let (g_l, g_r) = Self::stereo_gains(x);
                (inputs.0 * g_l, inputs.0.mul_add(g_r, inputs.1))
            }
        }
    }

    /// Generates the stereo gains for a specific x derived from pan
    #[inline]
    fn stereo_gains(x: f32) -> (f32, f32) {
        // truncation is the intented behavior
        #[allow(clippy::cast_possible_truncation)]
        // no sign loss: x is always positive
        #[allow(clippy::cast_sign_loss)]
        let idx = (x * TABLE_LENGTH_BY_4_F32) as usize;
        let gain_l = SINETABLE[idx + TABLE_LENGTH_BY_4_USIZE];
        let gain_r = SINETABLE[idx];

        // Assert correctness of wavetable optimization
        debug_assert_float_eq!(gain_l, (x * PI / 2.).cos(), abs <= 0.1, "gain_l panicked");
        debug_assert_float_eq!(gain_r, (x * PI / 2.).sin(), abs <= 0.1, "gain_r panicked");

        (gain_l, gain_r)
    }
}

#[cfg(test)]
mod test {
    use float_eq::assert_float_eq;

    use crate::{
        context::{Context, OfflineAudioContext},
        SampleRate,
    };

    use super::{StereoPannerNode, StereoPannerRenderer};
    const LENGTH: usize = 555;

    #[test]
    fn build_with_new() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _panner = StereoPannerNode::new(&context, None);
    }

    #[test]
    fn build_with_factory_func() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let _panner = context.create_stereo_panner();
    }

    #[test]
    fn assert_stereo_default_build() {
        let default_pan = 0.;

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let panner = StereoPannerNode::new(&context, None);

        context.start_rendering();

        let pan = panner.pan.value();
        assert_float_eq!(pan, default_pan, abs_all <= 0.);
    }

    #[test]
    fn setting_pan() {
        let default_pan = 0.;
        let new_pan = 0.1;

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let panner = StereoPannerNode::new(&context, None);

        let pan = panner.pan.value();
        assert_float_eq!(pan, default_pan, abs_all <= 0.);
        panner.pan().set_value(new_pan);

        context.start_rendering();

        let pan = panner.pan.value();
        assert_float_eq!(pan, new_pan, abs_all <= 0.);
    }

    #[test]
    fn panning_should_be_on_the_right() {
        let pan = 1.0;

        let (i_l, i_r) = StereoPannerRenderer::stereo_tick((1., 1.), pan);

        // i_l is not exactly 0. due to precision error in the wavetable used
        // to compute the panning gains
        // 0.00001 corresponds to a reduction of -100 dB, so even if the gain is not exactly 0.
        // it should not be audible
        assert_float_eq!(i_l, 0.0, abs <= 0.00001);
        assert_float_eq!(i_r, 2.0, abs <= 0.);
    }

    #[test]
    fn panning_should_be_on_the_left() {
        let pan = -1.0;

        let (i_l, i_r) = StereoPannerRenderer::stereo_tick((1., 1.), pan);

        assert_float_eq!(i_l, 2.0, abs <= 0.);
        assert_float_eq!(i_r, 0.0, abs <= 0.);
    }

    #[test]
    fn panning_should_be_in_the_middle() {
        let pan = 0.0;

        let (i_l, i_r) = StereoPannerRenderer::stereo_tick((1., 1.), pan);

        // i_l is not exactly 1. due to precision error in the wavetable used
        // to compute the panning gains
        // 0.1 corresponds to a difference of < 1 dB, so it should not be audible
        assert_float_eq!(i_l, 1.0, abs <= 0.1);
        assert_float_eq!(i_r, 1.0, abs <= 0.);
    }

    #[test]
    #[should_panic]
    fn setting_pan_more_than_1_should_fail() {
        let default_pan = 0.;
        let new_pan = 1.1;

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let panner = StereoPannerNode::new(&context, None);

        let pan = panner.pan.value();
        assert_float_eq!(pan, default_pan, abs_all <= 0.);
        panner.pan().set_value(new_pan);

        context.start_rendering();

        let pan = panner.pan.value();
        assert_float_eq!(pan, new_pan, abs_all <= 0.);
    }

    #[test]
    #[should_panic]
    fn setting_pan_less_than_minus1_should_fail() {
        let default_pan = 0.;
        let new_pan = -1.1;

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let panner = StereoPannerNode::new(&context, None);

        let pan = panner.pan.value();
        assert_float_eq!(pan, default_pan, abs_all <= 0.);
        panner.pan().set_value(new_pan);

        context.start_rendering();

        let pan = panner.pan.value();
        assert_float_eq!(pan, new_pan, abs_all <= 0.);
    }
}
