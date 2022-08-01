//! The stereo panner control and renderer parts
// #![warn(
//     clippy::all,
//     clippy::pedantic,
//     clippy::perf,
//     clippy::missing_docs_in_private_items
// )]

use float_eq::debug_assert_float_eq;
use std::f32::consts::PI;

use crate::RENDER_QUANTUM_SIZE;
use crate::{
    context::{AudioContextRegistration, AudioParamId, BaseAudioContext},
    param::{AudioParam, AudioParamDescriptor},
    render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope},
};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
    SINETABLE, TABLE_LENGTH_BY_4_F32, TABLE_LENGTH_BY_4_USIZE,
};

/// Options for constructing a [`StereoPannerOptions`]
// dictionary StereoPannerOptions : AudioNodeOptions {
//   float pan = 0;
// };
#[derive(Clone, Debug)]
pub struct StereoPannerOptions {
    /// initial value for the pan parameter
    pub pan: f32,
    /// audio node options
    pub channel_config: ChannelConfigOptions,
}

impl Default for StereoPannerOptions {
    fn default() -> Self {
        Self {
            pan: 0.,
            channel_config: ChannelConfigOptions {
                count: 2,
                mode: ChannelCountMode::ClampedMax,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// `StereoPannerNode` positions an incoming audio stream in a stereo image
pub struct StereoPannerNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// The position of the input in the output’s stereo image. -1 represents
    /// full left, +1 represents full right.
    pan: AudioParam,
}

impl AudioNode for StereoPannerNode {
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

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::ClampedMax
    }

    fn set_channel_count_mode(&self, v: ChannelCountMode) {
        if v == ChannelCountMode::Max {
            panic!("NotSupportedError: StereoPannerNode channel count mode cannot be set to max");
        }
        self.channel_config.set_count_mode(v);
    }

    fn set_channel_count(&self, v: usize) {
        if v > 2 {
            panic!("NotSupportedError: StereoPannerNode channel count cannot be greater than two");
        }
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
    pub fn new<C: BaseAudioContext>(context: &C, options: StereoPannerOptions) -> Self {
        context.register(move |registration| {
            assert!(
                options.channel_config.count <= 2,
                "NotSupportedError: channel count"
            );
            assert!(
                options.channel_config.mode != ChannelCountMode::Max,
                "NotSupportedError: count mode"
            );

            let pan_options = AudioParamDescriptor {
                min_value: -1.,
                max_value: 1.,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (pan_param, pan_proc) = context.create_audio_param(pan_options, &registration);

            pan_param.set_value(options.pan);

            let renderer = StereoPannerRenderer {
                pan: pan_proc,
                buffer: [[0.; 2]; RENDER_QUANTUM_SIZE],
            };

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
    buffer: [[f32; 2]; 128],
}

impl AudioProcessor for StereoPannerRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        if input.is_silent() {
            output.make_silent();
            return false;
        }

        output.set_number_of_channels(2);

        // a-rate param
        let pan_values = params.get(&self.pan);
        let mut buffer = self.buffer;

        match input.number_of_channels() {
            0 => (),
            1 => {
                buffer.iter_mut()
                    .zip(pan_values.iter())
                    .zip(input.channel_data(0).iter())
                    .for_each(|((o, pan), input)| {
                        *o = Self::mono_tick(*input, *pan);
                    });
            }
            2 => {
                buffer.iter_mut()
                    .zip(pan_values.iter())
                    .zip(input.channel_data(0).iter())
                    .zip(input.channel_data(1).iter())
                    .for_each(|(((o, pan), input_left), input_right)| {
                        *o = Self::stereo_tick(*input_left, *input_right, *pan);
                    });
            }
            _ => panic!("StereoPannerNode should not have more than 2 channels to process"),
        }

        output.channels_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(channel_number, channel)| {
                channel.iter_mut()
                    .zip(buffer.iter())
                    .for_each(|(o, i)| *o = i[channel_number]);
            });

        false
    }
}

impl StereoPannerRenderer {
    /// Generates the output samples for a mono input
    #[inline]
    fn mono_tick(input: f32, pan: f32) -> [f32; 2] {
        let x = (pan + 1.) * 0.5;
        let [gain_left, gain_right] = Self::stereo_gains(x);

        [input * gain_left, input * gain_right]
    }

    /// Generates the output samples for a stereo input
    #[inline]
    fn stereo_tick(input_left: f32, input_right: f32, pan: f32) -> [f32; 2] {
        if pan <= 0. {
            let x = pan + 1.;
            let [gain_left, gain_right] = Self::stereo_gains(x);
            [input_right.mul_add(gain_left, input_left), input_right * gain_right]
        } else {
            let x = pan;
            let [gain_left, gain_right] = Self::stereo_gains(x);
            [input_left * gain_left, input_left.mul_add(gain_right, input_right)]
        }
    }

    /// Generates the stereo gains for a specific x derived from pan
    #[inline]
    fn stereo_gains(x: f32) -> [f32; 2] {
        // truncation is the intented behavior
        #[allow(clippy::cast_possible_truncation)]
        // no sign loss: x is always positive
        #[allow(clippy::cast_sign_loss)]
        let idx = (x * TABLE_LENGTH_BY_4_F32) as usize;
        let gain_l = SINETABLE[idx + TABLE_LENGTH_BY_4_USIZE];
        let gain_r = SINETABLE[idx];

        // Assert correctness of wavetable optimization
        debug_assert_float_eq!(gain_l, (x * PI / 2.).cos(), abs <= 1e-7, "gain_l panicked");
        debug_assert_float_eq!(gain_r, (x * PI / 2.).sin(), abs <= 1e-7, "gain_r panicked");

        [gain_l, gain_r]
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{BaseAudioContext, OfflineAudioContext};

    use super::{StereoPannerNode, StereoPannerOptions, StereoPannerRenderer};

    #[test]
    fn test_constructor() {
        {
            let context = OfflineAudioContext::new(2, 1, 44_100.);
            let _panner = StereoPannerNode::new(&context, StereoPannerOptions::default());
        }

        {
            let context = OfflineAudioContext::new(2, 1, 44_100.);
            let _panner = context.create_stereo_panner();
        }

        {
            let context = OfflineAudioContext::new(2, 1, 44_100.);
            let panner = StereoPannerNode::new(&context, StereoPannerOptions::default());

            let default_pan = 0.;
            let pan = panner.pan.value();
            assert_float_eq!(pan, default_pan, abs_all <= 0.);
        }
    }

    #[test]
    fn panning_should_be_on_the_right() {
        let pan = 1.0;

        let [i_l, i_r] = StereoPannerRenderer::stereo_tick(1., 1., pan);

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

        let [i_l, i_r] = StereoPannerRenderer::stereo_tick(1., 1., pan);

        assert_float_eq!(i_l, 2.0, abs <= 0.);
        assert_float_eq!(i_r, 0.0, abs <= 0.);
    }

    #[test]
    fn panning_should_be_in_the_middle() {
        let pan = 0.0;

        let [i_l, i_r] = StereoPannerRenderer::stereo_tick(1., 1., pan);

        // i_l is not exactly 1. due to precision error in the wavetable used
        // to compute the panning gains
        // 0.1 corresponds to a difference of < 1 dB, so it should not be audible
        assert_float_eq!(i_l, 1.0, abs <= 0.1);
        assert_float_eq!(i_r, 1.0, abs <= 0.);
    }
}
