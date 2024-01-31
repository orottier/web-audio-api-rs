use float_eq::assert_float_eq;

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
use web_audio_api::node::{
    ChannelCountMode::{self, *},
    ChannelInterpretation::{self, *},
};
use web_audio_api::AudioBuffer;

fn setup_with_destination_channel_config(
    number_of_channels: usize,
    channel_interpretation: ChannelInterpretation,
) -> OfflineAudioContext {
    let context = OfflineAudioContext::new(number_of_channels, 128, 44_100.);
    let dest = context.destination();
    dest.set_channel_interpretation(channel_interpretation);
    context
}

fn run_with_intermediate_channel_config(
    mut context: OfflineAudioContext,
    number_of_channels: usize,
    channel_count_mode: ChannelCountMode,
    channel_interpretation: ChannelInterpretation,
) -> AudioBuffer {
    {
        // input signal
        let mut constant = context.create_constant_source();
        constant.start();

        // gain node is only added for mixing
        let gain = context.create_gain();
        gain.set_channel_count(number_of_channels);
        gain.set_channel_count_mode(channel_count_mode);
        gain.set_channel_interpretation(channel_interpretation);

        constant.connect(&gain);
        gain.connect(&context.destination());
    }

    context.start_rendering_sync()
}

const ZEROES: &[f32] = &[0.; 128];
const ONES: &[f32] = &[1.; 128];

#[test]
fn test_mono_speakers() {
    let context = setup_with_destination_channel_config(1, Speakers);
    let output = run_with_intermediate_channel_config(context, 1, Max, Speakers);

    assert_eq!(output.number_of_channels(), 1);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
}

#[test]
fn test_stereo_speakers() {
    let context = setup_with_destination_channel_config(2, Speakers);
    let output = run_with_intermediate_channel_config(context, 2, Max, Speakers);

    assert_eq!(output.number_of_channels(), 2);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(1), ONES, abs_all <= 0.);
}

#[test]
fn test_quad_speakers() {
    let context = setup_with_destination_channel_config(4, Speakers);
    let output = run_with_intermediate_channel_config(context, 4, Max, Speakers);

    assert_eq!(output.number_of_channels(), 4);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(1), ONES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(2), ZEROES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(3), ZEROES, abs_all <= 0.);
}

#[test]
fn test_mono_to_discrete_stereo() {
    let context = setup_with_destination_channel_config(2, Discrete);
    let output = run_with_intermediate_channel_config(context, 1, Max, Speakers);

    assert_eq!(output.number_of_channels(), 2);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(1), ZEROES, abs_all <= 0.);
}

#[test]
fn test_stereo_to_discrete_stereo() {
    let context = setup_with_destination_channel_config(2, Discrete);
    let output = run_with_intermediate_channel_config(context, 2, Max, Speakers);

    assert_eq!(output.number_of_channels(), 2);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
    assert_float_eq!(output.get_channel_data(1), ZEROES, abs_all <= 0.);
}

#[test]
fn test_stereo_to_discrete_mono() {
    let context = setup_with_destination_channel_config(1, Discrete);
    let output = run_with_intermediate_channel_config(context, 2, Max, Speakers);

    assert_eq!(output.number_of_channels(), 1);
    assert_float_eq!(output.get_channel_data(0), ONES, abs_all <= 0.);
}
