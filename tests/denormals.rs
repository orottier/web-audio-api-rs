use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

#[test]
fn test_flush_denormals() {
    let mut context = OfflineAudioContext::new(1, 128, 48000.);

    let mut signal = context.create_constant_source();
    signal.start();

    let gain1 = context.create_gain();
    gain1.gain().set_value(0.001);
    signal.connect(&gain1);

    let gain2 = context.create_gain();
    gain2.gain().set_value(f32::MIN_POSITIVE);
    gain1.connect(&gain2);

    let gain3 = context.create_gain();
    gain3.gain().set_value(f32::MAX);
    gain2.connect(&gain3);

    gain3.connect(&context.destination());

    let output = context.start_rendering_sync();

    // When denormals are flushed, we expect the output to be exactly 0.0
    // If not, the output will be ~0.004
    assert_eq!(output.get_channel_data(0), &[0.; 128][..]);
}
