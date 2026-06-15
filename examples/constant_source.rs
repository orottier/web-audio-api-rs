use std::{thread, time};
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// ConstantSourceNode example
//
// `cargo run --release --example constant_source`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example constant_source`
fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // use merger to pipe oscillators to right and left channels
    let merger = context.create_channel_merger(2);
    merger.connect(&context.destination());

    // left branch
    let gain_left = context.create_gain();
    gain_left.gain().set_value(0.);
    gain_left.connect_from_output_to_input(&merger, 0, 0);

    let mut src_left = context.create_oscillator();
    src_left.frequency().set_value(200.);
    src_left.connect(&gain_left);
    src_left.start();

    // right branch
    let gain_right = context.create_gain();
    gain_right.gain().set_value(0.);
    gain_right.connect_from_output_to_input(&merger, 0, 1);

    let mut src_right = context.create_oscillator();
    src_right.frequency().set_value(300.);
    src_right.connect(&gain_right);
    src_right.start();

    // control both left and right gains with constant source
    let mut constant_source = context.create_constant_source();
    constant_source.offset().set_value(0.);
    constant_source.connect(gain_left.gain());
    constant_source.connect(gain_right.gain());
    constant_source.start();

    let mut target = 0.;

    loop {
        let now = context.current_time();
        constant_source
            .offset()
            .set_target_at_time(target, now, 0.1);

        target = if target == 0. { 1. } else { 0. };

        thread::sleep(time::Duration::from_millis(1000));
    }
}
