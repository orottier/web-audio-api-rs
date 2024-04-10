use std::f32::consts::PI;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// AudioBuffer example
//
// `cargo run --release --example audio_buffer`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example audio_buffer`
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

    // create a 1 second buffer filled with a sine at 200Hz
    println!("> Play sine at 200Hz created manually in an AudioBuffer");

    let length = 4 * context.sample_rate() as usize;
    let sample_rate = context.sample_rate();
    let mut buffer = context.create_buffer(1, length, sample_rate);
    let mut sine = vec![];

    for i in 0..length {
        let phase = i as f32 / length as f32 * 2. * PI * 200.;
        sine.push(phase.sin());
    }

    buffer.copy_to_channel(&sine, 0);

    // play the buffer in a loop
    let mut src = context.create_buffer_source();
    src.set_buffer(buffer);

    src.set_loop(true);
    src.set_loop_start(1.1);
    src.set_loop_end(1.15);
    src.start();

    src.connect(&context.destination());
    drop(src);

    std::thread::sleep(std::time::Duration::from_secs(35));
}
