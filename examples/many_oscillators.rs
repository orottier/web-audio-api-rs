use std::{thread, time};
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Trigger many oscillators
//
// `cargo run --release --example many_oscillators`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example many_oscillators`
fn trigger_sine(context: &AudioContext) {
    let mut osc = context.create_oscillator();
    osc.connect(&context.destination());

    let now = context.current_time();
    osc.start_at(now);
    osc.stop_at(now + 0.03)
}

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

    // mimic setInterval
    loop {
        trigger_sine(&context);
        thread::sleep(time::Duration::from_millis(50));
    }
}
