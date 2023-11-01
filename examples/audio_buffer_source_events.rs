use std::fs::File;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// AudioBufferSource ended event example
//
// `cargo run --release --example audio_buffer_source_events`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example audio_buffer_source_events`
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

    let file = File::open("samples/sample.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    let mut src = context.create_buffer_source();
    src.connect(&context.destination());
    src.set_buffer(buffer);

    src.set_onended(|_| {
        println!("> Ended event triggered!");
    });

    let now = context.current_time();
    src.start_at(now);
    src.stop_at(now + 1.);

    std::thread::sleep(std::time::Duration::from_secs(4));
}
