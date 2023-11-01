use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Retrieve the output latency of the audio context
//
// `cargo run --release --example latency_attributes`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example latency_attributes`
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

    let mut sine = context.create_oscillator();
    sine.frequency().set_value(200.);
    sine.connect(&context.destination());

    sine.start();

    println!("- BaseLatency: {:?}", context.base_latency());

    loop {
        println!("-------------------------------------------------");
        println!("+ currentTime {:?}", context.current_time());
        println!("+ OutputLatency: {:?}", context.output_latency());

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
