use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Change audio context channel count
//
// `cargo run --release --example mono`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example mono`
fn main() {
    env_logger::init();

    // Create an audio context (default: stereo);
    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // Create an oscillator node with sine (default) type
    let mut osc = context.create_oscillator();

    // Connect osc to the destination node which is the default output device
    osc.connect(&context.destination());

    // Oscillator needs to be started explicitly
    osc.start();

    // Play for 2 seconds
    println!("stereo");
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Switch to mono
    context.destination().set_channel_count(1);

    // Play for 2 seconds
    println!("mono");
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Switch to stereo
    context.destination().set_channel_count(2);

    // Play for 2 seconds
    println!("stereo");
    std::thread::sleep(std::time::Duration::from_secs(2));
}
