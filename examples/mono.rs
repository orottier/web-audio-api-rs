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
    let context = match std::env::var("WEB_AUDIO_LATENCY") {
        Ok(val) => {
            if val == "playback" {
                AudioContext::new(AudioContextOptions {
                    latency_hint: AudioContextLatencyCategory::Playback,
                    ..AudioContextOptions::default()
                })
            } else {
                println!("Invalid WEB_AUDIO_LATENCY value, fall back to default");
                AudioContext::default()
            }
        }
        Err(_e) => AudioContext::default(),
    };

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();

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
