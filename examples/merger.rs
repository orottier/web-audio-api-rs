use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// ChannelMergerNode example
//
// `cargo run --release --example merger`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example merger`
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

    println!("Sample rate: {:?}", context.sample_rate());
    println!(
        "Available channels: {}",
        context.destination().max_channel_count()
    );

    println!("Force output to two channels");
    context.destination().set_channel_count(2);

    // Create an oscillator
    let mut left = context.create_oscillator();

    //Create an oscillator
    let mut right = context.create_oscillator();
    // set a different frequency to distinguish left from right osc
    right.frequency().set_value(1000.);

    // Create a merger
    let merger = context.create_channel_merger(2);

    // connect left osc to left input of the merger
    left.connect_from_output_to_input(&merger, 0, 0);
    // connect right osc to left input of the merger
    right.connect_from_output_to_input(&merger, 0, 1);

    // Connect the merger to speakers
    merger.connect(&context.destination());

    // Start the oscillators
    left.start();
    right.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
