use web_audio_api::context::{Context, AudioContext, AudioContextOptions, LatencyHint};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let options = AudioContextOptions {
        sample_rate: Some(48_000),
        channels: Some(2),
        latency_hint: Some(LatencyHint::Playback),
    };

    let context = AudioContext::new(Some(options));

    println!("Sample rate: {:?}", context.sample_rate());
    println!("Channels: {}", context.destination().max_channels_count());

    // Create an oscillator
    let left = context.create_oscillator();

    //Create an oscillator
    let right = context.create_oscillator();
    // set a different frequency to distinguish left from right osc
    right.frequency().set_value(1000.);

    // Create a merger
    let merger = context.create_channel_merger(2);

    // connect left osc to left input of the merger
    left.connect_at(&merger, 0, 0);
    // connect right osc to left input of the merger
    right.connect_at(&merger, 0, 1);

    // Connect the merger to speakers
    merger.connect(&context.destination());

    // Start the oscillators
    left.start();
    right.start();

    // connect left osc to splitter

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
