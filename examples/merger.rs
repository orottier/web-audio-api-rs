use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    println!("Sample rate: {:?}", context.sample_rate());
    println!(
        "Available channels: {}",
        context.destination().max_channels_count()
    );

    println!("Force output to two channels");
    context.destination().set_channel_count(2);

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

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
