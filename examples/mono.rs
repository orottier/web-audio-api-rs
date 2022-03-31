use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    // Create an audio context (default: stereo);
    let context = AudioContext::default();

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();

    // Connect osc to the destination node which is the default output device
    osc.connect(&context.destination());

    // Oscillator needs to be started explicitily
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
