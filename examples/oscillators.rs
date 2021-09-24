//! This example plays each oscillator type sequentially
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};

fn main() {
    // Create an audio context where all audio nodes lives
    let context = AudioContext::new();

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();

    // Connect osc to the destination node which is the default output device
    osc.connect(&context.destination());

    // Oscillator needs to be started explicitily
    osc.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(1));
    osc.set_type(OscillatorType::Square);
    std::thread::sleep(std::time::Duration::from_secs(1));
    osc.set_type(OscillatorType::Triangle);
    std::thread::sleep(std::time::Duration::from_secs(1));
    osc.set_type(OscillatorType::Sawtooth);
    std::thread::sleep(std::time::Duration::from_secs(1));
    osc.set_type(OscillatorType::Sine);
    std::thread::sleep(std::time::Duration::from_secs(1));
}
