use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::enumerate_devices;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    dbg!(enumerate_devices());

    // Create an audio context (default: stereo);
    let context = AudioContext::default();

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();
}
