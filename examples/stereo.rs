use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();
    let context = AudioContext::new(None);

    // Create an oscillator
    let osc = context.create_oscillator();

    // Create a stereo panner
    let panner = context.create_stereo_panner();

    panner.pan().set_value(-0.5);

    // Connect the oscillator to the panner
    osc.connect(&panner);

    // Connect the panner to speakers
    panner.connect(&context.destination());

    // Start the oscillator
    osc.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
