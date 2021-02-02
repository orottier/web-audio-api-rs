use web_audio_api::context::AudioContext;
use web_audio_api::node::{AudioNode, OscillatorNode, OscillatorOptions, OscillatorType};

fn main() {
    let context = AudioContext::new();

    let gain1 = context.create_gain();
    gain1.set_gain(0.5);
    gain1.connect(&context.destination());

    let mut opts = OscillatorOptions::default();
    opts.type_ = OscillatorType::Sawtooth;
    let osc = OscillatorNode::new(&context, opts);
    osc.connect(&gain1);
    //osc.start();

    /*
    let gain2 = context.create_gain();
    gain2.set_gain(0.5);
    gain2.connect(&context.destination());

    let osc2 = context.create_oscillator();
    osc2.set_frequency(445);
    osc2.connect(&gain2);
    //osc2.start();
    */

    std::thread::sleep(std::time::Duration::from_secs(2));

    osc.set_type(OscillatorType::Sine);
    //gain1.set_gain(0.2);

    std::thread::sleep(std::time::Duration::from_secs(1));

    //osc.set_frequency(1024);

    std::thread::sleep(std::time::Duration::from_secs(2));
}
