use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::AudioContext;
use web_audio_api::node::AudioScheduledSourceNode;
use web_audio_api::node::{AudioNode, OscillatorNode, OscillatorOptions, OscillatorType};

fn main() {
    let context = AudioContext::new();

    let merge = context.create_channel_merger(2);
    merge.connect(&context.destination());

    let gain = context.create_gain();
    gain.gain().set_value(0.1);
    gain.gain().set_value_at_time(0.9, 2.);
    gain.connect_at(&merge, 0, 0).unwrap();

    let split = context.create_channel_splitter(2);
    split.connect(&gain);
    split.connect_at(&merge, 0, 1).unwrap();

    let opts = OscillatorOptions {
        type_: OscillatorType::Sine,
        ..Default::default()
    };
    let osc = OscillatorNode::new(&context, opts);
    osc.connect(&split);
    osc.start();

    std::thread::sleep(std::time::Duration::from_secs(4));
}
