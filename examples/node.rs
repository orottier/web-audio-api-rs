use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::AudioScheduledSourceNode;
use web_audio_api::node::{AudioNode, JsWorkletNode};
use web_audio_api::worklet::AudioWorkletNodeOptions;
use web_audio_api::AudioRenderCapacityOptions;

fn main() {
    let context = AudioContext::default();

    JsWorkletNode::add_module("crush.js");
    JsWorkletNode::add_module("noise.js");

    let mut src = context.create_oscillator();
    src.frequency().set_value(5000.);
    src.start();

    // let src = JsWorkletNode::new(&context, "white-noise", AudioWorkletNodeOptions::default());
    let node = JsWorkletNode::new(&context, "bitcrusher", AudioWorkletNodeOptions::default());

    let param_bit_depth = node.parameters().get("bitDepth").unwrap();
    let param_reduction = node.parameters().get("frequencyReduction").unwrap();
    param_bit_depth.set_value_at_time(1., 0.);
    param_reduction.set_value_at_time(0.01, 0.);
    param_reduction.linear_ramp_to_value_at_time(0.1, 4.);
    param_reduction.exponential_ramp_to_value_at_time(0.01, 8.);

    src.connect(&node);
    node.connect(&context.destination());

    let cap = context.render_capacity();
    cap.set_onupdate(|e| println!("{e:?}"));
    cap.start(AudioRenderCapacityOptions {
        update_interval: 1.,
    });

    std::thread::sleep(std::time::Duration::from_secs(8));
}
