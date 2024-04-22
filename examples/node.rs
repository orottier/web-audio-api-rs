use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, JsWorkletNode};
use web_audio_api::worklet::AudioWorkletNodeOptions;
use web_audio_api::AudioRenderCapacityOptions;

fn main() {
    let context = AudioContext::default();
    let node = JsWorkletNode::new(&context, "test.js", AudioWorkletNodeOptions::default());
    node.connect(&context.destination());

    let cap = context.render_capacity();
    cap.set_onupdate(|e| println!("{e:?}"));
    cap.start(AudioRenderCapacityOptions {
        update_interval: 1.,
    });

    std::thread::sleep(std::time::Duration::from_secs(6));
}
