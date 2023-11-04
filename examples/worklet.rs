use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, AudioWorkletNode};

// AudioWorkletNode example
//
// `cargo run --release --example worklet`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example worklet`
fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // create a pass-through processor
    let process = |input: &[&[f32]], output: &mut [&mut [f32]]| {
        input
            .iter()
            .zip(output)
            .for_each(|(i, o)| o.copy_from_slice(i));
        false
    };

    let worklet = AudioWorkletNode::new(&context, process);
    worklet.connect(&context.destination());

    let mut osc = context.create_oscillator();
    osc.frequency().set_value(300.);
    osc.connect(&worklet);
    osc.start();

    std::thread::sleep(std::time::Duration::from_millis(5000));
}
