use rand::Rng;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// ScriptProcessorNode example
//
// `cargo run --release --example script_processor`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example script_processor`
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

    let node = context.create_script_processor(512, 1, 1);
    node.set_onaudioprocess(|mut e| {
        let mut rng = rand::thread_rng();
        e.output_buffer
            .get_channel_data_mut(0)
            .iter_mut()
            .zip(e.input_buffer.get_channel_data(0))
            .for_each(|(o, i)| *o = *i + rng.gen_range(-0.3..0.3));
    });

    let mut src = context.create_oscillator();
    src.frequency().set_value(400.);
    src.start();
    src.connect(&node);
    node.connect(&context.destination());

    std::thread::sleep(std::time::Duration::from_millis(5000));
}
