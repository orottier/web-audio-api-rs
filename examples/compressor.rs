use std::fs::File;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// DynamicsCompressorNode example
//
// `cargo run --release --example compressor`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example compressor`
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

    let file = File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    println!("> no compression");
    let mut src = context.create_buffer_source();
    src.connect(&context.destination());
    src.set_buffer(buffer.clone());
    src.start();

    std::thread::sleep(std::time::Duration::from_secs(3));

    println!("> compression (hard knee)");
    println!("+ attack: {:?}ms", 30);
    println!("+ release: {:?}ms", 100);
    println!("+ ratio: {:?}", 12);
    println!(">");

    for i in 0..6 {
        println!("+ threshold at {:?}", -10. * i as f32);

        let compressor = context.create_dynamics_compressor();
        compressor.connect(&context.destination());
        compressor.threshold().set_value(-10. * i as f32);
        compressor.knee().set_value(0.); // hard knee
        compressor.attack().set_value(0.03);
        compressor.release().set_value(0.1);

        let mut src = context.create_buffer_source();
        src.connect(&compressor);
        src.set_buffer(buffer.clone());
        src.start();

        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
