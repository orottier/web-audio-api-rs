use std::fs::File;
use std::{thread, time};

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ConvolverNode, ConvolverOptions};
use web_audio_api::AudioRenderCapacityOptions;

// ConvolverNode example
//
// `cargo run --release --example convolution`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example convolution`
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

    let cap = context.render_capacity();
    cap.set_onupdate(|e| println!("{e:?}"));
    cap.start(AudioRenderCapacityOptions {
        update_interval: 1.,
    });

    let file = File::open("samples/vocals-dry.wav").unwrap();
    let audio_buffer = context.decode_audio_data_sync(file).unwrap();

    let impulse_file1 = File::open("samples/small-room-response.wav").unwrap();
    let impulse_buffer1 = context.decode_audio_data_sync(impulse_file1).unwrap();

    let impulse_file2 = File::open("samples/parking-garage-response.wav").unwrap();
    let impulse_buffer2 = context.decode_audio_data_sync(impulse_file2).unwrap();

    let mut src = context.create_buffer_source();
    src.set_buffer(audio_buffer);

    let mut convolver = ConvolverNode::new(&context, ConvolverOptions::default());

    src.connect(&convolver);
    convolver.connect(&context.destination());

    src.start();

    println!("Dry");
    thread::sleep(time::Duration::from_millis(4_000));

    println!("Small room");
    convolver.set_buffer(impulse_buffer1);
    thread::sleep(time::Duration::from_millis(4_000));

    println!("Parking garage");
    convolver.set_buffer(impulse_buffer2);
    thread::sleep(time::Duration::from_millis(5_000));

    println!("Stop input - flush out remaining impulse response");
    src.stop();
    thread::sleep(time::Duration::from_millis(2_000));
}
