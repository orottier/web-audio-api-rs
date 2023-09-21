use std::f32::consts::PI;
use std::fs::File;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OverSampleType};

// WaveshaperNode example
//
// `cargo run --release --example waveshaper`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example waveshaper`

// use part of cosine, between [π, 2π] as shaping cureve
fn make_distortion_curve(size: usize) -> Vec<f32> {
    let mut curve = vec![0.; size];
    let mut phase = 0.;
    let phase_incr = PI / (size - 1) as f32;

    for s in curve.iter_mut() {
        *s = (PI + phase).cos();
        phase += phase_incr;
    }

    curve
}

fn main() {
    env_logger::init();

    println!("> gradually increase the amount of distortion applied on the sample");

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let file = File::open("samples/sample.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();
    let curve = make_distortion_curve(2048);

    let post_gain = context.create_gain();
    post_gain.connect(&context.destination());
    post_gain.gain().set_value(0.);

    let mut shaper = context.create_wave_shaper();
    shaper.set_oversample(OverSampleType::None);
    // shaper.set_oversample(OverSampleType::X2);
    // shaper.set_oversample(OverSampleType::X4);
    shaper.connect(&post_gain);
    shaper.set_curve(curve);

    let pre_gain = context.create_gain();
    pre_gain.connect(&shaper);
    pre_gain.gain().set_value(0.);

    for i in 1..10 {
        let gain = i as f32 * 2.;
        println!("+ pre gain: {gain:?}");

        pre_gain.gain().set_value(gain);
        post_gain.gain().set_value(1. / gain);

        let mut src = context.create_buffer_source();
        src.connect(&pre_gain);
        src.set_buffer(buffer.clone());
        src.start();

        // enjoy listening
        std::thread::sleep(std::time::Duration::from_secs(4));
    }
}
