use std::{env, thread, time};
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Amplitude Modulation synthesis
//
// `cargo run --release --example amplitude_modulation`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `playback` option which will increase the buffer size to 1024
//
// `cargo run --release --example amplitude_modulation -- playback`
fn main() {
    env_logger::init();

    let mut playback_option = false;
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && args[1] == "playback" {
        playback_option = true;
    }

    let context = if playback_option {
        println!("> Use playback latency hint");
        AudioContext::new(AudioContextOptions {
            latency_hint: AudioContextLatencyCategory::Playback,
            ..AudioContextOptions::default()
        })
    } else {
        AudioContext::default()
    };

    let modulated = context.create_gain();
    modulated.gain().set_value(0.5);
    modulated.connect(&context.destination());

    let carrier = context.create_oscillator();
    carrier.connect(&modulated);
    carrier.frequency().set_value(300.);

    // mod branch
    let depth = context.create_gain();
    depth.gain().set_value(0.5);
    depth.connect(modulated.gain());

    let modulator = context.create_oscillator();
    modulator.connect(&depth);
    modulator.frequency().set_value(1.);

    carrier.start();
    modulator.start();

    let mut flag = 1.;

    loop {
        let freq = flag * 300.;
        let when = context.current_time() + 10.;
        modulator
            .frequency()
            .linear_ramp_to_value_at_time(freq, when);

        flag = 1. - flag;

        thread::sleep(time::Duration::from_secs(10));
    }
}
