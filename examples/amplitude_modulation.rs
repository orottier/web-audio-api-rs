use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Amplitude Modulation synthesis example
//
// `cargo run --release --example amplitude_modulation`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example amplitude_modulation`
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

    let modulated = context.create_gain();
    modulated.gain().set_value(0.5);
    modulated.connect(&context.destination());

    let mut carrier = context.create_oscillator();
    carrier.connect(&modulated);
    carrier.frequency().set_value(300.);

    // mod branch
    let depth = context.create_gain();
    depth.gain().set_value(0.5);
    depth.connect(modulated.gain());

    let mut modulator = context.create_oscillator();
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

        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}
