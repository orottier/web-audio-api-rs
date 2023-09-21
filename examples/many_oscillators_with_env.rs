use rand::rngs::ThreadRng;
use rand::Rng;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Trigger many oscillators with an envelop
//
// `cargo run --release --example many_oscillators_with_env`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example many_oscillators_with_env`
fn trigger_sine(audio_context: &AudioContext, rng: &mut ThreadRng) {
    let env = audio_context.create_gain();
    env.gain().set_value(0.);
    env.connect(&audio_context.destination());

    let mut osc = audio_context.create_oscillator();
    osc.connect(&env);

    let now = audio_context.current_time();

    let freq = rng.gen_range(100..3000) as f32;
    osc.frequency().set_value(freq);

    env.gain()
        .set_value_at_time(0., now)
        .linear_ramp_to_value_at_time(0.1, now + 0.01)
        .exponential_ramp_to_value_at_time(0.0001, now + 2.);

    osc.start_at(now);
    osc.stop_at(now + 2.);
}

fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let audio_context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let mut rng = rand::thread_rng();
    let period = 50;

    // mimic setInterval
    loop {
        trigger_sine(&audio_context, &mut rng);
        std::thread::sleep(std::time::Duration::from_millis(period));
    }
}
