use rand::rngs::ThreadRng;
use rand::Rng;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Feedback delay example
//
// `cargo run --release --example feedback_delay`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example feedback_delay`
fn trigger_sine(context: &AudioContext, delay_input: &dyn AudioNode, rng: &mut ThreadRng) {
    let now = context.current_time();
    let base_freq = 100.;
    let num_partial = rng.gen_range(1..20) as f32;

    let env = context.create_gain();
    env.connect(delay_input);
    env.gain().set_value_at_time(0., now);
    env.gain()
        .linear_ramp_to_value_at_time(1. / num_partial, now + 0.02);
    env.gain()
        .exponential_ramp_to_value_at_time(0.0001, now + 1.);

    let mut osc = context.create_oscillator();
    osc.connect(&env);
    osc.frequency().set_value(base_freq * num_partial);
    osc.start_at(now);
    osc.stop_at(now + 1.);
}

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

    let mut rng = rand::thread_rng();

    // create feedback delay graph layout
    //                           |<- feedback <-|
    //            |-> pre-gain -----> delay ------>|
    // src ---> input ----------------------------------> output

    let output = context.create_gain();
    output.connect(&context.destination());

    let delay = context.create_delay(1.);
    delay.delay_time().set_value(0.3);
    delay.connect(&output);

    let feedback = context.create_gain();
    feedback.gain().set_value(0.85);
    feedback.connect(&delay);
    delay.connect(&feedback);

    let pre_gain = context.create_gain();
    pre_gain.gain().set_value(0.5);
    pre_gain.connect(&feedback);

    let input = context.create_gain();
    input.connect(&pre_gain);
    input.connect(&context.destination()); // direct sound

    loop {
        trigger_sine(&context, &input, &mut rng);

        let period = rng.gen_range(170..1000);
        std::thread::sleep(std::time::Duration::from_millis(period));
    }
}
