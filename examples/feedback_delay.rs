use rand::rngs::ThreadRng;
use rand::Rng;
use std::{thread, time};
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// run in release mode
// `cargo run --release --example feedback_delay`

fn trigger_sine(audio_context: &AudioContext, delay_input: &dyn AudioNode, rng: &mut ThreadRng) {
    let now = audio_context.current_time();
    let base_freq = 100.;
    let num_partial = rng.gen_range(1..20) as f32;

    let env = audio_context.create_gain();
    env.connect(delay_input);
    env.gain().set_value_at_time(0., now);
    env.gain()
        .linear_ramp_to_value_at_time(1. / num_partial, now + 0.02);
    env.gain()
        .exponential_ramp_to_value_at_time(0.0001, now + 1.);

    let osc = audio_context.create_oscillator();
    osc.connect(&env);
    osc.frequency().set_value(base_freq * num_partial);
    osc.start_at(now);
    osc.stop_at(now + 1.);
}

fn main() {
    let audio_context = AudioContext::new(None);
    let mut rng = rand::thread_rng();

    // create feedback delay graph layout
    //                           |<- feedback <-|
    //            |-> pre-gain -----> delay ------>|
    // src ---> input ----------------------------------> output

    let output = audio_context.create_gain();
    output.connect(&audio_context.destination());

    let delay = audio_context.create_delay(1.);
    delay.delay_time().set_value(0.3);
    delay.connect(&output);

    let feedback = audio_context.create_gain();
    feedback.gain().set_value(0.85);
    feedback.connect(&delay);
    delay.connect(&feedback);

    let pre_gain = audio_context.create_gain();
    pre_gain.gain().set_value(0.5);
    pre_gain.connect(&feedback);

    let input = audio_context.create_gain();
    input.connect(&pre_gain);
    input.connect(&audio_context.destination()); // direct sound

    loop {
        trigger_sine(&audio_context, &input, &mut rng);

        let period = rng.gen_range(170..1000);
        thread::sleep(time::Duration::from_millis(period));
    }
}
