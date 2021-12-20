// use rand::rngs::ThreadRng;
// use rand::Rng;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// run in release mode
// `cargo run --release --example cyclic_graph`

fn trigger_sine(context: &AudioContext) {
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.frequency().set_value(300.);

    let now = context.current_time();
    osc.start_at(now);
    osc.stop_at(now + 0.05);
}

fn main() {
    let context = AudioContext::new(None);

    let gain = context.create_gain();
    gain.gain().set_value(1.); // echo decay
    gain.connect(&context.destination());

    let period = 0.1;

    let delay = context.create_delay(3.);
    delay.delay_time().set_value(period);
    delay.connect(&gain);
    gain.connect(&delay);

    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.connect(&delay);
    osc.frequency().set_value(100.);

    let now = context.current_time();
    osc.start_at(now);
    osc.stop_at(now + 0.05);

    // mimic setInterval
    loop {
        trigger_sine(&context);

        // let period = rng.gen_range(170..1000);
        println!("> {:?}", context.current_time());
        thread::sleep(time::Duration::from_millis((period * 1000.) as u64));
    }
}
