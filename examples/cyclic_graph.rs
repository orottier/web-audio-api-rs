use rand::rngs::ThreadRng;
use rand::Rng;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// run in release mode
// `cargo run --release --example cyclic_graph`

fn trigger_sine(context: &AudioContext, rng: &mut ThreadRng, cycle: &dyn AudioNode) {
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.connect(cycle);

    let now = context.current_time();

    let freq = rng.gen_range(100..3000) as f32;
    osc.frequency().set_value(freq);

    osc.start_at(now);
    osc.stop_at(now + 0.1);
}

fn main() {
    let context = AudioContext::new(None);

    let gain = context.create_gain();
    gain.gain().set_value(0.5); // echo decay
    gain.connect(&context.destination());

    let delay = context.create_delay(1.);
    delay.delay_time().set_value(0.3);
    delay.connect(&gain);

    // add cycle to get echo effect
    gain.connect(&delay);

    let mut rng = rand::thread_rng();

    // mimic setInterval
    loop {
        trigger_sine(&context, &mut rng, &delay);

        let period = rng.gen_range(170..1000);
        thread::sleep(time::Duration::from_millis(period));
    }
}
