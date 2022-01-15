use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    let context = AudioContext::new(None);

    let am = context.create_gain();
    am.gain().set_value(0.5);
    am.connect(&context.destination());

    let carrier = context.create_oscillator();
    carrier.connect(&am);
    carrier.frequency().set_value(300.);

    // mod branch
    let depth = context.create_gain();
    depth.gain().set_value(0.5);
    depth.connect(am.gain());

    let modulator = context.create_oscillator();
    modulator.connect(&depth);
    modulator.frequency().set_value(1.);

    carrier.start();
    modulator.start();

    let mut flag = 0.;

    loop {
        let freq = if flag == 1. { 0. } else { 300. };
        let when = context.current_time() + 10.;
        modulator.frequency().linear_ramp_to_value_at_time(freq, when);

        flag = 1. - flag;

        thread::sleep(time::Duration::from_secs(10));
    }
}
