use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let audio_context = AudioContext::new(None);

    // use merger to pipe oscillators to right and left channels
    let merger = audio_context.create_channel_merger(2);
    merger.connect(&audio_context.destination());

    // left branch
    let gain_left = audio_context.create_gain();
    gain_left.gain().set_value(0.);
    gain_left.connect_at(&merger, 0, 0).ok(); // this is a bit weird

    let src_left = audio_context.create_oscillator();
    src_left.frequency().set_value(200.);
    src_left.connect(&gain_left);
    src_left.start();

    // right branch
    let gain_right = audio_context.create_gain();
    gain_right.gain().set_value(0.);
    gain_right.connect_at(&merger, 0, 1).ok(); // this is a bit weird

    let src_right = audio_context.create_oscillator();
    src_right.frequency().set_value(500.);
    src_right.connect(&gain_right);
    src_right.start();

    // control both left and right gains with constant source
    let constant_source = audio_context.create_constant_source();
    constant_source.offset().set_value(0.);
    constant_source.connect(gain_left.gain());
    constant_source.connect(gain_right.gain());
    constant_source.start();

    let mut target = 0.;

    loop {
        let now = audio_context.current_time();
        constant_source
            .offset()
            .set_target_at_time(target, now, 0.1);

        target = if target == 0. { 1. } else { 0. };

        thread::sleep(time::Duration::from_millis(1000));
    }
}
