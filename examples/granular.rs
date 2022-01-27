use std::fs::File;
use std::{thread, time};
use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::AudioNode;

// run in release mode
// cargo run --release --example granular

fn trigger_grain(
    audio_context: &AudioContext,
    audio_buffer: &AudioBuffer,
    position: f64,
    duration: f64,
) {
    let start_time = audio_context.current_time();

    let env = audio_context.create_gain();
    env.gain().set_value(0.);
    env.connect(&audio_context.destination());

    let src = audio_context.create_buffer_source();
    src.set_buffer(audio_buffer.clone());
    src.connect(&env);

    // ramp
    env.gain().set_value_at_time(0., start_time);
    env.gain()
        .linear_ramp_to_value_at_time(1., start_time + duration / 2.);
    env.gain()
        .linear_ramp_to_value_at_time(0., start_time + duration);

    src.start_at_with_offset(start_time, position);
    src.stop_at(start_time + duration);
}

fn main() {
    let audio_context = AudioContext::new(None);

    println!("++ scrub into file forward and backward at 0.5 speed");

    // grab audio buffer
    let file = File::open("samples/sample.wav").unwrap();
    let audio_buffer = audio_context.decode_audio_data_sync(file).unwrap();

    let period = 0.05;
    let grain_duration = 0.2;
    let mut position = 0.;
    let mut incr_position = period / 2.;

    loop {
        trigger_grain(&audio_context, &audio_buffer, position, grain_duration);

        if position + incr_position > audio_buffer.duration() - (grain_duration * 2.)
            || position + incr_position < 0.
        {
            incr_position *= -1.;
        }

        position += incr_position;

        thread::sleep(time::Duration::from_millis((period * 1000.) as u64));
    }
}
