use std::f32::consts::PI;

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::new(None);

    // create a 1 second buffer filled with a sine at 200Hz
    println!("> Play sine at 200Hz created manually in an AudioBuffer");

    let length = context.sample_rate() as usize;
    let sample_rate = context.sample_rate_raw();
    let mut buffer = context.create_buffer(1, length, sample_rate);
    let mut sine = vec![];

    for i in 0..length {
        let phase = i as f32 / length as f32 * 2. * PI * 200.;
        sine.push(phase.sin());
    }

    buffer.copy_to_channel(&sine, 0);

    // play the buffer in a loop
    let src = context.create_buffer_source();
    src.set_buffer(buffer.clone());
    src.set_loop(true);
    src.connect(&context.destination());
    src.start_at(context.current_time());
    src.stop_at(context.current_time() + 3.);

    std::thread::sleep(std::time::Duration::from_millis(3500));

    // play a sine at 200Hz
    println!("> Play sine at 200Hz from an OscillatorNode");

    let osc = context.create_oscillator();
    osc.frequency().set_value(200.);
    osc.connect(&context.destination());
    osc.start_at(context.current_time());
    osc.stop_at(context.current_time() + 3.);

    std::thread::sleep(std::time::Duration::from_millis(3500));
}
