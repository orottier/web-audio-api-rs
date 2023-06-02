use std::f32::consts::PI;

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    // create a 1 second buffer filled with a sine at 200Hz
    println!("> Play sine at 440Hz in AudioBufferSourceNode");

    let length = context.sample_rate() as usize;
    let sample_rate = context.sample_rate();
    let mut buffer = context.create_buffer(1, length, sample_rate);
    let mut sine = vec![];

    for i in 0..length {
        let phase = i as f32 / length as f32 * 2. * PI * 440.;
        sine.push(phase.sin());
    }

    buffer.copy_to_channel(&sine, 0);

    // play the buffer in a loop
    let src = context.create_buffer_source();
    src.set_buffer(buffer.clone());
    src.connect(&context.destination());
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(2000));

    println!("> Play sine at 440Hz w/ playback rate at 0.5");

    let src = context.create_buffer_source();
    src.set_buffer(buffer.clone());
    src.playback_rate().set_value(0.5);
    src.connect(&context.destination());
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3000));

    println!("> Play sine at 440Hz w/ detune at -1200.");

    let src = context.create_buffer_source();
    src.set_buffer(buffer.clone());
    src.detune().set_value(-1200.);
    src.connect(&context.destination());
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3000));
}
