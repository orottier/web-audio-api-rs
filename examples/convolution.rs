use std::fs::File;
use std::{thread, time};

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ConvolverNode, ConvolverOptions};

fn main() {
    // create an `AudioContext` and load a sound file
    let context = AudioContext::default();
    let file = File::open("samples/vocals-dry.wav").unwrap();
    let audio_buffer = context.decode_audio_data_sync(file).unwrap();

    let impulse_file = File::open("samples/parking-garage-response.wav").unwrap();
    let impulse_buffer = context.decode_audio_data_sync(impulse_file).unwrap();

    let src = context.create_buffer_source();
    src.set_buffer(audio_buffer);

    let opts = ConvolverOptions {
        buffer: Some(impulse_buffer),
        ..ConvolverOptions::default()
    };
    let convolve = ConvolverNode::new(&context, opts);

    src.connect(&convolve);
    convolve.connect(&context.destination());

    src.start();

    thread::sleep(time::Duration::from_millis(16_000));
}
