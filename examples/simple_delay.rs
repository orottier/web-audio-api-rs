use std::fs::File;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioNode;

fn main() {
    // create an `AudioContext` and load a sound file
    let context = AudioContext::new(None);
    let file = File::open("sample.wav").unwrap();
    let audio_buffer = context.decode_audio_data(file);

    // create a delay of 0.5s
    let delay = context.create_delay(1.);
    delay.delay_time().set_value(0.5);
    delay.connect(&context.destination());

    let src = context.create_buffer_source();
    src.set_buffer(audio_buffer);
    // connect to both delay and destination
    src.connect(&delay);
    src.connect(&context.destination());
    src.start();

    thread::sleep(time::Duration::from_millis(5000));
}
