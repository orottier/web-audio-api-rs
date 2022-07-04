use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    let file = File::open("samples/major-scale.ogg").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    // taken from https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_IIR_filters
    let feed_forward = vec![0.00020298, 0.0004059599, 0.00020298];
    let feed_backward = vec![1.0126964558, -1.9991880801, 0.9873035442];

    // Create an IIR filter node
    let iir = context.create_iir_filter(feed_forward, feed_backward);
    iir.connect(&context.destination());

    // Play the buffer
    let src = context.create_buffer_source();
    src.connect(&iir);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
