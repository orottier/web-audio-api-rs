use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    let file = File::open("samples/major-scale.ogg").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    let feedforward = vec![
        0.000016636797512844526,
        0.00003327359502568905,
        0.000016636797512844526,
    ];
    let feedback = vec![1.0, -1.9884300106225539, 0.9884965578126054];

    // Create an IIR filter node
    let iir = context.create_iir_filter(feedforward, feedback);
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
