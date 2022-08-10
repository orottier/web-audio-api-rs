use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    let file = File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    // these values correspond to a lowpass filter at 200Hz (calculated from biquad)
    let feedforward = vec![
        0.0002029799640409502,
        0.0004059599280819004,
        0.0002029799640409502,
    ];

    let feedback = vec![
        1.0126964557853775,
        -1.9991880801438362,
        0.9873035442146225,
    ];

    // Create an IIR filter node
    let iir = context.create_iir_filter(feedforward, feedback);
    iir.connect(&context.destination());

    // Play buffer and pipe to filter
    let src = context.create_buffer_source();
    src.connect(&iir);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    loop {
        std::thread::sleep(std::time::Duration::from_secs(4));
    }
}
