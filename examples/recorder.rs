//! Example showing how to record audio live from a running AudioContext
//!
//! Obviously you can use an OfflineAudioContext to render audio in a non-dynamic fashion

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    // Create an audio context where all audio nodes lives
    let context = AudioContext::new(None);

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();

    // Create a media destination node
    let dest = context.create_media_stream_destination();
    osc.connect(&dest);
    osc.start();

    // Handle recorded buffers
    println!("samples recorded:");
    let mut samples_recorded = 0;
    for item in dest.stream() {
        let buffer = item.unwrap();

        // You could write the samples to a file here.
        samples_recorded += buffer.length();
        print!("{}\r", samples_recorded);
    }
}
