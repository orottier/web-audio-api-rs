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

    // Create a media destination node that will ship the samples out of the audio graph
    let (sender, receiver) = crossbeam_channel::unbounded();
    let callback = move |buf| {
        // this will run on the render thread so it should not block
        sender.send(buf).unwrap();
    };
    let dest = context.create_media_stream_destination(callback);
    osc.connect(&dest);
    osc.start();

    // Handle recorded buffers
    println!("samples recorded:");
    let mut samples_recorded = 0;
    for buf in receiver.iter() {
        // You could write the samples to a file here.

        samples_recorded += buf.length();
        print!("{}\r", samples_recorded);
    }
}
