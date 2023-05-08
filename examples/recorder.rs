//! Example showing how to record audio live from a running AudioContext
//!
//! Obviously you can use an OfflineAudioContext to render audio in a non-dynamic fashion.
//!
//! Usage: `cargo run --release --example recorder > rec.wav`
//!
//! This will record 2 seconds of an oscillator to the file `rec.wav`

use std::io::Write;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media_recorder::MediaRecorder;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    // Create an audio context where all audio nodes lives
    let context = AudioContext::default();

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();

    // Connect oscillator to speakers
    osc.connect(&context.destination());

    // Create a media destination node
    let dest = context.create_media_stream_destination();
    osc.connect(&dest);

    // Start playing
    osc.start();

    let recorder = MediaRecorder::new(dest.stream());
    recorder.set_ondataavailable(move |event| {
        eprintln!(
            "timecode {:.6}, data size {}",
            event.timecode,
            event.blob.len()
        );
        std::io::stdout().write_all(&event.blob).unwrap();
    });
    recorder.start();

    std::thread::sleep(std::time::Duration::from_secs(2));

    // stop and wait for the final blob to flush
    let (send, recv) = crossbeam_channel::bounded(0);
    recorder.set_onstop(move || {
        let _ = send.send(());
    });
    recorder.stop();
    let _ = recv.recv();
}
