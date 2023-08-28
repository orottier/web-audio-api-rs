use std::io::Write;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::media_recorder::MediaRecorder;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Example showing how to record audio live from a running AudioContext.
// This will record 2 seconds of an oscillator to the file `rec.wav`.
//
// `cargo run --release --example recorder > rec.wav`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example recorder > rec.wav`
fn main() {
    env_logger::init();

    // Create an audio context where all audio nodes lives
    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // Create an oscillator node with sine (default) type
    let mut osc = context.create_oscillator();

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
    let (send, recv) = crossbeam_channel::bounded(1);
    recorder.set_onstop(move |_| {
        let _ = send.send(());
    });
    recorder.stop();
    let _ = recv.recv();
}
