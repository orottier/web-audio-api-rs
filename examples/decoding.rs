use std::fs::File;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Decode audio buffer from several format
//
// `cargo run --release --example decoding`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example decoding`
fn main() {
    env_logger::init();

    let files = [
        "samples/sample-faulty.wav",
        "samples/sample.wav",
        "samples/sample.flac",
        "samples/sample.ogg",
        "samples/sample.mp3",
        "samples/sample-aac.m4a",
        "samples/sample-alac.m4a",
        // cannot decode, format not supported or file corrupted
        "samples/empty_2c.wav",
        "samples/corrupt.wav",
        "samples/sample.aiff",
        "samples/sample.webm", // 48kHz,
    ];

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY")
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Ok("interactive") => AudioContextLatencyCategory::Interactive,
        Ok("balanced") => AudioContextLatencyCategory::Balanced,
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    for filepath in files.iter() {
        println!("> --------------------------------");

        let file = File::open(filepath).unwrap();
        let res = context.decode_audio_data_sync(file);

        match res {
            Ok(buffer) => {
                println!("> playing file: {filepath:?}");
                println!("> duration: {:?}", buffer.duration());
                println!("> length: {:?}", buffer.length());
                println!("> channels: {:?}", buffer.number_of_channels());
                println!("> sample rate: {:?}", buffer.sample_rate());
                println!("> --------------------------------");

                let mut src = context.create_buffer_source();
                src.connect(&context.destination());
                src.set_buffer(buffer);
                src.start();

                std::thread::sleep(std::time::Duration::from_secs(4));
            }
            Err(e) => {
                println!("> Error decoding audio file: {filepath:?}");
                eprintln!("> {e:?}");
                println!("> --------------------------------");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
}
