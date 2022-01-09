use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioNode;

fn main() {
    let files = [
        "samples/sample.wav",
        "samples/sample.flac",
        "samples/sample.ogg",
        "samples/sample.mp3",
        // not supported by decoder:
        // "samples/sample.aiff",
        // "samples/sample.webm", // 48kHz,
    ];

    let audio_context = AudioContext::new(None);

    for filepath in files.iter() {
        println!("> --------------------------------");
        println!("> reading file: {:?}", filepath);

        let file = File::open(filepath).unwrap();
        let buffer = audio_context.decode_audio_data(file).unwrap();

        println!("> duration: {:?}", buffer.duration());
        println!("> num samples: {:?}", buffer.get_channel_data(0).len());
        println!("> --------------------------------");

        let src = audio_context.create_buffer_source();
        src.connect(&audio_context.destination());
        src.set_buffer(buffer);
        src.start();

        std::thread::sleep(std::time::Duration::from_secs(4));
    }
}
