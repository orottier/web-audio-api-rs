use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioNode;

fn main() {
    let files = [
        "samples/sample.wav",
        "samples/sample.flac",
        "samples/sample.mp3",
        "samples/sample.ogg",
        // "samples/sample.aiff", // not supported by decoding
        // no opus
    ];

    let audio_context = AudioContext::new(None);

    for filepath in files.iter() {
        println!("> --------------------------------");
        println!("> reading file {:?}", filepath);

        let file = File::open(filepath).unwrap();
        let buffer = audio_context.decode_audio_data(file).unwrap();

        println!("> duration {:?}", buffer.duration());
        println!("> num samples {:?}", buffer.get_channel_data(0).len());

        let src = audio_context.create_buffer_source();
        src.connect(&audio_context.destination());
        src.set_buffer(buffer);
        src.start();

        std::thread::sleep(std::time::Duration::from_secs(4));
    }
}
