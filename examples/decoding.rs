use std::fs::File;
use web_audio_api::context::{Context, AudioContext};
use web_audio_api::node::AudioNode;

fn main() {
    // env_logger::init();
    let files = [
        "samples/sample-faulty.wav",
        "samples/sample.wav",
        "samples/sample.flac",
        "samples/sample.ogg",
        "samples/sample.mp3",
        // cannot decode, format not supported or file corrupted
        "samples/empty_2c.wav",
        "samples/corrupt.wav",
        "samples/sample.aiff",
        "samples/sample.webm", // 48kHz,
    ];

    let audio_context = AudioContext::new(None);

    for filepath in files.iter() {
        println!("> --------------------------------");

        let file = File::open(filepath).unwrap();
        let res = audio_context.decode_audio_data(file);

        match res {
            Ok(buffer) => {
                println!("> playing file: {:?}", filepath);
                println!("> duration: {:?}", buffer.duration());
                println!("> length: {:?}", buffer.length());
                println!("> channels: {:?}", buffer.number_of_channels());
                println!("> sample rate: {:?}", buffer.sample_rate());
                println!("> --------------------------------");

                let src = audio_context.create_buffer_source();
                src.connect(&audio_context.destination());
                src.set_buffer(buffer);
                src.start();

                std::thread::sleep(std::time::Duration::from_secs(4));
            }
            Err(e) => {
                println!("> Error decoding audio file: {:?}", filepath);
                eprintln!("> {:?}", e);
                println!("> --------------------------------");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
}
