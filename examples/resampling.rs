use std::fs::File;
use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let audio_context = AudioContext::default();
    println!(
        "> AudioContext sample_rate: {:?}",
        audio_context.sample_rate()
    );

    println!("--------------------------------------------------------------");
    println!("> Case 1: buffers are decoded at right sample rate by context");
    println!("--------------------------------------------------------------");

    let file_38000 = File::open("samples/sample-38000.wav").unwrap();
    let buffer_38000 = audio_context.decode_audio_data_sync(file_38000).unwrap();

    let file_44100 = File::open("samples/sample-44100.wav").unwrap();
    let buffer_44100 = audio_context.decode_audio_data_sync(file_44100).unwrap();

    let file_48000 = File::open("samples/sample-48000.wav").unwrap();
    let buffer_48000 = audio_context.decode_audio_data_sync(file_48000).unwrap();

    // audio context at default system sample rate
    println!(
        "+ playing sample-38000.wav - decoded sample rate: {:?}",
        buffer_38000.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_38000);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));

    println!(
        "+ playing sample-44100.wav - decoded sample rate: {:?}",
        buffer_44100.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_44100);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));

    println!(
        "+ playing sample-48000.wav - decoded sample rate: {:?}",
        buffer_48000.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_48000);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));

    // --------------------------------------------------------------
    println!("--------------------------------------------------------------");
    println!("> Case 2: buffers are decoded with another sample rate, then resampled by the AudioBufferSourceNode");
    println!("--------------------------------------------------------------");

    let audio_context_38000 = AudioContext::new(AudioContextOptions {
        sample_rate: Some(38000),
        latency_hint: None,
        number_of_channels: None,
    });
    let file_38000 = File::open("samples/sample-38000.wav").unwrap();
    let buffer_38000 = audio_context_38000
        .decode_audio_data_sync(file_38000)
        .unwrap();

    let audio_context_44100 = AudioContext::new(AudioContextOptions {
        sample_rate: Some(44100),
        latency_hint: None,
        number_of_channels: None,
    });
    let file_44100 = File::open("samples/sample-44100.wav").unwrap();
    let buffer_44100 = audio_context_44100
        .decode_audio_data_sync(file_44100)
        .unwrap();

    let audio_context_48000 = AudioContext::new(AudioContextOptions {
        sample_rate: Some(48000),
        latency_hint: None,
        number_of_channels: None,
    });
    let file_48000 = File::open("samples/sample-48000.wav").unwrap();
    let buffer_48000 = audio_context_48000
        .decode_audio_data_sync(file_48000)
        .unwrap();

    // audio context at default system sample rate
    println!(
        "+ playing sample-38000.wav - decoded sample rate: {:?}",
        buffer_38000.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_38000);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));

    println!(
        "+ playing sample-44100.wav - decoded sample rate: {:?}",
        buffer_44100.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_44100);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));

    println!(
        "+ playing sample-48000.wav - decoded sample rate: {:?}",
        buffer_48000.sample_rate()
    );
    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer_48000);
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(3500));
}
