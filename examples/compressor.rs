use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, DynamicsCompressorNode};

fn main() {
    env_logger::init();

    let context = AudioContext::default();

    let file = File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    println!("> no compression");
    let src = context.create_buffer_source();
    src.connect(&context.destination());
    src.set_buffer(buffer.clone());
    src.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(3));

    println!("> compression (hard knee)");
    println!("+ attack: {:?}ms", 30);
    println!("+ release: {:?}ms", 100);
    println!("+ ratio: {:?}", 12);
    println!(">");

    for i in 0..6 {
        println!("+ threshold at {:?}", -10. * i as f32);
        let compressor = DynamicsCompressorNode::new(&context, Default::default());
        compressor.connect(&context.destination());
        compressor.threshold().set_value(-10. * i as f32);
        compressor.knee().set_value(0.); // hard knee
        compressor.attack().set_value(0.03); // hard knee
        compressor.release().set_value(0.1); // hard knee

        let src = context.create_buffer_source();
        src.connect(&compressor);
        src.set_buffer(buffer.clone());
        src.start();

        // enjoy listening
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
