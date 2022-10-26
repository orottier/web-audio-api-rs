use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::enumerate_devices;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    let devices = enumerate_devices();
    dbg!(devices);

    println!("Choose output device, enter the 'device_id' and press <Enter>:");
    let sink_id = std::io::stdin().lines().next().unwrap().unwrap();

    // Create an audio context (default: stereo);
    let options = AudioContextOptions {
        sink_id: Some(sink_id),
        ..AudioContextOptions::default()
    };

    let mut context = AudioContext::new(options);
    println!("Playing beep for sink {:?}", context.sink_id());

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    loop {
        println!("Choose output device, enter the 'device_id' and press <Enter>:");
        let sink_id = std::io::stdin().lines().next().unwrap().unwrap();
        context.set_sink_id(sink_id);
        println!("Playing beep for sink {:?}", context.sink_id());
    }
}
