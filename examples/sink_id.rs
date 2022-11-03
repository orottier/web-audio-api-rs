use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::enumerate_devices;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    let devices = enumerate_devices();
    dbg!(devices);

    println!(
        "Enter the output 'device_id' and press <Enter>. Leave empty for AudioSinkType 'none'"
    );
    let input = std::io::stdin().lines().next().unwrap().unwrap();
    let sink_id = match input.trim() {
        "" => None,
        i => Some(i.to_string()),
    };

    // Create an audio context (default: stereo);
    let options = AudioContextOptions {
        sink_id: Some(sink_id),
        ..AudioContextOptions::default()
    };

    let context = AudioContext::new(options);
    println!("Playing beep for sink {:?}", context.sink_id());

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    loop {
        println!(
            "Enter the output 'device_id' and press <Enter>. Leave empty for AudioSinkType 'none'"
        );

        let input = std::io::stdin().lines().next().unwrap().unwrap();
        let sink_id = match input.trim() {
            "" => None,
            i => Some(i.to_string()),
        };

        context.set_sink_id_sync(sink_id).unwrap();
        println!("Playing beep for sink {:?}", context.sink_id());

        // with this info we can ensure the progression of time with any backend
        println!("Current time is now {:<3}", context.current_time());
    }
}
