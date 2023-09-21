use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::media_devices::{enumerate_devices_sync, MediaDeviceInfo, MediaDeviceInfoKind};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// Select output device example
//
// `cargo run --release --example sink_id`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example sink_id`

fn ask_sink_id() -> String {
    println!("Enter the output 'device_id' and press <Enter>");
    println!("- type 'none' to disable the output");
    println!("- Leave empty ('') for the default audio output device");

    std::io::stdin().lines().next().unwrap().unwrap()
}

fn main() {
    env_logger::init();

    let devices = enumerate_devices_sync();
    let output_devices: Vec<MediaDeviceInfo> = devices
        .into_iter()
        .filter(|d| d.kind() == MediaDeviceInfoKind::AudioOutput)
        .collect();

    dbg!(output_devices);

    let sink_id = ask_sink_id();

    // create context with selected sink id
    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        sink_id,
        ..AudioContextOptions::default()
    });

    println!("Playing beep for sink {:?}", context.sink_id());

    context.set_onsinkchange(|_| println!("sink change event"));

    // Create an oscillator node with sine (default) type
    let mut osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    loop {
        let sink_id = ask_sink_id();
        let result = context.set_sink_id_sync(sink_id);

        if let Err(err) = result {
            println!("Error setting sink id {}", err);
        }

        println!("Playing beep for sink {:?}", context.sink_id());

        // with this info we can ensure the progression of time with any backend
        println!("Current time is now {:<3}", context.current_time());
    }
}
