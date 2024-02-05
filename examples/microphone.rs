use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::media_devices;
use web_audio_api::media_devices::{enumerate_devices_sync, MediaDeviceInfo, MediaDeviceInfoKind};
use web_audio_api::media_devices::{MediaStreamConstraints, MediaTrackConstraints};
use web_audio_api::node::AudioNode;

// Pipe microphone stream into audio context
//
// `cargo run --release --example microphone`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example microphone`

fn ask_source_id() -> Option<String> {
    println!("Enter the input 'device_id' and press <Enter>");
    println!("- Leave empty ('') for the default audio input device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "" => None,
        i => Some(i.to_string()),
    }
}

fn ask_sink_id() -> String {
    println!("Enter the output 'device_id' and press <Enter>");
    println!("- type 'none' to disable the output");
    println!("- Leave empty ('') for the default audio output device");

    std::io::stdin().lines().next().unwrap().unwrap()
}

fn main() {
    env_logger::init();

    // select input and output devices
    let devices = enumerate_devices_sync();

    let input_devices: Vec<MediaDeviceInfo> = devices
        .into_iter()
        .filter(|d| d.kind() == MediaDeviceInfoKind::AudioInput)
        .collect();

    dbg!(input_devices);
    let source_id = ask_source_id();

    let devices = enumerate_devices_sync();

    let output_devices: Vec<MediaDeviceInfo> = devices
        .into_iter()
        .filter(|d| d.kind() == MediaDeviceInfoKind::AudioOutput)
        .collect();

    dbg!(output_devices);
    let sink_id = ask_sink_id();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        sink_id,
        ..AudioContextOptions::default()
    });

    let mut constraints = MediaTrackConstraints::default();
    constraints.device_id = source_id;
    // constraints.channel_count = Some(2);
    let stream_constraints = MediaStreamConstraints::AudioWithConstraints(constraints);
    let mic = media_devices::get_user_media_sync(stream_constraints);

    // create media stream source node with mic stream
    let stream_source = context.create_media_stream_source(&mic);
    stream_source.connect(&context.destination());

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    // println!("Closing microphone");
    // mic.get_tracks()[0].close();
    // std::thread::sleep(std::time::Duration::from_secs(2));
}
