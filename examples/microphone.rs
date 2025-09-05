use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::media_devices;
use web_audio_api::media_devices::{enumerate_devices_sync, MediaDeviceInfo, MediaDeviceInfoKind};
use web_audio_api::media_devices::{MediaStreamConstraints, MediaTrackConstraints};
use web_audio_api::media_recorder::{MediaRecorder, MediaRecorderOptions};
use web_audio_api::node::AudioNode;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    constraints.echo_cancellation = Some(true); // Enable echo cancellation
    let stream_constraints = MediaStreamConstraints::AudioWithConstraints(constraints);
    let mic = media_devices::get_user_media_sync(stream_constraints);

    println!("\n✓ Microphone stream created with echo cancellation enabled");
    println!("You should be able to speak without hearing feedback/echo.\n");

    // create media stream source node with mic stream
    let stream_source = context.create_media_stream_source(&mic);
    
    // Create a media stream destination to capture audio
    let dest = context.create_media_stream_destination();
    stream_source.connect(&dest);
    
    // Also connect to speakers so you can hear yourself
    stream_source.connect(&context.destination());
    
    // Create media recorder
    let options = MediaRecorderOptions::default(); // default to audio/wav
    let recorder = MediaRecorder::new(dest.stream(), options);
    
    // Create a file to write the recording
    let mut file = File::create("recording.wav").expect("Failed to create file");
    
    recorder.set_ondataavailable(move |event| {
        eprintln!(
            "Recording... timecode {:.3}s, chunk size {} bytes",
            event.timecode,
            event.blob.size()
        );
        file.write_all(&event.blob.data).unwrap();
    });
    
    // Set up signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    
    ctrlc::set_handler(move || {
        println!("\nReceived interrupt signal, stopping recording...");
        running_clone.store(false, Ordering::Relaxed);
    }).expect("Error setting Ctrl-C handler");
    
    // Start recording
    recorder.start();
    println!("Recording to 'recording.wav'... Press Ctrl+C to stop.");
    
    // Wait for interrupt
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    // Stop recording and wait for final data
    let (send, recv) = crossbeam_channel::bounded(1);
    recorder.set_onstop(move |_| {
        let _ = send.send(());
    });
    recorder.stop();
    let _ = recv.recv();
    
    println!("Recording saved successfully!");
}
