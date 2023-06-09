use std::env;

use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::{AudioNode};
use web_audio_api::media_devices::{
    enumerate_devices_sync, get_user_media_sync, MediaDeviceInfo, MediaDeviceInfoKind
};
use web_audio_api::media_devices::{MediaStreamConstraints, MediaTrackConstraints};

mod latency_tester;
use crate::latency_tester::LatencyTesterNode;

// Estimate the audio round-trip latency
//
// just plug your audio output into the audio input with some cable and run
// `cargo run --release`
//
// for testing a feedback delay line created to emulate the audio feedback, run with
// `cargo run --release -- test`

fn ask_source_id() -> Option<String> {
    println!("Enter the input 'device_id' and press <Enter>");
    println!("- Use 0 for the default audio input device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "0" => None,
        i => Some(i.to_string()),
    }
}

fn ask_sink_id() -> String {
    println!("Enter the output 'sink' and press <Enter>");
    println!("- Use 0 for the default audio output device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "0" => "none".to_string(),
        i => i.to_string(),
    }
}

fn main() {
    let mut test = false;
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && args[1] == "test" {
        test = true;
    }

    let (_context, latency_tester) = if test {
        // emulate loopback
        let latency: f32 = 0.017;
        println!("> testing: latency should be {:?}", latency);

        let context = AudioContext::new(AudioContextOptions {
            sample_rate: Some(48000.),
            ..AudioContextOptions::default()
        });

        let latency_tester = LatencyTesterNode::new(&context);
        latency_tester.connect(&context.destination());

        let delay = context.create_delay(latency.ceil() as f64);
        delay.delay_time().set_value(latency);

        latency_tester.connect(&delay);
        delay.connect(&latency_tester);

        (context, latency_tester)
    } else {
        // full round trip
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


        let context = AudioContext::new(AudioContextOptions {
            sink_id,
            sample_rate: Some(48000.),
            ..AudioContextOptions::default()
        });

        let latency_tester = LatencyTesterNode::new(&context);
        latency_tester.connect(&context.destination());

        // open mic stream and pipe into latency_tester
        let mut constraints = MediaTrackConstraints::default();
        constraints.device_id = source_id;
        let stream_constraints = MediaStreamConstraints::AudioWithConstraints(constraints);
        let mic = get_user_media_sync(stream_constraints);

        // create media stream source node with mic stream
        let stream_source = context.create_media_stream_source(&mic);
        stream_source.connect(&latency_tester);

        (context, latency_tester)
    };

    loop {
        println!("latency {:?}", latency_tester.calculated_delay());
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
