use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::media_devices::{enumerate_devices, MediaDeviceInfo, MediaDeviceInfoKind};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn ask_sink_id() -> String {
    println!("Enter the output 'device_id' and press <Enter>");
    println!("- Leave empty for AudioSinkType 'none'");
    println!("- Use 0 for the default audio output device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "0" => "none".to_string(),
        i => i.to_string(),
    }
}

fn main() {
    env_logger::init();

    let devices = enumerate_devices();
    let output_devices: Vec<MediaDeviceInfo> = devices
        .into_iter()
        .filter(|d| d.kind() == MediaDeviceInfoKind::AudioOutput)
        .collect();

    dbg!(output_devices);

    let sink_id = ask_sink_id();

    // Create an audio context (default: stereo);
    let options = AudioContextOptions {
        sink_id,
        ..AudioContextOptions::default()
    };

    let context = AudioContext::new(options);
    println!("Playing beep for sink {:?}", context.sink_id());

    context.set_onsinkchange(|_| println!("sink change event"));

    // Create an oscillator node with sine (default) type
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    loop {
        let sink_id = ask_sink_id();

        context.set_sink_id_sync(sink_id).unwrap();
        println!("Playing beep for sink {:?}", context.sink_id());

        // with this info we can ensure the progression of time with any backend
        println!("Current time is now {:<3}", context.current_time());
    }
}
