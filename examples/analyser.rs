use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// AnalyserNode example
//
// `cargo run --release --example analyser`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example analyser`
fn main() {
    env_logger::init();

    let context = match std::env::var("WEB_AUDIO_LATENCY") {
        Ok(val) => {
            if val == "playback" {
                AudioContext::new(AudioContextOptions {
                    latency_hint: AudioContextLatencyCategory::Playback,
                    ..AudioContextOptions::default()
                })
            } else {
                println!("Invalid WEB_AUDIO_LATENCY value, fall back to default");
                AudioContext::default()
            }
        }
        Err(_e) => AudioContext::default(),
    };

    let analyser = context.create_analyser();
    analyser.connect(&context.destination());

    let osc = context.create_oscillator();
    osc.frequency().set_value(200.);
    osc.connect(&analyser);
    osc.start();

    let mut bins = vec![0.; analyser.frequency_bin_count()];

    loop {
        analyser.get_float_frequency_data(&mut bins);
        println!("{:?}", &bins[0..20]); // print 20 first bins
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
