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

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let mut analyser = context.create_analyser();
    analyser.connect(&context.destination());

    let mut osc = context.create_oscillator();
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
