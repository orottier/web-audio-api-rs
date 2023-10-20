//! This example plays each oscillator type sequentially
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};
use web_audio_api::PeriodicWaveOptions;

// Oscillator types example
//
// `cargo run --release --example oscillators`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example oscillators`
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

    // Create an oscillator node with sine (default) type
    let mut osc = context.create_oscillator();

    // Connect osc to the destination node which is the default output device
    osc.connect(&context.destination());

    // defined periodic wave characteristics
    let options = PeriodicWaveOptions {
        real: Some(vec![0., 0.5, 0.5]),
        imag: Some(vec![0., 0., 0.]),
        disable_normalization: false,
    };

    // Create a custom periodic wave
    let periodic_wave = context.create_periodic_wave(options);

    // Oscillator needs to be started explicitly
    osc.start();

    let interval_time = 2.;

    println!("Sine tone sweep playing... 🎵🎵🎵");

    // Sine tone sweep
    osc.frequency()
        .linear_ramp_to_value_at_time(880., interval_time);
    std::thread::sleep(std::time::Duration::from_secs(interval_time as u64));

    println!("Square tone sweep playing...🎵🎵🎵");

    // Select Square as the oscillator type
    osc.set_type(OscillatorType::Square);
    // Square tone sweep
    osc.frequency()
        .linear_ramp_to_value_at_time(440., context.current_time() + interval_time);
    std::thread::sleep(std::time::Duration::from_secs(interval_time as u64));

    println!("Triangle tone sweep playing...🎵🎵🎵");

    // Select Triangle as the oscillator type
    osc.set_type(OscillatorType::Triangle);
    // Triangle tone sweep
    osc.frequency()
        .linear_ramp_to_value_at_time(880., context.current_time() + interval_time);
    std::thread::sleep(std::time::Duration::from_secs(interval_time as u64));

    println!("Sawtooth tone sweep playing...🎵🎵🎵");

    // Select Sawtooth as the oscillator type
    osc.set_type(OscillatorType::Sawtooth);
    // Sawtooth tone sweep
    osc.frequency()
        .linear_ramp_to_value_at_time(440., context.current_time() + interval_time);
    std::thread::sleep(std::time::Duration::from_secs(interval_time as u64));

    println!("Periodic wave tone sweep playing...🎵🎵🎵");

    // Select Sawtooth as the PeriodicWave type
    osc.set_periodic_wave(periodic_wave);
    // Custom periodic wave tone sweep
    osc.frequency()
        .linear_ramp_to_value_at_time(880., context.current_time() + interval_time);
    std::thread::sleep(std::time::Duration::from_secs(interval_time as u64));

    // enjoy listening
}
