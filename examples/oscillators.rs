//! This example plays each oscillator type sequentially
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, OscillatorType, PeriodicWaveOptions,
};

fn main() {
    // Create an audio context where all audio nodes lives
    let context = AudioContext::new();

    // Create an oscillator node with sine (default) type
    let mut osc = context.create_oscillator();

    // Connect osc to the destination node which is the default output device
    osc.connect(&context.destination());

    // defined periodic wave characteristics
    let options = Some(PeriodicWaveOptions {
        real: Some(vec![0., 0.5, 0.5]),
        imag: Some(vec![0., 0., 0.]),
        disable_normalization: Some(false),
    });

    // Create a custom periodic wave
    let periodic_wave = context.create_periodic_wave(options);

    // Oscillator needs to be started explicitily
    osc.start();

    println!("Sine tone sweep playing...");

    // Sine tone sweep
    for i in 0..1000 {
        osc.frequency().set_value(440. + i as f32);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    println!("Square tone sweep playing...");

    // Select Square as the oscillator type
    osc.set_type(OscillatorType::Square);

    // Square tone sweep
    for i in 0..1000 {
        osc.frequency().set_value(440. + i as f32);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    println!("Triangle tone sweep playing...");

    // Select Triangle as the oscillator type
    osc.set_type(OscillatorType::Triangle);

    // Triangle tone sweep
    for i in 0..1000 {
        osc.frequency().set_value(440. + i as f32);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    println!("Sawtooth tone sweep playing...");

    // Select Sawtooth as the oscillator type
    osc.set_type(OscillatorType::Sawtooth);

    // Sawtooth tone sweep
    for i in 0..1000 {
        osc.frequency().set_value(440. + i as f32);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    println!("Periodic wave tone sweep playing...");

    // Select Sawtooth as the PeriodicWave type
    osc.set_periodic_wave(periodic_wave);

    // Custom periodic wave tone sweep
    for i in 0..1000 {
        osc.frequency().set_value(440. + i as f32);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // enjoy listening
}
