use rand::rngs::ThreadRng;
use rand::Rng;
use std::fs::File;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::media::{MediaElement, WavDecoder};
use web_audio_api::node::{AudioControllableSourceNode, AudioScheduledSourceNode};
use web_audio_api::node::{AudioNode, MediaElementAudioSourceNode};

// run in release mode
// cargo run --release --example granular-original

fn trigger_grain(
    audio_context: &AudioContext,
    audio_buffer: &MediaElementAudioSourceNode,
    position: f64,
    duration: f64,
    rng: &mut ThreadRng,
) {
    // don't know the precision of sleep millis, but we add a small random
    // offset to avoid audible pitch due to period (even if with such period it
    // should be ok)
    let jitter = rng.gen_range(0..1000) as f64 * 3e-6;
    let start_time = audio_context.current_time() + jitter;

    let env = audio_context.create_gain();
    env.gain().set_value(0.);

    // need to protect my eardrums, VERY VERY LOUD!!
    let volume = audio_context.create_gain();
    volume.gain().set_value(0.001);
    env.connect(&volume);
    volume.connect(&audio_context.destination());

    audio_buffer.connect(&env);

    // ramp
    env.gain().set_value_at_time(0., start_time);
    env.gain()
        .linear_ramp_to_value_at_time(1., start_time + duration / 2.);
    env.gain()
        .linear_ramp_to_value_at_time(0., start_time + duration);

    audio_buffer.seek(position);
    audio_buffer.start_at(start_time);
    // I cannot stop the media element here, so the main() func will disconnect it
}

fn main() {
    let audio_context = AudioContext::new(None);

    println!("++ scrub into file forward and backward at 0.5 speed");

    // grab audio buffer
    let file = File::open("sample.wav").unwrap();
    let stream = WavDecoder::try_new(file).unwrap();
    let media = MediaElement::new(stream);
    let audio_buffer = audio_context.create_media_element_source(media);

    let mut rng = rand::thread_rng();

    let period = 0.05;
    let grain_duration = 0.2;
    let mut position = 0.;
    let mut incr_position = period / 2.;

    loop {
        trigger_grain(
            &audio_context,
            &audio_buffer,
            position,
            grain_duration,
            &mut rng,
        );

        let duration = 3.0f64; // Need to harcode here...
        if position + incr_position > duration - grain_duration || position + incr_position < 0. {
            incr_position *= -1.;
        }

        position += incr_position;

        thread::sleep(time::Duration::from_millis((period * 1000.) as u64));

        // Disconnect media, so the connected gain will drop
        // This differs from granular.rs which will fade out more nicely
        audio_buffer.disconnect_all();
    }
}
