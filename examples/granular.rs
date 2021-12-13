use rand::rngs::ThreadRng;
use rand::Rng;
use std::fs::File;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode};
use web_audio_api::audio_buffer::{AudioBuffer};

// experimental API
use web_audio_api::audio_buffer::{decode_audio_data};

// run in release mode
// cargo run --release --example granular

fn trigger_grain(
  audio_context: &AudioContext,
  audio_buffer: &AudioBuffer,
  position: f64,
  duration: f64,
  rng: &mut ThreadRng
) {
    // don't the precision of sleep millis, but we add a small random offset
    // to avoid audible pitch due to period
    let rand = rng.gen_range(0..1000) as f64;
    let jitter = rand * 3e-6;
    // println§

    let start_time = audio_context.current_time() + jitter;
    let grain_duration = 0.1;

    let env = audio_context.create_gain();
    env.gain().set_value(0.);
    env.connect(&audio_context.destination());

    let mut src = audio_context.create_buffer_source();
    src.set_buffer(&audio_buffer);
    src.connect(&env);

    // ramp
    env.gain().set_value_at_time(0., start_time);
    env.gain().linear_ramp_to_value_at_time(1., start_time + grain_duration / 2.);
    env.gain().linear_ramp_to_value_at_time(0., start_time + grain_duration);

    src.start_at_with_offset_and_duration(start_time, position, duration);
    src.stop_at(start_time + grain_duration);
}

fn main() {
    let audio_context = AudioContext::new(None);

    // grab audio buffer
    let file = File::open("sample.wav").unwrap();
    let audio_buffer = decode_audio_data(file);

    let mut rng = rand::thread_rng();

    // let period_ms = 50;
    let period = 0.05;
    let grain_duration = 0.2;
    let mut position = 0.;
    let mut incr_position = period / 2.;

    // should probably be more robust with a proper scheduler
    loop {
        // scrub forward and backward into buffer
        trigger_grain(
          &audio_context,
          &audio_buffer,
          position,
          grain_duration,
          &mut rng
        );

        // update position
        if position + incr_position > audio_buffer.duration() - grain_duration
          || position + incr_position < 0.
        {
          incr_position *= -1.;
        }

        position += incr_position;

        thread::sleep(time::Duration::from_millis((period * 1000.) as u64));
    }
}
