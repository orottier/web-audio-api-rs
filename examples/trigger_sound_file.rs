use std::rc::Rc;
use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode};

// experimental API
use web_audio_api::audio_buffer::{AudioBuffer, decode_audio_data};


fn main() {
  let context = AudioContext::new(None);
  // @note - `resume` does not seem to be needed

  // load and decode buffer
  let file = std::fs::File::open("sample.wav").unwrap();
  let audio_buffer = decode_audio_data(file);

  for i in 0..9 {
    // let offset = i as f64 / 1000.; // 10ms - look ok in --release
    let offset = i as f64 / 2.;

    let gain = if i % 4 == 0 { 1. } else { 0.3 };
    let env = context.create_gain();
    env.gain().set_value(gain);
    env.connect(&context.destination());

    let mut src = context.create_buffer_source();
    src.set_buffer(&audio_buffer);
    src.connect(&env);
    src.start_at(context.current_time() + offset);
  }

  std::thread::sleep(std::time::Duration::from_secs(8));
}
