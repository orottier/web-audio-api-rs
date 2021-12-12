use std::rc::Rc;
use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode};

// experimental API
use web_audio_api::audio_buffer::{AudioBuffer, decode_audio_data};


fn main() {
  let context = AudioContext::new(None);

  // load and decode buffer
  let file = std::fs::File::open("sample.wav").unwrap();
  let audio_buffer = decode_audio_data(file);

  let mut src = context.create_buffer_source();
  src.set_buffer(&audio_buffer);
  src.connect(&context.destination());
  src.start();

  // context.start_rendering();
  std::thread::sleep(std::time::Duration::from_secs(5));
}
