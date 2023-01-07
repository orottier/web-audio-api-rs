use std::fs::File;

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    let audio_context = AudioContext::default();

    let file = File::open("samples/sample.wav").unwrap();
    let buffer = audio_context.decode_audio_data_sync(file).unwrap();

    let src = audio_context.create_buffer_source();
    src.connect(&audio_context.destination());
    src.set_buffer(buffer);

    // @todo - should receive an event
    src.onended(Some(|| {
        println!("> Ended event triggered!");
    }));

    let now = audio_context.current_time();
    src.start_at(now);
    src.stop_at(now + 1.);

    std::thread::sleep(std::time::Duration::from_secs(4));
}
