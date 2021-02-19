use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::media::OggVorbisDecoder;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::new();

    // play background music
    let file = File::open("sample.ogg").unwrap();
    let media = OggVorbisDecoder::try_new(file).unwrap();
    let background = context.create_media_element_source(media);
    let gain = context.create_gain();
    gain.gain().set_value(0.5); // play at low volume
    background.connect(&gain);
    gain.connect(&context.destination());

    // mix in an oscillator sound
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
