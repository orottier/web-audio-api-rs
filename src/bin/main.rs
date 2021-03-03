use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::media::{MediaElement, OggVorbisDecoder};
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::new();

    // setup background music:
    // read from local file
    let file = File::open("sample.ogg").unwrap();
    // decode file to media stream
    let stream = OggVorbisDecoder::try_new(file).unwrap();
    // wrap stream in MediaElement, so we can control it (loop, play/pause)
    let media = MediaElement::new(stream);
    media.set_loop(true);
    media.set_loop_start(1.);
    media.set_loop_end(2.);
    // register as media element in the audio context
    let background = context.create_media_element_source(media);
    // use a gain node to control volume
    let gain = context.create_gain();
    // play at low volume
    gain.gain().set_value(0.5);
    // connect the media node to the gain node
    background.connect(&gain);
    // connect the gain node to the destination node (speakers)
    gain.connect(&context.destination());
    // start playback
    background.start();

    // mix in an oscillator sound
    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    osc.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
