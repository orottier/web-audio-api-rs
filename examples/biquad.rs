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
    // register as media element in the audio context
    let background = context.create_media_element_source(media);

    // use a gain node to control volume
    let gain = context.create_gain();
    // play at low volume
    gain.gain().set_value(0.5);

    // create a biquad filter
    let biquad = context.create_biquad_filter();

    // connect the media node to the gain node
    background.connect(&gain);
    // connect the gain node to the biquad node
    gain.connect(&biquad);
    // connect the biquad node to the destination node (speakers)
    biquad.connect(&context.destination());

    let frequency_hz = [500.0, 1000.];
    let mut mag_response = [0.; 2];
    let mut phase_response = [0.; 2];

    biquad.get_frequency_response(&frequency_hz, &mut mag_response, &mut phase_response);

    println!("{:?}", mag_response);

    // start playback
    background.set_loop(true);
    background.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
