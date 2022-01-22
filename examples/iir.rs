use std::fs::File;
use web_audio_api::context::{AudioContext, Context};
use web_audio_api::media::{MediaDecoder, MediaElement};
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::new(None);

    // setup background music:
    // read from local file
    let file = File::open("samples/major-scale.ogg").unwrap();
    // decode file to media stream
    let stream = MediaDecoder::try_new(file).unwrap();
    // wrap stream in MediaElement, so we can control it (loop, play/pause)
    let media = MediaElement::new(stream);
    // register as media element in the audio context
    let background = context.create_media_element_source(media);
    // start playback
    background.set_loop(true);
    background.start();

    let feedforward = vec![
        0.000016636797512844526,
        0.00003327359502568905,
        0.000016636797512844526,
    ];
    let feedback = vec![1.0, -1.9884300106225539, 0.9884965578126054];

    // Create an IIR filter node
    let iir = context.create_iir_filter(feedforward, feedback);

    // Connect background sound to IIR filter
    background.connect(&iir);
    // Connect IIR filter to the speakers
    iir.connect(&context.destination());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
