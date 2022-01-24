use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::{MediaDecoder, MediaElement};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();

    // construct the decoder
    let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
    let media = MediaDecoder::try_new(file).unwrap();

    // Wrap in a `MediaElement` so buffering/decoding does not take place on the render thread
    let element = MediaElement::new(media);

    // register the media element node
    let context = AudioContext::new(None);
    let node = context.create_media_element_source(element);

    // play media
    node.connect(&context.destination());
    node.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
