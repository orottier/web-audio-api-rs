use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::{MediaDecoder, MediaElement};
use web_audio_api::node::{AudioNode, MediaElementAudioSourceNode, MediaElementAudioSourceOptions};

fn main() {
    env_logger::init();

    // construct the decoder
    let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
    let stream = MediaDecoder::try_new(file).unwrap();
    // Wrap in a `MediaElement` so buffering/decoding does not take place on the render thread
    let media_element = MediaElement::new(stream);

    // register the media element node
    let context = AudioContext::new(None);

    let node = MediaElementAudioSourceNode::new(&context, MediaElementAudioSourceOptions {
        media_element
    });

    node.connect(&context.destination());
    // play media
    media_element.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
