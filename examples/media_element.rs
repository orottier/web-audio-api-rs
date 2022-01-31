use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::{MediaDecoder, MediaElement};
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();

    // build a decoded audio stream the decoder
    let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
    let stream = MediaDecoder::try_new(file).unwrap();
    // wrap in a `MediaElement` (buffering/decoding does not take place on the render thread)
    let media_element = MediaElement::new(stream);
    // pipe the media element into the web audio graph
    let context = AudioContext::new(None);
    let node = context.create_media_element_source(&media_element);
    node.connect(&context.destination());
    // start media playback
    media_element.start();
    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
