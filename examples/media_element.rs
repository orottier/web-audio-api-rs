use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::default();
    let mut media = MediaElement::new("samples/large-file.ogg");

    let src = context.create_media_element_source(&mut media);
    src.connect(&context.destination());

    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
