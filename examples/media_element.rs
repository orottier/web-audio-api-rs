use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::default();
    let mut media = MediaElement::new("samples/major-scale.ogg");
    media.set_loop(true);

    let src = context.create_media_element_source(&mut media);
    println!("Start playing");
    src.connect(&context.destination());

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Seek to frame 100_000");
    media.seek(100_000);

    loop {
        std::thread::sleep(std::time::Duration::from_millis(3000));
    }
}
