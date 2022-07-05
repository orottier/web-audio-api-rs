use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::default();
    let mut media = MediaElement::new("samples/major-scale.ogg");
    media.set_loop(true);

    let src = context.create_media_element_source(&mut media);
    src.connect(&context.destination());
    println!("Media Element ready");

    println!("Start playing");
    media.play();

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Seek to frame 100_000");
    media.seek(100_000);

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Pause");
    media.pause();
    std::thread::sleep(std::time::Duration::from_millis(1000));
    println!("Play");
    media.play();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(3000));
    }
}
