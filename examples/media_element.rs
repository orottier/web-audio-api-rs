use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::default();
    let mut media = MediaElement::new("samples/major-scale.ogg").unwrap();
    media.set_loop(true);

    let src = context.create_media_element_source(&mut media);
    src.connect(&context.destination());
    println!("Media Element ready");

    println!("Start playing");
    media.play();

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Current time is now {}", media.current_time());
    println!("Seek to 1 second");
    media.set_current_time(1.);

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Current time is now {}", media.current_time());
    println!("Pause");
    media.pause();
    std::thread::sleep(std::time::Duration::from_millis(1000));
    println!("Play");
    media.play();
    println!("Current time is now {}", media.current_time());

    loop {
        std::thread::sleep(std::time::Duration::from_millis(3000));
        println!("Current time is now {}", media.current_time());
    }
}
