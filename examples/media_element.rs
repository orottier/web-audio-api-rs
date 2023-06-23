use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::AudioNode;
use web_audio_api::MediaElement;

// Playback audio file with a MediaElement
//
// `cargo run --release --example media_element`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example media_element`
fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

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

    println!("Playback rate 1.25");
    media.set_playback_rate(1.25);

    std::thread::sleep(std::time::Duration::from_millis(3000));
    println!("Current time is now {}", media.current_time());
    println!("Pause");
    media.pause();
    std::thread::sleep(std::time::Duration::from_millis(1000));

    assert!(media.paused());
    assert!(media.loop_());

    println!("Play");
    media.play();
    println!("Current time is now {}", media.current_time());

    loop {
        std::thread::sleep(std::time::Duration::from_millis(3000));
        assert!(!media.paused());
        println!("Current time is now {}", media.current_time());
    }
}
