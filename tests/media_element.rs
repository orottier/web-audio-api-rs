use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::AudioNode;
use web_audio_api::MediaElement;

#[test]
fn test_media_element_source_progress() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    let mut media = MediaElement::new("samples/major-scale.ogg").unwrap();
    media.set_loop(true);

    let src = context.create_media_element_source(&mut media);
    src.connect(&context.destination());
    media.play();

    // TODO improve test setup of online AudioContext
    // <https://github.com/orottier/web-audio-api-rs/issues/323>
    // Sleep for a bit, make sure the audio thread has started
    std::thread::sleep(std::time::Duration::from_millis(100));

    // assert the media has progressed
    assert!(media.current_time() > 0.05);
}
