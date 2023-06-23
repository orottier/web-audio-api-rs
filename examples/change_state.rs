use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::media_devices;
use web_audio_api::media_devices::MediaStreamConstraints;
use web_audio_api::node::AudioNode;

// Changing state of an AudioContext (resume, suspend, close)
//
// `cargo run --release --example change_state`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example change_state`
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

    let stream = media_devices::get_user_media_sync(MediaStreamConstraints::Audio);
    // register as media element in the audio context
    let stream_source = context.create_media_stream_source(&stream);
    // connect the node to the destination node (speakers)
    stream_source.connect(&context.destination());

    // The Microphone will continue to run when either,
    // - the struct is still alive in the control thread
    // - the media stream is active in the render thread
    //
    // Let's drop it from the control thread so it's lifetime is bound by the render thread
    drop(stream);

    println!("Playback for 2 seconds");
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Pause context for 2 seconds - the mic will be left running");
    println!("Context state before suspend - {:?}", context.state());
    context.suspend_sync();
    println!("Context state after suspend - {:?}", context.state());
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Resume context for 2 seconds");
    println!("Context state before resume - {:?}", context.state());
    context.resume_sync();
    println!("Context state after resume - {:?}", context.state());
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Closing the context should halt the media stream source
    println!("Close context");
    println!("Context state before close - {:?}", context.state());
    context.close_sync();
    println!("Context state after close - {:?}", context.state());

    std::thread::sleep(std::time::Duration::from_secs(2));
}
