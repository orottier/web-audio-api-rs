use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media_devices;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();
    let context = AudioContext::default();

    let mic = media_devices::get_user_media();
    // register as media element in the audio context
    let background = context.create_media_stream_source(&mic);
    // connect the node to the destination node (speakers)
    background.connect(&context.destination());

    // The Microphone will continue to run when either,
    // - the struct is still alive in the control thread
    // - the media stream is active in the render thread
    //
    // Let's drop it from the control thread so it's lifetime is bound by the render thread
    drop(mic);

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
