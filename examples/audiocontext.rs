use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::default();

    let mic = Microphone::default();
    // register as media element in the audio context
    let background = context.create_media_stream_source(mic.stream());
    // connect the node to the destination node (speakers)
    background.connect(&context.destination());

    println!("Playback for 2 seconds");
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Pause context for 2 seconds");
    println!("Context state before suspend - {:?}", context.state());
    context.suspend_sync();
    println!("Context state after suspend - {:?}", context.state());
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Resume context for 2 seconds");
    println!("Context state before resume - {:?}", context.state());
    context.resume_sync();
    println!("Context state after resume - {:?}", context.state());
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Closing the mic should halt the media stream source
    println!("Close context");
    println!("Context state before close - {:?}", context.state());
    context.close_sync();
    println!("Context state after close - {:?}", context.state());

    std::thread::sleep(std::time::Duration::from_secs(2));
}
