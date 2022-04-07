use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();
    let context = AudioContext::default();

    let mic = Microphone::new();
    // register as media element in the audio context
    let background = context.create_media_stream_source(mic.stream());
    // connect the node to the destination node (speakers)
    background.connect(&context.destination());

    println!("Playback for 2 seconds");
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Pause mic for 2 seconds");
    mic.suspend();
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Resume mic for 2 seconds");
    mic.resume();
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Microphone is allowed to go out of scope, recording should continue
    println!("Drop mic and resume for 4 seconds");
    drop(mic);
    std::thread::sleep(std::time::Duration::from_secs(4));
}
