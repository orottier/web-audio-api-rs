use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media_devices;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();
    let context = AudioContext::default();

    // create media stream source node with mic stream
    let mic = media_devices::get_user_media();
    let stream_source = context.create_media_stream_source(&mic);
    stream_source.connect(&context.destination());

    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    // TODO implement controls on the microphone stream

    // println!("Playback for 2 seconds");
    // std::thread::sleep(std::time::Duration::from_secs(2));

    // println!("Pause mic for 2 seconds");
    // mic.suspend();
    // std::thread::sleep(std::time::Duration::from_secs(2));

    // println!("Resume mic for 2 seconds");
    // mic.resume();
    // std::thread::sleep(std::time::Duration::from_secs(2));

    // // Closing the mic should halt the media stream source
    // println!("Close mic - halting stream");
    // mic.close();
}
