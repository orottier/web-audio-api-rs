use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();

    let context = AudioContext::default();
    let mic = Microphone::default();

    // create media stream source node with mic stream
    let stream_source = context.create_media_stream_source(mic.stream());
    stream_source.connect(&context.destination());

    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    // note: uncomment to test controls over the mic instance,
    // this is maybe not the desired public interface, see MediaDevices API

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
