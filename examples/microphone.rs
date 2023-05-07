use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media_devices;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();
    let context = AudioContext::default();
    let mic = media_devices::get_user_media();

    // create media stream source node with mic stream
    let stream_source = context.create_media_stream_source(&mic);
    stream_source.connect(&context.destination());

    println!("Recording for 5 seconds");
    std::thread::sleep(std::time::Duration::from_secs(5));

    println!("Closing microphone");
    mic.get_tracks()[0].close();
    std::thread::sleep(std::time::Duration::from_secs(2));
}
