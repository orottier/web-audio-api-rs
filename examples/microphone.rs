use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::AudioNode;

fn main() {
    env_logger::init();
    let context = AudioContext::new();

    let stream = Microphone::new();
    // register as media element in the audio context
    let background = context.create_media_stream_source(stream);
    // connect the node to the destination node (speakers)
    background.connect(&context.destination());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
