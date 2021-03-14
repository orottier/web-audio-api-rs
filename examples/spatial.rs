use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioScheduledSourceNode;
use web_audio_api::node::{AudioNode, PannerNode, PannerOptions};

fn main() {
    let context = AudioContext::new();

    let tone = context.create_oscillator();
    tone.frequency().set_value_at_time(300.0f32, 0.);
    tone.start();

    let panner = context.create_panner();
    panner.position_y().set_value_at_time(4., 0.);
    tone.connect(&panner);
    panner.connect(&context.destination());

    let moving = context.create_oscillator();
    moving.start();
    moving.frequency().set_value_at_time(0.2, 0.);
    let gain = context.create_gain();
    gain.gain().set_value_at_time(4., 0.);
    moving.connect(&gain);
    gain.connect(panner.position_x());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
