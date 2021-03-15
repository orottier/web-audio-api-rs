use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioNode;
use web_audio_api::node::AudioScheduledSourceNode;

fn main() {
    let context = AudioContext::new();

    // Create a friendly tone
    let tone = context.create_oscillator();
    tone.frequency().set_value_at_time(300.0f32, 0.);
    tone.start();

    // Connect tone > panner node > destination node
    let panner = context.create_panner();
    tone.connect(&panner);
    panner.connect(&context.destination());

    // The panner node is 1 unit in front of listener
    panner.position_z().set_value_at_time(1., 0.);

    // And sweeps 10 units left to right, every second
    let moving = context.create_oscillator();
    moving.start();
    moving.frequency().set_value_at_time(1., 0.);
    let gain = context.create_gain();
    gain.gain().set_value_at_time(10., 0.);
    moving.connect(&gain);
    gain.connect(panner.position_x());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
