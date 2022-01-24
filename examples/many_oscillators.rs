use std::{thread, time};
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn trigger_sine(audio_context: &AudioContext) {
    let osc = audio_context.create_oscillator();
    osc.connect(&audio_context.destination());

    let now = audio_context.current_time();
    osc.start_at(now);
    osc.stop_at(now + 0.03)
}

fn main() {
    let audio_context = AudioContext::new(None);

    // mimic setInterval
    loop {
        trigger_sine(&audio_context);
        thread::sleep(time::Duration::from_millis(50));
    }
}
