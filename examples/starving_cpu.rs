use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorNode};

fn main() {
    env_logger::init();

    let context = AudioContext::default();

    let mut osc: Option<OscillatorNode> = None;

    loop {
        if let Some(prev_osc) = osc {
            prev_osc.disconnect();
        }

        let mut new_osc = context.create_oscillator();
        new_osc.connect(&context.destination());
        new_osc.start();

        osc = Some(new_osc);

        // reduce sleep duration to make the effect faster
        // ...I personally like the result but not sure this is what we expect :)
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}
