use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    println!("AudioContextLatencyCategory::Interactive");
    let context = AudioContext::default();

    let sine = context.create_oscillator();
    sine.frequency().set_value(200.);
    sine.connect(&context.destination());

    sine.start();

    println!("- BaseLatency: {:?}", context.base_latency());

    loop {
        println!("-------------------------------------------------");
        println!("+ currentTime {:?}", context.current_time());
        println!("+ OutputLatency: {:?}", context.output_latency());

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
