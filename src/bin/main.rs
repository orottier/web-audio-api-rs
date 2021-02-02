use web_audio_api::context::AudioContext;
use web_audio_api::node::AudioNode;

fn main() {
    let context = AudioContext::new();

    let osc = context.create_oscillator();
    osc.connect(&context.destination());
    //osc.start();

    let osc2 = context.create_oscillator();
    osc2.set_frequency(445);
    osc2.connect(&context.destination());
    //osc2.start();

    std::thread::sleep(std::time::Duration::from_secs(2));

    osc.set_frequency(1024);

    std::thread::sleep(std::time::Duration::from_secs(2));
}
