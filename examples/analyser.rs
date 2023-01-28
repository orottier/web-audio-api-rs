use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

fn main() {
    let context = AudioContext::default();

    let analyser = context.create_analyser();
    analyser.connect(&context.destination());

    let osc = context.create_oscillator();
    osc.frequency().set_value(200.);
    osc.connect(&analyser);
    osc.start();

    let mut bins = vec![0.; analyser.frequency_bin_count()];

    loop {
        analyser.get_float_frequency_data(&mut bins);
        println!("{:?}", &bins[0..20]); // print 20 first bins
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
