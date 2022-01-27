use std::f32::consts::PI;
use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, OverSampleType};

// use part of cosine, between [π, 2π] as shaping cureve
fn make_distortion_curve(size: usize) -> Vec<f32> {
    let mut curve = vec![0.; size];
    let mut phase = 0.;
    let phase_incr = PI / (size - 1) as f32;

    for s in curve.iter_mut() {
        *s = (PI + phase).cos();
        phase += phase_incr;
    }

    curve
}

fn main() {
    env_logger::init();

    println!("> gradually increase the amount of distortion applied on the sample");

    let context = AudioContext::new(None);

    let file = File::open("samples/sample.wav").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();
    let curve = make_distortion_curve(2048);

    let post_gain = context.create_gain();
    post_gain.connect(&context.destination());
    post_gain.gain().set_value(0.);

    let shaper = context.create_wave_shaper();
    shaper.set_oversample(OverSampleType::None);
    // shaper.set_oversample(OverSampleType::X2);
    // shaper.set_oversample(OverSampleType::X4);
    shaper.connect(&post_gain);
    shaper.set_curve(curve);

    let pre_gain = context.create_gain();
    pre_gain.connect(&shaper);
    pre_gain.gain().set_value(0.);

    for i in 1..10 {
        let gain = i as f32 * 2.;
        println!("+ pre gain: {:?}", gain);

        pre_gain.gain().set_value(gain);
        post_gain.gain().set_value(1. / gain);

        let src = context.create_buffer_source();
        src.connect(&pre_gain);
        src.set_buffer(buffer.clone());
        src.start();

        // enjoy listening
        std::thread::sleep(std::time::Duration::from_secs(4));
    }
}
