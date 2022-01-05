use std::f32::consts::PI;
use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::AudioNode;

/// Creates a sigmoid curve
/// see [curve graph](http://kevincennis.github.io/transfergraph/)
///
/// # Arguments
///
/// * `amount` - the steepness of the sigmoid curve (try value between 0-100)
fn make_distortion_curve(amount: usize) -> Vec<f32> {
    let n = 100;
    let mut curve = vec![0.; n];
    let deg = PI / 180.;
    for (i, c) in curve.iter_mut().enumerate() {
        let x = i as f32 * 2. / n as f32 - 1.;
        *c = (3.0 + amount as f32) * x * 20. * deg / (PI + amount as f32 * x.abs());
    }
    curve
}

fn main() {
    env_logger::init();
    let context = AudioContext::new(None);

    let file = File::open("sample.wav").unwrap();
    let buffer = context.decode_audio_data(file).unwrap();

    let shaper = context.create_wave_shaper();
    let curve = make_distortion_curve(40);
    shaper.connect(&context.destination());
    shaper.set_curve(curve);

    let src = context.create_buffer_source();
    src.connect(&shaper);
    src.set_buffer(buffer);

    src.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(5));
}
