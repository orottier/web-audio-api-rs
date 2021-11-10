use std::f32::consts::PI;
use std::fs::File;
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::media::{MediaElement, OggVorbisDecoder};
use web_audio_api::node::{
    AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode, OverSampleType,
    WaveShaperNode, WaveShaperOptions,
};

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

    // setup background music:
    // read from local file
    let file = File::open("sample.ogg").unwrap();
    // decode file to media stream
    let stream = OggVorbisDecoder::try_new(file).unwrap();
    // wrap stream in MediaElement, so we can control it (loop, play/pause)
    let media = MediaElement::new(stream);
    // register as media element in the audio context
    let background = context.create_media_element_source(media);

    // use a gain node to control volume
    let gain = context.create_gain();
    // play at low volume
    gain.gain().set_value(0.5);

    // Create the distortion curve
    let curve = make_distortion_curve(40);

    // Create wave shaper options
    let options = WaveShaperOptions {
        curve: Some(curve),
        ..Default::default()
    };
    // Create the waveshaper
    let mut shaper = WaveShaperNode::new(&context, Some(options));

    shaper.set_oversample(OverSampleType::None);

    // connect the media node to the gain node
    background.connect(&gain);
    // connect the gain node to the shaper node
    gain.connect(&shaper);
    // connect the shaper node to the destination node (speakers)
    shaper.connect(&context.destination());
    // start playback
    background.set_loop(true);
    background.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
