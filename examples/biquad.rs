use std::fs::File;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::{MediaDecoder, MediaElement};
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};

fn main() {
    env_logger::init();
    let context = AudioContext::new(None);

    // setup background music:
    // read from local file
    let file = File::open("samples/major-scale.ogg").unwrap();
    // decode file to media stream
    let stream = MediaDecoder::try_new(file).unwrap();
    // wrap stream in MediaElement, so we can control it (loop, play/pause)
    let media = MediaElement::new(stream);
    // register as media element in the audio context
    let background = context.create_media_element_source(media);
    // create a biquad filter
    let biquad = context.create_biquad_filter();
    // connect the media node to the gain node
    background.connect(&biquad);
    // connect the biquad node to the destination node (speakers)
    biquad.connect(&context.destination());

    let mut frequency_hz = [250., 500.0, 750.0, 1000., 1500.0, 2000.0, 4000.0];
    let mut mag_response = [0.; 7];
    let mut phase_response = [0.; 7];

    biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);

    println!("=================================");
    println!("Biquad filter frequency response:");
    println!("=================================");
    println!("Cutoff freq -- {} Hz", biquad.frequency().value());
    println!("Gain -- {}", biquad.gain().value());
    println!("Q factor -- {}", biquad.q().value());
    println!("---------------------------------");
    for i in 0..frequency_hz.len() {
        println!(
            "{} Hz --> {} dB",
            frequency_hz[i],
            20.0 * mag_response[i].log10()
        );
    }
    println!("---------------------------------");

    // start playback
    background.set_loop(true);
    background.start();

    biquad.frequency().set_value(125.);
    biquad.frequency().linear_ramp_to_value_at_time(3_000., 4.);
    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));

    biquad.get_frequency_response(&mut frequency_hz, &mut mag_response, &mut phase_response);
    println!("=================================");
    println!("Biquad filter frequency response:");
    println!("=================================");
    println!("Cutoff freq -- {} Hz", biquad.frequency().value());
    println!("Gain -- {}", biquad.gain().value());
    println!("Q factor -- {}", biquad.q().value());
    println!("---------------------------------");
    for i in 0..frequency_hz.len() {
        println!(
            "{} Hz --> {} dB",
            frequency_hz[i],
            20.0 * mag_response[i].log10()
        );
    }
    println!("---------------------------------");
}
