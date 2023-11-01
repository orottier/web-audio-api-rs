use std::fs::File;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, DistanceModelType, PannerNode, PannerOptions,
    PanningModelType,
};

// This example feature a 'true physics' Doppler effect.
//
// The basics are very simple, we just add a DelayNode that represents the finite speed of sound.
// Speed of sound = 343 m/s
// So a siren at 100 meters away from you will have a delay of 0.3 seconds. A siren near you
// obviously has no delay.
//
// We combine a delay node with a panner node that represents the moving siren. When the panner
// node moves closer to the listener, we decrease the delay time linearly. This gives the Doppler
// effect.
//
// `cargo run --release --example doppler`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example doppler`
fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let file = File::open("samples/siren.mp3").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();

    let mut src = context.create_buffer_source();
    src.set_buffer(buffer);
    src.set_loop(true);

    let opts = PannerOptions {
        panning_model: PanningModelType::EqualPower,
        distance_model: DistanceModelType::Inverse,
        position_x: 0.,
        position_y: 100., // siren starts 100 meters away
        position_z: 1.,   // we stand 1 meter away from the road
        orientation_x: 1.,
        orientation_y: 0.,
        orientation_z: 0.,
        ref_distance: 1.,
        max_distance: 10000.,
        rolloff_factor: 1.,
        // no cone effect:
        cone_inner_angle: 360.,
        cone_outer_angle: 0.,
        cone_outer_gain: 0.,
        ..PannerOptions::default()
    };
    let panner = PannerNode::new(&context, opts);
    // move the siren in 10 seconds from y = 100 to y = -100
    panner.position_y().linear_ramp_to_value_at_time(-100., 10.);

    // The delay starts with value 0.3, reaches 0 when the siren crosses us, then goes back to 0.3
    let delay = context.create_delay(1.);
    let doppler_max = 100. / 343.;
    delay.delay_time().set_value_at_time(doppler_max, 0.);
    delay.delay_time().linear_ramp_to_value_at_time(0., 5.);
    delay
        .delay_time()
        .linear_ramp_to_value_at_time(doppler_max, 10.);

    src.connect(&delay);
    delay.connect(&panner);
    panner.connect(&context.destination());
    src.start();

    std::thread::sleep(std::time::Duration::from_millis(10_000));
}
