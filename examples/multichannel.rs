use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ChannelInterpretation};

// Example of multichannel routing, for now the library can only handle up to
// 32 channels.
//
// The example can be tested with a virtual soundcard such as Blackhole
// https://github.com/ExistentialAudio/BlackHole
// - select backhole as the default system output
// - then use blackhole as input in another program to check the example output
// (see `multichannel.maxpat` if you have Max installed, @todo make a Pd version)
//
// `cargo run --release --example multichannel`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example multichannel`
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

    // this should be clamped to MAX_CHANNELS (32), even if the soundcard can provide more channels
    println!(
        "> Max channel count: {:?}",
        context.destination().max_channel_count()
    );

    let num_channels = context.destination().max_channel_count();

    context.destination().set_channel_count(num_channels);
    context
        .destination()
        .set_channel_interpretation(ChannelInterpretation::Discrete);

    let merger = context.create_channel_merger(num_channels);
    merger.set_channel_interpretation(ChannelInterpretation::Discrete);
    merger.connect(&context.destination());

    let mut output_channel = 0;

    loop {
        println!("- output sine in channel {:?}", output_channel);

        let now = context.current_time();

        let mut osc = context.create_oscillator();
        osc.connect_from_output_to_input(&merger, 0, output_channel);
        osc.frequency().set_value(200.);
        osc.start_at(now);
        osc.stop_at(now + 1.);

        output_channel = (output_channel + 1) % num_channels;

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
