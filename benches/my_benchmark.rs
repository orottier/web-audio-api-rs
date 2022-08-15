use iai::black_box;

use web_audio_api::context::BaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::AudioNode;
use web_audio_api::node::AudioScheduledSourceNode;

pub fn bench_ctor() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);
    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

pub fn bench_sine() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);
    let osc = ctx.create_oscillator();

    osc.connect(&ctx.destination());
    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

pub fn bench_sine_gain() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);
    let osc = ctx.create_oscillator();
    let gain = ctx.create_gain();

    osc.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

pub fn bench_sine_gain_delay() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);

    let osc = ctx.create_oscillator();
    let gain = ctx.create_gain();

    let delay = ctx.create_delay(0.3);
    delay.delay_time().set_value(0.2);

    osc.connect(&delay);
    delay.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

iai::main!(
    bench_ctor,
    bench_sine,
    bench_sine_gain,
    bench_sine_gain_delay
);
