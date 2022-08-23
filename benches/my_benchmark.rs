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

pub fn bench_buffer_src() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);

    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let src = ctx.create_buffer_source();
    src.connect(&ctx.destination());
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

pub fn bench_buffer_src_iir() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);
    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    // these values correspond to a lowpass filter at 200Hz (calculated from biquad)
    let feedforward = vec![
        0.0002029799640409502,
        0.0004059599280819004,
        0.0002029799640409502,
    ];

    let feedback = vec![1.0126964557853775, -1.9991880801438362, 0.9873035442146225];

    // Create an IIR filter node
    let iir = ctx.create_iir_filter(feedforward, feedback);
    iir.connect(&ctx.destination());

    // Play buffer and pipe to filter
    let src = ctx.create_buffer_source();
    src.connect(&iir);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

pub fn bench_buffer_src_biquad() {
    let ctx = OfflineAudioContext::new(2, black_box(48000), 48000.);
    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    // Create an biquad filter node (defaults to low pass)
    let biquad = ctx.create_biquad_filter();
    biquad.connect(&ctx.destination());
    biquad.frequency().set_value(200.);

    // Play buffer and pipe to filter
    let src = ctx.create_buffer_source();
    src.connect(&biquad);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), 48000);
}

iai::main!(
    bench_ctor,
    bench_sine,
    bench_sine_gain,
    bench_sine_gain_delay,
    bench_buffer_src,
    bench_buffer_src_iir,
);
