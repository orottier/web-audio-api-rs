// use iai::black_box;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use web_audio_api::context::BaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::AudioNode;
use web_audio_api::node::AudioScheduledSourceNode;

const SAMPLE_RATE: f32 = 48000.;
const DURATION: usize = 10;
const SAMPLES: usize = SAMPLE_RATE as usize * DURATION;

pub fn bench_ctor() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let osc = ctx.create_oscillator();

    osc.connect(&ctx.destination());
    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let osc = ctx.create_oscillator();
    let gain = ctx.create_gain();

    osc.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain_delay() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);

    let osc = ctx.create_oscillator();
    let gain = ctx.create_gain();

    let delay = ctx.create_delay(0.3);
    delay.delay_time().set_value(0.2);

    osc.connect(&delay);
    delay.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);

    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let src = ctx.create_buffer_source();
    src.connect(&ctx.destination());
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_resample(sample_rate: f32) {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), sample_rate);

    let file = include_bytes!("../samples/think-stereo-48000.wav").as_slice();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let src = ctx.create_buffer_source();
    src.connect(&ctx.destination());
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_delay() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);

    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let delay = ctx.create_delay(0.3);
    delay.delay_time().set_value(0.2);

    let src = ctx.create_buffer_source();
    src.set_buffer(buffer);
    src.start();

    src.connect(&delay);
    delay.connect(&ctx.destination());

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_iir() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_biquad() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_positional() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    // Create static panner node
    let panner = ctx.create_panner();
    panner.connect(&ctx.destination());
    panner.position_x().set_value(1.);
    panner.position_y().set_value(2.);
    panner.position_z().set_value(3.);
    panner.orientation_x().set_value(1.);
    panner.orientation_y().set_value(2.);
    panner.orientation_z().set_value(3.);

    // Play buffer and pipe to filter
    let src = ctx.create_buffer_source();
    src.connect(&panner);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_panning_automation() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let panner = ctx.create_stereo_panner();
    panner.connect(&ctx.destination());
    panner.pan().set_value_at_time(-1., 0.);
    panner.pan().set_value_at_time(0.2, 0.5);

    let src = ctx.create_buffer_source();
    src.connect(&panner);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

// This only benchmarks the render thread filling the analyser buffers.
// We don't request freq/time data because that happens off thread and there is no sensible way to
// benchmark this in deterministic way [citation needed].
pub fn bench_analyser_node() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let file = std::fs::File::open("samples/think-stereo-48000.wav").unwrap();
    let buffer = ctx.decode_audio_data_sync(file).unwrap();

    let analyser = ctx.create_analyser();
    analyser.connect(&ctx.destination());

    let src = ctx.create_buffer_source();
    src.connect(&analyser);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

/*
iai::main!(
    bench_ctor,
    bench_sine,
    bench_sine_gain,
    bench_sine_gain_delay,
    bench_buffer_src,
    bench_buffer_src_delay,
    bench_buffer_src_iir,
    bench_buffer_src_biquad,
    bench_stereo_positional,
    bench_stereo_panning_automation,
    bench_analyser_node,
);
*/

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("buffer src resample", |b| {
        b.iter(|| bench_buffer_src_resample(black_box(44100.)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
