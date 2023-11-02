#[cfg(feature = "iai")]
use iai::black_box;

#[cfg(not(feature = "iai"))]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use web_audio_api::context::BaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, PanningModelType};
use web_audio_api::AudioBuffer;

use std::fs::File;
use std::sync::OnceLock;

const SAMPLE_RATE: f32 = 48000.;
const DURATION: usize = 10;
const SAMPLES: usize = SAMPLE_RATE as usize * DURATION;
const SAMPLES_SHORT: usize = SAMPLE_RATE as usize; // only 1 second for heavy benchmarks

/// Load an audio buffer and cache the result
///
/// We don't want to measure the IO and decoding in most of our benchmarks, so by using this static
/// instance we avoid this in the criterion benchmarks because the file is already loaded in the
/// warmup phase.
fn get_audio_buffer(ctx: &OfflineAudioContext) -> AudioBuffer {
    static BUFFER: OnceLock<AudioBuffer> = OnceLock::new();
    BUFFER
        .get_or_init(|| {
            let file = File::open("samples/think-stereo-48000.wav").unwrap();
            ctx.decode_audio_data_sync(file).unwrap()
        })
        .clone()
}

pub fn bench_ctor() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

// This benchmark only makes sense in `iai`, because subsequent runs use the cached audiobuffer.
// However we would like to run this test here so the cache is filled for the subsequent benches.
pub fn bench_audio_buffer_decode() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);
    assert_eq!(buffer.length(), 101129);
}

pub fn bench_sine() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut osc = ctx.create_oscillator();

    osc.connect(&ctx.destination());
    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut osc = ctx.create_oscillator();
    let gain = ctx.create_gain();

    osc.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain_delay() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);

    let mut osc = ctx.create_oscillator();
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
    let buffer = get_audio_buffer(&ctx);

    let mut src = ctx.create_buffer_source();
    src.connect(&ctx.destination());
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_delay() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    let delay = ctx.create_delay(0.3);
    delay.delay_time().set_value(0.2);

    let mut src = ctx.create_buffer_source();
    src.set_buffer(buffer);
    src.start();

    src.connect(&delay);
    delay.connect(&ctx.destination());

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_iir() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

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
    let mut src = ctx.create_buffer_source();
    src.connect(&iir);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_biquad() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    // Create an biquad filter node (defaults to low pass)
    let biquad = ctx.create_biquad_filter();
    biquad.connect(&ctx.destination());
    biquad.frequency().set_value(200.);

    // Play buffer and pipe to filter
    let mut src = ctx.create_buffer_source();
    src.connect(&biquad);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_positional() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

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
    let mut src = ctx.create_buffer_source();
    src.connect(&panner);
    src.set_buffer(buffer);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_panning_automation() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    let panner = ctx.create_stereo_panner();
    panner.connect(&ctx.destination());
    panner.pan().set_value_at_time(-1., 0.);
    panner.pan().set_value_at_time(0.2, 0.5);

    let mut src = ctx.create_buffer_source();
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
    let buffer = get_audio_buffer(&ctx);

    let analyser = ctx.create_analyser();
    analyser.connect(&ctx.destination());

    let mut src = ctx.create_buffer_source();
    src.connect(&analyser);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_hrtf_panners() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES_SHORT), SAMPLE_RATE);

    let mut panner1 = ctx.create_panner();
    panner1.set_panning_model(PanningModelType::HRTF);
    panner1.position_x().set_value(10.0);
    panner1.connect(&ctx.destination());

    let mut panner2 = ctx.create_panner();
    panner2.set_panning_model(PanningModelType::HRTF);
    panner2.position_x().set_value(-10.0);
    panner2.connect(&ctx.destination());

    let mut osc = ctx.create_oscillator();
    osc.connect(&panner1);
    osc.connect(&panner2);
    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES_SHORT);
}

#[cfg(feature = "iai")]
iai::main!(
    bench_ctor,
    bench_audio_buffer_decode,
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
    bench_hrtf_panners,
);

#[cfg(not(feature = "iai"))]
fn criterion_ctor(c: &mut Criterion) {
    c.bench_function("bench_ctor", |b| b.iter(|| bench_ctor()));
}
#[cfg(not(feature = "iai"))]
fn criterion_audio_buffer_decode(c: &mut Criterion) {
    c.bench_function("bench_audio_buffer_decode", |b| {
        b.iter(|| bench_audio_buffer_decode())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_sine(c: &mut Criterion) {
    c.bench_function("bench_sine", |b| b.iter(|| bench_sine()));
}
#[cfg(not(feature = "iai"))]
fn criterion_sine_gain(c: &mut Criterion) {
    c.bench_function("bench_sine_gain", |b| b.iter(|| bench_sine_gain()));
}
#[cfg(not(feature = "iai"))]
fn criterion_sine_gain_delay(c: &mut Criterion) {
    c.bench_function("bench_sine_gain_delay", |b| {
        b.iter(|| bench_sine_gain_delay())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_buffer_src(c: &mut Criterion) {
    c.bench_function("bench_buffer_src", |b| b.iter(|| bench_buffer_src()));
}
#[cfg(not(feature = "iai"))]
fn criterion_buffer_src_delay(c: &mut Criterion) {
    c.bench_function("bench_buffer_src_delay", |b| {
        b.iter(|| bench_buffer_src_delay())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_buffer_src_iir(c: &mut Criterion) {
    c.bench_function("bench_buffer_src_iir", |b| {
        b.iter(|| bench_buffer_src_iir())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_buffer_src_biquad(c: &mut Criterion) {
    c.bench_function("bench_buffer_src_biquad", |b| {
        b.iter(|| bench_buffer_src_biquad())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_stereo_positional(c: &mut Criterion) {
    c.bench_function("bench_stereo_positional", |b| {
        b.iter(|| bench_stereo_positional())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_stereo_panning_automation(c: &mut Criterion) {
    c.bench_function("bench_stereo_panning_automation", |b| {
        b.iter(|| bench_stereo_panning_automation())
    });
}
#[cfg(not(feature = "iai"))]
fn criterion_analyser_node(c: &mut Criterion) {
    c.bench_function("bench_analyser_node", |b| b.iter(|| bench_analyser_node()));
}
#[cfg(not(feature = "iai"))]
fn criterion_hrtf_panners(c: &mut Criterion) {
    c.bench_function("bench_hrtf_panners", |b| b.iter(|| bench_hrtf_panners()));
}

#[cfg(not(feature = "iai"))]
criterion_group!(
    benches,
    criterion_ctor,
    criterion_audio_buffer_decode,
    criterion_sine,
    criterion_sine_gain,
    criterion_sine_gain_delay,
    criterion_buffer_src,
    criterion_buffer_src_delay,
    criterion_buffer_src_iir,
    criterion_buffer_src_biquad,
    criterion_stereo_positional,
    criterion_stereo_panning_automation,
    criterion_analyser_node,
    criterion_hrtf_panners
);

#[cfg(not(feature = "iai"))]
criterion_main!(benches);
