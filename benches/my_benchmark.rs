use std::fs::File;
use std::sync::OnceLock;

#[cfg(feature = "iai")]
use iai::black_box;

#[cfg(not(feature = "iai"))]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use paste::paste;

use web_audio_api::context::BaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, PanningModelType};
use web_audio_api::worklet::{AudioWorkletNode, AudioWorkletNodeOptions};
use web_audio_api::AudioBuffer;

const SAMPLE_RATE: f32 = 48000.;
const DURATION: usize = 10;
const SAMPLES: usize = SAMPLE_RATE as usize * DURATION;
const SAMPLES_SHORT: usize = SAMPLE_RATE as usize; // only 1 second for heavy benchmarks

mod worklet;

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
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

// This benchmark only makes sense in `iai`, because subsequent runs use the cached audiobuffer.
// However we would like to run this test here so the cache is filled for the subsequent benches.
pub fn bench_audio_buffer_decode() {
    let ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);
    assert_eq!(buffer.length(), 101129);
}

pub fn bench_constant_source() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut src = ctx.create_constant_source();

    src.connect(&ctx.destination());
    src.start_at(1.);
    src.stop_at(9.);

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut osc = ctx.create_oscillator();

    osc.connect(&ctx.destination());
    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut osc = ctx.create_oscillator();
    let gain = ctx.create_gain();
    gain.gain().set_value(0.5); // avoid happy path

    osc.connect(&gain);
    gain.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_sine_gain_delay() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);

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
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    let mut src = ctx.create_buffer_source();
    src.connect(&ctx.destination());
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_delay() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    let delay = ctx.create_delay(0.3);
    delay.delay_time().set_value(0.2);

    let mut src = ctx.create_buffer_source();
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    src.connect(&delay);
    delay.connect(&ctx.destination());

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_iir() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_buffer_src_biquad() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let buffer = get_audio_buffer(&ctx);

    // Create an biquad filter node (defaults to low pass)
    let biquad = ctx.create_biquad_filter();
    biquad.connect(&ctx.destination());
    biquad.frequency().set_value(200.);

    // Play buffer and pipe to filter
    let mut src = ctx.create_buffer_source();
    src.connect(&biquad);
    src.set_buffer(buffer);
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_positional() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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
    src.set_loop(true);
    src.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

pub fn bench_stereo_panning_automation() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
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
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES_SHORT), SAMPLE_RATE);

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

pub fn bench_sine_gain_with_worklet() {
    let mut ctx = OfflineAudioContext::new(2, black_box(SAMPLES), SAMPLE_RATE);
    let mut osc = ctx.create_oscillator();

    let options = AudioWorkletNodeOptions::default();
    let gain_worklet = AudioWorkletNode::new::<worklet::GainProcessor>(&ctx, options);
    gain_worklet
        .parameters()
        .get("gain")
        .unwrap()
        .set_value(0.5);

    osc.connect(&gain_worklet);
    gain_worklet.connect(&ctx.destination());

    osc.start();

    assert_eq!(ctx.start_rendering_sync().length(), SAMPLES);
}

macro_rules! iai_or_criterion {
    ( $( $func:ident ),+ $(,)* ) => {
        #[cfg(feature = "iai")]
        iai::main!(
            $($func,)*
        );

        paste! {
            $(
                #[cfg(not(feature = "iai"))]
                fn [<criterion_ $func>](c: &mut Criterion) {
                    c.bench_function(stringify!($func), |b| b.iter($func));
                }
            )*

            #[cfg(not(feature = "iai"))]
            criterion_group!(
                benches,
                $([<criterion_ $func>],)*
            );
        }

        #[cfg(not(feature = "iai"))]
        criterion_main!(benches);
    }
}

iai_or_criterion!(
    bench_ctor,
    bench_audio_buffer_decode,
    bench_constant_source,
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
    bench_sine_gain_with_worklet,
);
