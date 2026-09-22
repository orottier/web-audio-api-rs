//! Resource cleanup of `DelayNode`.
//!
//! Regression tests that a non-cyclic delay subgraph is disposed once every control handle is
//! dropped, while a delay inside a feedback cycle keeps rendering its tail.

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

/// A plain delay connected to the destination, then fully dropped, must still emit the delayed copy
/// of its finite input.
#[test]
fn plain_delay_fire_and_forget_still_delays() {
    let sample_rate = 48_000.0_f32;
    let length = sample_rate as usize;
    let mut context = OfflineAudioContext::new(1, length, sample_rate);

    {
        let mut src = context.create_constant_source();
        src.offset().set_value(1.0);

        let delay = context.create_delay(1.0);
        delay.delay_time().set_value(0.15);

        src.connect(&delay);
        delay.connect(&context.destination());

        src.start_at(0.0);
        src.stop_at(0.02);
    }

    let out = context.start_rendering_sync();
    let ch = out.get_channel_data(0);

    // a 20 ms pulse delayed by 150 ms: silence at 100 ms, the pulse at 160 ms
    assert!(
        ch[(0.10 * sample_rate) as usize].abs() < 1e-6,
        "output should be silent before the delay time"
    );
    assert!(
        ch[(0.16 * sample_rate) as usize].abs() > 0.5,
        "delayed signal missing"
    );
}

/// After the `DelayWriter` is reclaimed, the `DelayReader` must flush only what is still buffered
/// and then stop. It must not keep sweeping the ring buffer and re-emit its stale contents one
/// `max_delay_time` later.
#[test]
fn delay_fire_and_forget_has_no_phantom_echo() {
    const SAMPLE_RATE: f32 = 48_000.0;
    const MAX_DELAY: f64 = 1.0;
    const DELAY_TIME: f64 = 0.25;
    const BURST_LEN: f64 = 0.5;

    let render_len = (SAMPLE_RATE * 3.0) as usize;
    let mut ctx = OfflineAudioContext::new(1, render_len, SAMPLE_RATE);

    // a half-second burst of amplitude 1.0
    let burst_samples = (SAMPLE_RATE as f64 * BURST_LEN) as usize;
    let mut buffer = ctx.create_buffer(1, burst_samples, SAMPLE_RATE);
    buffer.copy_to_channel(&vec![1.0_f32; burst_samples], 0);

    let delay = ctx.create_delay(MAX_DELAY);
    delay.delay_time().set_value(DELAY_TIME as f32);
    delay.connect(&ctx.destination());

    let mut src = ctx.create_buffer_source();
    src.set_buffer(buffer);
    src.connect(&delay);
    src.start();

    // fire and forget: the render thread reclaims the DelayWriter as soon as the source ends
    drop(src);
    drop(delay);

    let rendered = ctx.start_rendering_sync();
    let channel = rendered.get_channel_data(0);

    let peak = |from: f64, to: f64| {
        let a = (from * SAMPLE_RATE as f64) as usize;
        let b = (to * SAMPLE_RATE as f64) as usize;
        channel[a..b].iter().fold(0.0_f32, |m, s| m.max(s.abs()))
    };

    // the legitimate delayed copy lands at [DELAY_TIME, DELAY_TIME + BURST_LEN]
    let legit = peak(DELAY_TIME, DELAY_TIME + BURST_LEN);
    // everything one full max-delay period later must be silent
    let phantom = peak(
        MAX_DELAY + DELAY_TIME - 0.02,
        MAX_DELAY + DELAY_TIME + BURST_LEN,
    );

    assert!(
        legit > 0.5,
        "expected the real delayed signal (peak {legit})"
    );
    assert!(
        phantom < 1e-3,
        "phantom echo: the burst is re-emitted {MAX_DELAY}s later (peak {phantom})"
    );
}

/// A delay in a feedback cycle keeps rendering its decaying echo tail after every handle is
/// dropped. Its reader flags the writer as breaking a cycle, so the writer reports side effects and
/// is not reclaimed while the loop is still ringing.
#[test]
fn feedback_delay_fire_and_forget_keeps_echoing() {
    let sample_rate = 48_000.0_f32;
    let length = sample_rate as usize; // 1 second
    let mut context = OfflineAudioContext::new(1, length, sample_rate);

    {
        // 20 ms DC pulse
        let mut src = context.create_constant_source();
        src.offset().set_value(1.0);

        let delay = context.create_delay(1.0);
        delay.delay_time().set_value(0.15);

        let feedback = context.create_gain();
        feedback.gain().set_value(0.6);

        //  src --> delay(writer)
        //          delay(reader) --> feedback --> delay(writer)   (cycle)
        //          delay(reader) --> destination
        src.connect(&delay);
        delay.connect(&feedback);
        feedback.connect(&delay);
        delay.connect(&context.destination());

        src.start_at(0.0);
        src.stop_at(0.02);

        // fire and forget: every handle drops here
    }

    let out = context.start_rendering_sync();
    let ch = out.get_channel_data(0);

    // the pulse is repeated every 150 ms, each repeat scaled by the 0.6 feedback gain
    let echo_early = ch[(0.16 * sample_rate) as usize].abs(); // direct delayed pulse (~1.0)
    let echo_mid = ch[(0.31 * sample_rate) as usize].abs(); // 1 feedback repeat  (~0.6)
    let echo_late = ch[(0.61 * sample_rate) as usize].abs(); // 3 feedback repeats (~0.216)

    assert!(echo_early > 0.5, "delayed signal missing");
    assert!(
        echo_late > 0.05,
        "feedback echoes died out - the writer was reclaimed"
    );
    assert!(
        echo_mid < echo_early && echo_late < echo_mid,
        "echo train is not decaying"
    );
}

/// A non-cyclic delay whose reader output is connected to nothing must be reclaimed from the render
/// graph once its handles are dropped - even while a source is still feeding its writer. Before the
/// fix, the writer lingered forever because it is registered as a cycle breaker.
#[cfg(feature = "diagnostics")]
#[test]
fn non_cyclic_delay_with_unused_output_is_disposed() {
    use std::sync::mpsc;
    use std::time::Duration;
    use web_audio_api::context::{AudioContext, AudioContextOptions};

    /// Poll the render graph for its current node count. The 'none' backend renders in real time,
    /// so callers sleep between snapshots to let the lifecycle collector run.
    fn node_count(context: &AudioContext) -> usize {
        let (tx, rx) = mpsc::channel();
        context.run_diagnostics(move |d| {
            let _ = tx.send(d);
        });
        rx.recv_timeout(Duration::from_secs(1))
            .expect("timed out waiting for diagnostics")
            .graph
            .node_count
    }

    let context = AudioContext::new(AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    });
    std::thread::sleep(Duration::from_millis(150));
    let baseline = node_count(&context);

    {
        let mut src = context.create_constant_source();
        src.offset().set_value(1.0);

        let delay = context.create_delay(0.1);
        delay.delay_time().set_value(0.05);

        src.connect(&delay);
        // NB. the delay output is deliberately left unconnected
        src.start(); // no stop: keeps feeding the writer

        std::thread::sleep(Duration::from_millis(100));
        assert!(
            node_count(&context) > baseline,
            "delay subgraph was never registered"
        );

        // fire and forget: drop the source and delay handles
    }

    std::thread::sleep(Duration::from_millis(600));

    assert_eq!(
        node_count(&context),
        baseline,
        "the unused non-cyclic delay subgraph was not reclaimed after its handles were dropped"
    );
}
