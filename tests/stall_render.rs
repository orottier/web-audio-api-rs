/// Tests to validate the render thread is not blocked by control thread locks
use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::AudioNode;
use web_audio_api::node::AudioScheduledSourceNode;
use web_audio_api::AudioRenderCapacityOptions;

use std::thread;
use std::time::Duration;

/*
 * These tests check that the render thread is never blocked, e.g. when the communication channels
 * are full. We force this by adding a 'runaway' event handler that blocks indefinitely.
 */

#[test]
fn test_capacity_handler() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    let cap = context.render_capacity();
    cap.set_onupdate(|_| {
        // block the event handler thread
        thread::sleep(Duration::from_secs(200));
    });
    cap.start(AudioRenderCapacityOptions {
        update_interval: 0.01,
    });

    // sleep for 250 milliseconds, allow the render thread to boot and progress
    thread::sleep(Duration::from_millis(250));
    assert!(context.current_time() >= 0.2);
}

#[test]
fn test_event_handler() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    for _ in 0..512 {
        let mut constant_source = context.create_constant_source();
        constant_source.connect(&context.destination());
        constant_source.start();
        constant_source.stop_at(0.001);
        constant_source.set_onended(|_| {
            // block the event handler thread
            thread::sleep(Duration::from_secs(200));
        });
    }

    // sleep for 250 milliseconds, allow the render thread to boot and progress
    thread::sleep(Duration::from_millis(200));
    assert!(context.current_time() >= 0.2);
}
