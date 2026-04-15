//! Test for the online AudioContext
//!
//! Our CI runner has no sound card enabled so these tests are 'compile time checks' and checks
//! using the 'none' audio backend.

use web_audio_api::context::{
    AudioContext, AudioContextOptions, AudioContextState, BaseAudioContext,
};
use web_audio_api::node::AudioNode;

use std::thread;
use std::time::Duration;
use web_audio_api::MAX_CHANNELS;

const TEST_TIMEOUT: Duration = Duration::from_secs(1);

fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

fn wait_until_current_time_at_least(context: &AudioContext, time: f64, message: &str) {
    let (done_tx, done_rx) = crossbeam_channel::bounded(1);
    let (stop_tx, stop_rx) = crossbeam_channel::bounded(1);

    thread::scope(|scope| {
        scope.spawn(|| loop {
            if context.current_time() >= time {
                let _ = done_tx.send(());
                break;
            }

            if stop_rx.try_recv().is_ok() {
                break;
            }

            thread::yield_now();
        });

        let result = done_rx.recv_timeout(TEST_TIMEOUT);
        if result.is_err() {
            let _ = stop_tx.send(());
        }

        assert!(result.is_ok(), "{message}");
    });
}

#[allow(dead_code)]
fn ensure_send_sync_static() {
    require_send_sync_static(AudioContext::default());

    let context = AudioContext::default();

    // All available nodes for BaseAudioContext
    require_send_sync_static(context.create_analyser());
    require_send_sync_static(context.create_biquad_filter());
    require_send_sync_static(context.create_buffer_source());
    require_send_sync_static(context.create_channel_merger(2));
    require_send_sync_static(context.create_channel_splitter(2));
    require_send_sync_static(context.create_constant_source());
    require_send_sync_static(context.create_convolver());
    require_send_sync_static(context.create_delay(1.));
    require_send_sync_static(context.create_dynamics_compressor());
    require_send_sync_static(context.create_gain());
    require_send_sync_static(context.create_iir_filter(vec![], vec![]));
    require_send_sync_static(context.create_oscillator());
    require_send_sync_static(context.create_panner());
    require_send_sync_static(
        context.create_periodic_wave(web_audio_api::PeriodicWaveOptions::default()),
    );
    require_send_sync_static(context.create_stereo_panner());
    require_send_sync_static(context.create_wave_shaper());

    // Available nodes for online AudioContext
    let media_track = web_audio_api::media_streams::MediaStreamTrack::from_iter(vec![]);
    let media_stream = web_audio_api::media_streams::MediaStream::from_tracks(vec![media_track]);
    require_send_sync_static(context.create_media_stream_source(&media_stream));
    require_send_sync_static(context.create_media_stream_destination());
    require_send_sync_static(
        context.create_media_stream_track_source(&media_stream.get_tracks()[0]),
    );
    let mut media_element = web_audio_api::MediaElement::new("").unwrap();
    require_send_sync_static(context.create_media_element_source(&mut media_element));

    // Provided nodes
    require_send_sync_static(context.destination());
    require_send_sync_static(context.listener());

    // AudioParams (borrow from their node, so do not test for 'static)
    let _: &(dyn Send + Sync) = context.listener().position_x();
}

#[allow(dead_code)]
fn ensure_audio_node_object_safe() {
    let context = AudioContext::default();
    let node = context.create_constant_source();
    let _object: Box<dyn AudioNode> = Box::new(node);
}

/*
 * AudioScheduledSourceNode trait is not object safe, see #249
 *
#[allow(dead_code)]
fn ensure_audio_scheduled_source_node_object_safe() {
    let context = AudioContext::default();
    let node = context.create_constant_source();
    let _object: Box<dyn AudioScheduledSourceNode> = Box::new(node);
}
*/

#[test]
fn test_none_sink_id() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };

    // construct with 'none' sink_id
    let context = AudioContext::new(options);
    assert_eq!(context.sink_id(), "none");

    // count the number of state changes
    let (state_change_tx, state_change_rx) = crossbeam_channel::bounded(4);
    context.set_onstatechange(move |_| {
        let _ = state_change_tx.send(());
    });

    assert!(
        state_change_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "timed out waiting for started state change"
    );

    // changing sink_id to 'none' again should make no changes
    let (sink_change_tx, sink_change_rx) = crossbeam_channel::bounded(1);
    context.set_onsinkchange(move |_| {
        let _ = sink_change_tx.try_send(());
    });
    context.set_sink_id_sync("none".into()).unwrap();
    assert_eq!(context.sink_id(), "none");
    assert!(sink_change_rx.try_recv().is_err());

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);

    assert!(
        state_change_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "timed out waiting for suspended state change"
    );

    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);

    assert!(
        state_change_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "timed out waiting for resumed state change"
    );

    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);
    assert!(sink_change_rx.try_recv().is_err());

    assert!(
        state_change_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "timed out waiting for closed state change"
    );
}

#[test]
fn test_weird_sample_rate() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        sample_rate: Some(24000.),
        ..AudioContextOptions::default()
    };

    // would crash due to <https://github.com/mrDIMAS/hrtf/issues/9>
    let _ = AudioContext::new(options);
}

#[test]
fn test_channels() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };

    let context = AudioContext::new(options);
    assert_eq!(context.destination().max_channel_count(), MAX_CHANNELS);
    assert_eq!(context.destination().channel_count(), 2);

    context.destination().set_channel_count(5);
    assert_eq!(context.destination().channel_count(), 5);
}

#[test]
fn test_panner_node_drop_panic() {
    // https://github.com/orottier/web-audio-api-rs/issues/369
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    // create a new panner and drop it
    let panner = context.create_panner();
    drop(panner);

    // allow the audio render thread to boot and handle adding and dropping the panner
    wait_until_current_time_at_least(
        &context,
        context.current_time() + 0.15,
        "timed out waiting for the dropped panner to be processed",
    );

    // creating a new panner node should not crash the render thread
    let mut _panner = context.create_panner();

    // A crashed thread will not fail the test (only if the main thread panics).
    // Instead inspect if there is progression of time in the audio context.
    wait_until_current_time_at_least(
        &context,
        context.current_time() + 0.15,
        "timed out waiting for time to progress after creating a new panner",
    );
}

#[test]
fn test_audioparam_outlives_audionode() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    // Create a node with an audioparam, drop to node but keep the audioparam
    let gain = context.create_gain();
    let gain_param = gain.gain().clone();
    drop(gain);

    // Start the audio graph, and give some time to drop the gain node (it has no inputs connected
    // so dynamic lifetime will drop the node);
    wait_until_current_time_at_least(
        &context,
        context.current_time() + 0.15,
        "timed out waiting for the dropped gain node to be processed",
    );

    // We still have a handle to the param, so that should not be removed from the audio graph.
    // So by updating the value, the render thread should not crash.
    gain_param.set_value(1.);

    // A crashed thread will not fail the test (only if the main thread panics).
    // Instead inspect if there is progression of time in the audio context.
    wait_until_current_time_at_least(
        &context,
        context.current_time() + 0.15,
        "timed out waiting for time to progress after updating an orphaned AudioParam",
    );
}

#[test]
fn test_closed() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);
    let node = context.create_gain();

    // Close the context
    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);

    // Should not be able to resume
    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Closed);

    // Drop the context (otherwise the comms channel is kept alive)
    drop(context);

    let (done_tx, done_rx) = crossbeam_channel::bounded(1);
    thread::spawn(move || {
        node.disconnect(); // should not panic
        let _ = done_tx.send(());
    });

    assert!(
        done_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "timed out waiting for node disconnect after closing context"
    );
}

#[test]
fn test_double_suspend() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);
    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);
    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);
}

#[test]
fn test_double_resume() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);
    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);
    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);
}

#[test]
fn test_double_close() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);
    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);
}

#[test]
fn test_suspend_then_close() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);
    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);
}

#[test]
fn test_control_messages_do_not_block_while_suspended() {
    let (done_tx, done_rx) = crossbeam_channel::bounded(1);

    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };
    let context = AudioContext::new(options);

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);

    thread::spawn(move || {
        // The control channel currently has a capacity of 256. Sending more messages while the
        // backend callback is suspended must not block the control thread.
        for i in 0..300 {
            context.destination().set_channel_count(1 + i % 2);
        }

        done_tx.send(()).unwrap();
    });

    assert!(
        done_rx.recv_timeout(TEST_TIMEOUT).is_ok(),
        "control messages blocked while the render callback was suspended"
    );
}
