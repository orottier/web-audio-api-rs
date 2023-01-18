//! Test for the online AudioContext
//!
//! Our CI runner has no sound card enabled so these tests are 'compile time checks' and checks
//! using the 'none' audio backend.

use web_audio_api::context::{
    AudioContext, AudioContextOptions, AudioContextState, BaseAudioContext,
};

use std::sync::atomic::{AtomicBool, Ordering};

fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

#[allow(dead_code)]
fn test_audio_context_send_sync() {
    let context = AudioContext::default();
    require_send_sync_static(context);
}

#[test]
fn test_none_sink_id() {
    let options = AudioContextOptions {
        sink_id: "none".into(),
        ..AudioContextOptions::default()
    };

    // construct with 'none' sink_id
    let context = AudioContext::new(options);
    assert_eq!(context.sink_id(), "none");

    // changing sink_id to 'none' again should make no changes
    let sink_stable = &*Box::leak(Box::new(AtomicBool::new(true)));
    context.set_onsinkchange(move |_| {
        sink_stable.store(false, Ordering::SeqCst);
    });
    context.set_sink_id_sync("none".into()).unwrap();
    assert_eq!(context.sink_id(), "none");

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);

    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);

    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);

    assert!(sink_stable.load(Ordering::SeqCst));
}
