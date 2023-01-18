//! Test for the online AudioContext
//!
//! Our CI runner has no sound card enabled so these tests are 'compile time checks' and checks
//! using the 'none' audio backend.

use web_audio_api::context::{
    AudioContext, AudioContextOptions, AudioContextState, BaseAudioContext,
};
use web_audio_api::node::AudioNode;

use std::sync::atomic::{AtomicBool, Ordering};

fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

#[allow(dead_code)]
fn test_audio_context_send_sync() {
    let context = AudioContext::default();
    require_send_sync_static(context);
}

#[allow(dead_code)]
fn ensure_audio_node_send_sync() {
    let context = AudioContext::default();
    let node = context.create_constant_source();
    require_send_sync_static(node);
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
