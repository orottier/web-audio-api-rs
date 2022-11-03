/// Test for the online AudioContext
///
/// Our CI runner has no sound card enabled so these tests are 'compile time checks' only, we
/// cannot run them.
use web_audio_api::context::{
    AudioContext, AudioContextOptions, AudioContextState, BaseAudioContext,
};

fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

#[allow(dead_code)]
fn test_audio_context_send_sync() {
    let context = AudioContext::default();
    require_send_sync_static(context);
}

#[test]
fn test_none_sink_id() {
    let options = AudioContextOptions {
        sink_id: Some(None),
        ..AudioContextOptions::default()
    };

    // construct with 'none' sink_id
    let context = AudioContext::new(options);
    assert!(context.sink_id().is_none());

    // changing sink_id to 'none' again should make no changes
    context.set_sink_id_sync(None).unwrap();
    assert!(context.sink_id().is_none());

    context.suspend_sync();
    assert_eq!(context.state(), AudioContextState::Suspended);

    context.resume_sync();
    assert_eq!(context.state(), AudioContextState::Running);

    context.close_sync();
    assert_eq!(context.state(), AudioContextState::Closed);
}
