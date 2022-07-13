/// Test for the online AudioContext
///
/// Our CI runner has no sound card enabled so these tests are 'compile time checks' only, we
/// cannot run them.
use web_audio_api::context::AudioContext;

fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

#[allow(dead_code)]
fn test_audio_context_send_sync() {
    let context = AudioContext::default();
    require_send_sync_static(context);
}
