//! The `BaseAudioContext` interface and the `AudioContext` and `OfflineAudioContext` types
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use std::ops::Range;

mod base;
pub use base::*;

mod concrete_base;
pub use concrete_base::*;

mod offline;
pub use offline::*;

mod online;
pub use online::*;

use crate::render::NodeIndex;

// magic node values
/// Destination node id is always at index 0
const DESTINATION_NODE_ID: u64 = 0;
/// listener node id is always at index 1
const LISTENER_NODE_ID: u64 = 1;
/// listener audio parameters ids are always at index 2 through 10
const LISTENER_PARAM_IDS: Range<u64> = 2..11;

/// Unique identifier for audio nodes.
///
/// Used for internal bookkeeping.
#[derive(Debug)]
pub(crate) struct AudioNodeId(u64);

/// Unique identifier for audio params.
///
/// Store these in your `AudioProcessor` to get access to `AudioParam` values.
pub struct AudioParamId(u64);

// bit contrived, but for type safety only the context mod can access the inner u64
impl From<&AudioParamId> for NodeIndex {
    fn from(i: &AudioParamId) -> Self {
        Self(i.0)
    }
}

/// Describes the current state of the `AudioContext`
#[derive(Debug, Copy, Clone)]
pub enum AudioContextState {
    /// This context is currently suspended (context time is not proceeding,
    /// audio hardware may be powered down/released).
    Suspended,
    /// Audio is being processed.
    Running,
    /// This context has been released, and can no longer be used to process audio.
    /// All system audio resources have been released.
    Closed,
}

/// Handle of the [`AudioNode`](crate::node::AudioNode) to its associated [`BaseAudioContext`].
///
/// This allows for communication with the render thread and lifetime management.
///
/// The only way to construct this object is by calling [`ConcreteBaseAudioContext::register`]
pub struct AudioContextRegistration {
    /// the audio context in wich nodes and connections lives
    context: ConcreteBaseAudioContext,
    /// identify a specific `AudioNode`
    id: AudioNodeId,
}

impl AudioContextRegistration {
    /// get the audio node id of the registration
    // false positive: AudioContextRegistration is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub(crate) fn id(&self) -> &AudioNodeId {
        &self.id
    }
    /// get the context of the registration
    // false positive: AudioContextRegistration is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub(crate) fn context(&self) -> &ConcreteBaseAudioContext {
        &self.context
    }
}

impl Drop for AudioContextRegistration {
    fn drop(&mut self) {
        self.context.mark_node_dropped(self.id.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::AudioNode;
    use crate::SampleRate;

    use float_eq::assert_float_eq;

    fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

    #[test]
    fn test_audio_context_registration_traits() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        let registration = context.mock_registration();

        // we want to be able to ship AudioNodes to another thread, so the Registration should be
        // Send Sync and 'static
        require_send_sync_static(registration);
    }

    #[test]
    fn test_sample_rate_length() {
        let context = OfflineAudioContext::new(1, 48000, SampleRate(96000));
        assert_float_eq!(context.sample_rate(), 96000., abs_all <= 0.);
        assert_eq!(context.sample_rate_raw(), SampleRate(96000));
        assert_eq!(context.length(), 48000);
    }

    #[test]
    fn test_decode_audio_data() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/sample.wav").unwrap();
        let audio_buffer = context.decode_audio_data_sync(file).unwrap();

        assert_eq!(audio_buffer.sample_rate_raw(), SampleRate(44100));
        assert_eq!(audio_buffer.length(), 142_187);
        assert_eq!(audio_buffer.number_of_channels(), 2);
        assert_float_eq!(audio_buffer.duration(), 3.224, abs_all <= 0.001);

        let left_start = &audio_buffer.get_channel_data(0)[0..100];
        let right_start = &audio_buffer.get_channel_data(1)[0..100];
        // assert distinct two channel data
        assert!(left_start != right_start);
    }

    // #[test]
    // disabled: symphonia cannot handle empty WAV-files
    #[allow(dead_code)]
    fn test_decode_audio_data_empty() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/empty_2c.wav").unwrap();
        let audio_buffer = context.decode_audio_data_sync(file).unwrap();
        assert_eq!(audio_buffer.length(), 0);
    }

    #[test]
    fn test_decode_audio_data_decoding_error() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/corrupt.wav").unwrap();
        assert!(context.decode_audio_data_sync(file).is_err());
    }

    #[test]
    fn test_create_buffer() {
        let number_of_channels = 3;
        let length = 2000;
        let sample_rate = SampleRate(96_000);

        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let buffer = context.create_buffer(number_of_channels, length, sample_rate);

        assert_eq!(buffer.number_of_channels(), 3);
        assert_eq!(buffer.length(), 2000);
        assert_float_eq!(buffer.sample_rate(), 96000., abs_all <= 0.);
    }

    #[test]
    fn test_registration() {
        let context = OfflineAudioContext::new(1, 48000, SampleRate(96000));
        let dest = context.destination();
        assert!(dest.context() == context.base());
    }
}
