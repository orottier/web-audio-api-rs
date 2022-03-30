//! The `BaseAudioContext` interface and the `AudioContext` and `OfflineAudioContext` types
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;

mod base;
pub use base::*;

mod offline;
pub use offline::*;

mod online;
pub use online::*;

use crate::message::ControlMessage;
use crate::node::{self, AudioNode};
use crate::param::{AudioParam, AudioParamEvent};
use crate::render::{AudioProcessor, NodeIndex};
use crate::spatial::{AudioListener, AudioListenerParams};
use crate::SampleRate;

// magic node values
/// Destination node id is always at index 0
const DESTINATION_NODE_ID: u64 = 0;
/// listener node id is always at index 1
const LISTENER_NODE_ID: u64 = 1;
/// listener audio parameters ids are always at index 2 through 10
const LISTENER_PARAM_IDS: Range<u64> = 2..11;

/// The struct that corresponds to the Javascript `BaseAudioContext` object.
///
/// Please note that in rust, we need to differentiate between the [`BaseAudioContext`] trait and
/// the [`ConcreteBaseAudioContext`] concrete implementation.
///
/// This object is returned from the `base()` method on [`AudioContext`] and
/// [`OfflineAudioContext`], or the `context()` method on `AudioNode`s.
///
/// The `ConcreteBaseAudioContext` allows for cheap cloning (using an `Arc` internally).
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct ConcreteBaseAudioContext {
    /// inner makes `ConcreteBaseAudioContext` cheap to clone
    inner: Arc<ConcreteBaseAudioContextInner>,
}

impl PartialEq for ConcreteBaseAudioContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Inner representation of the `ConcreteBaseAudioContext`
///
/// These fields are wrapped inside an `Arc` in the actual `ConcreteBaseAudioContext`.
struct ConcreteBaseAudioContextInner {
    /// sample rate in Hertz
    sample_rate: SampleRate,
    /// number of speaker output channels
    number_of_channels: u32,
    /// incrementing id to assign to audio nodes
    node_id_inc: AtomicU64,
    /// destination node's current channel count
    destination_channel_count: Arc<AtomicUsize>,
    /// message channel from control to render thread
    render_channel: Sender<ControlMessage>,
    /// control messages that cannot be sent immediately
    queued_messages: Mutex<Vec<ControlMessage>>,
    /// number of frames played
    frames_played: Arc<AtomicU64>,
    /// control msg to add the AudioListener, to be sent when the first panner is created
    queued_audio_listener_msgs: Mutex<Vec<ControlMessage>>,
    /// AudioListener fields
    listener_params: Option<AudioListenerParams>,
    /// Denotes if this AudioContext is offline or not
    offline: bool,
}

impl BaseAudioContext for ConcreteBaseAudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        self
    }
}

/// Unique identifier for audio nodes.
///
/// Used for internal bookkeeping.
#[derive(Debug)]
pub struct AudioNodeId(u64);

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

/// Handle of the [`node::AudioNode`] to its associated [`BaseAudioContext`].
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
    pub fn id(&self) -> &AudioNodeId {
        &self.id
    }
    /// get the context of the registration
    // false positive: AudioContextRegistration is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn context(&self) -> &ConcreteBaseAudioContext {
        &self.context
    }
}

impl Drop for AudioContextRegistration {
    fn drop(&mut self) {
        // do not drop magic nodes
        let magic = self.id.0 == DESTINATION_NODE_ID
            || self.id.0 == LISTENER_NODE_ID
            || LISTENER_PARAM_IDS.contains(&self.id.0);

        if !magic {
            let message = ControlMessage::FreeWhenFinished { id: self.id.0 };

            // Sending the message will fail when the render thread has already shut down.
            // This is fine
            let _r = self.context.inner.render_channel.send(message);
        }
    }
}

impl ConcreteBaseAudioContext {
    /// Creates a `BaseAudioContext` instance
    fn new(
        sample_rate: SampleRate,
        number_of_channels: u32,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
        offline: bool,
    ) -> Self {
        // placeholder for destination node channel count, will patch later
        let tmp_dest_channel_count = Arc::default();

        let base_inner = ConcreteBaseAudioContextInner {
            sample_rate,
            number_of_channels,
            render_channel,
            queued_messages: Mutex::new(Vec::new()),
            node_id_inc: AtomicU64::new(0),
            destination_channel_count: tmp_dest_channel_count,
            frames_played,
            queued_audio_listener_msgs: Mutex::new(Vec::new()),
            listener_params: None,
            offline,
        };
        let base = Self {
            inner: Arc::new(base_inner),
        };

        let (listener_params, dest_channels) = {
            // Register magical nodes. We should not store the nodes inside our context since that
            // will create a cyclic reference, but we can reconstruct a new instance on the fly
            // when requested
            let dest = node::AudioDestinationNode::new(&base, number_of_channels as usize);
            let dest_channels = dest.into_raw_parts().into_count();
            let listener = crate::spatial::AudioListenerNode::new(&base);

            let listener_params = listener.into_fields();
            let AudioListener {
                position_x,
                position_y,
                position_z,
                forward_x,
                forward_y,
                forward_z,
                up_x,
                up_y,
                up_z,
            } = listener_params;

            let listener_params = AudioListenerParams {
                position_x: position_x.into_raw_parts(),
                position_y: position_y.into_raw_parts(),
                position_z: position_z.into_raw_parts(),
                forward_x: forward_x.into_raw_parts(),
                forward_y: forward_y.into_raw_parts(),
                forward_z: forward_z.into_raw_parts(),
                up_x: up_x.into_raw_parts(),
                up_y: up_y.into_raw_parts(),
                up_z: up_z.into_raw_parts(),
            };

            (listener_params, dest_channels)
        }; // nodes will drop now, so base.inner has no copies anymore

        let mut base = base;
        let mut inner_mut = Arc::get_mut(&mut base.inner).unwrap();
        inner_mut.listener_params = Some(listener_params);
        inner_mut.destination_channel_count = dest_channels;

        // validate if the hardcoded node IDs line up
        debug_assert_eq!(
            base.inner.node_id_inc.load(Ordering::Relaxed),
            LISTENER_PARAM_IDS.end,
        );

        base
    }

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.inner.sample_rate.0 as f32
    }

    /// The raw sample rate of the `AudioContext` (which has more precision than the float
    /// [`sample_rate()`](BaseAudioContext::sample_rate) value).
    #[must_use]
    pub fn sample_rate_raw(&self) -> SampleRate {
        self.inner.sample_rate
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    #[must_use]
    // web audio api specification requires that `current_time` returns an f64
    // std::sync::AtomicsF64 is not currently implemented in the standard library
    // Currently, we have no other choice than casting an u64 into f64, with possible loss of precision
    #[allow(clippy::cast_precision_loss)]
    pub fn current_time(&self) -> f64 {
        self.inner.frames_played.load(Ordering::SeqCst) as f64 / f64::from(self.inner.sample_rate.0)
    }

    /// Number of channels for the audio destination
    #[must_use]
    pub fn channels(&self) -> u32 {
        self.inner.number_of_channels
    }

    /// Construct a new pair of [`node::AudioNode`] and [`AudioProcessor`]
    ///
    /// The `AudioNode` lives in the user-facing control thread. The Processor is sent to the render thread.
    #[allow(clippy::missing_panics_doc)]
    pub fn register<
        T: node::AudioNode,
        F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
    >(
        &self,
        f: F,
    ) -> T {
        // create unique identifier for this node
        let id = self.inner.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let node_id = AudioNodeId(id);
        let registration = AudioContextRegistration {
            id: node_id,
            context: self.clone(),
        };

        // create the node and its renderer
        let (node, render) = (f)(registration);

        // pass the renderer to the audio graph
        let message = ControlMessage::RegisterNode {
            id,
            node: render,
            inputs: node.number_of_inputs() as usize,
            outputs: node.number_of_outputs() as usize,
            channel_config: node.channel_config().clone(),
        };

        // if this is the AudioListener or its params, do not add it to the graph just yet
        if id == LISTENER_NODE_ID || LISTENER_PARAM_IDS.contains(&id) {
            let mut queued_audio_listener_msgs =
                self.inner.queued_audio_listener_msgs.lock().unwrap();
            queued_audio_listener_msgs.push(message);
        } else {
            self.inner.render_channel.send(message).unwrap();
            self.resolve_queued_control_msgs(id);
        }

        node
    }

    /// Release queued control messages to the render thread that were blocking on the availability
    /// of the Node with the given `id`
    fn resolve_queued_control_msgs(&self, id: u64) {
        // resolve control messages that depend on this registration
        let mut queued = self.inner.queued_messages.lock().unwrap();
        let mut i = 0; // waiting for Vec::drain_filter to stabilize
        while i < queued.len() {
            if matches!(&queued[i], ControlMessage::ConnectNode {to, ..} if *to == id) {
                let m = queued.remove(i);
                self.inner.render_channel.send(m).unwrap();
            } else {
                i += 1;
            }
        }
    }

    /// Connects the output of the `from` audio node to the input of the `to` audio node
    pub(crate) fn connect(&self, from: &AudioNodeId, to: &AudioNodeId, output: u32, input: u32) {
        let message = ControlMessage::ConnectNode {
            from: from.0,
            to: to.0,
            output,
            input,
        };
        self.inner.render_channel.send(message).unwrap();
    }

    /// Schedule a connection of an `AudioParam` to the `AudioNode` it belongs to
    ///
    /// It is not performed immediately as the `AudioNode` is not registered at this point.
    fn queue_audio_param_connect(&self, param: &AudioParam, audio_node: &AudioNodeId) {
        let message = ControlMessage::ConnectNode {
            from: param.id().0,
            to: audio_node.0,
            output: 0,
            input: u32::MAX, // audio params connect to the 'hidden' input port
        };
        self.inner.queued_messages.lock().unwrap().push(message);
    }

    /// Disconnects all outputs of the audio node that go to a specific destination node.
    pub(crate) fn disconnect_from(&self, from: &AudioNodeId, to: &AudioNodeId) {
        let message = ControlMessage::DisconnectNode {
            from: from.0,
            to: to.0,
        };
        self.inner.render_channel.send(message).unwrap();
    }

    /// Disconnects all outgoing connections from the audio node.
    pub(crate) fn disconnect(&self, from: &AudioNodeId) {
        let message = ControlMessage::DisconnectAll { from: from.0 };
        self.inner.render_channel.send(message).unwrap();
    }

    /// Pass an `AudioParam::AudioParamEvent` to the render thread
    ///
    /// This clunky setup (wrapping a Sender in a message sent by another Sender) ensures
    /// automation events will never be handled out of order.
    pub(crate) fn pass_audio_param_event(
        &self,
        to: &Sender<AudioParamEvent>,
        event: AudioParamEvent,
    ) {
        let message = ControlMessage::AudioParamEvent {
            to: to.clone(),
            event,
        };
        self.inner.render_channel.send(message).unwrap();
    }

    /// Attach the 9 `AudioListener` coordinates to a `PannerNode`
    pub(crate) fn connect_listener_to_panner(&self, panner: &AudioNodeId) {
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 0, 1);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 1, 2);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 2, 3);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 3, 4);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 4, 5);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 5, 6);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 6, 7);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 7, 8);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 8, 9);
    }

    /// Add the [`AudioListener`] to the audio graph (if not already)
    pub(crate) fn ensure_audio_listener_present(&self) {
        let mut queued_audio_listener_msgs = self.inner.queued_audio_listener_msgs.lock().unwrap();
        let mut released = false;
        while let Some(message) = queued_audio_listener_msgs.pop() {
            // add the AudioListenerRenderer to the graph
            self.inner.render_channel.send(message).unwrap();
            released = true;
        }

        if released {
            // connect the AudioParamRenderers to the Listener
            self.resolve_queued_control_msgs(LISTENER_NODE_ID);

            // hack: Connect the listener to the destination node to force it to render at each
            // quantum. Abuse the magical u32::MAX port so it acts as an AudioParam and has no side
            // effects
            self.connect(
                &AudioNodeId(LISTENER_NODE_ID),
                &AudioNodeId(DESTINATION_NODE_ID),
                0,
                u32::MAX,
            );
        }
    }

    /// Returns true if this is `OfflineAudioContext` (false when it is an `AudioContext`)
    pub(crate) fn offline(&self) -> bool {
        self.inner.offline
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

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
    fn test_sample_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(96000));
        assert_float_eq!(context.sample_rate(), 96000., abs_all <= 0.);
        assert_eq!(context.sample_rate_raw(), SampleRate(96000));
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
}
