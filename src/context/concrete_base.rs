//! The `ConcreteBaseAudioContext` type

use crate::context::{
    AudioContextRegistration, AudioContextState, AudioNodeId, BaseAudioContext,
    DESTINATION_NODE_ID, LISTENER_NODE_ID, LISTENER_PARAM_IDS,
};
use crate::events::{EventDispatch, EventHandler, EventLoop, EventType};
use crate::message::ControlMessage;
use crate::node::{AudioDestinationNode, AudioNode, ChannelConfig, ChannelConfigOptions};
use crate::param::AudioParam;
use crate::render::AudioProcessor;
use crate::spatial::AudioListenerParams;

use crate::AudioListener;

use crossbeam_channel::{Receiver, SendError, Sender};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

/// The struct that corresponds to the Javascript `BaseAudioContext` object.
///
/// This object is returned from the `base()` method on
/// [`AudioContext`](crate::context::AudioContext) and
/// [`OfflineAudioContext`](crate::context::OfflineAudioContext), and the `context()` method on
/// `AudioNode`s.
///
/// The `ConcreteBaseAudioContext` allows for cheap cloning (using an `Arc` internally).
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
#[doc(hidden)]
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
    sample_rate: f32,
    /// max number of speaker output channels
    max_channel_count: usize,
    /// incrementing id to assign to audio nodes
    node_id_inc: AtomicU64,
    /// receiver for decommissioned AudioNodeIds, which can be reused
    node_id_consumer: Mutex<llq::Consumer<AudioNodeId>>,
    /// destination node's current channel count
    destination_channel_config: ChannelConfig,
    /// message channel from control to render thread
    render_channel: RwLock<Sender<ControlMessage>>,
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
    /// Describes the current state of the `ConcreteBaseAudioContext`
    state: AtomicU8,
    /// Stores the event handlers
    event_loop: EventLoop,
    /// Sender for events that will be handled by the EventLoop
    event_send: Option<Sender<EventDispatch>>,
}

impl BaseAudioContext for ConcreteBaseAudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        self
    }

    fn register<
        T: AudioNode,
        F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
    >(
        &self,
        f: F,
    ) -> T {
        // create unique identifier for this node
        let id = if let Some(available_id) = self.inner.node_id_consumer.lock().unwrap().pop() {
            llq::Node::into_inner(available_id)
        } else {
            AudioNodeId(self.inner.node_id_inc.fetch_add(1, Ordering::Relaxed))
        };

        let registration = AudioContextRegistration {
            id,
            context: self.clone(),
        };

        // create the node and its renderer
        let (node, render) = (f)(registration);

        // pass the renderer to the audio graph
        let message = ControlMessage::RegisterNode {
            id,
            reclaim_id: llq::Node::new(id),
            node: render,
            inputs: node.number_of_inputs(),
            outputs: node.number_of_outputs(),
            channel_config: node.channel_config().clone(),
        };

        // if this is the AudioListener or its params, do not add it to the graph just yet
        if id == LISTENER_NODE_ID || LISTENER_PARAM_IDS.contains(&id.0) {
            let mut queued_audio_listener_msgs =
                self.inner.queued_audio_listener_msgs.lock().unwrap();
            queued_audio_listener_msgs.push(message);
        } else {
            self.send_control_msg(message).unwrap();
            self.resolve_queued_control_msgs(id);
        }

        node
    }
}

impl ConcreteBaseAudioContext {
    /// Creates a `BaseAudioContext` instance
    pub(super) fn new(
        sample_rate: f32,
        max_channel_count: usize,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
        event_channel: Option<(Sender<EventDispatch>, Receiver<EventDispatch>)>,
        offline: bool,
        node_id_consumer: llq::Consumer<AudioNodeId>,
    ) -> Self {
        let event_loop = EventLoop::new();
        let (event_send, event_recv) = match event_channel {
            None => (None, None),
            Some((send, recv)) => (Some(send), Some(recv)),
        };

        let base_inner = ConcreteBaseAudioContextInner {
            sample_rate,
            max_channel_count,
            render_channel: RwLock::new(render_channel),
            queued_messages: Mutex::new(Vec::new()),
            node_id_inc: AtomicU64::new(0),
            node_id_consumer: Mutex::new(node_id_consumer),
            destination_channel_config: ChannelConfigOptions::default().into(),
            frames_played,
            queued_audio_listener_msgs: Mutex::new(Vec::new()),
            listener_params: None,
            offline,
            state: AtomicU8::new(AudioContextState::Suspended as u8),
            event_loop: event_loop.clone(),
            event_send,
        };
        let base = Self {
            inner: Arc::new(base_inner),
        };

        // Online AudioContext should start with stereo channels by default
        let initial_channel_count = if offline {
            max_channel_count
        } else {
            2.min(max_channel_count)
        };

        let (listener_params, destination_channel_config) = {
            // Register magical nodes. We should not store the nodes inside our context since that
            // will create a cyclic reference, but we can reconstruct a new instance on the fly
            // when requested
            let dest = AudioDestinationNode::new(&base, initial_channel_count);
            let destination_channel_config = dest.into_channel_config();
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

            (listener_params, destination_channel_config)
        }; // Nodes will drop now, so base.inner has no copies anymore

        let mut base = base;
        let inner_mut = Arc::get_mut(&mut base.inner).unwrap();
        inner_mut.listener_params = Some(listener_params);
        inner_mut.destination_channel_config = destination_channel_config;

        // Validate if the hardcoded node IDs line up
        debug_assert_eq!(
            base.inner.node_id_inc.load(Ordering::Relaxed),
            LISTENER_PARAM_IDS.end,
        );

        // For an online AudioContext, pre-create the HRTF-database for panner nodes
        if !offline {
            crate::node::load_hrtf_processor(sample_rate as u32);
        }

        // Boot the event loop thread that handles the events spawned by the render thread
        // (we don't do this for offline rendering because it makes little sense, the graph cannot
        // be mutated once rendering has started anyway)
        if let Some(event_channel) = event_recv {
            event_loop.run(event_channel);
        }

        base
    }

    pub(crate) fn send_control_msg(
        &self,
        msg: ControlMessage,
    ) -> Result<(), SendError<ControlMessage>> {
        self.inner.render_channel.read().unwrap().send(msg)
    }

    pub(crate) fn send_event(&self, msg: EventDispatch) -> Result<(), SendError<EventDispatch>> {
        match self.inner.event_send.as_ref() {
            Some(s) => s.send(msg),
            None => Err(SendError(msg)),
        }
    }

    pub(crate) fn lock_control_msg_sender(&self) -> RwLockWriteGuard<'_, Sender<ControlMessage>> {
        self.inner.render_channel.write().unwrap()
    }

    /// Inform render thread that the control thread `AudioNode` no langer has any handles
    pub(super) fn mark_node_dropped(&self, id: AudioNodeId) {
        // do not drop magic nodes
        let magic = id == DESTINATION_NODE_ID
            || id == LISTENER_NODE_ID
            || LISTENER_PARAM_IDS.contains(&id.0);

        if !magic {
            let message = ControlMessage::FreeWhenFinished { id };

            // Sending the message will fail when the render thread has already shut down.
            // This is fine
            let _r = self.send_control_msg(message);
        }
    }

    /// Inform render thread that this node can act as a cycle breaker
    #[doc(hidden)]
    pub fn mark_cycle_breaker(&self, reg: &AudioContextRegistration) {
        let id = reg.id();
        let message = ControlMessage::MarkCycleBreaker { id };

        // Sending the message will fail when the render thread has already shut down.
        // This is fine
        let _r = self.send_control_msg(message);
    }

    /// `ChannelConfig` of the `AudioDestinationNode`
    pub(super) fn destination_channel_config(&self) -> ChannelConfig {
        self.inner.destination_channel_config.clone()
    }

    /// Returns the `AudioListener` which is used for 3D spatialization
    pub(super) fn listener(&self) -> AudioListener {
        // instruct to BaseContext to add the AudioListener if it has not already
        self.base().ensure_audio_listener_present();

        let mut ids = LISTENER_PARAM_IDS.map(|i| AudioContextRegistration {
            id: AudioNodeId(i),
            context: self.clone(),
        });
        let params = self.inner.listener_params.as_ref().unwrap();

        AudioListener {
            position_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_x.clone()),
            position_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_y.clone()),
            position_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_z.clone()),
            forward_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_x.clone()),
            forward_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_y.clone()),
            forward_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_z.clone()),
            up_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_x.clone()),
            up_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_y.clone()),
            up_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_z.clone()),
        }
    }

    /// Returns state of current context
    #[must_use]
    pub(super) fn state(&self) -> AudioContextState {
        self.inner.state.load(Ordering::SeqCst).into()
    }

    /// Updates state of current context
    pub(super) fn set_state(&self, state: AudioContextState) {
        self.inner.state.store(state as u8, Ordering::SeqCst);
    }

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[must_use]
    pub(super) fn sample_rate(&self) -> f32 {
        self.inner.sample_rate
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    #[must_use]
    // web audio api specification requires that `current_time` returns an f64
    // std::sync::AtomicsF64 is not currently implemented in the standard library
    // Currently, we have no other choice than casting an u64 into f64, with possible loss of precision
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn current_time(&self) -> f64 {
        self.inner.frames_played.load(Ordering::SeqCst) as f64 / self.inner.sample_rate as f64
    }

    /// Maximum available channels for the audio destination
    #[must_use]
    pub(crate) fn max_channel_count(&self) -> usize {
        self.inner.max_channel_count
    }

    /// Release queued control messages to the render thread that were blocking on the availability
    /// of the Node with the given `id`
    fn resolve_queued_control_msgs(&self, id: AudioNodeId) {
        // resolve control messages that depend on this registration
        let mut queued = self.inner.queued_messages.lock().unwrap();
        let mut i = 0; // waiting for Vec::drain_filter to stabilize
        while i < queued.len() {
            if matches!(&queued[i], ControlMessage::ConnectNode {to, ..} if *to == id) {
                let m = queued.remove(i);
                self.send_control_msg(m).unwrap();
            } else {
                i += 1;
            }
        }
    }

    /// Connects the output of the `from` audio node to the input of the `to` audio node
    pub(crate) fn connect(&self, from: AudioNodeId, to: AudioNodeId, output: usize, input: usize) {
        let message = ControlMessage::ConnectNode {
            from,
            to,
            output,
            input,
        };
        self.send_control_msg(message).unwrap();
    }

    /// Schedule a connection of an `AudioParam` to the `AudioNode` it belongs to
    ///
    /// It is not performed immediately as the `AudioNode` is not registered at this point.
    pub(super) fn queue_audio_param_connect(&self, param: &AudioParam, audio_node: AudioNodeId) {
        let message = ControlMessage::ConnectNode {
            from: param.registration().id(),
            to: audio_node,
            output: 0,
            input: usize::MAX, // audio params connect to the 'hidden' input port
        };
        self.inner.queued_messages.lock().unwrap().push(message);
    }

    /// Disconnects all outputs of the audio node that go to a specific destination node.
    pub(crate) fn disconnect_from(&self, from: AudioNodeId, to: AudioNodeId) {
        let message = ControlMessage::DisconnectNode { from, to };
        self.send_control_msg(message).unwrap();
    }

    /// Disconnects all outgoing connections from the audio node.
    pub(crate) fn disconnect(&self, from: AudioNodeId) {
        let message = ControlMessage::DisconnectAll { from };
        self.send_control_msg(message).unwrap();
    }

    /// Connect the `AudioListener` to a `PannerNode`
    pub(crate) fn connect_listener_to_panner(&self, panner: AudioNodeId) {
        self.connect(LISTENER_NODE_ID, panner, 0, usize::MAX);
    }

    /// Add the [`AudioListener`] to the audio graph (if not already)
    pub(crate) fn ensure_audio_listener_present(&self) {
        let mut queued_audio_listener_msgs = self.inner.queued_audio_listener_msgs.lock().unwrap();
        let mut released = false;
        while let Some(message) = queued_audio_listener_msgs.pop() {
            // add the AudioListenerRenderer to the graph
            self.send_control_msg(message).unwrap();
            released = true;
        }

        if released {
            // connect the AudioParamRenderers to the Listener
            self.resolve_queued_control_msgs(LISTENER_NODE_ID);

            // hack: Connect the listener to the destination node to force it to render at each
            // quantum. Abuse the magical usize::MAX port so it acts as an AudioParam and has no side
            // effects
            self.connect(LISTENER_NODE_ID, DESTINATION_NODE_ID, 0, usize::MAX);
        }
    }

    /// Returns true if this is `OfflineAudioContext` (false when it is an `AudioContext`)
    pub(crate) fn offline(&self) -> bool {
        self.inner.offline
    }

    pub(crate) fn set_event_handler(&self, event: EventType, callback: EventHandler) {
        self.inner.event_loop.set_handler(event, callback);
    }

    pub(crate) fn clear_event_handler(&self, event: EventType) {
        self.inner.event_loop.clear_handler(event);
    }
}
