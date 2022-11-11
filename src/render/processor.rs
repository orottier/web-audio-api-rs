//! Audio processing code that runs on the audio rendering thread
use crate::context::AudioParamId;
use crate::events::{EventType, TriggerEventMessage};
use crate::RENDER_QUANTUM_SIZE;

use super::{graph::Node, AudioRenderQuantum, NodeIndex};

use crossbeam_channel::Sender;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};
use std::ops::Deref;

#[non_exhaustive] // we may want to add user-provided blobs to this later
/// The execution context of all AudioProcessors in a given AudioContext
///
/// This struct currently only contains information about the progress of time. In a future
/// version, it should be possible to add arbitrary data. For example, multiple processors might
/// share a buffer defining a wavetable or an impulse response.
pub struct RenderScope {
    pub current_frame: u64,
    pub current_time: f64,
    pub sample_rate: f32,

    pub(crate) node_id: Cell<u64>,
    pub(crate) event_sender: Option<Sender<TriggerEventMessage>>,
}

impl RenderScope {
    pub(crate) fn send_event(&self, event_type: EventType) {
        if let Some(sender) = self.event_sender.as_ref() {
            let message = TriggerEventMessage {
                node_id: self.node_id.get(),
                event_type,
            };
            let _ = sender.try_send(message);
        }
    }
}

/// Interface for audio processing code that runs on the audio rendering thread.
///
/// Note that the AudioProcessor is typically constructed together with an
/// [`AudioNode`](crate::node::AudioNode) (the user facing object that lives in the control
/// thread). See [`BaseAudioContext::register`](crate::context::BaseAudioContext::register).
///
/// Check the `examples/worklet.rs` file for example usage of this trait.
pub trait AudioProcessor: Send {
    /// Audio processing function
    ///
    /// # Arguments
    ///
    /// - inputs: readonly array of input buffers
    /// - outputs: array of output buffers
    /// - params: available `AudioParam`s for this processor
    /// - timestamp: time of the start of this render quantum
    /// - sample_rate: sample rate of this render quantum
    ///
    /// # Return value
    ///
    /// The return value (bool) of this callback controls the lifetime of the processor.
    ///
    /// - return `false` when the node only transforms their inputs, and as such can be removed when
    /// the inputs are disconnected (e.g. GainNode)
    /// - return `true` for some time when the node still outputs after the inputs are disconnected
    /// (e.g. DelayNode)
    /// - return `true` as long as this node is a source of output (e.g. OscillatorNode)
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        scope: &RenderScope,
    ) -> bool;
}

struct DerefAudioRenderQuantumChannel<'a>(std::cell::Ref<'a, Node>);

impl Deref for DerefAudioRenderQuantumChannel<'_> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        let buffer = self.0.get_buffer();
        let len = if buffer.single_valued() {
            1
        } else {
            RENDER_QUANTUM_SIZE
        };

        &buffer.channel_data(0)[..len]
    }
}

/// Accessor for current [`crate::param::AudioParam`] values
///
/// Provided to implementations of [`AudioProcessor`] in the render thread
pub struct AudioParamValues<'a> {
    nodes: &'a FxHashMap<NodeIndex, RefCell<Node>>,
}

impl<'a> AudioParamValues<'a> {
    pub(crate) fn from(nodes: &'a FxHashMap<NodeIndex, RefCell<Node>>) -> Self {
        Self { nodes }
    }

    pub(crate) fn get_raw(&self, index: &AudioParamId) -> AudioRenderQuantum {
        self.nodes
            .get(&index.into())
            .unwrap()
            .borrow()
            .get_buffer()
            .clone()
    }

    /// Get the computed values for the given [`crate::param::AudioParam`]
    ///
    /// For k-rate params or if the (a-rate) parameter is constant for this block, it will
    /// provide a slice of length 1. In other cases, i.e. a-rate param with scheduled
    /// automations it will provide a slice of length [`crate::RENDER_QUANTUM_SIZE`]
    #[allow(clippy::missing_panics_doc)]
    pub fn get(&self, index: &AudioParamId) -> impl std::ops::Deref<Target = [f32]> + '_ {
        DerefAudioRenderQuantumChannel(self.nodes.get(&index.into()).unwrap().borrow())
    }
}
