//! Audio processing code that runs on the audio rendering thread
use crate::context::{AudioNodeId, AudioParamId};
use crate::events::{ErrorEvent, EventDispatch};
use crate::{Event, RENDER_QUANTUM_SIZE};

use super::{graph::Node, AudioRenderQuantum};

use crossbeam_channel::Sender;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};

use std::any::Any;
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

    pub(crate) node_id: Cell<AudioNodeId>,
    pub(crate) event_sender: Option<Sender<EventDispatch>>,
}

impl RenderScope {
    pub(crate) fn send_ended_event(&self) {
        if let Some(sender) = self.event_sender.as_ref() {
            // sending could fail if the channel is saturated or the main thread is shutting down
            let _ = sender.try_send(EventDispatch::ended(self.node_id.get()));
        }
    }

    pub(crate) fn report_error(&self, error: Box<dyn Any + Send>) {
        pub fn type_name_of_val<T: ?Sized>(_val: &T) -> &'static str {
            std::any::type_name::<T>()
        }
        let message = if let Some(v) = error.downcast_ref::<String>() {
            v.to_string()
        } else if let Some(v) = error.downcast_ref::<&str>() {
            v.to_string()
        } else {
            type_name_of_val(&error).to_string()
        };
        eprintln!(
            "Panic occurred in Audio Processor: '{}'. Removing node from graph.",
            &message
        );

        if let Some(sender) = self.event_sender.as_ref() {
            let event = ErrorEvent {
                message,
                error,
                event: Event {
                    type_: "ErrorEvent",
                },
            };
            let _ = sender.try_send(EventDispatch::processor_error(self.node_id.get(), event));
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
        params: AudioParamValues<'_>,
        scope: &RenderScope,
    ) -> bool;

    /// Handle incoming messages from the linked AudioNode
    ///
    /// By overriding this method you can add a handler for messages sent from the control thread
    /// via
    /// [`AudioContextRegistration::post_message`](crate::context::AudioContextRegistration::post_message).
    /// This will not be necessary for most processors.
    ///
    /// Receivers are supposed to consume the content of `msg`. The content of `msg` might
    /// also be replaced by cruft that needs to be deallocated outside of the render thread
    /// afterwards, e.g. when replacing an internal buffer.
    ///
    /// This method is just a shim of the full
    /// [`MessagePort`](https://webaudio.github.io/web-audio-api/#dom-audioworkletprocessor-port)
    /// `onmessage` functionality of the AudioWorkletProcessor.
    #[allow(unused_variables)]
    fn onmessage(&mut self, msg: &mut dyn Any) {
        log::warn!("Ignoring incoming message");
    }
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
    nodes: &'a FxHashMap<AudioNodeId, RefCell<Node>>,
}

impl<'a> AudioParamValues<'a> {
    pub(crate) fn from(nodes: &'a FxHashMap<AudioNodeId, RefCell<Node>>) -> Self {
        Self { nodes }
    }

    /// Get the computed values for the given [`crate::param::AudioParam`]
    ///
    /// For k-rate params or if the (a-rate) parameter is constant for this block, it will provide
    /// a slice of length 1. In other cases, i.e. a-rate param with scheduled automations it will
    /// provide a slice of length equal to the render quantum size (default: 128)
    #[allow(clippy::missing_panics_doc)]
    pub fn get(&self, index: &AudioParamId) -> impl Deref<Target = [f32]> + '_ {
        DerefAudioRenderQuantumChannel(self.nodes.get(&index.into()).unwrap().borrow())
    }

    pub(crate) fn listener_params(&self) -> [impl Deref<Target = [f32]> + '_; 9] {
        crate::context::LISTENER_AUDIO_PARAM_IDS.map(|p| self.get(&p))
    }
}
