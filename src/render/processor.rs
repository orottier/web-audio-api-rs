//! Audio processing code that runs on the audio rendering thread

use std::collections::HashMap;

use crate::context::AudioParamId;
use crate::SampleRate;

use super::{graph::Node, AudioRenderQuantum, NodeIndex};

#[non_exhaustive] // we may want to add user-provided blobs to this later
/// The execution context of all AudioProcessors in a given AudioContext
///
/// This struct currently only contains information about the progress of time. In a future
/// version, it should be possible to add arbitrary data. For example, multiple processors might
/// share a buffer defining a wavetable or an impulse response.
pub struct RenderScope {
    pub current_frame: u64,
    pub current_time: f64,
    pub sample_rate: SampleRate,
}

/// Interface for audio processing code that runs on the audio rendering thread.
///
/// Note that the AudioProcessor is typically constructed together with an [`crate::node::AudioNode`]
/// (the user facing object that lives in the control thread). See [`crate::context::ConcreteBaseAudioContext::register`].
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

/// Accessor for current [`crate::param::AudioParam`] values
///
/// Provided to implementations of [`AudioProcessor`] in the render thread
pub struct AudioParamValues<'a> {
    nodes: &'a HashMap<NodeIndex, Node>,
}

impl<'a> AudioParamValues<'a> {
    pub(crate) fn from(nodes: &'a HashMap<NodeIndex, Node>) -> Self {
        Self { nodes }
    }

    pub(crate) fn get_raw(&self, index: &AudioParamId) -> &AudioRenderQuantum {
        self.nodes.get(&index.into()).unwrap().get_buffer()
    }

    /// Get the computed values for the given [`crate::param::AudioParam`]
    ///
    /// For both A & K-rate params, it will provide a slice of length [`crate::RENDER_QUANTUM_SIZE`]
    pub fn get(&self, index: &AudioParamId) -> &[f32] {
        &self.get_raw(index).channel_data(0)[..]
    }
}
