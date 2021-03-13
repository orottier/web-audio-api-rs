//! Audio processing code that runs on the audio rendering thread

use std::collections::HashMap;

use crate::buffer::AudioBuffer;
use crate::context::AudioParamId;
use crate::graph::{Node, NodeIndex};
use crate::SampleRate;

/// Interface for audio processing code that runs on the audio rendering thread.
///
/// Note that the AudioProcessor is typically constructed together with an `AudioNode`
/// (the user facing object that lives in the control thread). See `[crate::context::BaseAudioContext::register]`.
pub trait AudioProcessor: Send {
    /// Render an audio quantum for the given timestamp and input buffers
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    );

    /// Indicate if this Node currently has tail-time, meaning it can provide output when no inputs are supplied.
    ///
    /// Tail time is `true` for source nodes (as long as they are still generating audio).
    ///
    /// Tail time is `false` for nodes that only transform their inputs.
    fn tail_time(&self) -> bool;
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

    pub(crate) fn get_raw(&self, index: &AudioParamId) -> &AudioBuffer {
        self.nodes.get(&index.into()).unwrap().get_buffer()
    }

    /// Get the computed values for the given [`crate::param::AudioParam`]
    ///
    /// For both A & K-rate params, it will provide a slice of length [`crate::BUFFER_SIZE`]
    pub fn get(&self, index: &AudioParamId) -> &[f32] {
        self.get_raw(index).channel_data(0).unwrap().as_slice()
    }
}
