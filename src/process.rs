//! Audio processing code that runs on the audio rendering thread

use crate::buffer::AudioBuffer;
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
