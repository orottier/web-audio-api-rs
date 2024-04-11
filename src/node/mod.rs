//! The AudioNode interface and concrete types

use std::f32::consts::PI;
use std::sync::OnceLock;

use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::AudioBufferIter;

// traits
mod audio_node;
pub use audio_node::*;
mod scheduled_source;
pub use scheduled_source::*;

// nodes
mod analyser;
pub use analyser::*;
mod audio_buffer_source;
pub use audio_buffer_source::*;
mod biquad_filter;
pub use biquad_filter::*;
mod channel_merger;
pub use channel_merger::*;
mod channel_splitter;
pub use channel_splitter::*;
mod constant_source;
pub use constant_source::*;
mod convolver;
pub use convolver::*;
mod delay;
pub use delay::*;
mod destination;
pub use destination::*;
mod dynamics_compressor;
pub use dynamics_compressor::*;
mod gain;
pub use gain::*;
mod iir_filter;
pub use iir_filter::*;
mod media_element_source;
pub use media_element_source::*;
mod media_stream_destination;
pub use media_stream_destination::*;
mod media_stream_source;
pub use media_stream_source::*;
mod media_stream_track_source;
pub use media_stream_track_source::*;
mod oscillator;
pub use oscillator::*;
mod panner;
pub use panner::*;
mod script_processor;
pub use script_processor::*;
mod stereo_panner;
pub use stereo_panner::*;
mod waveshaper;
pub use waveshaper::*;

pub(crate) const TABLE_LENGTH_USIZE: usize = 8192;
pub(crate) const TABLE_LENGTH_BY_4_USIZE: usize = TABLE_LENGTH_USIZE / 4;

pub(crate) const TABLE_LENGTH_F32: f32 = TABLE_LENGTH_USIZE as f32;
pub(crate) const TABLE_LENGTH_BY_4_F32: f32 = TABLE_LENGTH_BY_4_USIZE as f32;

/// Precomputed sine table
pub(crate) fn precomputed_sine_table() -> &'static [f32] {
    static INSTANCE: OnceLock<Vec<f32>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        // Compute one period sine wavetable of size TABLE_LENGTH
        (0..TABLE_LENGTH_USIZE)
            .map(|x| ((x as f32) * 2.0 * PI * (1. / (TABLE_LENGTH_F32))).sin())
            .collect()
    })
}

// `MediaStreamRenderer` is internally used by `MediaElementAudioSourceNode` and
// `MediaStreamAudioSourceNode`.
struct MediaStreamRenderer<R> {
    stream: R,
    finished: bool,
}

impl<R> MediaStreamRenderer<R> {
    fn new(stream: R) -> Self {
        Self {
            stream,
            // scheduler,
            finished: false,
        }
    }
}

impl<R: AudioBufferIter> AudioProcessor for MediaStreamRenderer<R> {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // @note - maybe we need to disciminate between a paused and depleted term
        match self.stream.next() {
            Some(Ok(buffer)) => {
                let channels = buffer.number_of_channels();
                output.set_number_of_channels(channels);
                output
                    .channels_mut()
                    .iter_mut()
                    .zip(buffer.channels())
                    .for_each(|(o, i)| o.copy_from_slice(i.as_slice()));
            }
            Some(Err(e)) => {
                log::warn!("Error playing audio stream: {}", e);
                self.finished = true; // halt playback
                output.make_silent()
            }
            None => {
                if !self.finished {
                    log::debug!("Stream finished");
                    self.finished = true;
                }
                output.make_silent()
            }
        }

        !self.finished
    }
}
