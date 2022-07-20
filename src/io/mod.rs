use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use cpal::{Stream, StreamConfig};

use crate::buffer::AudioBuffer;
use crate::AtomicF64;
use crate::context::AudioContextOptions;
use crate::message::ControlMessage;

mod backend_cpal;

/// Builds the output
pub(crate) fn build_output(
    frames_played: Arc<AtomicU64>,
    output_latency: Arc<AtomicF64>,
    options: AudioContextOptions,
) -> (Stream, StreamConfig, Sender<ControlMessage>) {
    backend_cpal::build_output(
        frames_played,
        output_latency,
        options,
        )
}

pub(crate) fn build_input(options: AudioContextOptions) -> (Stream, StreamConfig, Receiver<AudioBuffer>) {
    backend_cpal::build_input(options)
}

