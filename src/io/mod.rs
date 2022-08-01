//! Audio input/output interfaces

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextLatencyCategory, AudioContextOptions};
use crate::message::ControlMessage;
use crate::RENDER_QUANTUM_SIZE;

#[cfg(feature = "cpal")]
mod backend_cpal;

#[cfg(feature = "cubeb")]
mod backend_cubeb;

/// Set up an output stream (speakers) bases on the selected features (cubeb/cpal/none)
pub(crate) fn build_output(
    options: AudioContextOptions,
    frames_played: Arc<AtomicU64>,
) -> (Box<dyn AudioBackend>, Sender<ControlMessage>) {
    #[cfg(feature = "cubeb")]
    {
        let (b, s) = backend_cubeb::CubebBackend::build_output(options, frames_played);
        (Box::new(b), s)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let (b, s) = backend_cpal::CpalBackend::build_output(options, frames_played);
        (Box::new(b), s)
    }
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }
}

/// Set up an input stream (microphone) bases on the selected features (cubeb/cpal/none)
#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub(crate) fn build_input(
    options: AudioContextOptions,
) -> (Box<dyn AudioBackend>, Receiver<AudioBuffer>) {
    #[cfg(feature = "cubeb")]
    {
        let (b, r) = backend_cubeb::CubebBackend::build_input(options);
        (Box::new(b), r)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let (b, r) = backend_cpal::CpalBackend::build_input(options);
        (Box::new(b), r)
    }
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }
}

/// Interface for audio backends
pub(crate) trait AudioBackend: Send + Sync + 'static {
    /// Setup a new output stream (speakers)
    fn build_output(
        options: AudioContextOptions,
        frames_played: Arc<AtomicU64>,
    ) -> (Self, Sender<ControlMessage>)
    where
        Self: Sized;

    /// Setup a new input stream (microphone capture)
    fn build_input(options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized;

    /// Resume or start the stream
    fn resume(&self) -> bool;

    /// Suspend the stream
    fn suspend(&self) -> bool;

    /// Close the stream, freeing all resources. It cannot be started again after closing.
    fn close(&self);

    /// Sample rate of the stream
    fn sample_rate(&self) -> f32;

    /// Number of channels of the stream
    fn number_of_channels(&self) -> usize;

    /// Output latency of the stream in seconds
    ///
    /// This is the difference between the time the backend acquires the data in the callback and
    /// the listener can hear the sound.
    fn output_latency(&self) -> f64;

    /// Clone the stream reference
    fn boxed_clone(&self) -> Box<dyn AudioBackend>;
}

/// Calculate buffer size in frames for a given latency category
fn buffer_size_for_latency_category(
    latency_cat: AudioContextLatencyCategory,
    sample_rate: f32,
) -> usize {
    // at 44100Hz sample rate (this could be even more relaxed):
    // Interactive: 128 samples is 2,9ms
    // Balanced:    512 samples is 11,6ms
    // Playback:    1024 samples is 23,2ms
    match latency_cat {
        AudioContextLatencyCategory::Interactive => RENDER_QUANTUM_SIZE,
        AudioContextLatencyCategory::Balanced => RENDER_QUANTUM_SIZE * 4,
        AudioContextLatencyCategory::Playback => RENDER_QUANTUM_SIZE * 8,
        // buffer_size is always positive and truncation is the desired behavior
        #[allow(clippy::cast_sign_loss)]
        #[allow(clippy::cast_possible_truncation)]
        AudioContextLatencyCategory::Custom(latency) => {
            if latency <= 0. {
                panic!(
                    "RangeError - Invalid custom latency: {:?}, should be strictly positive",
                    latency
                );
            }

            let buffer_size = (latency * sample_rate as f64) as usize;
            buffer_size.next_power_of_two()
        }
    }
}
