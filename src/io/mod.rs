//! Audio input/output interfaces

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextLatencyCategory, AudioContextOptions};
use crate::message::ControlMessage;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

#[cfg(feature = "cpal")]
mod backend_cpal;

#[cfg(feature = "cubeb")]
mod backend_cubeb;

/// List the available media output devices, such as speakers, headsets, loopbacks, etc
///
/// The media device_id can be used to specify the `sink_id` of the AudioContext
pub fn enumerate_devices() -> Vec<MediaDeviceInfo> {
    #[cfg(feature = "cubeb")]
    {
        backend_cubeb::CubebBackend::enumerate_devices()
    }

    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        backend_cpal::CpalBackend::enumerate_devices()
    }

    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
}

/// Set up an output stream (speakers) bases on the selected features (cubeb/cpal/none)
pub(crate) fn build_output(
    options: AudioContextOptions,
    frames_played: Arc<AtomicU64>,
) -> (
    Box<dyn AudioBackend>,
    Sender<ControlMessage>,
    Receiver<AudioRenderCapacityLoad>,
) {
    #[cfg(feature = "cubeb")]
    {
        let (b, s, r) = backend_cubeb::CubebBackend::build_output(options, frames_played);
        (Box::new(b), s, r)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let (b, s, r) = backend_cpal::CpalBackend::build_output(options, frames_played);
        (Box::new(b), s, r)
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
    ) -> (
        Self,
        Sender<ControlMessage>,
        Receiver<AudioRenderCapacityLoad>,
    )
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

    fn enumerate_devices() -> Vec<MediaDeviceInfo>
    where
        Self: Sized;
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

/// Describes input/output type of a media device
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MediaDeviceInfoKind {
    VideoInput,
    AudioInput,
    AudioOutput,
}

/// Describes a single media input or output device
#[derive(Clone, Debug)]
pub struct MediaDeviceInfo {
    device_id: String,
    group_id: Option<String>,
    kind: MediaDeviceInfoKind,
    label: String,
}

impl MediaDeviceInfo {
    pub(crate) fn new(
        device_id: String,
        group_id: Option<String>,
        kind: MediaDeviceInfoKind,
        label: String,
    ) -> Self {
        Self {
            device_id,
            group_id,
            kind,
            label,
        }
    }

    /// Identifier for the represented device
    ///
    /// The current implementation is not stable across sessions so you should not persist this
    /// value
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Two devices have the same group identifier if they belong to the same physical device
    pub fn group_id(&self) -> Option<&str> {
        self.group_id.as_deref()
    }

    /// Enumerated value that is either "videoinput", "audioinput" or "audiooutput".
    pub fn kind(&self) -> MediaDeviceInfoKind {
        self.kind
    }

    /// Friendly label describing this device
    pub fn label(&self) -> &str {
        &self.label
    }
}
