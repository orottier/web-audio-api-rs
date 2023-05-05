//! Audio input/output interfaces

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextLatencyCategory, AudioContextOptions};
use crate::events::EventDispatch;
use crate::media_devices::MediaDeviceInfo;
use crate::media_streams::{MediaStream, MediaStreamTrack};
use crate::message::ControlMessage;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

mod none;

#[cfg(feature = "cpal")]
mod cpal;

#[cfg(feature = "cubeb")]
mod cubeb;

#[cfg(any(feature = "cubeb", feature = "cpal"))]
mod microphone;

#[derive(Debug)]
pub(crate) struct ControlThreadInit {
    pub frames_played: Arc<AtomicU64>,
    pub ctrl_msg_send: Sender<ControlMessage>,
    pub load_value_recv: Receiver<AudioRenderCapacityLoad>,
    pub event_send: Sender<EventDispatch>,
    pub event_recv: Receiver<EventDispatch>,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderThreadInit {
    pub frames_played: Arc<AtomicU64>,
    pub ctrl_msg_recv: Receiver<ControlMessage>,
    pub load_value_send: Sender<AudioRenderCapacityLoad>,
    pub event_send: Sender<EventDispatch>,
}

pub(crate) fn thread_init() -> (ControlThreadInit, RenderThreadInit) {
    // track number of frames - synced from render thread to control thread
    let frames_played = Arc::new(AtomicU64::new(0));
    // communication channel for ctrl msgs to the render thread
    let (ctrl_msg_send, ctrl_msg_recv) = crossbeam_channel::unbounded();
    // communication channel for render load values
    let (load_value_send, load_value_recv) = crossbeam_channel::bounded(1);
    // communication channel for events for render thread to control thread
    let (event_send, event_recv) = crossbeam_channel::unbounded();

    let control_thread_init = ControlThreadInit {
        frames_played: frames_played.clone(),
        ctrl_msg_send,
        load_value_recv,
        event_send: event_send.clone(),
        event_recv,
    };

    let render_thread_init = RenderThreadInit {
        frames_played,
        ctrl_msg_recv,
        load_value_send,
        event_send,
    };

    (control_thread_init, render_thread_init)
}

/// Set up an output stream (speakers) bases on the selected features (cubeb/cpal/none)
pub(crate) fn build_output(
    options: AudioContextOptions,
    render_thread_init: RenderThreadInit,
) -> Box<dyn AudioBackendManager> {
    if options.sink_id == "none" {
        let backend = none::NoneBackend::build_output(options, render_thread_init);
        return Box::new(backend);
    }

    #[cfg(feature = "cubeb")]
    {
        let backend = cubeb::CubebBackend::build_output(options, render_thread_init);
        Box::new(backend)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let backend = cpal::CpalBackend::build_output(options, render_thread_init);
        Box::new(backend)
    }
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }
}

/// Set up an input stream (microphone) bases on the selected features (cubeb/cpal/none)
pub(crate) fn build_input(options: AudioContextOptions) -> MediaStream {
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }

    #[cfg(any(feature = "cubeb", feature = "cpal"))]
    {
        let (backend, receiver) = {
            #[cfg(feature = "cubeb")]
            {
                cubeb::CubebBackend::build_input(options)
            }

            #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
            {
                cpal::CpalBackend::build_input(options)
            }
        };

        let media_iter = microphone::MicrophoneStream::new(receiver, Box::new(backend));
        let track = MediaStreamTrack::from_iter(media_iter);
        MediaStream::from_tracks(vec![track])
    }
}

/// Interface for audio backends
pub(crate) trait AudioBackendManager: Send + Sync + 'static {
    /// Setup a new output stream (speakers)
    fn build_output(options: AudioContextOptions, render_thread_init: RenderThreadInit) -> Self
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

    /// The audio output device - `""` means the default device
    fn sink_id(&self) -> &str;

    /// Clone the stream reference
    fn boxed_clone(&self) -> Box<dyn AudioBackendManager>;

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

pub(crate) fn enumerate_devices() -> Vec<MediaDeviceInfo> {
    #[cfg(feature = "cubeb")]
    {
        crate::io::cubeb::CubebBackend::enumerate_devices()
    }

    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        crate::io::cpal::CpalBackend::enumerate_devices()
    }

    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
}
