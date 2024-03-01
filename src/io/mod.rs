//! Audio input/output interfaces

use std::sync::atomic::{AtomicU64, AtomicU8};
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextLatencyCategory, AudioContextOptions, AudioContextState};
use crate::events::EventDispatch;
use crate::media_devices::MediaDeviceInfo;
use crate::media_streams::{MediaStream, MediaStreamTrack};
use crate::message::ControlMessage;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

mod none;
pub(crate) use none::NoneBackend;

#[cfg(feature = "cpal")]
mod cpal;

#[cfg(feature = "cubeb")]
mod cubeb;

#[cfg(any(feature = "cubeb", feature = "cpal"))]
mod microphone;

#[derive(Debug)]
pub(crate) struct ControlThreadInit {
    pub state: Arc<AtomicU8>,
    pub frames_played: Arc<AtomicU64>,
    pub ctrl_msg_send: Sender<ControlMessage>,
    pub load_value_recv: Receiver<AudioRenderCapacityLoad>,
    pub event_send: Sender<EventDispatch>,
    pub event_recv: Receiver<EventDispatch>,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderThreadInit {
    pub state: Arc<AtomicU8>,
    pub frames_played: Arc<AtomicU64>,
    pub ctrl_msg_recv: Receiver<ControlMessage>,
    pub load_value_send: Sender<AudioRenderCapacityLoad>,
    pub event_send: Sender<EventDispatch>,
}

pub(crate) fn thread_init() -> (ControlThreadInit, RenderThreadInit) {
    // Track audio context state - synced from render thread to control thread
    let state = Arc::new(AtomicU8::new(AudioContextState::Suspended as u8));

    // Track number of frames - synced from render thread to control thread
    let frames_played = Arc::new(AtomicU64::new(0));

    // Communication channel for ctrl msgs from the control thread to the render thread.
    // Use a bounded channel for real-time safety. A maximum of 256 control messages (add/remove
    // node, settings, ..) will be handled per render quantum. The control thread will block when
    // the capacity is reached.
    let (ctrl_msg_send, ctrl_msg_recv) = crossbeam_channel::bounded(256);

    // Communication channel for render load values.
    // A dedicated thread is consuming these messages so there is no need for buffering.
    let (load_value_send, load_value_recv) = crossbeam_channel::bounded(1);

    // Communication channel for events from the render thread to the control thread.
    // Use a bounded channel for real-time safety. A maximum of 256 events (node ended, error, ..)
    // will be sent per render quantum. Excess events are dropped when the capacity is reached.
    let (event_send, event_recv) = crossbeam_channel::bounded(256);

    let control_thread_init = ControlThreadInit {
        state: Arc::clone(&state),
        frames_played: Arc::clone(&frames_played),
        ctrl_msg_send,
        load_value_recv,
        event_send: event_send.clone(),
        event_recv,
    };

    let render_thread_init = RenderThreadInit {
        state,
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
        let backend = NoneBackend::build_output(options, render_thread_init);
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
pub(crate) fn build_input(
    options: AudioContextOptions,
    number_of_channels: Option<u32>,
) -> MediaStream {
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }

    #[cfg(any(feature = "cubeb", feature = "cpal"))]
    {
        let (backend, receiver) = {
            #[cfg(feature = "cubeb")]
            {
                cubeb::CubebBackend::build_input(options, number_of_channels)
            }

            #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
            {
                cpal::CpalBackend::build_input(options, number_of_channels)
            }
        };

        let media_iter = microphone::MicrophoneStream::new(receiver, Box::new(backend));
        let track = MediaStreamTrack::from_iter(media_iter);
        MediaStream::from_tracks(vec![track])
    }
}

/// Interface for audio backends
pub(crate) trait AudioBackendManager: Send + Sync + 'static {
    /// Name of the concrete implementation - for debug purposes
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Setup a new output stream (speakers)
    fn build_output(options: AudioContextOptions, render_thread_init: RenderThreadInit) -> Self
    where
        Self: Sized;

    /// Setup a new input stream (microphone capture)
    fn build_input(
        options: AudioContextOptions,
        number_of_channels: Option<u32>,
    ) -> (Self, Receiver<AudioBuffer>)
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

    fn enumerate_devices_sync() -> Vec<MediaDeviceInfo>
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
            assert!(
                latency > 0.,
                "RangeError - Invalid custom latency: {:?}, should be strictly positive",
                latency
            );

            let buffer_size = (latency * sample_rate as f64) as usize;
            buffer_size.next_power_of_two()
        }
    }
}

pub(crate) fn enumerate_devices_sync() -> Vec<MediaDeviceInfo> {
    #[cfg(feature = "cubeb")]
    {
        cubeb::CubebBackend::enumerate_devices_sync()
    }

    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        cpal::CpalBackend::enumerate_devices_sync()
    }

    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
}
