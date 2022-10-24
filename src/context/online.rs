//! The `AudioContext` type and constructor options
use crate::context::{AudioContextState, BaseAudioContext, ConcreteBaseAudioContext};
use crate::io::{self, AudioBackend};
use crate::media::{MediaElement, MediaStream};
use crate::node::{self, ChannelConfigOptions};
use crate::AudioRenderCapacity;

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Identify the type of playback, which affects tradeoffs
/// between audio output latency and power consumption
#[derive(Clone, Debug)]
pub enum AudioContextLatencyCategory {
    /// Balance audio output latency and power consumption.
    Balanced,
    /// Provide the lowest audio output latency possible without glitching. This is the default.
    Interactive,
    /// Prioritize sustained playback without interruption
    /// over audio output latency. Lowest power consumption.
    Playback,
    /// Specify the number of seconds of latency
    /// this latency is not guaranted to be applied,
    /// it depends on the audio hardware capabilities
    Custom(f64),
}

impl Default for AudioContextLatencyCategory {
    fn default() -> Self {
        Self::Interactive
    }
}

/// Specify the playback configuration for the [`AudioContext`] constructor.
///
/// All fields are optional and will default to the value best suited for interactive playback on
/// your hardware configuration.
///
/// Check the documentation of the [`AudioContext` constructor](AudioContext::new) for usage
/// instructions.
#[derive(Clone, Debug, Default)]
pub struct AudioContextOptions {
    /// Identify the type of playback, which affects
    /// tradeoffs between audio output latency and power consumption
    pub latency_hint: AudioContextLatencyCategory,
    /// Sample rate of the audio Context and audio output hardware
    pub sample_rate: Option<f32>,
}

/// This interface represents an audio graph whose `AudioDestinationNode` is routed to a real-time
/// output device that produces a signal directed at the user.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct AudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// audio backend (play/pause functionality)
    backend: Box<dyn AudioBackend>,
    /// Provider for rendering performance metrics
    render_capacity: AudioRenderCapacity,
}

impl BaseAudioContext for AudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new(AudioContextOptions::default())
    }
}

impl AudioContext {
    /// Creates and returns a new `AudioContext` object.
    ///
    /// This will play live audio on the default output device.
    ///
    /// ```no_run
    /// use web_audio_api::context::{AudioContext, AudioContextLatencyCategory, AudioContextOptions};
    ///
    /// // Request a sample rate of 44.1 kHz and default latency (buffer size 128, if available)
    /// let opts = AudioContextOptions {
    ///     sample_rate: Some(44100.),
    ///     latency_hint: AudioContextLatencyCategory::Interactive,
    /// };
    ///
    /// // Setup the audio context that will emit to your speakers
    /// let context = AudioContext::new(opts);
    ///
    /// // Alternatively, use the default constructor to get the best settings for your hardware
    /// // let context = AudioContext::default();
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(options: AudioContextOptions) -> Self {
        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        // select backend based on cargo features
        let (backend, sender, cap_recv) = io::build_output(options, frames_played_clone);

        let base = ConcreteBaseAudioContext::new(
            backend.sample_rate(),
            backend.number_of_channels(),
            frames_played,
            sender,
            false,
        );
        base.set_state(AudioContextState::Running);

        // setup AudioRenderCapacity for this context
        let base_clone = base.clone();
        let render_capacity = AudioRenderCapacity::new(base_clone, cap_recv);

        Self {
            base,
            backend,
            render_capacity,
        }
    }

    /// This represents the number of seconds of processing latency incurred by
    /// the `AudioContext` passing the audio from the `AudioDestinationNode`
    /// to the audio subsystem.
    // We don't do any buffering between rendering the audio and sending
    // it to the audio subsystem, so this value is zero. (see Gecko)
    #[allow(clippy::unused_self)]
    #[must_use]
    pub const fn base_latency(&self) -> f64 {
        0.
    }

    /// The estimation in seconds of audio output latency, i.e., the interval
    /// between the time the UA requests the host system to play a buffer and
    /// the time at which the first sample in the buffer is actually processed
    /// by the audio output device.
    #[must_use]
    pub fn output_latency(&self) -> f64 {
        self.backend.output_latency()
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn suspend_sync(&self) {
        if self.backend.suspend() {
            self.base().set_state(AudioContextState::Suspended);
        }
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume_sync(&self) {
        if self.backend.resume() {
            self.base().set_state(AudioContextState::Running);
        }
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic when this function is called multiple times
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn close_sync(&self) {
        self.backend.close();

        self.base().set_state(AudioContextState::Closed);
    }

    /// Creates a [`MediaStreamAudioSourceNode`](node::MediaStreamAudioSourceNode) from a
    /// [`MediaStream`]
    #[must_use]
    pub fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode {
        let opts = node::MediaStreamAudioSourceOptions {
            media_stream: media,
        };
        node::MediaStreamAudioSourceNode::new(self, opts)
    }

    /// Creates a [`MediaStreamAudioDestinationNode`](node::MediaStreamAudioDestinationNode)
    #[must_use]
    pub fn create_media_stream_destination(&self) -> node::MediaStreamAudioDestinationNode {
        let opts = ChannelConfigOptions::default();
        node::MediaStreamAudioDestinationNode::new(self, opts)
    }

    /// Creates a [`MediaElementAudioSourceNode`](node::MediaElementAudioSourceNode) from a
    /// [`MediaElement`]
    #[must_use]
    pub fn create_media_element_source(
        &self,
        media_element: &mut MediaElement,
    ) -> node::MediaElementAudioSourceNode {
        let opts = node::MediaElementAudioSourceOptions { media_element };
        node::MediaElementAudioSourceNode::new(self, opts)
    }

    /// Returns an [`AudioRenderCapacity`] instance associated with an AudioContext.
    #[must_use]
    pub fn render_capacity(&self) -> &AudioRenderCapacity {
        &self.render_capacity
    }
}
