//! The `AudioContext` type and constructor options
use crate::context::{AudioContextState, BaseAudioContext, ConcreteBaseAudioContext};
use crate::media::{MediaElement, MediaStream};
use crate::node::{self, ChannelConfigOptions};
use crate::AtomicF64;

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

#[cfg(not(test))]
use std::sync::Mutex;

#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Stream};

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
    /// cpal stream (play/pause functionality)
    #[cfg(not(test))] // in tests, do not set up a cpal Stream
    stream: ThreadSafeClosableStream,
    /// delay between render and actual system audio output
    output_latency: Arc<AtomicF64>,
}

#[cfg(not(test))]
mod private {
    use super::*;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<Stream>>>);

    impl ThreadSafeClosableStream {
        pub fn new(stream: Stream) -> Self {
            Self(Arc::new(Mutex::new(Some(stream))))
        }

        pub fn close(&self) {
            self.0.lock().unwrap().take(); // will Drop
        }

        pub fn resume(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.play() {
                    panic!("Error resuming cpal stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn suspend(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.pause() {
                    panic!("Error suspending cpal stream: {:?}", e);
                }
                return true;
            }

            false
        }
    }

    // SAFETY:
    // The cpal `Stream` is marked !Sync and !Send because some platforms are not thread-safe
    // https://github.com/RustAudio/cpal/commit/33ddf749548d87bf54ce18eb342f954cec1465b2
    // Since we wrap the Stream in a Mutex, we should be fine
    unsafe impl Sync for ThreadSafeClosableStream {}
    unsafe impl Send for ThreadSafeClosableStream {}
}

#[cfg(not(test))]
use private::ThreadSafeClosableStream;

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
    #[cfg(not(test))]
    #[must_use]
    pub fn new(options: AudioContextOptions) -> Self {
        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        let output_latency = Arc::new(AtomicF64::new(0.));
        let output_latency_clone = output_latency.clone();

        let (stream, config, sender) =
            io::build_output(frames_played_clone, output_latency_clone, options);

        let number_of_channels = usize::from(config.channels);
        let sample_rate = config.sample_rate.0 as f32;

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            false,
        );
        base.set_state(AudioContextState::Running);

        Self {
            base,
            stream: ThreadSafeClosableStream::new(stream),
            output_latency,
        }
    }

    #[cfg(test)] // in tests, do not set up a cpal Stream
    #[allow(clippy::must_use_candidate)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(options: AudioContextOptions) -> Self {
        let sample_rate = options.sample_rate.unwrap_or(44100.);
        let number_of_channels = 2;

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let frames_played = Arc::new(AtomicU64::new(0));
        let output_latency = Arc::new(AtomicF64::new(0.));

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            false,
        );
        base.set_state(AudioContextState::Running);

        Self {
            base,
            output_latency,
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
        self.output_latency.load()
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// This function operates synchronously and might block the current thread.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn suspend_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        if self.stream.suspend() {
            self.base().set_state(AudioContextState::Suspended);
        }
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[cfg(not(test))]
    pub async fn suspend(&self) {
        // The cpal backend does not provide callback info after suspending, so just async run the
        // sync version. When we add other backends this might change.

        // make 'static vars for ease of use
        let stream = self.stream.clone();
        let base = self.base().clone();

        async move {
            if stream.suspend() {
                base.set_state(AudioContextState::Suspended);
            }
        }
        .await
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// This function operates synchronously and might block the current thread.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        if self.stream.resume() {
            self.base().set_state(AudioContextState::Running);
        }
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[cfg(not(test))]
    pub async fn resume(&self) {
        // The cpal backend does not provide callback info after resuming, so just async run the
        // sync version. When we add other backends this might change.

        // make 'static vars for ease of use
        let stream = self.stream.clone();
        let base = self.base().clone();

        async move {
            if stream.resume() {
                base.set_state(AudioContextState::Running);
            }
        }
        .await
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// This function operates synchronously and might block the current thread.
    ///
    /// # Panics
    ///
    /// Will panic when this function is called multiple times
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn close_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.close();

        self.base().set_state(AudioContextState::Closed);
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// # Panics
    ///
    /// Will panic when this function is called multiple times
    #[cfg(not(test))]
    pub async fn close(&self) {
        // The cpal backend does not provide callback info after resuming, so just async run the
        // sync version. When we add other backends this might change.

        // make 'static vars for ease of use
        let stream = self.stream.clone();
        let base = self.base().clone();

        async move {
            stream.close();
            base.set_state(AudioContextState::Closed);
        }
        .await
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
}
