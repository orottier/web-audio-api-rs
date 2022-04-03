//! The `AudioContext` type and constructor options
use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};
use crate::SampleRate;
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
    Specific(f64),
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
    pub sample_rate: Option<u32>,
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
    stream: Mutex<Option<Stream>>,
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
    ///     sample_rate: Some(44100),
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

        let (stream, config, sender) = io::build_output(frames_played_clone, options);
        let number_of_channels = u32::from(config.channels);
        let sample_rate = SampleRate(config.sample_rate.0);

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            false,
        );

        Self {
            base,
            stream: Mutex::new(Some(stream)),
        }
    }

    #[cfg(test)] // in tests, do not set up a cpal Stream
    #[allow(clippy::must_use_candidate)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(options: AudioContextOptions) -> Self {
        let sample_rate = SampleRate(options.sample_rate.unwrap_or(44_100));
        let number_of_channels = 2u32;

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let frames_played = Arc::new(AtomicU64::new(0));

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            false,
        );

        Self { base }
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn suspend_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        if let Some(s) = self.stream.lock().unwrap().as_ref() {
            if let Err(e) = s.pause() {
                panic!("Error suspending cpal stream: {:?}", e);
            }
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        if let Some(s) = self.stream.lock().unwrap().as_ref() {
            if let Err(e) = s.play() {
                panic!("Error resuming cpal stream: {:?}", e);
            }
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn close_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.lock().unwrap().take(); // will Drop
    }
}
