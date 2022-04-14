//! The `AudioContext` type and constructor options
use crate::context::{AudioContextState, BaseAudioContext, ConcreteBaseAudioContext};
use crate::media::MediaStream;
use crate::node::{self, ChannelConfigOptions};
use crate::{AtomicF64, SampleRate, TIME_ORIGIN};

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

/// Struct returned by `get_output_timestamp` that associate estimated playback
/// context time (i.e. `current_time - output_latency`) to a monotonic global clock
/// (i.e. <https://doc.rust-lang.org/std/time/struct.Instant.html>)
#[derive(Clone, Debug, Default)]
pub struct AudioTimestamp {
    /// Represents a point in the time coordinate system of BaseAudioContext’s currentTime.
    pub context_time: f64,
    /// Represents a point in the time coordinate system of a Performance interface implementation
    pub performance_time: f64,
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

        let output_latency = Arc::new(AtomicF64::new(0.));
        let output_latency_clone = output_latency.clone();

        let (stream, config, sender) =
            io::build_output(frames_played_clone, output_latency_clone, options);

        let number_of_channels = usize::from(config.channels);
        let sample_rate = SampleRate(config.sample_rate.0);

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            output_latency,
            sender,
            false,
        );
        base.set_state(AudioContextState::Running);

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
        let number_of_channels = 2;

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let frames_played = Arc::new(AtomicU64::new(0));
        let output_latency = Arc::new(AtomicF64::new(0.));

        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            output_latency,
            sender,
            false,
        );
        base.set_state(AudioContextState::Running);

        Self { base }
    }

    /// This represents the number of seconds of processing latency incurred by
    /// the `AudioContext` passing the audio from the `AudioDestinationNode`
    /// to the audio subsystem.
    // We don't do any buffering between rendering the audio and sending
    // it to the audio subsystem, so this value is zero. (see Gecko)
    #[must_use]
    pub fn base_latency(&self) -> f64 {
        self.base().base_latency()
    }

    /// The estimation in seconds of audio output latency, i.e., the interval
    /// between the time the UA requests the host system to play a buffer and
    /// the time at which the first sample in the buffer is actually processed
    /// by the audio output device.
    #[must_use]
    pub fn output_latency(&self) -> f64 {
        self.base().output_latency()
    }

    /// Returns a new `AudioTimestamp` instance containing two related audio stream
    /// position values for the context: the `context_time` member contains the time of
    /// the sample frame which is currently being rendered by the audio output device
    /// (i.e. output audio stream position), in the same units and origin as context’s
    /// `current_time`; the `performance_time` member contains the time estimating the
    /// moment when the sample frame corresponding to the stored `context_time` value was
    /// rendered by the audio output device, in milliseconds using `TIME_ORIGIN` as
    /// the origin.
    #[must_use]
    pub fn get_output_timestamp(&self) -> AudioTimestamp {
        // @todo - [spec] If the context’s rendering graph has not yet processed a block
        // of audio, then getOutputTimestamp call returns an AudioTimestamp instance
        // with both members containing zero.

        let output_latency = self.output_latency();
        // estimated current_time (i.e. position) at output
        let context_time = (self.current_time() - output_latency).max(0.);
        // estimated performance_time at output
        let now = TIME_ORIGIN.elapsed();
        let performance_time_sec =
            now.as_secs() as f64 + now.subsec_nanos() as f64 * 1e-9 - output_latency;
        // performance time is in ms
        let performance_time = (performance_time_sec * 1000.).max(0.);

        AudioTimestamp {
            context_time,
            performance_time,
        }
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        if let Some(s) = self.stream.lock().unwrap().as_ref() {
            if let Err(e) = s.play() {
                panic!("Error resuming cpal stream: {:?}", e);
            }
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn close_sync(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.lock().unwrap().take(); // will Drop
        self.base().set_state(AudioContextState::Closed);
    }

    /// Creates a `MediaStreamAudioSourceNode` from a [`MediaStream`]
    #[must_use]
    pub fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode {
        let opts = node::MediaStreamAudioSourceOptions {
            media_stream: media,
        };
        node::MediaStreamAudioSourceNode::new(self.base(), opts)
    }

    /// Creates a `MediaStreamAudioDestinationNode`
    #[must_use]
    pub fn create_media_stream_destination(&self) -> node::MediaStreamAudioDestinationNode {
        let opts = ChannelConfigOptions::default();
        node::MediaStreamAudioDestinationNode::new(self.base(), opts)
    }
}
