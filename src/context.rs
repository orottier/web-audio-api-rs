//! The `BaseAudioContext` interface and the `AudioContext` and `OfflineAudioContext` types
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use delegate_attr::delegate;

// magic node values
/// Destination node id is always at index 0
const DESTINATION_NODE_ID: u64 = 0;
/// listener node id is always at index 1
const LISTENER_NODE_ID: u64 = 1;
/// listener audio parameters ids are always at index 2 through 12
const LISTENER_PARAM_IDS: Range<u64> = 2..12;

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::media::{MediaDecoder, MediaElement, MediaStream};
use crate::message::ControlMessage;
use crate::node::{
    self, AnalyserOptions, AudioBufferSourceOptions, AudioNode, ChannelConfigOptions,
    ChannelCountMode, ChannelInterpretation, ChannelMergerOptions, ChannelSplitterOptions,
    ConstantSourceOptions, DelayOptions, GainOptions, IirFilterOptions, PannerOptions,
};
use crate::param::{AudioParam, AudioParamEvent, AudioParamOptions};
use crate::periodic_wave::{PeriodicWave, PeriodicWaveOptions};
use crate::render::{AudioProcessor, NodeIndex, RenderThread};
use crate::spatial::{AudioListener, AudioListenerParams};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Stream};

use crossbeam_channel::Sender;

/// The `BaseAudioContext` interface represents an audio-processing graph built from audio modules
/// linked together, each represented by an `AudioNode`. An audio context controls both the creation
/// of the nodes it contains and the execution of the audio processing, or decoding.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
struct BaseAudioContext {
    /// inner makes `BaseAudioContextInner` cheap to clone
    inner: Arc<BaseAudioContextInner>,
}

impl PartialEq for BaseAudioContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Inner representation of the `BaseAudioContext`
/// 
/// Required for internal synchronization.
struct BaseAudioContextInner {
    /// sample rate in Hertz
    sample_rate: SampleRate,
    /// number of speaker output channels
    channels: u32,
    /// incrementing id to assign to audio nodes
    node_id_inc: AtomicU64,
    /// message channel from control to render thread
    render_channel: Sender<ControlMessage>,
    /// control messages that cannot be sent immediately
    queued_messages: Mutex<Vec<ControlMessage>>,
    /// number of frames played
    frames_played: Arc<AtomicU64>,
    /// AudioListener fields
    listener_params: Option<AudioListenerParams>,
}

macro_rules! delegate_context {
    ($name:ident, $parent:ident) => {
        impl Context for $name {
            fn decode_audio_data<R: std::io::Read + Send + 'static>(
                &self,
                input: R,
            ) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
                self.$parent.decode_audio_data(input)
            }

            fn create_buffer(
                &self,
                number_of_channels: usize,
                length: usize,
                sample_rate: SampleRate,
            ) -> AudioBuffer {
                self.$parent
                    .create_buffer(number_of_channels, length, sample_rate)
            }

            fn create_oscillator(&self) -> node::OscillatorNode {
                self.$parent.create_oscillator()
            }

            fn create_stereo_panner(&self) -> node::StereoPannerNode {
                self.$parent.create_stereo_panner()
            }

            fn create_gain(&self) -> node::GainNode {
                self.$parent.create_gain()
            }

            fn create_constant_source(&self) -> node::ConstantSourceNode {
                self.$parent.create_constant_source()
            }

            fn create_iir_filter(
                &self,
                feedforward: Vec<f64>,
                feedback: Vec<f64>,
            ) -> node::IirFilterNode {
                self.$parent.create_iir_filter(feedforward, feedback)
            }

            fn create_delay(&self, max_delay_time: f64) -> node::DelayNode {
                self.$parent.create_delay(max_delay_time)
            }

            fn create_biquad_filter(&self) -> node::BiquadFilterNode {
                self.$parent.create_biquad_filter()
            }

            fn create_wave_shaper(&self) -> node::WaveShaperNode {
                self.$parent.create_wave_shaper()
            }

            fn create_channel_splitter(&self, number_of_outputs: u32) -> node::ChannelSplitterNode {
                self.$parent.create_channel_splitter(number_of_outputs)
            }

            fn create_channel_merger(&self, number_of_inputs: u32) -> node::ChannelMergerNode {
                self.$parent.create_channel_merger(number_of_inputs)
            }

            fn create_media_stream_source<M: MediaStream>(
                &self,
                media: M,
            ) -> node::MediaStreamAudioSourceNode {
                self.$parent.create_media_stream_source(media)
            }

            fn create_media_element_source(
                &self,
                media: MediaElement,
            ) -> node::MediaElementAudioSourceNode {
                self.$parent.create_media_element_source(media)
            }

            fn create_buffer_source(&self) -> node::AudioBufferSourceNode {
                self.$parent.create_buffer_source()
            }

            fn create_panner(&self) -> node::PannerNode {
                self.$parent.create_panner()
            }

            fn create_analyser(&self) -> node::AnalyserNode {
                self.$parent.create_analyser()
            }

            fn create_periodic_wave(&self, options: Option<PeriodicWaveOptions>) -> PeriodicWave {
                self.$parent.create_periodic_wave(options)
            }

            fn create_audio_param(
                &self,
                opts: AudioParamOptions,
                dest: &AudioNodeId,
            ) -> (crate::param::AudioParam, AudioParamId) {
                self.$parent.create_audio_param(opts, dest)
            }

            fn destination(&self) -> node::AudioDestinationNode {
                self.$parent.destination()
            }

            fn listener(&self) -> AudioListener {
                self.$parent.listener()
            }

            #[must_use]
            fn sample_rate(&self) -> f32 {
                self.$parent.sample_rate()
            }

            #[must_use]
            fn sample_rate_raw(&self) -> SampleRate {
                self.$parent.sample_rate_raw()
            }

            fn current_time(&self) -> f64 {
                self.$parent.current_time()
            }

            fn channels(&self) -> u32 {
                self.$parent.channels()
            }

            #[allow(clippy::missing_panics_doc)]
            fn register<
                T: node::AudioNode,
                F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
            >(
                &self,
                f: F,
            ) -> T {
                self.$parent.register(f)
            }

            #[cfg(test)]
            fn mock_registration(&self) -> AudioContextRegistration {
                self.$parent.mock_registration()
            }
        }
    };
}

/// Retrieve the `BaseAudioContext` from the concrete `AudioContext`
#[allow(clippy::module_name_repetitions)]
pub trait Context {
    /// Decode an [`AudioBuffer`] from a given input stream.
    ///
    /// The current implementation can decode FLAC, Opus, PCM, Vorbis, and Wav.
    ///
    /// In addition to the official spec, the input parameter can be any byte stream (not just an
    /// array). This means you can decode audio data from a file, network stream, or in memory
    /// buffer, and any other [`std::io::Read`] implementor. The data if buffered internally so you
    /// should not wrap the source in a `BufReader`.
    ///
    /// This function operates synchronously, which may be undesirable on the control thread. The
    /// example shows how to avoid this.
    ///
    /// # Errors
    ///
    /// This method returns an Error in various cases (IO, mime sniffing, decoding).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::io::Cursor;
    /// use web_audio_api::SampleRate;
    /// use web_audio_api::context::{Context, OfflineAudioContext};
    ///
    /// let input = Cursor::new(vec![0; 32]); // or a File, TcpStream, ...
    ///
    /// let context = OfflineAudioContext::new(2, 44_100, SampleRate(44_100));
    /// let handle = std::thread::spawn(move || context.decode_audio_data(input));
    ///
    /// // do other things
    ///
    /// // await result from the decoder thread
    /// let decode_buffer_result = handle.join();
    /// ```
    fn decode_audio_data<R: std::io::Read + Send + 'static>(
        &self,
        input: R,
    ) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>>;

    /// Create an new "in-memory" `AudioBuffer` with the given number of channels,
    /// length (i.e. number of samples per channel) and sample rate.
    ///
    /// Note: In most cases you will want the sample rate to match the current
    /// audio context sample rate.
    fn create_buffer(
        &self,
        number_of_channels: usize,
        length: usize,
        sample_rate: SampleRate,
    ) -> AudioBuffer;

    /// Creates an `OscillatorNode`, a source representing a periodic waveform. It basically
    /// generates a tone.
    fn create_oscillator(&self) -> node::OscillatorNode;

    /// Creates an `StereoPannerNode` to pan a stereo output
    fn create_stereo_panner(&self) -> node::StereoPannerNode;

    /// Creates an `GainNode`, to control audio volume
    fn create_gain(&self) -> node::GainNode;

    /// Creates an `ConstantSourceNode`, a source representing a constant value
    fn create_constant_source(&self) -> node::ConstantSourceNode;

    /// Creates an `IirFilterNode`
    ///
    /// # Arguments
    ///
    /// * `feedforward` - An array of the feedforward (numerator) coefficients for the transfer function of the IIR filter.
    /// The maximum length of this array is 20
    /// * `feedback` - An array of the feedback (denominator) coefficients for the transfer function of the IIR filter.
    /// The maximum length of this array is 20
    fn create_iir_filter(&self, feedforward: Vec<f64>, feedback: Vec<f64>) -> node::IirFilterNode;

    /// Creates a `DelayNode`, delaying the audio signal
    fn create_delay(&self, max_delay_time: f64) -> node::DelayNode;

    /// Creates an `BiquadFilterNode` which implements a second order filter
    fn create_biquad_filter(&self) -> node::BiquadFilterNode;

    /// Creates a `WaveShaperNode`
    fn create_wave_shaper(&self) -> node::WaveShaperNode;

    /// Creates a `ChannelSplitterNode`
    fn create_channel_splitter(&self, number_of_outputs: u32) -> node::ChannelSplitterNode;

    /// Creates a `ChannelMergerNode`
    fn create_channel_merger(&self, number_of_inputs: u32) -> node::ChannelMergerNode;

    /// Creates a `MediaStreamAudioSourceNode` from a `MediaElement`
    fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode;

    /// Creates a `MediaElementAudioSourceNode` from a `MediaElement`
    ///
    /// Note: do not forget to `start()` the node.
    fn create_media_element_source(&self, media: MediaElement)
        -> node::MediaElementAudioSourceNode;

    /// Creates an `AudioBufferSourceNode`
    ///
    /// Note: do not forget to `start()` the node.
    fn create_buffer_source(&self) -> node::AudioBufferSourceNode;

    /// Creates a `PannerNode`
    fn create_panner(&self) -> node::PannerNode;

    /// Creates a `AnalyserNode`
    fn create_analyser(&self) -> node::AnalyserNode;

    /// Creates a periodic wave
    fn create_periodic_wave(&self, options: Option<PeriodicWaveOptions>) -> PeriodicWave;

    /// Create an `AudioParam`.
    ///
    /// Call this inside the `register` closure when setting up your `AudioNode`
    fn create_audio_param(
        &self,
        opts: AudioParamOptions,
        dest: &AudioNodeId,
    ) -> (crate::param::AudioParam, AudioParamId);

    /// Returns an `AudioDestinationNode` representing the final destination of all audio in the
    /// context. It can be thought of as the audio-rendering device.
    fn destination(&self) -> node::AudioDestinationNode;

    /// Returns the `AudioListener` which is used for 3D spatialization
    fn listener(&self) -> AudioListener;

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[must_use]
    fn sample_rate(&self) -> f32;

    /// The raw sample rate of the `AudioContext` (which has more precision than the float
    /// [`sample_rate()`](Context::sample_rate) value).
    #[must_use]
    fn sample_rate_raw(&self) -> SampleRate;

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    fn current_time(&self) -> f64;

    /// Number of channels for the audio destination
    fn channels(&self) -> u32;

    /// Construct a new pair of [`node::AudioNode`] and [`AudioProcessor`]
    ///
    /// The `AudioNode` lives in the user-facing control thread. The Processor is sent to the render thread.
    #[allow(clippy::missing_panics_doc)]
    fn register<
        T: node::AudioNode,
        F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
    >(
        &self,
        f: F,
    ) -> T;

    #[cfg(test)]
    fn mock_registration(&self) -> AudioContextRegistration;
}

delegate_context!(AudioContext, base);
delegate_context!(OfflineAudioContext, base);

pub(crate) mod private {
    pub trait Private {}
}

pub trait ConnectListenerToPannerRole: self::private::Private {
    /// Attach the 9 `AudioListener` coordinates to a `PannerNode`
    fn connect_listener_to_panner(&self, panner: &crate::context::AudioNodeId);
}

impl private::Private for BaseAudioContext {}

impl ConnectListenerToPannerRole for BaseAudioContext {
    fn connect_listener_to_panner(&self, panner: &crate::context::AudioNodeId) {
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 0, 1);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 1, 2);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 2, 3);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 3, 4);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 4, 5);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 5, 6);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 6, 7);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 7, 8);
        self.connect(&AudioNodeId(LISTENER_NODE_ID), panner, 8, 9);
    }
}

macro_rules! impl_connect_listener_to_panner {
    ($name:ident, $parent:ident) => {
        impl crate::context::private::Private for $name {}

        impl crate::context::ConnectListenerToPannerRole for $name {
            fn connect_listener_to_panner(&self, panner: &crate::context::AudioNodeId) {
                self.$parent.connect_listener_to_panner(panner)
            }
        }
    };
}

impl_connect_listener_to_panner!(AudioContext, base);
impl_connect_listener_to_panner!(OfflineAudioContext, base);

/// Identify the type of playback, which affects tradeoffs
/// between audio output latency and power consumption
pub enum LatencyHint {
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

/// Specify the playback configuration
/// in non web context, it is the only way to specify
/// the system configuration
pub struct AudioContextOptions {
    /// Identify the type of playback, which affects
    /// tradeoffs between audio output latency and power consumption
    pub latency_hint: Option<LatencyHint>,
    /// Sample rate of the audio Context and audio output hardware
    pub sample_rate: Option<u32>,
    /// Number of output channels of destination node and audio output hardware
    pub channels: Option<u16>,
}

/// This interface represents an audio graph whose `AudioDestinationNode` is routed to a real-time
/// output device that produces a signal directed at the user.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct AudioContext {
    /// represents the underlying `BaseAudioContext`
    base: BaseAudioContext,

    /// cpal stream (play/pause functionality)
    #[cfg(not(test))] // in tests, do not set up a cpal Stream
    stream: Stream,
}

/// The `OfflineAudioContext` doesn't render the audio to the device hardware; instead, it generates
/// it, as fast as it can, and outputs the result to an `AudioBuffer`.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct OfflineAudioContext {
    /// represents the underlying `BaseAudioContext`
    base: BaseAudioContext,
    /// the size of the buffer in sample-frames
    length: usize,
    /// the rendering 'thread', fully controlled by the offline context
    renderer: RenderThread,
}

impl AudioContext {
    /// Creates and returns a new `AudioContext` object.
    /// This will play live audio on the default output
    // options is passed by value to be conform to the specification interface
    #[allow(clippy::needless_pass_by_value)]
    #[cfg(not(test))]
    #[must_use]
    pub fn new(options: Option<AudioContextOptions>) -> Self {
        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        let (stream, config, sender) = io::build_output(frames_played_clone, options.as_ref());
        let channels = u32::from(config.channels);
        let sample_rate = SampleRate(config.sample_rate.0);

        let base = BaseAudioContext::new(sample_rate, channels, frames_played, sender);

        Self { base, stream }
    }

    #[cfg(test)] // in tests, do not set up a cpal Stream
    #[allow(clippy::must_use_candidate)]
    pub fn new(options: Option<AudioContextOptions>) -> Self {
        let options = options.unwrap_or(AudioContextOptions {
            latency_hint: Some(LatencyHint::Interactive),
            sample_rate: Some(44_100),
            channels: Some(2),
        });

        let sample_rate = SampleRate(options.sample_rate.unwrap_or(44_100));
        let channels = u32::from(options.channels.unwrap_or(2));
        let (sender, _receiver) = crossbeam_channel::unbounded();
        let frames_played = Arc::new(AtomicU64::new(0));
        let base = BaseAudioContext::new(sample_rate, channels, frames_played, sender);

        Self { base }
    }

    /// Suspends the progression of time in the audio context, temporarily halting audio hardware
    /// access and reducing CPU/battery usage in the process.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn suspend(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.pause().unwrap();
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
    // false positive due to #[cfg(not(test))]
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.play().expect("Audio device refuse to play");
    }
}

/// Unique identifier for audio nodes.
///
/// Used for internal bookkeeping.
#[derive(Debug)]
pub struct AudioNodeId(u64);

/// Unique identifier for audio params.
///
/// Store these in your `AudioProcessor` to get access to `AudioParam` values.
pub struct AudioParamId(u64);

// bit contrived, but for type safety only the context mod can access the inner u64
impl From<&AudioParamId> for NodeIndex {
    fn from(i: &AudioParamId) -> Self {
        Self(i.0)
    }
}

/// Handle of the [`node::AudioNode`] to its associated [`BaseAudioContext`].
///
/// This allows for communication with the render thread and lifetime management.
///
/// The only way to construct this object is by calling [`BaseAudioContext::register`]
pub struct AudioContextRegistration {
    /// the audio context in wich nodes and connections lives
    context: BaseAudioContext,
    /// identify a specific `AudioNode`
    id: AudioNodeId,
}

delegate_context!(AudioContextRegistration, context);

#[delegate(self.context)]
impl AudioContextRegistration {
    pub(crate) fn connect(&self, from: &AudioNodeId, to: &AudioNodeId, output: u32, input: u32);

    pub(crate) fn disconnect(&self, from: &AudioNodeId, to: &AudioNodeId);

    pub(crate) fn disconnect_all(&self, from: &AudioNodeId);

    pub(crate) fn pass_audio_param_event(&self, to: &Sender<AudioParamEvent>, event: AudioParamEvent);
}

impl AudioContextRegistration {
    /// get the audio node id of the registration
    // false positive: AudioContextRegistration is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn id(&self) -> &AudioNodeId {
        &self.id
    }
}

impl PartialEq for AudioContextRegistration {
    fn eq(&self, other: &Self) -> bool {
        self.context.eq(&other.context)
    }
}

impl Drop for AudioContextRegistration {
    fn drop(&mut self) {
        // do not drop magic nodes
        let magic = self.id.0 == DESTINATION_NODE_ID
            || self.id.0 == LISTENER_NODE_ID
            || LISTENER_PARAM_IDS.contains(&self.id.0);

        if !magic {
            let message = ControlMessage::FreeWhenFinished { id: self.id.0 };
            self.context.inner.render_channel.send(message).unwrap();
        }
    }
}

impl Context for BaseAudioContext {
    fn decode_audio_data<R: std::io::Read + Send + 'static>(
        &self,
        input: R,
    ) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
        // Set up a media decoder, consume the stream in full and construct a single buffer out of it
        let mut buffer = MediaDecoder::try_new(input)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .reduce(|mut accum, item| {
                accum.extend(&item);
                accum
            })
            // if there are no samples decoded, return an empty buffer
            .unwrap_or_else(|| AudioBuffer::from(vec![vec![]], self.sample_rate_raw()));

        // resample to desired rate (no-op if already matching)
        buffer.resample(self.sample_rate_raw());

        Ok(buffer)
    }

    fn create_buffer(
        &self,
        number_of_channels: usize,
        length: usize,
        sample_rate: SampleRate,
    ) -> AudioBuffer {
        let options = AudioBufferOptions {
            number_of_channels,
            length,
            sample_rate,
        };

        AudioBuffer::new(options)
    }

    fn create_oscillator(&self) -> node::OscillatorNode {
        node::OscillatorNode::new(self, None)
    }

    fn create_stereo_panner(&self) -> node::StereoPannerNode {
        node::StereoPannerNode::new(self, None)
    }

    fn create_gain(&self) -> node::GainNode {
        node::GainNode::new(self, GainOptions::default())
    }

    fn create_constant_source(&self) -> node::ConstantSourceNode {
        node::ConstantSourceNode::new(self, ConstantSourceOptions::default())
    }

    fn create_iir_filter(&self, feedforward: Vec<f64>, feedback: Vec<f64>) -> node::IirFilterNode {
        let options = IirFilterOptions {
            channel_config: ChannelConfigOptions::default(),
            feedforward,
            feedback,
        };
        node::IirFilterNode::new(self, options)
    }

    fn create_delay(&self, max_delay_time: f64) -> node::DelayNode {
        let opts = node::DelayOptions {
            max_delay_time,
            ..DelayOptions::default()
        };
        node::DelayNode::new(self, opts)
    }

    fn create_biquad_filter(&self) -> node::BiquadFilterNode {
        node::BiquadFilterNode::new(self, None)
    }

    fn create_wave_shaper(&self) -> node::WaveShaperNode {
        node::WaveShaperNode::new(self, None)
    }

    fn create_channel_splitter(&self, number_of_outputs: u32) -> node::ChannelSplitterNode {
        let opts = node::ChannelSplitterOptions {
            number_of_outputs,
            ..ChannelSplitterOptions::default()
        };
        node::ChannelSplitterNode::new(self, opts)
    }

    fn create_channel_merger(&self, number_of_inputs: u32) -> node::ChannelMergerNode {
        let opts = node::ChannelMergerOptions {
            number_of_inputs,
            ..ChannelMergerOptions::default()
        };
        node::ChannelMergerNode::new(self, opts)
    }

    fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode {
        let channel_config = ChannelConfigOptions {
            count: 1,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Speakers,
        };
        let opts = node::MediaStreamAudioSourceNodeOptions {
            media,
            channel_config,
        };
        node::MediaStreamAudioSourceNode::new(self, opts)
    }

    fn create_media_element_source(
        &self,
        media: MediaElement,
    ) -> node::MediaElementAudioSourceNode {
        let channel_config = ChannelConfigOptions {
            count: 1,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Speakers,
        };
        let opts = node::MediaElementAudioSourceNodeOptions {
            media,
            channel_config,
        };
        node::MediaElementAudioSourceNode::new(self, opts)
    }

    fn create_buffer_source(&self) -> node::AudioBufferSourceNode {
        node::AudioBufferSourceNode::new(self, AudioBufferSourceOptions::default())
    }

    fn create_panner(&self) -> node::PannerNode {
        node::PannerNode::new(self, PannerOptions::default())
    }

    fn create_analyser(&self) -> node::AnalyserNode {
        node::AnalyserNode::new(self, AnalyserOptions::default())
    }

    fn create_periodic_wave(&self, options: Option<PeriodicWaveOptions>) -> PeriodicWave {
        PeriodicWave::new(self, options)
    }

    fn create_audio_param(
        &self,
        opts: AudioParamOptions,
        dest: &AudioNodeId,
    ) -> (crate::param::AudioParam, AudioParamId) {
        let param = self.register(move |registration| {
            let (node, proc) = crate::param::audio_param_pair(opts, registration);

            (node, Box::new(proc))
        });

        // Connect the param to the node, once the node is registered inside the audio graph.
        self.queue_audio_param_connect(&param, dest);

        let proc_id = AudioParamId(param.id().0);
        (param, proc_id)
    }

    fn destination(&self) -> node::AudioDestinationNode {
        let registration = AudioContextRegistration {
            id: AudioNodeId(DESTINATION_NODE_ID),
            context: self.clone(),
        };
        node::AudioDestinationNode {
            registration,
            channel_count: self.channels() as usize,
        }
    }

    fn listener(&self) -> AudioListener {
        let mut ids = LISTENER_PARAM_IDS.map(|i| AudioContextRegistration {
            id: AudioNodeId(i),
            context: self.clone(),
        });
        let params = self.inner.listener_params.as_ref().unwrap();

        AudioListener {
            position_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_x.clone()),
            position_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_y.clone()),
            position_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_z.clone()),
            forward_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_x.clone()),
            forward_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_y.clone()),
            forward_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_z.clone()),
            up_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_x.clone()),
            up_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_y.clone()),
            up_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_z.clone()),
        }
    }

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    fn sample_rate(&self) -> f32 {
        self.inner.sample_rate.0 as f32
    }

    /// The raw sample rate of the `AudioContext` (which has more precision than the float
    /// [`sample_rate()`](Context::sample_rate) value).
    #[must_use]
    fn sample_rate_raw(&self) -> SampleRate {
        self.inner.sample_rate
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    #[must_use]
    // web audio api specification requires that `current_time` returns an f64
    // std::sync::AtomicsF64 is not currently implemented in the standard library
    // Currently, we have no other choice than casting an u64 into f64, with possible loss of precision
    #[allow(clippy::cast_precision_loss)]
    fn current_time(&self) -> f64 {
        self.inner.frames_played.load(Ordering::SeqCst) as f64 / f64::from(self.inner.sample_rate.0)
    }

    #[must_use]
    fn channels(&self) -> u32 {
        self.inner.channels
    }

    /// Construct a new pair of [`node::AudioNode`] and [`AudioProcessor`]
    ///
    /// The `AudioNode` lives in the user-facing control thread. The Processor is sent to the render thread.
    #[allow(clippy::missing_panics_doc)]
    fn register<
        T: node::AudioNode,
        F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
    >(
        &self,
        f: F,
    ) -> T {
        // create unique identifier for this node
        let id = self.inner.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let node_id = AudioNodeId(id);
        let registration = AudioContextRegistration {
            id: node_id,
            context: self.clone(),
        };

        // create the node and its renderer
        let (node, render) = (f)(registration);

        // pass the renderer to the audio graph
        let message = ControlMessage::RegisterNode {
            id,
            node: render,
            inputs: node.number_of_inputs() as usize,
            outputs: node.number_of_outputs() as usize,
            channel_config: node.channel_config_cloned(),
        };
        self.inner.render_channel.send(message).unwrap();

        // resolve control messages that depend on this registration
        let mut queued = self.inner.queued_messages.lock().unwrap();
        let mut i = 0; // waiting for Vec::drain_filter to stabilize
        while i < queued.len() {
            if matches!(&queued[i], ControlMessage::ConnectNode {to, ..} if *to == id) {
                let m = queued.remove(i);
                self.inner.render_channel.send(m).unwrap();
            } else {
                i += 1;
            }
        }

        node
    }

    #[cfg(test)]
    fn mock_registration(&self) -> AudioContextRegistration {
        AudioContextRegistration {
            id: AudioNodeId(0),
            context: self.clone(),
        }
    }
}

impl BaseAudioContext {
    /// Creates a `BaseAudioContext` instance
    fn new(
        sample_rate: SampleRate,
        channels: u32,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
    ) -> Self {
        let base_inner = BaseAudioContextInner {
            sample_rate,
            channels,
            render_channel,
            queued_messages: Mutex::new(Vec::new()),
            node_id_inc: AtomicU64::new(0),
            frames_played,
            listener_params: None,
        };
        let base = Self {
            inner: Arc::new(base_inner),
        };

        let listener_params = {
            // Register magical nodes. We should not store the nodes inside our context since that
            // will create a cyclic reference, but we can reconstruct a new instance on the fly
            // when requested

            let dest = node::AudioDestinationNode::new(&base, channels as usize);
            let listener = crate::spatial::AudioListenerNode::new(&base);

            // hack: Connect the listener to the destination node to force it to render at each
            // quantum. Abuse the magical u32::MAX port so it acts as an AudioParam and has no side
            // effects
            base.connect(listener.id(), dest.id(), 0, u32::MAX);

            let listener_params = listener.into_fields();
            let AudioListener {
                position_x,
                position_y,
                position_z,
                forward_x,
                forward_y,
                forward_z,
                up_x,
                up_y,
                up_z,
            } = listener_params;

            AudioListenerParams {
                position_x: position_x.into_raw_parts(),
                position_y: position_y.into_raw_parts(),
                position_z: position_z.into_raw_parts(),
                forward_x: forward_x.into_raw_parts(),
                forward_y: forward_y.into_raw_parts(),
                forward_z: forward_z.into_raw_parts(),
                up_x: up_x.into_raw_parts(),
                up_y: up_y.into_raw_parts(),
                up_z: up_z.into_raw_parts(),
            }
        }; // nodes will drop now, so base.inner has no copies anymore

        let mut base = base;
        let mut inner_mut = Arc::get_mut(&mut base.inner).unwrap();
        inner_mut.listener_params = Some(listener_params);

        base
    }
}

/// for [BaseAudioContext::create_audio_param]
impl BaseAudioContext {
    /// Schedule a connection of an `AudioParam` to the `AudioNode` it belongs to
    ///
    /// It is not performed immediately as the `AudioNode` is not registered at this point.
    fn queue_audio_param_connect(&self, param: &AudioParam, audio_node: &AudioNodeId) {
        let message = ControlMessage::ConnectNode {
            from: param.id().0,
            to: audio_node.0,
            output: 0,
            input: u32::MAX, // audio params connect to the 'hidden' input port
        };
        self.inner.queued_messages.lock().unwrap().push(message);
    }
}

impl BaseAudioContext {
    /// connects the output of the `from` audio node to the input of the `to` audio node
    pub(crate) fn connect(&self, from: &AudioNodeId, to: &AudioNodeId, output: u32, input: u32) {
        let message = ControlMessage::ConnectNode {
            from: from.0,
            to: to.0,
            output,
            input,
        };
        self.inner.render_channel.send(message).unwrap();
    }

    /// connects the `from` audio node to the `to` audio node
    pub(crate) fn disconnect(&self, from: &AudioNodeId, to: &AudioNodeId) {
        let message = ControlMessage::DisconnectNode {
            from: from.0,
            to: to.0,
        };
        self.inner.render_channel.send(message).unwrap();
    }

    /// disconnects all the audio nodes
    pub(crate) fn disconnect_all(&self, from: &AudioNodeId) {
        let message = ControlMessage::DisconnectAll { from: from.0 };
        self.inner.render_channel.send(message).unwrap();
    }

    /// Pass an `AudioParam::AudioParamEvent` to the render thread
    ///
    /// This clunky setup (wrapping a Sender in a message sent by another Sender) ensures
    /// automation events will never be handled out of order.
    pub(crate) fn pass_audio_param_event(&self, to: &Sender<AudioParamEvent>, event: AudioParamEvent) {
        let message = ControlMessage::AudioParamEvent {
            to: to.clone(),
            event,
        };
        self.inner.render_channel.send(message).unwrap();
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OfflineAudioContext {
    /// Creates an `OfflineAudioContext` instance
    ///
    /// # Arguments
    ///
    /// * `channels` - number of output channels to render
    /// * `length` - length of the rendering audio buffer
    /// * `sample_rate` - output sample rate
    #[must_use]
    pub fn new(channels: u32, length: usize, sample_rate: SampleRate) -> Self {
        // communication channel to the render thread
        let (sender, receiver) = crossbeam_channel::unbounded();

        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        // setup the render 'thread', which will run inside the control thread
        let renderer = RenderThread::new(
            sample_rate,
            channels as usize,
            receiver,
            frames_played_clone,
        );

        // first, setup the base audio context
        let base = BaseAudioContext::new(sample_rate, channels, frames_played, sender);

        Self {
            base,
            length,
            renderer,
        }
    }

    /// `OfflineAudioContext` doesn't start rendering automatically
    /// You need to call this function to start the audio rendering
    pub fn start_rendering(&mut self) -> AudioBuffer {
        // make buffer_size always a multiple of RENDER_QUANTUM_SIZE, so we can still render piecewise with
        // the desired number of frames.
        let buffer_size =
            (self.length + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE * RENDER_QUANTUM_SIZE;

        let mut buf = self.renderer.render_audiobuffer(buffer_size);
        let _split = buf.split_off(self.length);
        buf
    }

    /// get the length of rendering audio buffer
    // false positive: OfflineAudioContext is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

    #[test]
    fn test_audio_context_registration_traits() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        let registration = context.mock_registration();

        // we want to be able to ship AudioNodes to another thread, so the Registration should be
        // Send Sync and 'static
        require_send_sync_static(registration);
    }

    #[test]
    fn test_sample_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(96000));
        assert_float_eq!(context.sample_rate(), 96000., abs_all <= 0.);
        assert_eq!(context.sample_rate_raw(), SampleRate(96000));
    }

    #[test]
    fn test_decode_audio_data() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/sample.wav").unwrap();
        let audio_buffer = context.decode_audio_data(file).unwrap();

        assert_eq!(audio_buffer.sample_rate_raw(), SampleRate(44100));
        assert_eq!(audio_buffer.length(), 142_187);
        assert_eq!(audio_buffer.number_of_channels(), 2);
        assert_float_eq!(audio_buffer.duration(), 3.224, abs_all <= 0.001);

        let left_start = &audio_buffer.get_channel_data(0)[0..100];
        let right_start = &audio_buffer.get_channel_data(1)[0..100];
        // assert distinct two channel data
        assert!(left_start != right_start);
    }

    // #[test]
    // disabled: symphonia cannot handle empty WAV-files
    #[allow(dead_code)]
    fn test_decode_audio_data_empty() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/empty_2c.wav").unwrap();
        let audio_buffer = context.decode_audio_data(file).unwrap();
        assert_eq!(audio_buffer.length(), 0);
    }

    #[test]
    fn test_decode_audio_data_decoding_error() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(44100));
        let file = std::fs::File::open("samples/corrupt.wav").unwrap();
        assert!(context.decode_audio_data(file).is_err());
    }
}
