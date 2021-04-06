//! The BaseAudioContext interface and the AudioContext and OfflineAudioContext types

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};

// magic node values
const DESTINATION_NODE_ID: u64 = 0;
const LISTENER_NODE_ID: u64 = 1;
const LISTENER_PARAM_IDS: Range<u64> = 2..12;

#[cfg(not(test))]
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};

use crate::buffer::{ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::graph::{NodeIndex, RenderThread};
use crate::media::{MediaElement, MediaStream};
use crate::message::ControlMessage;
use crate::node;
use crate::node::AudioNode;
use crate::param::{AudioParam, AudioParamOptions};
use crate::process::AudioProcessor2;
use crate::spatial::{AudioListener, AudioListenerParams};
use crate::SampleRate;

/// The BaseAudioContext interface represents an audio-processing graph built from audio modules
/// linked together, each represented by an AudioNode. An audio context controls both the creation
/// of the nodes it contains and the execution of the audio processing, or decoding.
pub struct BaseAudioContext {
    /// sample rate in Hertz
    sample_rate: SampleRate,
    /// number of speaker output channels
    channels: u32,
    /// incrementing id to assign to audio nodes
    node_id_inc: AtomicU64,
    /// mpsc channel from control to render thread
    render_channel: Sender<ControlMessage>,
    /// number of frames played
    frames_played: AtomicU64,
    /// AudioListener fields
    listener_params: Option<AudioListenerParams>,
}

/// Retrieve the BaseAudioContext from the concrete AudioContext
pub trait AsBaseAudioContext {
    fn base(&self) -> &BaseAudioContext;

    /// Creates an OscillatorNode, a source representing a periodic waveform. It basically
    /// generates a tone.
    fn create_oscillator(&self) -> node::OscillatorNode {
        node::OscillatorNode::new(self.base(), Default::default())
    }

    /// Creates an GainNode, to control audio volume
    fn create_gain(&self) -> node::GainNode {
        node::GainNode::new(self.base(), Default::default())
    }

    /// Creates an ConstantSourceNode, a source representing a constant value
    fn create_constant_source(&self) -> node::ConstantSourceNode {
        node::ConstantSourceNode::new(self.base(), Default::default())
    }

    /// Creates a DelayNode, delaying the audio signal
    fn create_delay(&self) -> node::DelayNode {
        node::DelayNode::new(self.base(), Default::default())
    }

    /// Creates a ChannelSplitterNode
    fn create_channel_splitter(&self, number_of_outputs: u32) -> node::ChannelSplitterNode {
        let opts = node::ChannelSplitterOptions {
            number_of_outputs,
            ..Default::default()
        };
        node::ChannelSplitterNode::new(self.base(), opts)
    }

    /// Creates a ChannelMergerNode
    fn create_channel_merger(&self, number_of_inputs: u32) -> node::ChannelMergerNode {
        let opts = node::ChannelMergerOptions {
            number_of_inputs,
            ..Default::default()
        };
        node::ChannelMergerNode::new(self.base(), opts)
    }

    /// Creates a MediaStreamAudioSourceNode from a MediaElement
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
        node::MediaStreamAudioSourceNode::new(self.base(), opts)
    }

    /// Creates a MediaElementAudioSourceNode from a MediaElement
    fn create_media_element_source<M: MediaStream>(
        &self,
        media: MediaElement<M>,
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
        node::MediaElementAudioSourceNode::new(self.base(), opts)
    }

    /// Creates an AudioBufferSourceNode
    fn create_buffer_source(&self) -> node::AudioBufferSourceNode {
        node::AudioBufferSourceNode::new(self.base(), Default::default())
    }

    /// Creates a PannerNode
    fn create_panner(&self) -> node::PannerNode {
        node::PannerNode::new(self.base(), Default::default())
    }

    /// Create an AudioParam.
    ///
    /// Call this inside the `register` closure when setting up your AudioNode
    fn create_audio_param(
        &self,
        opts: AudioParamOptions,
        dest: &AudioNodeId,
    ) -> (crate::param::AudioParam<'_>, AudioParamId) {
        let param = self.base().register(move |registration| {
            let (node, proc) = crate::param::audio_param_pair(opts, registration);

            (node, Box::new(proc))
        });

        // audio params are connected to the 'hidden' u32::MAX input. TODO make nicer
        self.base().connect(param.id(), dest, 0, u32::MAX);

        let proc_id = AudioParamId(param.id().0);
        (param, proc_id)
    }

    /// Returns an AudioDestinationNode representing the final destination of all audio in the
    /// context. It can be thought of as the audio-rendering device.
    fn destination(&self) -> node::DestinationNode {
        let registration = AudioContextRegistration {
            id: AudioNodeId(DESTINATION_NODE_ID),
            context: &self.base(),
        };
        node::DestinationNode {
            registration,
            channel_count: self.base().channels as usize,
        }
    }

    /// Returns the AudioListener which is used for 3D spatialization
    fn listener(&self) -> AudioListener {
        let mut ids = LISTENER_PARAM_IDS.map(|i| AudioContextRegistration {
            id: AudioNodeId(i),
            context: self.base(),
        });
        let params = self.base().listener_params.as_ref().unwrap();

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

    /// The sample rate (in sample-frames per second) at which the AudioContext handles audio.
    fn sample_rate(&self) -> SampleRate {
        self.base().sample_rate()
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    fn current_time(&self) -> f64 {
        self.base().current_time()
    }

    #[cfg(test)]
    fn mock_registration(&self) -> AudioContextRegistration {
        AudioContextRegistration {
            id: AudioNodeId(0),
            context: self.base(),
        }
    }
}

impl AsBaseAudioContext for BaseAudioContext {
    fn base(&self) -> &BaseAudioContext {
        &self
    }
}

/// This interface represents an audio graph whose AudioDestinationNode is routed to a real-time
/// output device that produces a signal directed at the user.
pub struct AudioContext {
    base: BaseAudioContext,

    /// cpal stream (play/pause functionality)
    #[cfg(not(test))] // in tests, do not set up a cpal Stream
    stream: Stream,
}

impl AsBaseAudioContext for AudioContext {
    fn base(&self) -> &BaseAudioContext {
        &self.base
    }
}

/// The OfflineAudioContext doesn't render the audio to the device hardware; instead, it generates
/// it, as fast as it can, and outputs the result to an AudioBuffer.
pub struct OfflineAudioContext {
    base: BaseAudioContext,

    /// the size of the buffer in sample-frames
    length: usize,
    /// the rendered audio data
    buffer: Vec<f32>,
    /// the rendering 'thread', fully controlled by the offline context
    render: RenderThread,
}

impl AsBaseAudioContext for OfflineAudioContext {
    fn base(&self) -> &BaseAudioContext {
        &self.base
    }
}

impl AudioContext {
    /// Creates and returns a new AudioContext object.
    /// This will play live audio on the default output
    #[cfg(not(test))]
    pub fn new() -> Self {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
        let sample_format = supported_config.sample_format();

        // set max buffer size, note: this defines only the upper bound (on my machine!)
        let mut config: StreamConfig = supported_config.into();
        config.buffer_size = cpal::BufferSize::Fixed(crate::BUFFER_SIZE);

        let sample_rate = SampleRate(config.sample_rate.0);
        let channels = config.channels as u32;

        // communication channel to the render thread
        let (sender, receiver) = mpsc::channel();

        // first, setup the base audio context
        let base = BaseAudioContext::new(sample_rate, channels, sender);

        // spawn the render thread
        let mut render = RenderThread::new(sample_rate, channels as usize, receiver);
        let stream = match sample_format {
            SampleFormat::F32 => {
                device.build_output_stream(&config, move |data, _c| render.render(data), err_fn)
            }
            _ => unimplemented!(),
        }
        .unwrap();

        stream.play().unwrap();

        Self { base, stream }
    }

    #[cfg(test)] // in tests, do not set up a cpal Stream
    pub fn new() -> Self {
        let sample_rate = SampleRate(44_100);
        let channels = 2;
        let (sender, _receiver) = mpsc::channel();

        let base = BaseAudioContext::new(sample_rate, channels, sender);

        Self { base }
    }

    /// Suspends the progression of time in the audio context, temporarily halting audio hardware
    /// access and reducing CPU/battery usage in the process.
    pub fn suspend(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.pause().unwrap()
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    pub fn resume(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.play().unwrap()
    }
}

/// Unique identifier for audio nodes.
///
/// Used for internal bookkeeping.
pub struct AudioNodeId(u64);

/// Unique identifier for audio params.
///
/// Store these in your AudioProcessor to get access to AudioParam values.
pub struct AudioParamId(u64);

// bit contrived, but for type safety only the context lib can access the inner u64
impl From<&AudioParamId> for NodeIndex {
    fn from(i: &AudioParamId) -> Self {
        NodeIndex(i.0)
    }
}

/// Handle of the [`node::AudioNode`] to its associated [`BaseAudioContext`].
///
/// This allows for communication with the render thread and lifetime management.
///
/// The only way to construct this object is by calling [`BaseAudioContext::register`]
pub struct AudioContextRegistration<'a> {
    context: &'a BaseAudioContext,
    id: AudioNodeId,
}

impl<'a> AudioContextRegistration<'a> {
    pub fn id(&self) -> &AudioNodeId {
        &self.id
    }
    pub fn context(&self) -> &BaseAudioContext {
        self.context
    }
}

impl<'a> Drop for AudioContextRegistration<'a> {
    fn drop(&mut self) {
        // do not drop magic nodes
        let magic = self.id.0 == DESTINATION_NODE_ID
            || self.id.0 == LISTENER_NODE_ID
            || LISTENER_PARAM_IDS.contains(&self.id.0);

        if !magic {
            let message = ControlMessage::FreeWhenFinished { id: self.id.0 };
            self.context.render_channel.send(message).unwrap();
        }
    }
}

impl BaseAudioContext {
    fn new(sample_rate: SampleRate, channels: u32, render_channel: Sender<ControlMessage>) -> Self {
        let base = Self {
            sample_rate,
            channels,
            render_channel,
            node_id_inc: AtomicU64::new(0),
            frames_played: AtomicU64::new(0),
            listener_params: None,
        };

        let listener_params = {
            // Register magical nodes.  We cannot store the destination node inside our context
            // since it borrows base, but we can reconstruct a new instance on the fly when
            // requested

            let dest = node::DestinationNode::new(&base, channels as usize);
            assert_eq!(dest.registration().id.0, DESTINATION_NODE_ID); // we rely on this elsewhere

            let listener = crate::spatial::AudioListenerNode::new(&base);
            assert_eq!(listener.registration().id.0, LISTENER_NODE_ID); // we rely on this elsewhere
            assert_eq!(
                base.node_id_inc.load(Ordering::SeqCst),
                LISTENER_PARAM_IDS.end - 1
            );

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
        };

        let mut base = base;
        base.listener_params = Some(listener_params);

        base
    }

    /// The sample rate (in sample-frames per second) at which the AudioContext handles audio.
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    pub fn current_time(&self) -> f64 {
        self.frames_played.load(Ordering::SeqCst) as f64 / self.sample_rate.0 as f64
    }

    /// Number of channels for the audio destination
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Construct a new pair of [`node::AudioNode`] and [`AudioProcessor`]
    ///
    /// The AudioNode lives in the user-facing control thread. The Processor is sent to the render thread.
    pub fn register<
        'a,
        T: node::AudioNode,
        F: FnOnce(AudioContextRegistration<'a>) -> (T, Box<dyn AudioProcessor2>),
    >(
        &'a self,
        f: F,
    ) -> T {
        // create unique identifier for this node
        let id = self.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let node_id = AudioNodeId(id);
        let registration = AudioContextRegistration {
            id: node_id,
            context: &self,
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
        self.render_channel.send(message).unwrap();

        node
    }

    pub(crate) fn connect(&self, from: &AudioNodeId, to: &AudioNodeId, output: u32, input: u32) {
        let message = ControlMessage::ConnectNode {
            from: from.0,
            to: to.0,
            output,
            input,
        };
        self.render_channel.send(message).unwrap();
    }

    pub(crate) fn disconnect(&self, from: &AudioNodeId, to: &AudioNodeId) {
        let message = ControlMessage::DisconnectNode {
            from: from.0,
            to: to.0,
        };
        self.render_channel.send(message).unwrap();
    }

    pub(crate) fn disconnect_all(&self, from: &AudioNodeId) {
        let message = ControlMessage::DisconnectAll { from: from.0 };
        self.render_channel.send(message).unwrap();
    }

    /// Attach the 9 AudioListener coordinates to a PannerNode
    pub(crate) fn connect_listener_to_panner(&self, panner: &AudioNodeId) {
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

impl Default for AudioContext {
    fn default() -> Self {
        Self::new()
    }
}

impl OfflineAudioContext {
    pub fn new(channels: u32, length: usize, sample_rate: SampleRate) -> Self {
        // communication channel to the render thread
        let (sender, receiver) = mpsc::channel();

        // first, setup the base audio context
        let base = BaseAudioContext::new(sample_rate, channels, sender);

        // setup the render 'thread', which will run inside the control thread
        let render = RenderThread::new(sample_rate, channels as usize, receiver);

        // pre-allocate enough space (todo, round to multiple of channels * buffer_size?)
        let buffer = vec![0.; length];

        Self {
            base,
            length,
            buffer,
            render,
        }
    }

    pub fn start_rendering(&mut self) -> &[f32] {
        for quantum in self.buffer.chunks_mut(crate::BUFFER_SIZE as usize) {
            self.render.render(quantum)
        }

        self.buffer.as_slice()
    }

    pub fn length(&self) -> usize {
        self.length
    }
}
