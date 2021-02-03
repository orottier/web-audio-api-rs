use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use cpal::{Stream, StreamConfig};

use crate::control::ControlMessage;
use crate::graph::RenderThread;
use crate::node;

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;

/// The AudioContext interface represents an audio-processing graph built from audio modules linked
/// together, each represented by an AudioNode. An audio context controls both the creation of the
/// nodes it contains and the execution of the audio processing, or decoding.
pub struct AudioContext {
    /// cpal stream (play/pause functionality)
    stream: Stream, // todo should be in render thread?
    /// sample rate in Hertz
    sample_rate: u32,
    /// number of speaker output channels
    channels: usize,
    /// incrementing id to assign to audio nodes
    node_id_inc: AtomicU64,
    /// mpsc channel from control to render thread
    render_channel: Sender<ControlMessage>,
}

impl AudioContext {
    /// Creates and returns a new AudioContext object.
    /// This will play live audio on the default output
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

        dbg!(&config);

        let sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;

        // construct graph for the render thread
        let dest = crate::node::DestinationRenderer { channels };
        let (sender, receiver) = mpsc::channel();
        let mut render = RenderThread::new(dest, sample_rate, channels, receiver);

        let stream = match sample_format {
            SampleFormat::F32 => {
                device.build_output_stream(&config, move |data, _c| render.render(data), err_fn)
            }
            _ => unimplemented!(),
        }
        .unwrap();

        stream.play().unwrap();

        Self {
            stream,
            sample_rate,
            channels,
            node_id_inc: AtomicU64::new(1),
            render_channel: sender,
        }
    }

    /// Suspends the progression of time in the audio context, temporarily halting audio hardware
    /// access and reducing CPU/battery usage in the process.
    pub fn suspend(&self) {
        self.stream.pause().unwrap()
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    pub fn resume(&self) {
        self.stream.play().unwrap()
    }

    /// Creates an OscillatorNode, a source representing a periodic waveform. It basically
    /// generates a tone.
    pub fn create_oscillator(&self) -> node::OscillatorNode {
        self.create_oscillator_with(node::OscillatorOptions::default())
    }

    pub(crate) fn create_oscillator_with(
        &self,
        options: node::OscillatorOptions,
    ) -> node::OscillatorNode {
        let id = self.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let frequency = Arc::new(AtomicU32::new(options.frequency));
        let type_ = Arc::new(AtomicU32::new(options.type_ as u32));

        let render = node::OscillatorRenderer {
            frequency: frequency.clone(),
            type_: type_.clone(),
        };
        let message = ControlMessage::RegisterNode {
            id,
            node: Box::new(render),
            buffer: vec![0.; crate::BUFFER_SIZE as usize],
        };
        self.render_channel.send(message).unwrap();

        node::OscillatorNode {
            context: &self,
            id,
            frequency,
            type_,
        }
    }

    /// Creates an GainNode, to control audio volume
    pub fn create_gain(&self) -> node::GainNode {
        self.create_gain_with(node::GainOptions::default())
    }

    pub(crate) fn create_gain_with(&self, options: node::GainOptions) -> node::GainNode {
        let id = self.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let gain = Arc::new(AtomicU32::new((options.gain * 100.) as u32));

        let render = node::GainRenderer { gain: gain.clone() };
        let message = ControlMessage::RegisterNode {
            id,
            node: Box::new(render),
            buffer: vec![0.; crate::BUFFER_SIZE as usize],
        };
        self.render_channel.send(message).unwrap();

        node::GainNode {
            context: &self,
            id,
            gain,
        }
    }

    /// Creates a DelayNode, delaying the audio signal
    pub fn create_delay(&self) -> node::DelayNode {
        self.create_delay_with(node::DelayOptions::default())
    }

    pub(crate) fn create_delay_with(&self, options: node::DelayOptions) -> node::DelayNode {
        let id = self.node_id_inc.fetch_add(1, Ordering::SeqCst);
        let render_quanta = Arc::new(AtomicU32::new(options.render_quanta));

        let cap = (options.render_quanta * crate::BUFFER_SIZE) as usize;
        let delay_buffer = Vec::with_capacity(cap);
        let render = node::DelayRenderer {
            render_quanta: render_quanta.clone(),
            delay_buffer,
            index: 0,
        };

        let message = ControlMessage::RegisterNode {
            id,
            node: Box::new(render),
            buffer: vec![0.; crate::BUFFER_SIZE as usize],
        };
        self.render_channel.send(message).unwrap();

        node::DelayNode {
            context: &self,
            id,
            render_quanta,
        }
    }

    pub(crate) fn connect(&self, from: u64, to: u64, channel: usize) {
        println!("connecting {} to {}", from, to);
        let message = ControlMessage::ConnectNode { from, to, channel };
        self.render_channel.send(message).unwrap();
    }

    /// Returns an AudioDestinationNode representing the final destination of all audio in the
    /// context. It can be thought of as the audio-rendering device.
    pub fn destination(&self) -> node::DestinationNode {
        node::DestinationNode {
            context: &self,
            id: 0,
            channels: self.channels,
        }
    }

    /// The sample rate (in sample-frames per second) at which the AudioContext handles audio.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new()
    }
}
