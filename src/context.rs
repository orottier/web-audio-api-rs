use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use cpal::{Stream, StreamConfig};

use crate::node::{OscillatorNode, OscillatorNodeInner};

use std::sync::Arc;
use std::sync::Mutex;

/// The AudioContext interface represents an audio-processing graph built from audio modules linked
/// together, each represented by an AudioNode. An audio context controls both the creation of the
/// nodes it contains and the execution of the audio processing, or decoding. You need to create an
/// AudioContext before you do anything else, as everything happens inside a context.
pub struct AudioContext {
    stream: Stream,
    sample_rate: u32,
    channels: u32,
    graph: Arc<Mutex<AudioGraph>>,
}

pub struct AudioGraph {
    timestamp: f64,
    pub children: Vec<Arc<OscillatorNodeInner>>,
}

impl AudioGraph {
    fn build_output<T: Sample>(
        &mut self,
        data: &mut [T],
        _info: &cpal::OutputCallbackInfo,
        sample_rate: u32,
        channels: u32,
    ) {
        // assume mono input, stereo output
        let channels = channels as usize;
        let len = data.len() / channels;

        // clear slice, it may be re-used
        for d in data.iter_mut() {
            *d = Sample::from(&0.);
        }

        // mix signal from all child nodes, prevent allocations
        for c in &self.children {
            let frames = data.chunks_mut(channels);
            let mono = c.signal(self.timestamp, len, sample_rate);

            for (frame, v) in frames.zip(mono) {
                for sample in frame.iter_mut() {
                    let value = Sample::from(&(v + sample.to_f32()));
                    *sample = value;
                }
            }
        }

        // progress (fake) time
        self.timestamp += len as f64 / sample_rate as f64;
    }
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
        let config: StreamConfig = supported_config.into();

        let sample_rate = config.sample_rate.0;
        let channels = config.channels as u32;

        let graph = Arc::new(Mutex::new(AudioGraph {
            timestamp: 0.,
            children: vec![],
        }));

        let stream_graph = graph.clone();
        let stream = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                &config,
                move |d, c| {
                    let mut graph = stream_graph.lock().unwrap();
                    graph.build_output::<f32>(d, c, sample_rate, channels)
                },
                err_fn,
            ),
            _ => unimplemented!(),
        }
        .unwrap();

        stream.play().unwrap();

        Self {
            stream,
            graph,
            channels,
            sample_rate,
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
    pub fn create_oscillator(&self) -> crate::node::OscillatorNode {
        OscillatorNode::new(440)
    }

    /// Returns an AudioDestinationNode representing the final destination of all audio in the
    /// context. It can be thought of as the audio-rendering device.
    pub fn destination(&mut self) -> &Mutex<AudioGraph> {
        // todo actually return an AudioDestinationNode
        self.graph.as_ref()
    }
}
