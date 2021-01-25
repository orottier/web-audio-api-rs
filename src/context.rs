use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use cpal::{Stream, StreamConfig};

/// The AudioContext interface represents an audio-processing graph built from audio modules linked
/// together, each represented by an AudioNode. An audio context controls both the creation of the
/// nodes it contains and the execution of the audio processing, or decoding. You need to create an
/// AudioContext before you do anything else, as everything happens inside a context.
pub struct AudioContext {
    stream: Stream,
    sample_rate: u32,
    channels: u16,
}

fn build_output<T: Sample>(data: &mut [T], _: &cpal::OutputCallbackInfo) {
    for sample in data.iter_mut() {
        *sample = Sample::from(&0.0);
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

        let stream = match sample_format {
            SampleFormat::F32 => {
                device.build_output_stream(&config, |d, c| build_output::<f32>(d, c), err_fn)
            }
            _ => unimplemented!(),
        }
        .unwrap();

        stream.play().unwrap();

        Self {
            stream,
            channels: config.channels,
            sample_rate: config.sample_rate.0,
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
}
