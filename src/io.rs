#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,

    // clippy::missing_docs_in_private_items
)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::message::ControlMessage;
use crate::{SampleRate, BUFFER_SIZE};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, SampleFormat, Stream, StreamConfig, SupportedBufferSize,
};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextOptions, LatencyHint};
use crate::graph::RenderThread;
use crate::media::MicrophoneRender;

use crossbeam_channel::{Receiver, Sender};
use log::warn;

/// Creates an output stream
///
/// # Arguments:
///
/// * `device` - the output audio device on which the stream is created
/// * `sample_format` - audio sample format of the stream
/// * `config` - stream configuration
/// * `render` - the render thread which process the audio data
fn spawn_output_stream(
    device: &Device,
    sample_format: SampleFormat,
    config: &StreamConfig,
    mut render: RenderThread,
) -> Result<Stream, BuildStreamError> {
    let err_fn = |err| log::error!("an error occurred on the output audio stream: {}", err);

    match sample_format {
        SampleFormat::F32 => {
            device.build_output_stream(config, move |d: &mut [f32], _c| render.render(d), err_fn)
        }
        SampleFormat::U16 => {
            device.build_output_stream(config, move |d: &mut [u16], _c| render.render(d), err_fn)
        }
        SampleFormat::I16 => {
            device.build_output_stream(config, move |d: &mut [i16], _c| render.render(d), err_fn)
        }
    }
}

/// Creates an input stream
///
/// # Arguments:
///
/// * `device` - the input audio device on which the stream is created
/// * `sample_format` - audio sample format of the stream
/// * `config` - stream configuration
/// * `render` - the render thread which process the audio data
fn spawn_input_stream(
    device: &Device,
    sample_format: SampleFormat,
    config: &StreamConfig,
    render: MicrophoneRender,
) -> Result<Stream, BuildStreamError> {
    let err_fn = |err| log::error!("an error occurred on the input audio stream: {}", err);

    match sample_format {
        SampleFormat::F32 => {
            device.build_input_stream(config, move |d: &[f32], _c| render.render(d), err_fn)
        }
        SampleFormat::U16 => {
            device.build_input_stream(config, move |d: &[u16], _c| render.render(d), err_fn)
        }
        SampleFormat::I16 => {
            device.build_input_stream(config, move |d: &[i16], _c| render.render(d), err_fn)
        }
    }
}

struct StreamConfigBuilder {
    device: cpal::Device,
    supported: cpal::SupportedStreamConfig,
    prefered: cpal::StreamConfig,
}

impl StreamConfigBuilder {
    fn new() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");

        log::info!("Host: {:?}", host.id());
        log::info!("Output device: {:?}", device.name());

        let supported = Self::get_supported_config(&device);

        Self {
            device,
            supported: supported.clone(),
            prefered: supported.into(),
        }
    }

    fn get_supported_config(device: &cpal::Device) -> cpal::SupportedStreamConfig {
        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("error while querying configs");

        supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate()
    }

    fn get_buffer_size(&self, options: Option<&AudioContextOptions>) -> u32 {
        let default_buffer_size = match self.supported.buffer_size() {
            SupportedBufferSize::Range { min, .. } => crate::BUFFER_SIZE.max(*min),
            SupportedBufferSize::Unknown => BUFFER_SIZE,
        };

        match options {
            Some(opts) => match opts.latency_hint.as_ref() {
                None => (default_buffer_size + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE,
                Some(l) => {
                    return match l {
                        LatencyHint::Interactive => default_buffer_size,
                        LatencyHint::Balanced => match self.supported.buffer_size() {
                            SupportedBufferSize::Range { max, .. } => {
                                let b = (default_buffer_size * 2).min(*max);
                                (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE
                            }
                            SupportedBufferSize::Unknown => {
                                let b = default_buffer_size * 2;
                                (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE
                            }
                        },
                        LatencyHint::Playback => match self.supported.buffer_size() {
                            SupportedBufferSize::Range { max, .. } => {
                                let b = (default_buffer_size * 4).min(*max);
                                (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE
                            }
                            SupportedBufferSize::Unknown => {
                                let b = default_buffer_size * 4;
                                (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE
                            }
                        },
                        LatencyHint::Specific(t) => {
                            let b = t * f64::from(self.prefered.sample_rate.0);
                            (b as u32 + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE
                        }
                    }
                }
            },
            None => (default_buffer_size + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE,
        }
    }

    fn with_sample_rate(mut self, options: Option<&AudioContextOptions>) -> Self {
        if let Some(opts) = options {
            if let Some(fs) = opts.sample_rate {
                self.prefered.sample_rate.0 = fs;
            }
        };

        self
    }

    fn with_latency_hint(mut self, options: Option<&AudioContextOptions>) -> Self {
        let buffer_size = self.get_buffer_size(options);
        self.prefered.buffer_size = cpal::BufferSize::Fixed(buffer_size);
        self
    }

    fn with_channels(mut self, options: Option<&AudioContextOptions>) -> Self {
        if let Some(opts) = options {
            if let Some(chs) = opts.channels {
                self.prefered.channels = chs;
            }
        };

        self
    }

    fn build(self) -> StreamConfigs {
        StreamConfigs::new(self)
    }
}

struct StreamConfigs {
    sample_format: cpal::SampleFormat,
    prefered: cpal::StreamConfig,
    fallback: cpal::StreamConfig,
}

impl StreamConfigs {
    fn new(builder: StreamConfigBuilder) -> Self {
        let StreamConfigBuilder {
            supported,
            prefered,
            ..
        } = builder;

        let sample_format = supported.sample_format();

        Self {
            sample_format,
            prefered,
            fallback: supported.into(),
        }
    }
}

struct OutputStreamer {
    device: cpal::Device,
    configs: StreamConfigs,
    frames_played: Arc<AtomicU64>,
    sender: Option<Sender<ControlMessage>>,
    stream: Option<Stream>,
    falled_back: bool,
}

impl OutputStreamer {
    fn new(configs: StreamConfigs, frames_played: Arc<AtomicU64>) -> Self {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");

        Self {
            device,
            configs,
            frames_played,
            sender: None,
            stream: None,
            falled_back: false,
        }
    }

    fn spawn(mut self) -> Result<Self, Self> {
        // try with prefered config
        let config = &self.configs.prefered;

        // Creates the render thread
        let sample_rate = SampleRate(config.sample_rate.0);

        // communication channel to the render thread
        let (sender, receiver) = crossbeam_channel::unbounded();

        self.sender = Some(sender);

        // spawn the render thread
        let renderer = RenderThread::new(
            sample_rate,
            config.channels as usize,
            receiver,
            self.frames_played.clone(),
        );

        let spawned =
            spawn_output_stream(&self.device, self.configs.sample_format, config, renderer);

        match spawned {
            Ok(stream) => {
                self.stream = Some(stream);
                Ok(self)
            }
            Err(e) => {
                warn!("Output stream build failed with prefered config: {}", e);
                Err(self)
            }
        }
    }

    fn play(self) -> Self {
        self.stream
            .as_ref()
            .expect("Stream needs to exist to be played")
            .play()
            .expect("Stream refused to play");
        self
    }

    fn get_output_stream(self) -> (Stream, StreamConfig, Sender<ControlMessage>) {
        if self.falled_back {
            (
                self.stream.unwrap(),
                self.configs.fallback,
                self.sender.unwrap(),
            )
        } else {
            (
                self.stream.unwrap(),
                self.configs.prefered,
                self.sender.unwrap(),
            )
        }
    }
}

trait OrFallback {
    fn or_fallback(self) -> OutputStreamer;
}

impl OrFallback for Result<OutputStreamer, OutputStreamer> {
    fn or_fallback(self) -> OutputStreamer {
        match self {
            Ok(streamer) => streamer,
            Err(mut streamer) => {
                // try with fallback config
                streamer.falled_back = true;
                let config = &streamer.configs.fallback;

                // Creates the renderer thread
                let sample_rate = SampleRate(config.sample_rate.0);

                // communication channel to the render thread
                let (sender, receiver) = crossbeam_channel::unbounded();

                streamer.sender = Some(sender);

                // spawn the render thread
                let renderer = RenderThread::new(
                    sample_rate,
                    config.channels as usize,
                    receiver,
                    streamer.frames_played.clone(),
                );

                let spawned = spawn_output_stream(
                    &streamer.device,
                    streamer.configs.sample_format,
                    config,
                    renderer,
                );
                let stream = spawned.expect("OutputStream build failed with default config");
                streamer.stream = Some(stream);
                streamer
            }
        }
    }
}

/// Builds the output
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn build_output(
    frames_played: Arc<AtomicU64>,
    options: Option<AudioContextOptions>,
) -> (Stream, StreamConfig, Sender<ControlMessage>) {
    let options = options.as_ref();
    let configs = StreamConfigBuilder::new()
        .with_sample_rate(options)
        .with_latency_hint(options)
        .with_channels(options)
        .build();

    let streamer = OutputStreamer::new(configs, frames_played)
        .spawn()
        .or_fallback()
        .play();

    streamer.get_output_stream()
}

/// Builds the input
pub fn build_input() -> (Stream, StreamConfig, Receiver<AudioBuffer>) {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");
    log::info!("Input device: {:?}", device.name());

    let mut supported_configs_range = device
        .supported_input_configs()
        .expect("error while querying configs");
    let supported_config = supported_configs_range
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();

    let sample_format = supported_config.sample_format();

    // clone the config, we may need to fall back on it later
    let default_config: StreamConfig = supported_config.clone().into();

    // determine best buffer size. Spec requires BUFFER_SIZE, but that might not be available
    let mut buffer_size = match supported_config.buffer_size() {
        SupportedBufferSize::Range { min, .. } => crate::BUFFER_SIZE.max(*min),
        SupportedBufferSize::Unknown => BUFFER_SIZE,
    };
    // make buffer_size always a multiple of BUFFER_SIZE, so we can still render piecewise with
    // the desired number of frames.
    buffer_size = (buffer_size + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;

    let mut config: StreamConfig = supported_config.into();
    config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
    let sample_rate = SampleRate(config.sample_rate.0);
    let channels = config.channels as usize;

    let smoothing = 3; // todo, use buffering to smooth frame drops
    let (sender, mut receiver) = crossbeam_channel::bounded(smoothing);
    let renderer = MicrophoneRender::new(channels, sample_rate, sender);

    let maybe_stream = spawn_input_stream(&device, sample_format, &config, renderer);
    // our BUFFER_SIZEd config may not be supported, in that case, use the default config
    let stream = match maybe_stream {
        Ok(stream) => stream,
        Err(e) => {
            log::warn!(
                "Output stream failed to build: {:?}, retry with default config {:?}",
                e,
                default_config
            );

            // setup a new comms channel
            let (sender, receiver2) = crossbeam_channel::bounded(smoothing);
            receiver = receiver2; // overwrite earlier

            let renderer = MicrophoneRender::new(channels, sample_rate, sender);
            spawn_input_stream(&device, sample_format, &default_config, renderer)
                .expect("Unable to spawn input stream with default config")
        }
    };

    // Required because some hosts don't play the stream automatically
    stream.play().expect("Input stream refused to play");

    (stream, config, receiver)
}
