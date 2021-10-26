use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::message::ControlMessage;
use crate::{SampleRate, BUFFER_SIZE};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, SampleFormat, SampleRate as CpalSampleRate, Stream, StreamConfig,
    SupportedBufferSize,
};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextOptions, LatencyHint};
use crate::graph::RenderThread;
use crate::media::MicrophoneRender;

use crossbeam_channel::{Receiver, Sender};

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

pub(crate) fn build_output(
    frames_played: Arc<AtomicU64>,
    options: Option<AudioContextOptions>,
) -> (Stream, StreamConfig, Sender<ControlMessage>) {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    log::info!("Output device: {:?}", device.name());

    let mut supported_configs_range = device
        .supported_output_configs()
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

    let mut primary_config: StreamConfig = supported_config.clone().into();
    primary_config.buffer_size = cpal::BufferSize::Fixed(buffer_size);

    let mut config: StreamConfig = supported_config.clone().into();

    match options {
        Some(opts) => {
            config.sample_rate = CpalSampleRate(opts.sample_rate.unwrap_or(config.sample_rate.0));
            config.channels = opts.channels.unwrap_or(config.channels);

            match opts.latency_hint {
                None => {
                    config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                }
                Some(l) => match l {
                    LatencyHint::Interactive => {
                        config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                    }
                    LatencyHint::Balanced => match supported_config.buffer_size() {
                        SupportedBufferSize::Range { max, .. } => {
                            let b = (buffer_size * 2).min(*max);
                            let buffer_size = (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;
                            config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                        }
                        SupportedBufferSize::Unknown => {
                            let b = buffer_size * 2;
                            let buffer_size = (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;
                            config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                        }
                    },
                    LatencyHint::Playback => match supported_config.buffer_size() {
                        SupportedBufferSize::Range { max, .. } => {
                            let b = (buffer_size * 4).min(*max);
                            let buffer_size = (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;
                            config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                        }
                        SupportedBufferSize::Unknown => {
                            let b = buffer_size * 4;
                            let buffer_size = (b + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;
                            config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                        }
                    },
                    LatencyHint::Specific(t) => {
                        let b = t / config.sample_rate.0 as f32;
                        let buffer_size = (b as u32 + BUFFER_SIZE - 1) / BUFFER_SIZE * BUFFER_SIZE;
                        config.buffer_size = cpal::BufferSize::Fixed(buffer_size);
                    }
                },
            }
        }
        None => {
            config = primary_config;
        }
    }
    let sample_rate = SampleRate(config.sample_rate.0);
    let channels = config.channels as u32;

    // communication channel to the render thread
    let (mut sender, receiver) = crossbeam_channel::unbounded();

    // spawn the render thread
    let frames_played_clone = frames_played.clone();
    let render = RenderThread::new(
        sample_rate,
        channels as usize,
        receiver,
        frames_played_clone,
    );

    let maybe_stream = spawn_output_stream(&device, sample_format, &config, render);
    // our BUFFER_SIZEd config may not be supported, in that case, use the default config
    let stream = match maybe_stream {
        Ok(stream) => {
            log::warn!("Input stream build sucessfully with config: {:?}", config);
            stream
        }
        Err(e) => {
            log::warn!(
                "Input stream failed to build: {:?}, retry with default config {:?}",
                e,
                default_config
            );

            // setup a new comms channel
            let (sender2, receiver) = crossbeam_channel::unbounded();
            sender = sender2; // overwrite earlier

            let render = RenderThread::new(sample_rate, channels as usize, receiver, frames_played);

            spawn_output_stream(&device, sample_format, &default_config, render)
                .expect("Unable to spawn output stream with default config")
        }
    };

    // Required because some hosts don't play the stream automatically
    stream.play().expect("Output stream refused to play");

    (stream, config, sender)
}

pub(crate) fn build_input() -> (Stream, StreamConfig, Receiver<AudioBuffer>) {
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
    let render = MicrophoneRender::new(channels, sample_rate, sender);

    let maybe_stream = spawn_input_stream(&device, sample_format, &config, render);
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

            let render = MicrophoneRender::new(channels, sample_rate, sender);
            spawn_input_stream(&device, sample_format, &default_config, render)
                .expect("Unable to spawn input stream with default config")
        }
    };

    // Required because some hosts don't play the stream automatically
    stream.play().expect("Input stream refused to play");

    (stream, config, receiver)
}
