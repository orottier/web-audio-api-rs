use crate::BUFFER_SIZE;
use std::default::Default;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, SampleFormat, Stream, StreamConfig, SupportedBufferSize,
};

use crate::graph::RenderThread;
use crate::media::MicrophoneRender;

pub(crate) struct OutputBuilder {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
}

impl OutputBuilder {
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

        let sample_format = supported_config.sample_format();

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

        Self {
            device,
            config,
            sample_format,
        }
    }

    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    pub fn build(self, mut render: RenderThread) -> Result<Stream, BuildStreamError> {
        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
        match self.sample_format {
            SampleFormat::F32 => self.device.build_output_stream(
                &self.config,
                move |d: &mut [f32], _c| render.render(d),
                err_fn,
            ),
            SampleFormat::U16 => self.device.build_output_stream(
                &self.config,
                move |d: &mut [u16], _c| render.render(d),
                err_fn,
            ),
            SampleFormat::I16 => self.device.build_output_stream(
                &self.config,
                move |d: &mut [i16], _c| render.render(d),
                err_fn,
            ),
        }
    }
}

impl Default for OutputBuilder {
    fn default() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("error while querying configs");

        let supported_config = if cfg!(target_os = "linux") {
            device
                .default_output_config()
                .expect("default config not found")
        } else {
            supported_configs_range
                .next()
                .expect("no supported config?!")
                .with_max_sample_rate()
        };

        let sample_format = supported_config.sample_format();

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

        Self {
            device,
            config,
            sample_format,
        }
    }
}

pub(crate) struct InputBuilder {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
}

impl InputBuilder {
    pub fn new() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no input device available");
        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();

        let sample_format = supported_config.sample_format();

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

        Self {
            device,
            config,
            sample_format,
        }
    }

    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    pub fn build(self, render: MicrophoneRender) -> Stream {
        let err_fn = |err| eprintln!("an error occurred on the input audio stream: {}", err);
        let stream = match self.sample_format {
            SampleFormat::F32 => self.device.build_input_stream(
                &self.config,
                move |d: &[f32], _c| render.render(d),
                err_fn,
            ),
            SampleFormat::U16 => self.device.build_input_stream(
                &self.config,
                move |d: &[u16], _c| render.render(d),
                err_fn,
            ),
            SampleFormat::I16 => self.device.build_input_stream(
                &self.config,
                move |d: &[i16], _c| render.render(d),
                err_fn,
            ),
        }
        .unwrap();

        stream.play().unwrap();

        stream
    }
}
