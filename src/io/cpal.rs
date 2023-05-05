//! Audio IO management API
use std::sync::Arc;
use std::sync::Mutex;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, OutputCallbackInfo, SampleFormat, Stream, StreamConfig,
    SupportedBufferSize,
};

use super::{AudioBackendManager, RenderThreadInit};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::io::microphone::MicrophoneRender;
use crate::media_devices::{MediaDeviceInfo, MediaDeviceInfoKind};
use crate::render::RenderThread;
use crate::AtomicF64;

use crossbeam_channel::Receiver;

mod private {
    use super::*;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<Stream>>>);

    impl ThreadSafeClosableStream {
        pub fn new(stream: Stream) -> Self {
            Self(Arc::new(Mutex::new(Some(stream))))
        }

        pub fn close(&self) {
            self.0.lock().unwrap().take(); // will Drop
        }

        pub fn resume(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.play() {
                    panic!("Error resuming cpal stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn suspend(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.pause() {
                    panic!("Error suspending cpal stream: {:?}", e);
                }
                return true;
            }

            false
        }
    }

    // SAFETY:
    // The cpal `Stream` is marked !Sync and !Send because some platforms are not thread-safe
    // https://github.com/RustAudio/cpal/commit/33ddf749548d87bf54ce18eb342f954cec1465b2
    // Since we wrap the Stream in a Mutex, we should be fine
    unsafe impl Sync for ThreadSafeClosableStream {}
    unsafe impl Send for ThreadSafeClosableStream {}
}
use private::ThreadSafeClosableStream;

/// Audio backend using the `cpal` library
#[derive(Clone)]
pub struct CpalBackend {
    stream: ThreadSafeClosableStream,
    output_latency: Arc<AtomicF64>,
    sample_rate: f32,
    number_of_channels: usize,
    sink_id: String,
}

impl AudioBackendManager for CpalBackend {
    fn build_output(options: AudioContextOptions, render_thread_init: RenderThreadInit) -> Self
    where
        Self: Sized,
    {
        let host = cpal::default_host();
        log::info!("Host: {:?}", host.id());

        let RenderThreadInit {
            frames_played,
            ctrl_msg_recv,
            load_value_send,
            event_send,
        } = render_thread_init;

        let device = if options.sink_id.is_empty() {
            host.default_output_device()
                .expect("no output device available")
        } else {
            Self::enumerate_devices()
                .into_iter()
                .find(|e| e.device_id() == options.sink_id)
                .map(|e| *e.device().downcast::<cpal::Device>().unwrap())
                .unwrap()
        };

        log::info!("Output device: {:?}", device.name());

        let supported = device
            .default_output_config()
            .expect("error while querying config");

        let mut prefered: StreamConfig = supported.clone().into();

        // set specific sample rate if requested
        if let Some(sample_rate) = options.sample_rate {
            crate::assert_valid_sample_rate(sample_rate);
            prefered.sample_rate.0 = sample_rate as u32;
        }

        // always try to set a decent buffer size
        let buffer_size = super::buffer_size_for_latency_category(
            options.latency_hint,
            prefered.sample_rate.0 as f32,
        ) as u32;

        let clamped_buffer_size: u32 = match supported.buffer_size() {
            SupportedBufferSize::Unknown => buffer_size,
            SupportedBufferSize::Range { min, max } => buffer_size.clamp(*min, *max),
        };

        prefered.buffer_size = cpal::BufferSize::Fixed(clamped_buffer_size);

        let output_latency = Arc::new(AtomicF64::new(0.));
        let mut number_of_channels = usize::from(prefered.channels);
        let mut sample_rate = prefered.sample_rate.0 as f32;

        let renderer = RenderThread::new(
            sample_rate,
            prefered.channels as usize,
            ctrl_msg_recv.clone(),
            frames_played.clone(),
            Some(load_value_send.clone()),
            Some(event_send.clone()),
        );

        log::debug!(
            "Attempt output stream with prefered config: {:?}",
            &prefered
        );
        let spawned = spawn_output_stream(
            &device,
            supported.sample_format(),
            &prefered,
            renderer,
            output_latency.clone(),
        );

        let stream = match spawned {
            Ok(stream) => {
                log::debug!("Output stream set up successfully");
                stream
            }
            Err(e) => {
                log::warn!("Output stream build failed with prefered config: {}", e);

                let supported_config: StreamConfig = supported.clone().into();
                number_of_channels = usize::from(supported_config.channels);
                sample_rate = supported_config.sample_rate.0 as f32;

                log::debug!(
                    "Attempt output stream with fallback config: {:?}",
                    &supported_config
                );

                let renderer = RenderThread::new(
                    sample_rate,
                    supported_config.channels as usize,
                    ctrl_msg_recv,
                    frames_played,
                    Some(load_value_send),
                    Some(event_send),
                );

                let spawned = spawn_output_stream(
                    &device,
                    supported.sample_format(),
                    &supported_config,
                    renderer,
                    output_latency.clone(),
                );
                spawned.expect("OutputStream build failed with default config")
            }
        };

        stream.play().expect("Stream refused to play");

        CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency,
            sample_rate,
            number_of_channels,
            sink_id: options.sink_id,
        }
    }

    fn build_input(options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized,
    {
        // @todo - enable device selection, i.e. device_id from enumerate_devices
        // see https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no input device available");

        log::info!("Input device: {:?}", device.name());

        let supported = device
            .default_input_config()
            .expect("error while querying configs");

        // clone the config, we may need to fall back on it later
        let mut prefered: StreamConfig = supported.clone().into();

        // set specific sample rate if requested
        if let Some(sample_rate) = options.sample_rate {
            crate::assert_valid_sample_rate(sample_rate);
            prefered.sample_rate.0 = sample_rate as u32;
        }

        // always try to set a decent buffer size
        let buffer_size = super::buffer_size_for_latency_category(
            options.latency_hint,
            prefered.sample_rate.0 as f32,
        ) as u32;

        let clamped_buffer_size: u32 = match supported.buffer_size() {
            SupportedBufferSize::Unknown => buffer_size,
            SupportedBufferSize::Range { min, max } => buffer_size.clamp(*min, *max),
        };

        prefered.buffer_size = cpal::BufferSize::Fixed(clamped_buffer_size);

        let mut number_of_channels = usize::from(prefered.channels);
        let mut sample_rate = prefered.sample_rate.0 as f32;

        let smoothing = 3; // todo, use buffering to smooth frame drops
        let (sender, mut receiver) = crossbeam_channel::bounded(smoothing);
        let renderer = MicrophoneRender::new(number_of_channels, sample_rate, sender);

        let maybe_stream =
            spawn_input_stream(&device, supported.sample_format(), &prefered, renderer);

        // the required block size prefered config may not be supported, in that
        // case, fallback the supported config
        let stream = match maybe_stream {
            Ok(stream) => stream,
            Err(e) => {
                log::warn!(
                    "Output stream failed to build: {:?}, retry with default config {:?}",
                    e,
                    prefered
                );

                let supported_config: StreamConfig = supported.clone().into();
                number_of_channels = usize::from(supported_config.channels);
                sample_rate = supported_config.sample_rate.0 as f32;

                // setup a new comms channel
                let (sender, receiver2) = crossbeam_channel::bounded(smoothing);
                receiver = receiver2; // overwrite earlier

                let renderer = MicrophoneRender::new(number_of_channels, sample_rate, sender);

                let spawned = spawn_input_stream(
                    &device,
                    supported.sample_format(),
                    &supported_config,
                    renderer,
                );
                spawned.expect("Unable to spawn input stream with default config")
            }
        };

        // Required because some hosts don't play the stream automatically
        stream.play().expect("Input stream refused to play");

        let backend = CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency: Arc::new(AtomicF64::new(0.)),
            sample_rate,
            number_of_channels,
            sink_id: "".into(),
        };

        (backend, receiver)
    }

    fn resume(&self) -> bool {
        self.stream.resume()
    }

    fn suspend(&self) -> bool {
        self.stream.suspend()
    }

    fn close(&self) {
        self.stream.close()
    }

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn number_of_channels(&self) -> usize {
        self.number_of_channels
    }

    fn output_latency(&self) -> f64 {
        self.output_latency.load()
    }

    fn sink_id(&self) -> &str {
        self.sink_id.as_str()
    }

    fn boxed_clone(&self) -> Box<dyn AudioBackendManager> {
        Box::new(self.clone())
    }

    fn enumerate_devices() -> Vec<MediaDeviceInfo>
    where
        Self: Sized,
    {
        let mut index = 0;

        let mut inputs: Vec<MediaDeviceInfo> = cpal::default_host()
            .devices()
            .unwrap()
            .filter(|d| d.default_input_config().is_ok())
            .map(|d| {
                index += 1;

                MediaDeviceInfo::new(
                    format!("{}", index),
                    None,
                    MediaDeviceInfoKind::AudioInput,
                    d.name().unwrap(),
                    Box::new(d),
                )
            })
            .collect();

        let mut outputs: Vec<MediaDeviceInfo> = cpal::default_host()
            .devices()
            .unwrap()
            .filter(|d| d.default_output_config().is_ok())
            .map(|d| {
                index += 1;

                MediaDeviceInfo::new(
                    format!("{}", index),
                    None,
                    MediaDeviceInfoKind::AudioOutput,
                    d.name().unwrap(),
                    Box::new(d),
                )
            })
            .collect();

        inputs.append(&mut outputs);

        inputs
    }
}

fn latency_in_seconds(infos: &OutputCallbackInfo) -> f64 {
    let timestamp = infos.timestamp();
    let delta = timestamp
        .playback
        .duration_since(&timestamp.callback)
        .unwrap();
    delta.as_secs() as f64 + delta.subsec_nanos() as f64 * 1e-9
}

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
    output_latency: Arc<AtomicF64>,
) -> Result<Stream, BuildStreamError> {
    let err_fn = |err| log::error!("an error occurred on the output audio stream: {}", err);

    match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            config,
            move |d: &mut [f32], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::F64 => device.build_output_stream(
            config,
            move |d: &mut [f64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::U8 => device.build_output_stream(
            config,
            move |d: &mut [u8], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            config,
            move |d: &mut [u16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::U32 => device.build_output_stream(
            config,
            move |d: &mut [u32], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::U64 => device.build_output_stream(
            config,
            move |d: &mut [u64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::I8 => device.build_output_stream(
            config,
            move |d: &mut [i8], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            move |d: &mut [i16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::I32 => device.build_output_stream(
            config,
            move |d: &mut [i32], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        SampleFormat::I64 => device.build_output_stream(
            config,
            move |d: &mut [i64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
            None,
        ),
        _ => panic!("Unknown cpal output sample format"),
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
            device.build_input_stream(config, move |d: &[f32], _c| render.render(d), err_fn, None)
        }
        SampleFormat::F64 => {
            device.build_input_stream(config, move |d: &[f64], _c| render.render(d), err_fn, None)
        }
        SampleFormat::U8 => {
            device.build_input_stream(config, move |d: &[u8], _c| render.render(d), err_fn, None)
        }
        SampleFormat::U16 => {
            device.build_input_stream(config, move |d: &[u16], _c| render.render(d), err_fn, None)
        }
        SampleFormat::U32 => {
            device.build_input_stream(config, move |d: &[u32], _c| render.render(d), err_fn, None)
        }
        SampleFormat::U64 => {
            device.build_input_stream(config, move |d: &[u64], _c| render.render(d), err_fn, None)
        }
        SampleFormat::I8 => {
            device.build_input_stream(config, move |d: &[i8], _c| render.render(d), err_fn, None)
        }
        SampleFormat::I16 => {
            device.build_input_stream(config, move |d: &[i16], _c| render.render(d), err_fn, None)
        }
        SampleFormat::I32 => {
            device.build_input_stream(config, move |d: &[i32], _c| render.render(d), err_fn, None)
        }
        SampleFormat::I64 => {
            device.build_input_stream(config, move |d: &[i64], _c| render.render(d), err_fn, None)
        }
        _ => panic!("Unknown cpal input sample format"),
    }
}
