//! Audio IO management API
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, OutputCallbackInfo, SampleFormat, Stream, StreamConfig,
    SupportedBufferSize,
};
use crossbeam_channel::Receiver;

use super::{AudioBackendManager, RenderThreadInit};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextLatencyCategory;
use crate::context::AudioContextOptions;
use crate::io::microphone::MicrophoneRender;
use crate::media_devices::{MediaDeviceInfo, MediaDeviceInfoKind};
use crate::render::RenderThread;
use crate::{AtomicF64, MAX_CHANNELS};

// I doubt this construct is entirely safe. Stream is not Send/Sync (probably for a good reason) so
// it should be managed from a single thread instead.
// <https://github.com/orottier/web-audio-api-rs/issues/357>
mod private {
    use super::*;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<Stream>>>);

    impl ThreadSafeClosableStream {
        pub fn new(stream: Stream) -> Self {
            #[allow(clippy::arc_with_non_send_sync)]
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

fn get_host() -> cpal::Host {
    #[cfg(feature = "cpal-jack")]
    {
        // seems to be always Some when jack is installed,
        // even if it's not running
        if let Some(jack_id) = cpal::available_hosts()
            .into_iter()
            .find(|id| *id == cpal::HostId::Jack)
        {
            let jack_host = cpal::host_from_id(jack_id).unwrap();

            // if jack is not running, the host can't access devices
            // fallback to default host
            return match jack_host.devices() {
                Ok(devices) => {
                    // no jack devices found, jack is not running
                    if devices.count() == 0 {
                        log::warn!("No jack devices found, fallback to default host");
                        cpal::default_host()
                    } else {
                        jack_host
                    }
                }
                // cpal does not seems to return Err at this point
                // but just in case, fallback to default host
                Err(_) => cpal::default_host(),
            };
        }
    }

    cpal::default_host()
}

/// Audio backend using the `cpal` library
#[derive(Clone)]
#[allow(unused)]
pub(crate) struct CpalBackend {
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
        let host = get_host();

        log::info!("Audio Output Host: cpal {:?}", host.id());

        let RenderThreadInit {
            state,
            frames_played,
            ctrl_msg_recv,
            load_value_send,
            event_send,
        } = render_thread_init;

        let device = if options.sink_id.is_empty() {
            host.default_output_device()
                .expect("InvalidStateError - no output device available")
        } else {
            Self::enumerate_devices_sync()
                .into_iter()
                .find(|e| e.device_id() == options.sink_id)
                .map(|e| *e.device().downcast::<cpal::Device>().unwrap())
                .unwrap_or_else(|| {
                    host.default_output_device()
                        .expect("InvalidStateError - no output device available")
                })
        };

        log::info!("Output device: {:?}", device.name());

        let default_device_config = device
            .default_output_config()
            .expect("InvalidStateError - error while querying device output config");

        // we grab the largest number of channels provided by the soundcard
        // clamped to MAX_CHANNELS, this value cannot be changed by the user
        let number_of_channels = usize::from(default_device_config.channels()).min(MAX_CHANNELS);

        // override default device configuration with the options provided by
        // the user when creating the `AudioContext`
        let mut preferred_config: StreamConfig = default_device_config.clone().into();
        // make sure the number of channels is clamped to MAX_CHANNELS
        preferred_config.channels = number_of_channels as u16;

        // set specific sample rate if requested
        if let Some(sample_rate) = options.sample_rate {
            crate::assert_valid_sample_rate(sample_rate);
            preferred_config.sample_rate.0 = sample_rate as u32;
        }

        // always try to set a decent buffer size
        let buffer_size = super::buffer_size_for_latency_category(
            options.latency_hint,
            preferred_config.sample_rate.0 as f32,
        ) as u32;

        let clamped_buffer_size: u32 = match default_device_config.buffer_size() {
            SupportedBufferSize::Unknown => buffer_size,
            SupportedBufferSize::Range { min, max } => buffer_size.clamp(*min, *max),
        };

        preferred_config.buffer_size = cpal::BufferSize::Fixed(clamped_buffer_size);

        // On android detected range for the buffer size seems to be too big, use default buffer size instead
        // See https://github.com/orottier/web-audio-api-rs/issues/515
        if cfg!(target_os = "android") {
            if let AudioContextLatencyCategory::Balanced
            | AudioContextLatencyCategory::Interactive = options.latency_hint
            {
                preferred_config.buffer_size = cpal::BufferSize::Default;
            }
        }

        // report the picked sample rate to the render thread, i.e. if the requested
        // sample rate is not supported by the hardware, it will fallback to the
        // default device sample rate
        let mut sample_rate = preferred_config.sample_rate.0 as f32;

        // shared atomic to report output latency to the control thread
        let output_latency = Arc::new(AtomicF64::new(0.));

        let mut renderer = RenderThread::new(
            sample_rate,
            preferred_config.channels as usize,
            ctrl_msg_recv.clone(),
            Arc::clone(&state),
            Arc::clone(&frames_played),
            event_send.clone(),
        );
        renderer.set_load_value_sender(load_value_send.clone());
        renderer.spawn_garbage_collector_thread();

        log::debug!(
            "Attempt output stream with preferred config: {:?}",
            &preferred_config
        );

        let spawned = spawn_output_stream(
            &device,
            default_device_config.sample_format(),
            &preferred_config,
            renderer,
            Arc::clone(&output_latency),
        );

        let stream = match spawned {
            Ok(stream) => {
                log::debug!("Output stream set up successfully");
                stream
            }
            Err(e) => {
                log::warn!("Output stream build failed with preferred config: {}", e);

                let mut supported_config: StreamConfig = default_device_config.clone().into();
                // make sure number of channels is clamped to MAX_CHANNELS
                supported_config.channels = number_of_channels as u16;
                // fallback to device default sample rate
                sample_rate = supported_config.sample_rate.0 as f32;

                log::debug!(
                    "Attempt output stream with fallback config: {:?}",
                    &supported_config
                );

                let mut renderer = RenderThread::new(
                    sample_rate,
                    supported_config.channels as usize,
                    ctrl_msg_recv,
                    state,
                    frames_played,
                    event_send,
                );
                renderer.set_load_value_sender(load_value_send);
                renderer.spawn_garbage_collector_thread();

                let spawned = spawn_output_stream(
                    &device,
                    default_device_config.sample_format(),
                    &supported_config,
                    renderer,
                    Arc::clone(&output_latency),
                );

                spawned
                    .expect("InvalidStateError - Unable to spawn output stream with default config")
            }
        };

        // Required because some hosts don't play the stream automatically
        stream
            .play()
            .expect("InvalidStateError - Output stream refused to play");

        CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency,
            sample_rate,
            number_of_channels,
            sink_id: options.sink_id,
        }
    }

    fn build_input(
        options: AudioContextOptions,
        number_of_channels: Option<u32>,
    ) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized,
    {
        let host = get_host();

        log::info!("Audio Input Host: cpal {:?}", host.id());

        let device = if options.sink_id.is_empty() {
            host.default_input_device()
                .expect("InvalidStateError - no input device available")
        } else {
            Self::enumerate_devices_sync()
                .into_iter()
                .find(|e| e.device_id() == options.sink_id)
                .map(|e| *e.device().downcast::<cpal::Device>().unwrap())
                .unwrap_or_else(|| {
                    host.default_input_device()
                        .expect("InvalidStateError - no input device available")
                })
        };

        log::info!("Input device: {:?}", device.name());

        let supported = device
            .default_input_config()
            .expect("InvalidStateError - error while querying device input config");

        // clone the config, we may need to fall back on it later
        let mut preferred: StreamConfig = supported.clone().into();

        if let Some(number_of_channels) = number_of_channels {
            preferred.channels = number_of_channels as u16;
        }

        // set specific sample rate if requested
        if let Some(sample_rate) = options.sample_rate {
            crate::assert_valid_sample_rate(sample_rate);
            preferred.sample_rate.0 = sample_rate as u32;
        }

        // always try to set a decent buffer size
        let buffer_size = super::buffer_size_for_latency_category(
            options.latency_hint,
            preferred.sample_rate.0 as f32,
        ) as u32;

        let clamped_buffer_size: u32 = match supported.buffer_size() {
            SupportedBufferSize::Unknown => buffer_size,
            SupportedBufferSize::Range { min, max } => buffer_size.clamp(*min, *max),
        };

        preferred.buffer_size = cpal::BufferSize::Fixed(clamped_buffer_size);
        let mut sample_rate = preferred.sample_rate.0 as f32;
        let mut number_of_channels = preferred.channels as usize;

        let smoothing = 3; // todo, use buffering to smooth frame drops
        let (sender, mut receiver) = crossbeam_channel::bounded(smoothing);
        let renderer = MicrophoneRender::new(number_of_channels, sample_rate, sender);

        log::debug!(
            "Attempt input stream with preferred config: {:?}",
            &preferred
        );

        let spawned = spawn_input_stream(&device, supported.sample_format(), &preferred, renderer);

        // the required block size preferred config may not be supported, in that
        // case, fallback the supported config
        let stream = match spawned {
            Ok(stream) => {
                log::debug!("Input stream set up successfully");
                stream
            }
            Err(e) => {
                log::warn!("Output stream build failed with preferred config: {}", e);

                let supported_config: StreamConfig = supported.clone().into();
                // fallback to device default sample rate and channel count
                number_of_channels = usize::from(supported_config.channels);
                sample_rate = supported_config.sample_rate.0 as f32;

                log::debug!(
                    "Attempt output stream with fallback config: {:?}",
                    &supported_config
                );

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
                spawned
                    .expect("InvalidStateError - Unable to spawn input stream with default config")
            }
        };

        // Required because some hosts don't play the stream automatically
        stream
            .play()
            .expect("InvalidStateError - Input stream refused to play");

        let backend = CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency: Arc::new(AtomicF64::new(0.)),
            sample_rate,
            number_of_channels,
            sink_id: options.sink_id,
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
        self.output_latency.load(Ordering::Relaxed)
    }

    fn sink_id(&self) -> &str {
        self.sink_id.as_str()
    }

    fn enumerate_devices_sync() -> Vec<MediaDeviceInfo>
    where
        Self: Sized,
    {
        let host = get_host();

        let input_devices = host.input_devices().unwrap().map(|d| {
            let num_channels = d.default_input_config().unwrap().channels();
            (d, MediaDeviceInfoKind::AudioInput, num_channels)
        });

        let output_devices = host.output_devices().unwrap().map(|d| {
            let num_channels = d.default_output_config().unwrap().channels();
            (d, MediaDeviceInfoKind::AudioOutput, num_channels)
        });

        // cf. https://github.com/orottier/web-audio-api-rs/issues/356
        let mut list = Vec::<MediaDeviceInfo>::new();

        for (device, kind, num_channels) in input_devices.chain(output_devices) {
            let mut index = 0;

            loop {
                let device_id = crate::media_devices::DeviceId::as_string(
                    kind,
                    "cpal".to_string(),
                    device.name().unwrap(),
                    num_channels,
                    index,
                );

                if !list.iter().any(|d| d.device_id() == device_id) {
                    let device = MediaDeviceInfo::new(
                        device_id,
                        None,
                        kind,
                        device.name().unwrap(),
                        Box::new(device),
                    );

                    list.push(device);
                    break;
                } else {
                    index += 1;
                }
            }
        }

        list
    }
}

fn latency_in_seconds(infos: &OutputCallbackInfo) -> f64 {
    let timestamp = infos.timestamp();
    timestamp
        .playback
        .duration_since(&timestamp.callback)
        .map(|delta| delta.as_secs() as f64 + delta.subsec_nanos() as f64 * 1e-9)
        .unwrap_or(0.0)
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
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::F64 => device.build_output_stream(
            config,
            move |d: &mut [f64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::U8 => device.build_output_stream(
            config,
            move |d: &mut [u8], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            config,
            move |d: &mut [u16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::U32 => device.build_output_stream(
            config,
            move |d: &mut [u32], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::U64 => device.build_output_stream(
            config,
            move |d: &mut [u64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::I8 => device.build_output_stream(
            config,
            move |d: &mut [i8], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            move |d: &mut [i16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::I32 => device.build_output_stream(
            config,
            move |d: &mut [i32], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::I64 => device.build_output_stream(
            config,
            move |d: &mut [i64], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i), Ordering::Relaxed);
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
