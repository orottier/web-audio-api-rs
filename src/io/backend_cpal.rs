//! Audio IO management API
use std::convert::TryFrom;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::Mutex;

use crate::message::ControlMessage;
use crate::{AtomicF64, AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, OutputCallbackInfo, SampleFormat, SampleRate as CpalSampleRate,
    Stream, StreamConfig, SupportedBufferSize,
};

use super::{AudioBackend, MediaDeviceInfo, MediaDeviceInfoKind};
use crate::buffer::AudioBuffer;
use crate::context::{AudioContextLatencyCategory, AudioContextOptions};
use crate::media::MicrophoneRender;
use crate::render::RenderThread;

use crossbeam_channel::{Receiver, Sender};

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
    sink_id: Option<String>,
}

impl AudioBackend for CpalBackend {
    fn build_output(
        options: AudioContextOptions,
        frames_played: Arc<AtomicU64>,
    ) -> (
        Self,
        Sender<ControlMessage>,
        Receiver<AudioRenderCapacityLoad>,
    )
    where
        Self: Sized,
    {
        let host = cpal::default_host();
        log::info!("Host: {:?}", host.id());

        let device = match &options.sink_id {
            None => host
                .default_output_device()
                .expect("no output device available"),
            Some(d) => Self::enumerate_devices()
                .into_iter()
                .find(|e| e.device_id() == d)
                .map(|e| *e.device().downcast::<cpal::Device>().unwrap())
                .unwrap(),
        };

        log::info!("Output device: {:?}", device.name());

        let mut builder = StreamConfigsBuilder::new(device);

        // set specific sample rate if requested
        if let Some(sample_rate) = options.sample_rate {
            builder.with_sample_rate(sample_rate);
        }

        // always try to set a decent buffer size
        builder.with_latency_hint(options.latency_hint);

        let configs = builder.build();

        let output_latency = Arc::new(AtomicF64::new(0.));
        let streamer = OutputStreamer::new(configs, frames_played, output_latency.clone())
            .spawn()
            .or_fallback()
            .play();

        let (stream, config, sender, cap_receiver) = streamer.get_output_stream();
        let number_of_channels = usize::from(config.channels);
        let sample_rate = config.sample_rate.0 as f32;

        let backend = CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency,
            sample_rate,
            number_of_channels,
            sink_id: options.sink_id,
        };

        (backend, sender, cap_receiver)
    }

    fn build_input(options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized,
    {
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

        // determine best buffer size. Spec requires RENDER_QUANTUM_SIZE, but that might not be available
        let buffer_size: u32 = u32::try_from(RENDER_QUANTUM_SIZE).unwrap();
        let mut input_buffer_size = match supported_config.buffer_size() {
            SupportedBufferSize::Range { min, .. } => buffer_size.max(*min),
            SupportedBufferSize::Unknown => buffer_size,
        };
        // make buffer_size always a multiple of RENDER_QUANTUM_SIZE, so we can still render piecewise with
        // the desired number of frames.
        input_buffer_size = (input_buffer_size + buffer_size - 1) / buffer_size * buffer_size;

        let mut config: StreamConfig = supported_config.into();
        config.buffer_size = cpal::BufferSize::Fixed(input_buffer_size);
        if let Some(sample_rate) = options.sample_rate {
            config.sample_rate = CpalSampleRate(sample_rate as u32);
        }

        let sample_rate = config.sample_rate.0 as f32;
        let channels = config.channels as usize;

        let smoothing = 3; // todo, use buffering to smooth frame drops
        let (sender, mut receiver) = crossbeam_channel::bounded(smoothing);
        let renderer = MicrophoneRender::new(channels, sample_rate, sender);

        let maybe_stream = spawn_input_stream(&device, sample_format, &config, renderer);
        // our RENDER_QUANTUM_SIZEd config may not be supported, in that case, use the default config
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

        let number_of_channels = usize::from(config.channels);
        let sample_rate = config.sample_rate.0 as f32;

        let backend = CpalBackend {
            stream: ThreadSafeClosableStream::new(stream),
            output_latency: Arc::new(AtomicF64::new(0.)),
            sample_rate,
            number_of_channels,
            sink_id: None,
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

    fn sink_id(&self) -> Option<&str> {
        self.sink_id.as_deref()
    }

    fn boxed_clone(&self) -> Box<dyn AudioBackend> {
        Box::new(self.clone())
    }

    fn enumerate_devices() -> Vec<MediaDeviceInfo>
    where
        Self: Sized,
    {
        cpal::default_host()
            .devices()
            .unwrap()
            .filter(|d| d.default_output_config().is_ok())
            .enumerate()
            .map(|(i, d)| {
                MediaDeviceInfo::new(
                    format!("{}", i + 1),
                    None,
                    MediaDeviceInfoKind::AudioOutput,
                    d.name().unwrap(),
                    Box::new(d),
                )
            })
            .collect()
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
        ),
        SampleFormat::U16 => device.build_output_stream(
            config,
            move |d: &mut [u16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            move |d: &mut [i16], i: &OutputCallbackInfo| {
                render.render(d);
                output_latency.store(latency_in_seconds(i));
            },
            err_fn,
        ),
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

/// This struct helps to build `StreamConfigs`
struct StreamConfigsBuilder {
    // This is not a dead code, this field is used by OutputStreamer
    #[allow(dead_code)]
    /// The device on which the stream will be build
    device: cpal::Device,
    /// the device supported config from wich all the other configs are derived
    supported: cpal::SupportedStreamConfig,
    /// the prefered config is a primary config optionnaly modified by the user options `AudioContextOptions`
    prefered: cpal::StreamConfig,
}

impl StreamConfigsBuilder {
    /// creates the `StreamConfigBuilder`
    fn new(device: cpal::Device) -> Self {
        let supported = device
            .default_output_config()
            .expect("error while querying configs");

        Self {
            device,
            supported: supported.clone(),
            prefered: supported.into(),
        }
    }

    /// set preferred sample rate
    fn with_sample_rate(&mut self, sample_rate: f32) {
        crate::assert_valid_sample_rate(sample_rate);
        self.prefered.sample_rate.0 = sample_rate as u32;
    }

    /// define requested hardware buffer size
    #[allow(clippy::needless_pass_by_value)]
    fn with_latency_hint(&mut self, latency_hint: AudioContextLatencyCategory) {
        let buffer_size = super::buffer_size_for_latency_category(
            latency_hint,
            self.prefered.sample_rate.0 as f32,
        ) as u32;

        let clamped_buffer_size: u32 = match self.supported.buffer_size() {
            SupportedBufferSize::Unknown => buffer_size,
            SupportedBufferSize::Range { min, max } => buffer_size.clamp(*min, *max),
        };

        self.prefered.buffer_size = cpal::BufferSize::Fixed(clamped_buffer_size);
    }

    /// builds `StreamConfigs`
    fn build(self) -> StreamConfigs {
        StreamConfigs::new(self)
    }
}

/// `StreamConfigs` contains configs data
/// required to build an output stream on
/// a prefered config or a fallback config in case of failure
struct StreamConfigs {
    /// the requested sample format of the output stream
    sample_format: cpal::SampleFormat,
    /// the prefered config of the output stream
    prefered: cpal::StreamConfig,
    /// in case of failure to build the stream with `prefered`
    /// a fallback config is used to spawn the stream
    fallback: cpal::StreamConfig,
}

impl StreamConfigs {
    /// creates a stream configs with the data prepared by the builder
    fn new(builder: StreamConfigsBuilder) -> Self {
        let StreamConfigsBuilder {
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

/// `OutputStreamer` is used to spawn an output stream
struct OutputStreamer {
    /// The audio device on which the output stream is broadcast
    device: cpal::Device,
    /// The configs on which the output stream can be build
    configs: StreamConfigs,
    /// `frames_played` act as a time reference when processing
    frames_played: Arc<AtomicU64>,
    /// delay between render and actual system audio output
    output_latency: Arc<AtomicF64>,
    /// communication channel between control and render thread (sender part)
    sender: Option<Sender<ControlMessage>>,
    /// communication channel for render load values
    cap_receiver: Option<Receiver<AudioRenderCapacityLoad>>,
    /// the output stream
    stream: Option<Stream>,
    /// a flag to know if the output stream has been build with prefered config
    /// or fallback config
    falled_back: bool,
}

impl OutputStreamer {
    /// creates an `OutputStreamer`
    fn new(
        configs: StreamConfigs,
        frames_played: Arc<AtomicU64>,
        output_latency: Arc<AtomicF64>,
    ) -> Self {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");

        Self {
            device,
            configs,
            frames_played,
            output_latency,
            sender: None,
            cap_receiver: None,
            stream: None,
            falled_back: false,
        }
    }

    /// spawns the output stram with prefered config
    fn spawn(mut self) -> Result<Self, Self> {
        // try with prefered config
        let config = &self.configs.prefered;

        // Creates the render thread
        let sample_rate = config.sample_rate.0 as f32;

        // communication channel for ctrl msgs to the render thread
        let (sender, receiver) = crossbeam_channel::unbounded();
        // communication channel for render load values
        let (cap_sender, cap_receiver) = crossbeam_channel::bounded(1);

        self.sender = Some(sender);
        self.cap_receiver = Some(cap_receiver);

        // spawn the render thread
        let renderer = RenderThread::new(
            sample_rate,
            config.channels as usize,
            receiver,
            self.frames_played.clone(),
            Some(cap_sender),
        );

        log::debug!("Attempt output stream with prefered config: {:?}", &config);
        let spawned = spawn_output_stream(
            &self.device,
            self.configs.sample_format,
            config,
            renderer,
            self.output_latency.clone(),
        );

        match spawned {
            Ok(stream) => {
                log::debug!("Output stream set up successfully");
                self.stream = Some(stream);
                Ok(self)
            }
            Err(e) => {
                log::warn!("Output stream build failed with prefered config: {}", e);
                Err(self)
            }
        }
    }

    /// playes the output stream
    fn play(self) -> Self {
        self.stream
            .as_ref()
            .expect("Stream needs to exist to be played")
            .play()
            .expect("Stream refused to play");
        self
    }

    /// returns the output stream infos
    fn get_output_stream(
        self,
    ) -> (
        Stream,
        StreamConfig,
        Sender<ControlMessage>,
        Receiver<AudioRenderCapacityLoad>,
    ) {
        if self.falled_back {
            (
                self.stream.unwrap(),
                self.configs.fallback,
                self.sender.unwrap(),
                self.cap_receiver.unwrap(),
            )
        } else {
            (
                self.stream.unwrap(),
                self.configs.prefered,
                self.sender.unwrap(),
                self.cap_receiver.unwrap(),
            )
        }
    }
}

/// adds a fallback path to `OutputStreamer`
trait OrFallback {
    /// falls back if previous attempt failed
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
                let sample_rate = config.sample_rate.0 as f32;

                // communication channel to the render thread
                let (sender, receiver) = crossbeam_channel::unbounded();
                // communication channel for render load values
                let (cap_sender, cap_receiver) = crossbeam_channel::bounded(1);

                streamer.sender = Some(sender);
                streamer.cap_receiver = Some(cap_receiver);

                // spawn the render thread
                let renderer = RenderThread::new(
                    sample_rate,
                    config.channels as usize,
                    receiver,
                    streamer.frames_played.clone(),
                    Some(cap_sender),
                );

                let spawned = spawn_output_stream(
                    &streamer.device,
                    streamer.configs.sample_format,
                    config,
                    renderer,
                    streamer.output_latency.clone(),
                );
                let stream = spawned.expect("OutputStream build failed with default config");
                streamer.stream = Some(stream);
                streamer
            }
        }
    }
}
