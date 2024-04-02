use std::sync::Arc;

use super::{AudioBackendManager, RenderThreadInit};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::io::microphone::MicrophoneRender;
use crate::media_devices::{MediaDeviceInfo, MediaDeviceInfoKind};
use crate::render::RenderThread;
use crate::{MAX_CHANNELS, RENDER_QUANTUM_SIZE};

use cubeb::{Context, DeviceId, DeviceType, StereoFrame, Stream, StreamParams};

use crossbeam_channel::Receiver;

// erase type of `Frame` in cubeb `Stream<Frame>`
struct BoxedStream(Box<dyn CubebStream>);

trait CubebStream {
    fn delegate_start(&self) -> Result<(), cubeb::Error>;
    fn delegate_stop(&self) -> Result<(), cubeb::Error>;
    fn delegate_latency(&self) -> Result<u32, cubeb::Error>;
}

impl<F> CubebStream for Stream<F> {
    fn delegate_start(&self) -> Result<(), cubeb::Error> {
        self.start()
    }
    fn delegate_stop(&self) -> Result<(), cubeb::Error> {
        self.stop()
    }
    fn delegate_latency(&self) -> Result<u32, cubeb::Error> {
        self.latency()
    }
}

// I doubt this construct is entirely safe. Stream is not Send/Sync (probably for a good reason) so
// it should be managed from a single thread instead.
// <https://github.com/orottier/web-audio-api-rs/issues/357>
mod private {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<BoxedStream>>>);

    impl ThreadSafeClosableStream {
        pub fn new<F: 'static>(stream: Stream<F>) -> Self {
            let boxed_stream = BoxedStream(Box::new(stream));
            #[allow(clippy::arc_with_non_send_sync)]
            Self(Arc::new(Mutex::new(Some(boxed_stream))))
        }

        pub fn close(&self) {
            self.suspend();
            self.0.lock().unwrap().take();
        }

        pub fn resume(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.0.delegate_start() {
                    panic!("Error resuming cubeb stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn suspend(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.0.delegate_stop() {
                    panic!("Error suspending cubeb stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn output_latency(&self, sample_rate: f32) -> f64 {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                match s.0.delegate_latency() {
                    Err(e) => panic!("Error getting cubeb latency: {:?}", e),
                    Ok(frames) => return frames as f64 / sample_rate as f64,
                }
            }

            0.
        }
    }

    // SAFETY:
    // The cubeb `Stream` is marked !Sync and !Send because some platforms are not thread-safe
    // Since we wrap the Stream in a Mutex, we should be fine
    unsafe impl Sync for ThreadSafeClosableStream {}
    unsafe impl Send for ThreadSafeClosableStream {}
}
use private::ThreadSafeClosableStream;

fn init_output_backend<const N: usize>(
    ctx: &Context,
    params: StreamParams,
    buffer_size: u32,
    device: Option<DeviceId>,
    mut renderer: RenderThread,
) -> ThreadSafeClosableStream {
    let mut builder = cubeb::StreamBuilder::<[f32; N]>::new();

    match device {
        None => builder.default_output(&params),
        Some(devid) => builder.output(devid, &params),
    };

    builder
        .name("Cubeb web_audio_api")
        .latency(buffer_size)
        .data_callback(move |_input, output| {
            // `output` is `&mut [[f32; N]]`, a slice of slices.
            // The renderer just wants a single slice, flatten it.
            // Inspired by the unstable feature <https://github.com/rust-lang/rust/pull/95579>
            {
                let output: &mut [f32] =
                    // SAFETY: `[T]` is layout-identical to `[T; N]`
                    unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), RENDER_QUANTUM_SIZE * N) };
                renderer.render(output);
            }

            output.len() as isize
        })
        .state_callback(|state| {
            println!("stream state changed: {state:?}");
        });

    let stream = builder
        .init(ctx)
        .expect("InvalidStateError - Failed to create cubeb stream");
    ThreadSafeClosableStream::new(stream)
}

/// Audio backend using the `cubeb` library
#[derive(Clone)]
pub(crate) struct CubebBackend {
    stream: ThreadSafeClosableStream,
    sample_rate: f32,
    number_of_channels: usize,
    sink_id: String,
}

impl AudioBackendManager for CubebBackend {
    fn build_output(options: AudioContextOptions, render_thread_init: RenderThreadInit) -> Self
    where
        Self: Sized,
    {
        let RenderThreadInit {
            state,
            frames_played,
            ctrl_msg_recv,
            load_value_send,
            event_send,
        } = render_thread_init;

        // Set up cubeb context
        let ctx = Context::init(None, None).unwrap();
        log::info!("Audio Output Host: cubeb {:?}", ctx.backend_id());

        // Use user requested sample rate, or else the device preferred one
        let device_sample_rate = ctx.preferred_sample_rate().map(|v| v as f32).ok();
        let sample_rate = options.sample_rate.or(device_sample_rate).unwrap_or(48000.);

        let number_of_channels = ctx
            .max_channel_count()
            .map(|v| v as usize)
            .ok()
            .unwrap_or(2);

        // clamp the requested stream number of channels to MAX_CHANNELS even if
        // the soundcard can provide more channels
        let number_of_channels = number_of_channels.min(MAX_CHANNELS);

        let layout = match number_of_channels {
            1 => cubeb::ChannelLayout::MONO,
            2 => cubeb::ChannelLayout::STEREO,
            4 => cubeb::ChannelLayout::QUAD,
            _ => cubeb::ChannelLayout::UNDEFINED, // TODO, does this work?
        };

        let mut renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            ctrl_msg_recv,
            state,
            frames_played,
            event_send,
        );
        renderer.set_load_value_sender(load_value_send);
        renderer.spawn_garbage_collector_thread();

        let params = cubeb::StreamParamsBuilder::new()
            .format(cubeb::SampleFormat::Float32NE) // use float (native endian)
            .rate(sample_rate as u32)
            .channels(number_of_channels as u32)
            .layout(layout)
            .take();

        // Calculate ideal latency
        let buffer_size_req =
            super::buffer_size_for_latency_category(options.latency_hint, sample_rate) as u32;
        let min_latency = ctx
            .min_latency(&params)
            .ok()
            .unwrap_or(RENDER_QUANTUM_SIZE as u32);
        let buffer_size = buffer_size_req.max(min_latency);

        let device = if options.sink_id.is_empty() {
            None
        } else {
            Self::enumerate_devices_sync()
                .into_iter()
                .find(|e| e.device_id() == options.sink_id)
                .map(|e| *e.device().downcast::<DeviceId>().unwrap())
        };

        let stream = match number_of_channels {
            // so sorry, but I need to constify the non-const `number_of_channels`
            1 => init_output_backend::<1>(&ctx, params, buffer_size, device, renderer),
            2 => init_output_backend::<2>(&ctx, params, buffer_size, device, renderer),
            3 => init_output_backend::<3>(&ctx, params, buffer_size, device, renderer),
            4 => init_output_backend::<4>(&ctx, params, buffer_size, device, renderer),
            5 => init_output_backend::<5>(&ctx, params, buffer_size, device, renderer),
            6 => init_output_backend::<6>(&ctx, params, buffer_size, device, renderer),
            7 => init_output_backend::<7>(&ctx, params, buffer_size, device, renderer),
            8 => init_output_backend::<8>(&ctx, params, buffer_size, device, renderer),
            9 => init_output_backend::<9>(&ctx, params, buffer_size, device, renderer),
            10 => init_output_backend::<10>(&ctx, params, buffer_size, device, renderer),
            11 => init_output_backend::<11>(&ctx, params, buffer_size, device, renderer),
            12 => init_output_backend::<12>(&ctx, params, buffer_size, device, renderer),
            13 => init_output_backend::<13>(&ctx, params, buffer_size, device, renderer),
            14 => init_output_backend::<14>(&ctx, params, buffer_size, device, renderer),
            15 => init_output_backend::<15>(&ctx, params, buffer_size, device, renderer),
            16 => init_output_backend::<16>(&ctx, params, buffer_size, device, renderer),
            17 => init_output_backend::<17>(&ctx, params, buffer_size, device, renderer),
            18 => init_output_backend::<18>(&ctx, params, buffer_size, device, renderer),
            19 => init_output_backend::<19>(&ctx, params, buffer_size, device, renderer),
            20 => init_output_backend::<20>(&ctx, params, buffer_size, device, renderer),
            21 => init_output_backend::<21>(&ctx, params, buffer_size, device, renderer),
            22 => init_output_backend::<22>(&ctx, params, buffer_size, device, renderer),
            23 => init_output_backend::<23>(&ctx, params, buffer_size, device, renderer),
            24 => init_output_backend::<24>(&ctx, params, buffer_size, device, renderer),
            25 => init_output_backend::<25>(&ctx, params, buffer_size, device, renderer),
            26 => init_output_backend::<26>(&ctx, params, buffer_size, device, renderer),
            27 => init_output_backend::<27>(&ctx, params, buffer_size, device, renderer),
            28 => init_output_backend::<28>(&ctx, params, buffer_size, device, renderer),
            29 => init_output_backend::<29>(&ctx, params, buffer_size, device, renderer),
            30 => init_output_backend::<30>(&ctx, params, buffer_size, device, renderer),
            31 => init_output_backend::<31>(&ctx, params, buffer_size, device, renderer),
            32 => init_output_backend::<32>(&ctx, params, buffer_size, device, renderer),
            _ => unreachable!(),
        };

        let backend = CubebBackend {
            stream,
            number_of_channels,
            sample_rate,
            sink_id: options.sink_id,
        };

        backend.resume();

        backend
    }

    fn build_input(
        options: AudioContextOptions,
        _number_of_channels: Option<u32>,
    ) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized,
    {
        /* Set up a dedicated stream for input capturing
         *
         * This is not how it should be, we should link the input stream together
         * with the output stream in one go. However, this means that we always
         * capture mic input, even if the user does not want it - TODO
         */

        // Set up cubeb context
        let ctx = Context::init(None, None).unwrap();
        log::info!("Audio Input Host: cubeb {:?}", ctx.backend_id());

        // Use user requested sample rate, or else the device preferred one
        let device_sample_rate = ctx.preferred_sample_rate().map(|v| v as f32).ok();
        let sample_rate = options.sample_rate.or(device_sample_rate).unwrap_or(48000.);

        // TODO support all channel configs
        let _max_channel_count = ctx.max_channel_count().map(|v| v as usize).ok();
        const NUMBER_OF_INPUT_CHANNELS: usize = 2;
        let layout = cubeb::ChannelLayout::STEREO;

        let params = cubeb::StreamParamsBuilder::new()
            .format(cubeb::SampleFormat::Float32NE) // use float (native endian)
            .rate(sample_rate as u32)
            .channels(NUMBER_OF_INPUT_CHANNELS as u32)
            .layout(layout)
            .take();

        // Calculate ideal latency
        let buffer_size_req =
            super::buffer_size_for_latency_category(options.latency_hint, sample_rate) as u32;
        let min_latency = ctx
            .min_latency(&params)
            .ok()
            .unwrap_or(RENDER_QUANTUM_SIZE as u32);
        let buffer_size = buffer_size_req.max(min_latency);

        let device = if options.sink_id.is_empty() {
            None
        } else {
            Self::enumerate_devices_sync()
                .into_iter()
                .find(|e| e.device_id() == options.sink_id)
                .map(|e| *e.device().downcast::<DeviceId>().unwrap())
        };

        let smoothing = 3; // todo, use buffering to smooth frame drops
        let (sender, receiver) = crossbeam_channel::bounded(smoothing);
        let renderer = MicrophoneRender::new(NUMBER_OF_INPUT_CHANNELS, sample_rate, sender);

        // Microphone input is always assumed STEREO (TODO)
        let mut builder = cubeb::StreamBuilder::<StereoFrame<f32>>::new();

        match device {
            None => builder.default_input(&params),
            Some(devid) => builder.input(devid, &params),
        };

        builder
            .name("Cubeb web_audio_api (mono)")
            .latency(buffer_size)
            .data_callback(move |input, _output| {
                let mut tmp = [0.; RENDER_QUANTUM_SIZE * NUMBER_OF_INPUT_CHANNELS];
                tmp.chunks_mut(NUMBER_OF_INPUT_CHANNELS)
                    .zip(input)
                    .for_each(|(t, i)| {
                        t[0] = i.l;
                        t[1] = i.r;
                    });
                renderer.render(&tmp);
                input.len() as isize
            })
            .state_callback(|state| {
                println!("stream state changed: {state:?}");
            });

        let stream = builder
            .init(&ctx)
            .expect("InvalidStateError - Failed to create cubeb stream");

        stream.start().unwrap();

        let backend = CubebBackend {
            stream: ThreadSafeClosableStream::new(stream),
            number_of_channels: NUMBER_OF_INPUT_CHANNELS,
            sample_rate,
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
        self.stream.output_latency(self.sample_rate)
    }

    fn sink_id(&self) -> &str {
        self.sink_id.as_str()
    }

    fn enumerate_devices_sync() -> Vec<MediaDeviceInfo>
    where
        Self: Sized,
    {
        let context = Context::init(None, None).unwrap();

        let inputs = context.enumerate_devices(DeviceType::INPUT).unwrap();
        let input_devices = inputs.iter().map(|d| (d, MediaDeviceInfoKind::AudioInput));

        let outputs = context.enumerate_devices(DeviceType::OUTPUT).unwrap();
        let output_devices = outputs
            .iter()
            .map(|d| (d, MediaDeviceInfoKind::AudioOutput));

        let mut list = Vec::<MediaDeviceInfo>::new();

        for (device, kind) in input_devices.chain(output_devices) {
            let mut index = 0;

            loop {
                let device_id = crate::media_devices::DeviceId::as_string(
                    kind,
                    "cubeb".to_string(),
                    device.friendly_name().unwrap().into(),
                    device.max_channels().try_into().unwrap(),
                    index,
                );

                if !list.iter().any(|d| d.device_id() == device_id) {
                    let device = MediaDeviceInfo::new(
                        device_id,
                        device.group_id().map(str::to_string),
                        kind,
                        device.friendly_name().unwrap().into(),
                        Box::new(device.devid()),
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
