use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::{AudioBackend, MediaDeviceInfo, MediaDeviceInfoKind};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::media::MicrophoneRender;
use crate::message::ControlMessage;
use crate::render::RenderThread;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

use cubeb::{Context, DeviceType, StereoFrame, Stream, StreamParams};

use crossbeam_channel::{Receiver, Sender};

// erase type of `Frame` in cubeb `Stream<Frame>`
pub struct BoxedStream(Box<dyn CubebStream>);

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

mod private {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<BoxedStream>>>);

    impl ThreadSafeClosableStream {
        pub fn new<F: 'static>(stream: Stream<F>) -> Self {
            let boxed_stream = BoxedStream(Box::new(stream));
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
    mut renderer: RenderThread,
) -> ThreadSafeClosableStream {
    let mut builder = cubeb::StreamBuilder::<[f32; N]>::new();

    builder
        .name("Cubeb web_audio_api")
        .default_output(&params)
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
            println!("stream state changed: {:?}", state);
        });

    let stream = builder.init(ctx).expect("Failed to create cubeb stream");
    ThreadSafeClosableStream::new(stream)
}

/// Audio backend using the `cubeb` library
#[derive(Clone)]
pub struct CubebBackend {
    stream: ThreadSafeClosableStream,
    sample_rate: f32,
    number_of_channels: usize,
}

impl AudioBackend for CubebBackend {
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
        // Set up cubeb context
        let ctx = Context::init(None, None).unwrap();

        // Use user requested sample rate, or else the device preferred one
        let device_sample_rate = ctx.preferred_sample_rate().map(|v| v as f32).ok();
        let sample_rate = options.sample_rate.or(device_sample_rate).unwrap_or(48000.);

        let number_of_channels = ctx
            .max_channel_count()
            .map(|v| v as usize)
            .ok()
            .unwrap_or(2);
        crate::assert_valid_number_of_channels(number_of_channels);

        let layout = match number_of_channels {
            1 => cubeb::ChannelLayout::MONO,
            2 => cubeb::ChannelLayout::STEREO,
            4 => cubeb::ChannelLayout::QUAD,
            _ => cubeb::ChannelLayout::UNDEFINED, // TODO, does this work?
        };

        // Set up render thread
        let (sender, receiver) = crossbeam_channel::unbounded();
        let (cap_sender, cap_receiver) = crossbeam_channel::bounded(1);
        let renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            frames_played,
            Some(cap_sender),
        );

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

        let stream = match number_of_channels {
            // so sorry, but I need to constify the non-const `number_of_channels`
            1 => init_output_backend::<1>(&ctx, params, buffer_size, renderer),
            2 => init_output_backend::<2>(&ctx, params, buffer_size, renderer),
            3 => init_output_backend::<3>(&ctx, params, buffer_size, renderer),
            4 => init_output_backend::<4>(&ctx, params, buffer_size, renderer),
            5 => init_output_backend::<5>(&ctx, params, buffer_size, renderer),
            6 => init_output_backend::<6>(&ctx, params, buffer_size, renderer),
            7 => init_output_backend::<7>(&ctx, params, buffer_size, renderer),
            8 => init_output_backend::<8>(&ctx, params, buffer_size, renderer),
            9 => init_output_backend::<9>(&ctx, params, buffer_size, renderer),
            10 => init_output_backend::<10>(&ctx, params, buffer_size, renderer),
            11 => init_output_backend::<11>(&ctx, params, buffer_size, renderer),
            12 => init_output_backend::<12>(&ctx, params, buffer_size, renderer),
            13 => init_output_backend::<13>(&ctx, params, buffer_size, renderer),
            14 => init_output_backend::<14>(&ctx, params, buffer_size, renderer),
            15 => init_output_backend::<15>(&ctx, params, buffer_size, renderer),
            16 => init_output_backend::<16>(&ctx, params, buffer_size, renderer),
            17 => init_output_backend::<17>(&ctx, params, buffer_size, renderer),
            18 => init_output_backend::<18>(&ctx, params, buffer_size, renderer),
            19 => init_output_backend::<19>(&ctx, params, buffer_size, renderer),
            20 => init_output_backend::<20>(&ctx, params, buffer_size, renderer),
            21 => init_output_backend::<21>(&ctx, params, buffer_size, renderer),
            22 => init_output_backend::<22>(&ctx, params, buffer_size, renderer),
            23 => init_output_backend::<23>(&ctx, params, buffer_size, renderer),
            24 => init_output_backend::<24>(&ctx, params, buffer_size, renderer),
            25 => init_output_backend::<25>(&ctx, params, buffer_size, renderer),
            26 => init_output_backend::<26>(&ctx, params, buffer_size, renderer),
            27 => init_output_backend::<27>(&ctx, params, buffer_size, renderer),
            28 => init_output_backend::<28>(&ctx, params, buffer_size, renderer),
            29 => init_output_backend::<29>(&ctx, params, buffer_size, renderer),
            30 => init_output_backend::<30>(&ctx, params, buffer_size, renderer),
            31 => init_output_backend::<31>(&ctx, params, buffer_size, renderer),
            32 => init_output_backend::<32>(&ctx, params, buffer_size, renderer),
            _ => unreachable!(),
        };

        let backend = CubebBackend {
            stream,
            number_of_channels,
            sample_rate,
        };

        backend.resume();

        (backend, sender, cap_receiver)
    }

    fn build_input(options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
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

        // Use user requested sample rate, or else the device preferred one
        let device_sample_rate = ctx.preferred_sample_rate().map(|v| v as f32).ok();
        let sample_rate = options.sample_rate.or(device_sample_rate).unwrap_or(48000.);

        // TODO support all channel configs
        let _max_channel_count = ctx.max_channel_count().map(|v| v as usize).ok();
        let number_of_channels = 2;
        let layout = cubeb::ChannelLayout::STEREO;

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

        let smoothing = 3; // todo, use buffering to smooth frame drops
        let (sender, receiver) = crossbeam_channel::bounded(smoothing);
        let renderer = MicrophoneRender::new(number_of_channels, sample_rate, sender);

        // Microphone input is always assumed STEREO (TODO)
        let mut builder = cubeb::StreamBuilder::<StereoFrame<f32>>::new();
        builder
            .name("Cubeb web_audio_api (mono)")
            .default_input(&params)
            .latency(buffer_size)
            .data_callback(move |input, _output| {
                let mut tmp = [0.; RENDER_QUANTUM_SIZE * 2];
                tmp.chunks_mut(number_of_channels)
                    .zip(input)
                    .for_each(|(t, i)| {
                        t[0] = i.l;
                        t[1] = i.r;
                    });
                renderer.render(&tmp);
                input.len() as isize
            })
            .state_callback(|state| {
                println!("stream state changed: {:?}", state);
            });

        let stream = builder.init(&ctx).expect("Failed to create cubeb stream");

        stream.start().unwrap();

        let backend = CubebBackend {
            stream: ThreadSafeClosableStream::new(stream),
            number_of_channels,
            sample_rate,
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

    fn boxed_clone(&self) -> Box<dyn AudioBackend> {
        Box::new(self.clone())
    }

    fn enumerate_devices() -> Vec<MediaDeviceInfo>
    where
        Self: Sized,
    {
        Context::init(None, None)
            .unwrap()
            .enumerate_devices(DeviceType::OUTPUT)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, d)| {
                MediaDeviceInfo::new(
                    format!("{}", i + 1),
                    d.group_id().map(str::to_string),
                    MediaDeviceInfoKind::AudioOutput,
                    d.friendly_name().unwrap().into(),
                )
            })
            .collect()
    }
}
