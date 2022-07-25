use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::AudioBackend;

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::media::MicrophoneRender;
use crate::message::ControlMessage;
use crate::render::RenderThread;
use crate::RENDER_QUANTUM_SIZE;

use cubeb::{Context, StereoFrame, Stream};

use crossbeam_channel::{Receiver, Sender};

type Frame = StereoFrame<f32>;

mod private {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone)]
    pub struct ThreadSafeClosableStream(Arc<Mutex<Option<Stream<Frame>>>>);

    impl ThreadSafeClosableStream {
        pub fn new(stream: Stream<Frame>) -> Self {
            Self(Arc::new(Mutex::new(Some(stream))))
        }

        pub fn close(&self) {
            self.suspend();
            self.0.lock().unwrap().take();
        }

        pub fn resume(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.start() {
                    panic!("Error resuming cubeb stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn suspend(&self) -> bool {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                if let Err(e) = s.stop() {
                    panic!("Error suspending cubeb stream: {:?}", e);
                }
                return true;
            }

            false
        }

        pub fn output_latency(&self, sample_rate: f32) -> f64 {
            if let Some(s) = self.0.lock().unwrap().as_ref() {
                match s.latency() {
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
    ) -> (Self, Sender<ControlMessage>)
    where
        Self: Sized,
    {
        // Set up cubeb context
        let ctx = Context::init(None, None).unwrap();

        // Use user requested sample rate, or else the device preferred one
        let device_sample_rate = ctx.preferred_sample_rate().map(|v| v as f32).ok();
        let sample_rate = options.sample_rate.or(device_sample_rate).unwrap_or(48000.);

        // TODO support all channel configs
        let _max_channel_count = ctx.max_channel_count().map(|v| v as usize).ok();
        let number_of_channels = 2;
        let layout = cubeb::ChannelLayout::STEREO;

        // Set up render thread
        let (sender, receiver) = crossbeam_channel::unbounded();
        let mut renderer =
            RenderThread::new(sample_rate, number_of_channels, receiver, frames_played);

        let params = cubeb::StreamParamsBuilder::new()
            .format(cubeb::SampleFormat::Float32LE) // TODO may not be available for device
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

        let mut builder = cubeb::StreamBuilder::<Frame>::new();
        builder
            .name("Cubeb web_audio_api (mono)")
            .default_output(&params)
            .default_input(&params)
            .latency(buffer_size)
            .data_callback(move |_input, output| {
                // TODO can we avoid the temp buffer?
                let mut tmp = [0.; 128 * 2];
                renderer.render(&mut tmp);
                for (f, t) in output.iter_mut().zip(tmp.chunks(number_of_channels)) {
                    f.l = t[0];
                    f.r = t[1];
                }
                output.len() as isize
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

        (backend, sender)
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
            .format(cubeb::SampleFormat::Float32LE) // TODO may not be available for device
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

        let mut builder = cubeb::StreamBuilder::<Frame>::new();
        builder
            .name("Cubeb web_audio_api (mono)")
            .default_output(&params)
            .default_input(&params)
            .latency(buffer_size)
            .data_callback(move |input, _output| {
                // TODO can we avoid the temp buffer?
                let mut tmp = [0.; 128 * 2];
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
}
