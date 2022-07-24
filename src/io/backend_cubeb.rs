use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::AudioBackend;

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::message::ControlMessage;
use crate::render::RenderThread;
use crate::AtomicF64;

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
        frames_played: Arc<AtomicU64>,
        output_latency: Arc<AtomicF64>,
        options: AudioContextOptions,
    ) -> (Self, Sender<ControlMessage>)
    where
        Self: Sized,
    {
        // TODO support all channel configs
        let number_of_channels = 2;
        let layout = cubeb::ChannelLayout::STEREO;

        // TODO get preferred sample rate from Device
        let sample_rate = options.sample_rate.unwrap_or(48_000.);

        // Set up render thread
        let (sender, receiver) = crossbeam_channel::unbounded();
        let mut renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            frames_played,
            output_latency,
        );

        // Set up cubeb
        let ctx = Context::init(None, None).unwrap();
        let params = cubeb::StreamParamsBuilder::new()
            .format(cubeb::SampleFormat::Float32LE) // TODO may not be available for device
            .rate(sample_rate as u32)
            .channels(number_of_channels as u32)
            .layout(layout)
            .take();

        let mut builder = cubeb::StreamBuilder::<Frame>::new();
        builder
            .name("Cubeb web_audio_api (mono)")
            .default_output(&params)
            .latency(128)
            .data_callback(move |_, output| {
                // TODO can we avoid the temp buffer?
                let mut tmp = [0.; 128 * 2];
                renderer.render(&mut tmp, 0.);
                for (f, t) in output.iter_mut().zip(tmp.chunks(2)) {
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

    fn build_input(_options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized,
    {
        todo!()
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

    fn boxed_clone(&self) -> Box<dyn AudioBackend> {
        Box::new(self.clone())
    }
}
