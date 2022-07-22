use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::message::ControlMessage;
use crate::AtomicF64;

#[cfg(feature = "cpal")]
mod backend_cpal;

#[cfg(feature = "cubeb")]
mod backend_cubeb;

#[allow(unused_variables)]
pub(crate) fn build_output(
    frames_played: Arc<AtomicU64>,
    output_latency: Arc<AtomicF64>,
    options: AudioContextOptions,
) -> (Box<dyn AudioBackend>, Sender<ControlMessage>) {
    #[cfg(feature = "cubeb")]
    {
        let (b, s) =
            backend_cubeb::CubebBackend::build_output(frames_played, output_latency, options);
        (Box::new(b), s)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let (b, s) =
            backend_cpal::CpalBackend::build_output(frames_played, output_latency, options);
        (Box::new(b), s)
    }
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }
}

#[cfg(any(feature = "cubeb", feature = "cpal"))]
pub(crate) fn build_input(
    options: AudioContextOptions,
) -> (Box<dyn AudioBackend>, Receiver<AudioBuffer>) {
    #[cfg(feature = "cubeb")]
    {
        let (b, r) = backend_cubeb::CubebBackend::build_input(options);
        (Box::new(b), r)
    }
    #[cfg(all(not(feature = "cubeb"), feature = "cpal"))]
    {
        let (b, r) = backend_cpal::CpalBackend::build_input(options);
        (Box::new(b), r)
    }
    #[cfg(all(not(feature = "cubeb"), not(feature = "cpal")))]
    {
        panic!("No audio backend available, enable the 'cpal' or 'cubeb' feature")
    }
}

pub(crate) trait AudioBackend: Send + Sync + 'static {
    fn build_output(
        frames_played: Arc<AtomicU64>,
        output_latency: Arc<AtomicF64>,
        options: AudioContextOptions,
    ) -> (Self, Sender<ControlMessage>)
    where
        Self: Sized;

    fn build_input(options: AudioContextOptions) -> (Self, Receiver<AudioBuffer>)
    where
        Self: Sized;
    fn resume(&self) -> bool;
    fn suspend(&self) -> bool;
    fn close(&self);

    fn sample_rate(&self) -> f32;
    fn number_of_channels(&self) -> usize;
    fn boxed_clone(&self) -> Box<dyn AudioBackend>;
}
