use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::buffer::AudioBuffer;
use crate::context::AudioContextOptions;
use crate::message::ControlMessage;
use crate::AtomicF64;

mod backend_cpal;
pub(crate) use backend_cpal::CpalBackend;

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
