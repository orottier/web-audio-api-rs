//! The `OfflineAudioContext` type
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::buffer::AudioBuffer;
use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};
use crate::render::RenderThread;
use crate::{AtomicF64, SampleRate, RENDER_QUANTUM_SIZE};

/// The `OfflineAudioContext` doesn't render the audio to the device hardware; instead, it generates
/// it, as fast as it can, and outputs the result to an `AudioBuffer`.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct OfflineAudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// the size of the buffer in sample-frames
    length: usize,
    /// the rendering 'thread', fully controlled by the offline context
    renderer: RenderThread,
}

impl BaseAudioContext for OfflineAudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }
}

impl OfflineAudioContext {
    /// Creates an `OfflineAudioContext` instance
    ///
    /// # Arguments
    ///
    /// * `channels` - number of output channels to render
    /// * `length` - length of the rendering audio buffer
    /// * `sample_rate` - output sample rate
    #[must_use]
    pub fn new(number_of_channels: usize, length: usize, sample_rate: SampleRate) -> Self {
        // communication channel to the render thread
        let (sender, receiver) = crossbeam_channel::unbounded();

        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        // this is irrelevant for offline context, so we just put 0.
        let output_latency = Arc::new(AtomicF64::new(0.));
        let output_latency_clone = output_latency.clone();

        // setup the render 'thread', which will run inside the control thread
        let renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            frames_played_clone,
            output_latency_clones,
        );

        // first, setup the base audio context
        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            output_latency,
            sender,
            true,
        );

        Self {
            base,
            length,
            renderer,
        }
    }

    /// Given the current connections and scheduled changes, starts rendering audio.
    ///
    /// This function will block the current thread and returns the rendered `AudioBuffer`
    /// synchronously. An async version is currently not implemented.
    pub fn start_rendering_sync(&mut self) -> AudioBuffer {
        // make buffer_size always a multiple of RENDER_QUANTUM_SIZE, so we can still render piecewise with
        // the desired number of frames.
        let buffer_size =
            (self.length + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE * RENDER_QUANTUM_SIZE;

        let mut buf = self.renderer.render_audiobuffer(buffer_size);
        let _split = buf.split_off(self.length);

        buf
    }

    /// get the length of rendering audio buffer
    // false positive: OfflineAudioContext is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
}
