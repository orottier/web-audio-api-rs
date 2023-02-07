//! The `OfflineAudioContext` type
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::buffer::AudioBuffer;
use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};
use crate::render::RenderThread;
use crate::{assert_valid_sample_rate, RENDER_QUANTUM_SIZE};

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
    renderer: SingleUseRenderThread,
}

mod private {
    use super::*;

    pub(crate) struct SingleUseRenderThread(RenderThread);

    impl SingleUseRenderThread {
        pub fn new(rt: RenderThread) -> Self {
            Self(rt)
        }

        pub fn render_audiobuffer(self, buffer_size: usize) -> AudioBuffer {
            self.0.render_audiobuffer(buffer_size)
        }
    }

    // SAFETY:
    // The RenderThread is not Sync since it contains `AudioRenderQuantum`s (which use Rc) and `dyn
    // AudioProcessor` which may not allow sharing between threads. However we mark the
    // SingleUseRenderThread as Sync because it can only run once (and thus on a single thread)
    // NB: the render thread should never hand out the contained `Rc` and `AudioProcessor`s
    unsafe impl Sync for SingleUseRenderThread {}
}
use private::SingleUseRenderThread;

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
    #[allow(clippy::missing_panics_doc)]
    pub fn new(number_of_channels: usize, length: usize, sample_rate: f32) -> Self {
        assert_valid_sample_rate(sample_rate);

        // communication channel to the render thread
        let (sender, receiver) = crossbeam_channel::unbounded();

        let graph = crate::render::graph::Graph::new();
        let message = crate::message::ControlMessage::Startup { graph };
        sender.send(message).unwrap();

        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = frames_played.clone();

        // setup the render 'thread', which will run inside the control thread
        let renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            frames_played_clone,
            None,
            None,
        );

        // first, setup the base audio context
        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            None,
            true,
        );

        Self {
            base,
            length,
            renderer: SingleUseRenderThread::new(renderer),
        }
    }

    /// Given the current connections and scheduled changes, starts rendering audio.
    ///
    /// This function will block the current thread and returns the rendered `AudioBuffer`
    /// synchronously. An async version is currently not implemented.
    pub fn start_rendering_sync(self) -> AudioBuffer {
        // make buffer_size always a multiple of RENDER_QUANTUM_SIZE, so we can still render piecewise with
        // the desired number of frames.
        let buffer_size =
            (self.length + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE * RENDER_QUANTUM_SIZE;

        let mut buffer = self.renderer.render_audiobuffer(buffer_size);
        buffer.split_off(self.length);

        buffer
    }

    /// get the length of rendering audio buffer
    // false positive: OfflineAudioContext is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn render_empty_graph() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);
        let buffer = context.start_rendering_sync();

        assert_eq!(buffer.number_of_channels(), 2);
        assert_eq!(buffer.length(), 555);
        assert_float_eq!(buffer.get_channel_data(0), &[0.; 555][..], abs_all <= 0.);
        assert_float_eq!(buffer.get_channel_data(1), &[0.; 555][..], abs_all <= 0.);
    }
}
