//! The `OfflineAudioContext` type
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::buffer::AudioBuffer;
use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};
use crate::render::RenderThread;
use crate::{assert_valid_sample_rate, RENDER_QUANTUM_SIZE};

pub(crate) type OfflineAudioContextCallback = dyn FnOnce(&mut OfflineAudioContext);

/// The `OfflineAudioContext` doesn't render the audio to the device hardware; instead, it generates
/// it, as fast as it can, and outputs the result to an `AudioBuffer`.
// the naming comes from the web audio specification
#[allow(clippy::module_name_repetitions)]
pub struct OfflineAudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// the size of the buffer in sample-frames
    length: usize,
    /// the rendering 'thread', fully controlled by the offline context
    renderer: Option<RenderThread>,
    /// callbacks to run at certain render quanta (via `suspend`)
    suspend_callbacks: HashMap<usize, Box<OfflineAudioContextCallback>>,
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
    #[allow(clippy::missing_panics_doc)]
    pub fn new(number_of_channels: usize, length: usize, sample_rate: f32) -> Self {
        assert_valid_sample_rate(sample_rate);

        // communication channel to the render thread,
        // unbounded is fine because it does not need to be realtime safe
        let (sender, receiver) = crossbeam_channel::unbounded();

        let (node_id_producer, node_id_consumer) = llq::Queue::new().split();
        let graph = crate::render::graph::Graph::new(node_id_producer);
        let message = crate::message::ControlMessage::Startup { graph };
        sender.send(message).unwrap();

        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = Arc::clone(&frames_played);

        // setup the render 'thread', which will run inside the control thread
        let renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            frames_played_clone,
        );

        // first, setup the base audio context
        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            frames_played,
            sender,
            None,
            true,
            node_id_consumer,
        );

        Self {
            base,
            length,
            renderer: Some(renderer),
            suspend_callbacks: HashMap::new(),
        }
    }

    /// Given the current connections and scheduled changes, starts rendering audio.
    ///
    /// This function will block the current thread and returns the rendered `AudioBuffer`
    /// synchronously. An async version is currently not implemented.
    ///
    /// # Panics
    ///
    /// Panics if this method is called multiple times
    pub fn start_rendering_sync(&mut self) -> AudioBuffer {
        let renderer = match self.renderer.take() {
            None => panic!("InvalidStateError: Cannot call `startRendering` twice"),
            Some(v) => v,
        };
        renderer.render_audiobuffer_sync(
            self.length,
            std::mem::take(&mut self.suspend_callbacks),
            self,
        )
    }

    /// get the length of rendering audio buffer
    // false positive: OfflineAudioContext is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Schedules a suspension of the time progression in the audio context at the specified time
    /// and runs a callback.
    ///
    /// The specified time is quantized and rounded up to the render quantum size.
    ///
    /// # Panics
    ///
    /// Panics if the quantized frame number
    ///
    /// - is negative or
    /// - is less than or equal to the current time or
    /// - is greater than or equal to the total render duration or
    /// - is scheduled by another suspend for the same time
    pub fn suspend_at<F: FnOnce(&mut Self) + 'static>(&mut self, suspend_time: f64, callback: F) {
        let quantum = (suspend_time * self.base.sample_rate() as f64 / RENDER_QUANTUM_SIZE as f64)
            .ceil() as usize;
        match self.suspend_callbacks.entry(quantum) {
            Entry::Occupied(_) => panic!(
                "InvalidStateError: cannot suspend multiple times at the same render quantum"
            ),
            Entry::Vacant(e) => {
                e.insert(Box::new(callback));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn render_empty_graph() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);
        let buffer = context.start_rendering_sync();

        assert_eq!(context.length(), 555);

        assert_eq!(buffer.number_of_channels(), 2);
        assert_eq!(buffer.length(), 555);
        assert_float_eq!(buffer.get_channel_data(0), &[0.; 555][..], abs_all <= 0.);
        assert_float_eq!(buffer.get_channel_data(1), &[0.; 555][..], abs_all <= 0.);
    }

    #[test]
    #[should_panic]
    fn render_twice_panics() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);
        let _ = context.start_rendering_sync();
        let _ = context.start_rendering_sync();
    }
}
