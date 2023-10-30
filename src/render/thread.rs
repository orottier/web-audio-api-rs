//! Communicates with the control thread and ships audio samples to the hardware

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use dasp_sample::FromSample;

use super::{AudioRenderQuantum, GarbageCollector, RenderScope};
use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::context::AudioNodeId;
use crate::events::EventDispatch;
use crate::message::ControlMessage;
use crate::node::ChannelInterpretation;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

use super::graph::Graph;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Option<Graph>,
    sample_rate: f32,
    /// number of channels of the backend stream, i.e. sound card number of
    /// channels clamped to MAX_CHANNELS
    number_of_channels: usize,
    frames_played: Arc<AtomicU64>,
    receiver: Option<Receiver<ControlMessage>>,
    buffer_offset: Option<(usize, AudioRenderQuantum)>,
    load_value_sender: Option<Sender<AudioRenderCapacityLoad>>,
    event_sender: Option<Sender<EventDispatch>>,
    garbage_collector: GarbageCollector,
}

// SAFETY:
// The RenderThread is not Send/Sync since it contains `AudioRenderQuantum`s (which use Rc), but
// these are only accessed within the same thread (the render thread). Due to the cpal constraints
// we can neither move the RenderThread object into the render thread, nor can we initialize the
// Rc's in that thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}
unsafe impl Send for RenderThread {}
unsafe impl Sync for RenderThread {}

impl RenderThread {
    pub fn new(
        sample_rate: f32,
        number_of_channels: usize,
        receiver: Receiver<ControlMessage>,
        frames_played: Arc<AtomicU64>,
    ) -> Self {
        Self {
            graph: None,
            sample_rate,
            number_of_channels,
            frames_played,
            receiver: Some(receiver),
            buffer_offset: None,
            load_value_sender: None,
            event_sender: None,
            garbage_collector: GarbageCollector::default(),
        }
    }

    pub(crate) fn set_event_channels(
        &mut self,
        load_value_sender: Sender<AudioRenderCapacityLoad>,
        event_sender: Sender<EventDispatch>,
    ) {
        self.load_value_sender = Some(load_value_sender);
        self.event_sender = Some(event_sender);
    }

    pub(crate) fn spawn_garbage_collector_thread(&mut self) {
        self.garbage_collector.spawn_garbage_collector_thread()
    }

    fn handle_control_messages(&mut self) {
        let receiver = match &self.receiver {
            None => return,
            Some(receiver) => receiver,
        };

        for msg in receiver.try_iter() {
            use ControlMessage::*;

            match msg {
                RegisterNode {
                    id: node_id,
                    reclaim_id,
                    node,
                    inputs,
                    outputs,
                    channel_config,
                } => {
                    self.graph.as_mut().unwrap().add_node(
                        node_id,
                        reclaim_id,
                        node,
                        inputs,
                        outputs,
                        channel_config,
                    );
                }
                ConnectNode {
                    from,
                    to,
                    output,
                    input,
                } => {
                    self.graph
                        .as_mut()
                        .unwrap()
                        .add_edge((from, output), (to, input));
                }
                DisconnectNode { from, to } => {
                    self.graph.as_mut().unwrap().remove_edge(from, to);
                }
                DisconnectAll { from } => {
                    self.graph.as_mut().unwrap().remove_edges_from(from);
                }
                FreeWhenFinished { id } => {
                    self.graph.as_mut().unwrap().mark_free_when_finished(id);
                }
                MarkCycleBreaker { id } => {
                    self.graph.as_mut().unwrap().mark_cycle_breaker(id);
                }
                Shutdown { sender } => {
                    let _ = sender.send(self.graph.take().unwrap());
                    self.receiver = None;
                    return; // no further handling of ctrl msgs
                }
                Startup { graph } => {
                    // Obtaining the current thread id invokes an allocation (on OSX) so let's take
                    // this hit on audio graph startup, so subsequent calls (needed for crossbeam)
                    // don't need to.
                    assert_no_alloc::permit_alloc(|| std::thread::current().id());
                    debug_assert!(self.graph.is_none());
                    self.graph = Some(graph);
                }
                NodeMessage { id, mut msg } => {
                    self.graph.as_mut().unwrap().route_message(id, msg.as_mut());
                    self.garbage_collector.deallocate_async(msg);
                }
            }
        }
    }

    // Render method of the `OfflineAudioContext::start_rendering_sync`
    // This method is not spec compliant and obviously marked as synchronous, so we
    // don't launch a thread.
    //
    // cf. https://webaudio.github.io/web-audio-api/#dom-offlineaudiocontext-startrendering
    pub fn render_audiobuffer_sync(mut self, length: usize) -> AudioBuffer {
        let options = AudioBufferOptions {
            number_of_channels: self.number_of_channels,
            length,
            sample_rate: self.sample_rate,
        };

        let mut buffer = AudioBuffer::new(options);
        let num_frames = (length + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE;

        for _ in 0..num_frames {
            // Handle addition/removal of nodes/edges
            self.handle_control_messages();

            // Update time
            let current_frame = self
                .frames_played
                .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::SeqCst);
            let current_time = current_frame as f64 / self.sample_rate as f64;

            let scope = RenderScope {
                current_frame,
                current_time,
                sample_rate: self.sample_rate,
                event_sender: self.event_sender.clone(),
                node_id: Cell::new(AudioNodeId(0)), // placeholder value
                garbage_collector: self.garbage_collector.clone(),
            };

            // Render audio graph
            let graph = self.graph.as_mut().unwrap();

            // For x64 and aarch, process with denormal floats disabled (for performance, #194)
            #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
            let rendered = no_denormals::no_denormals(|| graph.render(&scope));
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            let rendered = graph.render(&scope);

            rendered.channels().iter().enumerate().for_each(
                |(channel_number, rendered_channel)| {
                    buffer.copy_to_channel_with_offset(
                        rendered_channel,
                        channel_number,
                        current_frame as usize,
                    );
                },
            );
        }

        buffer
    }

    pub fn render<S: FromSample<f32> + Clone>(&mut self, output_buffer: &mut [S]) {
        // Collect timing information
        let render_start = Instant::now();

        // Perform actual rendering
        assert_no_alloc::assert_no_alloc(|| {
            // For x64 and aarch, process with denormal floats disabled (for performance, #194)
            #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
            no_denormals::no_denormals(|| self.render_inner(output_buffer));
            #[cfg(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64"
            )))]
            self.render_inner(output_buffer);
        });

        // calculate load value and ship to control thread
        if let Some(load_value_sender) = &self.load_value_sender {
            let duration = render_start.elapsed().as_micros() as f64 / 1E6;
            let max_duration = RENDER_QUANTUM_SIZE as f64 / self.sample_rate as f64;
            let load_value = duration / max_duration;
            let render_timestamp =
                self.frames_played.load(Ordering::SeqCst) as f64 / self.sample_rate as f64;
            let load_value_data = AudioRenderCapacityLoad {
                render_timestamp,
                load_value,
            };
            let _ = load_value_sender.try_send(load_value_data);
        }
    }

    fn render_inner<S: FromSample<f32> + Clone>(&mut self, mut output_buffer: &mut [S]) {
        // There may be audio frames left over from the previous render call,
        // if the cpal buffer size did not align with our internal RENDER_QUANTUM_SIZE
        if let Some((offset, prev_rendered)) = self.buffer_offset.take() {
            let leftover_len = (RENDER_QUANTUM_SIZE - offset) * self.number_of_channels;
            // split the leftover frames slice, to fit in `buffer`
            let (first, next) = output_buffer.split_at_mut(leftover_len.min(output_buffer.len()));

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = first.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = prev_rendered.channel_data(i)[offset..].iter();
                for (sample, input) in output.zip(channel) {
                    let value = S::from_sample_(*input);
                    *sample = value;
                }
            }

            // exit early if we are done filling the buffer with the previously rendered data
            if next.is_empty() {
                self.buffer_offset = Some((
                    offset + first.len() / self.number_of_channels,
                    prev_rendered,
                ));
                return;
            }

            // if there's still space left in the buffer, continue rendering
            output_buffer = next;
        }

        // handle addition/removal of nodes/edges
        self.handle_control_messages();

        // if the thread is still booting, or shutting down, fill with silence
        if !self.graph.as_ref().is_some_and(Graph::is_active) {
            output_buffer.fill(S::from_sample_(0.));
            return;
        }

        // The audio graph is rendered in chunks of RENDER_QUANTUM_SIZE frames.  But some audio backends
        // may not be able to emit chunks of this size.
        let chunk_size = RENDER_QUANTUM_SIZE * self.number_of_channels;

        for data in output_buffer.chunks_mut(chunk_size) {
            // update time
            let current_frame = self
                .frames_played
                .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::SeqCst);
            let current_time = current_frame as f64 / self.sample_rate as f64;

            let scope = RenderScope {
                current_frame,
                current_time,
                sample_rate: self.sample_rate,
                event_sender: self.event_sender.clone(),
                node_id: Cell::new(AudioNodeId(0)), // placeholder value
                garbage_collector: self.garbage_collector.clone(),
            };

            // render audio graph, clone it in case we need to mutate/store the value later
            let mut destination_buffer = self.graph.as_mut().unwrap().render(&scope).clone();

            // online AudioContext allows channel count to be less than the number
            // of channels of the backend stream, i.e. number of channels of the
            // soundcard clamped to MAX_CHANNELS.
            if destination_buffer.number_of_channels() < self.number_of_channels {
                destination_buffer.mix(self.number_of_channels, ChannelInterpretation::Discrete);
            }

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = data.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = destination_buffer.channel_data(i).iter();
                for (sample, input) in output.zip(channel) {
                    let value = S::from_sample_(*input);
                    *sample = value;
                }
            }

            if data.len() != chunk_size {
                // this is the last chunk, and it contained less than RENDER_QUANTUM_SIZE samples
                let channel_offset = data.len() / self.number_of_channels;
                debug_assert!(channel_offset < RENDER_QUANTUM_SIZE);
                self.buffer_offset = Some((channel_offset, destination_buffer));
            }

            // handle addition/removal of nodes/edges
            self.handle_control_messages();
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        self.garbage_collector
            .deallocate_async(llq::Node::new(Box::new(
                super::TerminateGarbageCollectorThread,
            )));
        log::info!("Audio render thread has been dropped");
    }
}
