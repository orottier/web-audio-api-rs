//! Communicates with the control thread and ships audio samples to the hardware

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};

use super::{AudioRenderQuantum, NodeIndex};
use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::events::EventEmitterMessage;
use crate::message::ControlMessage;
use crate::node::ChannelInterpretation;
use crate::render::RenderScope;
use crate::{AudioRenderCapacityLoad, RENDER_QUANTUM_SIZE};

use super::graph::Graph;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Option<Graph>,
    sample_rate: f32,
    number_of_channels: usize,
    frames_played: Arc<AtomicU64>,
    receiver: Option<Receiver<ControlMessage>>,
    buffer_offset: Option<(usize, AudioRenderQuantum)>,
    load_value_sender: Option<Sender<AudioRenderCapacityLoad>>,
    event_sender: Option<Sender<EventEmitterMessage>>,
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
        load_value_sender: Option<Sender<AudioRenderCapacityLoad>>,
        event_sender: Option<Sender<EventEmitterMessage>>,
    ) -> Self {
        Self {
            graph: None,
            sample_rate,
            number_of_channels,
            frames_played,
            receiver: Some(receiver),
            buffer_offset: None,
            load_value_sender,
            event_sender,
        }
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
                    id,
                    node,
                    inputs,
                    outputs,
                    channel_config,
                } => {
                    self.graph.as_mut().unwrap().add_node(
                        NodeIndex(id),
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
                        .add_edge((NodeIndex(from), output), (NodeIndex(to), input));
                }
                DisconnectNode { from, to } => {
                    self.graph
                        .as_mut()
                        .unwrap()
                        .remove_edge(NodeIndex(from), NodeIndex(to));
                }
                DisconnectAll { from } => {
                    self.graph
                        .as_mut()
                        .unwrap()
                        .remove_edges_from(NodeIndex(from));
                }
                FreeWhenFinished { id } => {
                    self.graph
                        .as_mut()
                        .unwrap()
                        .mark_free_when_finished(NodeIndex(id));
                }
                AudioParamEvent { to, event } => {
                    to.send(event).expect("Audioparam disappeared unexpectedly")
                }
                MarkCycleBreaker { id } => {
                    self.graph
                        .as_mut()
                        .unwrap()
                        .mark_cycle_breaker(NodeIndex(id));
                }
                Shutdown { sender } => {
                    let _ = sender.send(self.graph.take().unwrap());
                    self.receiver = None;
                    return; // no further handling of ctrl msgs
                }
                Startup { graph } => {
                    self.graph = Some(graph);
                }
            }
        }
    }

    // render method of the OfflineAudioContext
    pub fn render_audiobuffer(mut self, length: usize) -> AudioBuffer {
        // assert input was properly sized
        debug_assert_eq!(length % RENDER_QUANTUM_SIZE, 0);

        let options = AudioBufferOptions {
            number_of_channels: self.number_of_channels,
            length: 0,
            sample_rate: self.sample_rate,
        };

        let mut buf = AudioBuffer::new(options);

        for _ in 0..length / RENDER_QUANTUM_SIZE {
            // handle addition/removal of nodes/edges
            self.handle_control_messages();

            // update time
            let current_frame = self
                .frames_played
                .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::SeqCst);
            let current_time = current_frame as f64 / self.sample_rate as f64;

            let scope = RenderScope {
                current_frame,
                current_time,
                sample_rate: self.sample_rate,
            };

            // render audio graph
            let event_sender = self.event_sender.as_ref(); // None
            let rendered = self.graph.as_mut().unwrap().render(&scope, &event_sender);

            buf.extend_alloc(&rendered);
        }

        buf
    }

    pub fn render<S: crate::Sample>(&mut self, buffer: &mut [S]) {
        // collect timing information
        let render_start = Instant::now();

        // perform actual rendering
        self.render_inner(buffer);

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

    fn render_inner<S: crate::Sample>(&mut self, mut buffer: &mut [S]) {
        // There may be audio frames left over from the previous render call,
        // if the cpal buffer size did not align with our internal RENDER_QUANTUM_SIZE
        if let Some((offset, prev_rendered)) = self.buffer_offset.take() {
            let leftover_len = (RENDER_QUANTUM_SIZE - offset) * self.number_of_channels;
            // split the leftover frames slice, to fit in `buffer`
            let (first, next) = buffer.split_at_mut(leftover_len.min(buffer.len()));

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = first.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = prev_rendered.channel_data(i)[offset..].iter();
                for (sample, input) in output.zip(channel) {
                    let value = crate::Sample::from::<f32>(input);
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
            buffer = next;
        }

        // handle addition/removal of nodes/edges
        self.handle_control_messages();

        // if the thread is still booting, or shutting down, fill with silence
        if self.graph.is_none() {
            buffer.fill(crate::Sample::from(&0.));
            return;
        }

        // The audio graph is rendered in chunks of RENDER_QUANTUM_SIZE frames.  But some audio backends
        // may not be able to emit chunks of this size.
        let chunk_size = RENDER_QUANTUM_SIZE * self.number_of_channels;

        for data in buffer.chunks_mut(chunk_size) {
            // update time
            let current_frame = self
                .frames_played
                .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::SeqCst);
            let current_time = current_frame as f64 / self.sample_rate as f64;

            let scope = RenderScope {
                current_frame,
                current_time,
                sample_rate: self.sample_rate,
            };

            // render audio graph
            let event_sender = self.event_sender.as_ref();
            let mut rendered = self.graph.as_mut().unwrap().render(&scope, &event_sender);

            // online AudioContext allows channel count to be less than no of hardware channels
            if rendered.number_of_channels() != self.number_of_channels {
                rendered.mix(self.number_of_channels, ChannelInterpretation::Discrete);
            }

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = data.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = rendered.channel_data(i).iter();
                for (sample, input) in output.zip(channel) {
                    let value = crate::Sample::from::<f32>(input);
                    *sample = value;
                }
            }

            if data.len() != chunk_size {
                // this is the last chunk, and it contained less than RENDER_QUANTUM_SIZE samples
                let channel_offset = data.len() / self.number_of_channels;
                debug_assert!(channel_offset < RENDER_QUANTUM_SIZE);
                self.buffer_offset = Some((channel_offset, rendered));
            }

            // handle addition/removal of nodes/edges
            self.handle_control_messages();
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        log::info!("Audio render thread has been dropped");
    }
}
