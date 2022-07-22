//! Communicates with the control thread and ships audio samples to the hardware

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;

use super::{AudioRenderQuantum, NodeIndex};
use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::message::ControlMessage;
use crate::node::ChannelInterpretation;
use crate::render::RenderScope;
use crate::{AtomicF64, RENDER_QUANTUM_SIZE};

use super::graph::Graph;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Graph,
    sample_rate: f32,
    number_of_channels: usize,
    frames_played: Arc<AtomicU64>,
    output_latency: Arc<AtomicF64>,
    receiver: Receiver<ControlMessage>,
    buffer_offset: Option<(usize, AudioRenderQuantum)>,
}

// SAFETY:
// The RenderThread is not Send/Sync since it contains `AudioRenderQuantum`s (which use Rc), but
// these are only accessed within the same thread (the render thread). Due to the cpal constraints
// we can neither move the RenderThread object into the render thread, nor can we initialize the
// Rc's in that thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for RenderThread {}
unsafe impl Sync for RenderThread {}

impl RenderThread {
    pub fn new(
        sample_rate: f32,
        number_of_channels: usize,
        receiver: Receiver<ControlMessage>,
        frames_played: Arc<AtomicU64>,
        output_latency: Arc<AtomicF64>,
    ) -> Self {
        Self {
            graph: Graph::new(),
            sample_rate,
            number_of_channels,
            frames_played,
            output_latency,
            receiver,
            buffer_offset: None,
        }
    }

    fn handle_control_messages(&mut self) {
        for msg in self.receiver.try_iter() {
            use ControlMessage::*;

            match msg {
                RegisterNode {
                    id,
                    node,
                    inputs,
                    outputs,
                    channel_config,
                } => {
                    self.graph
                        .add_node(NodeIndex(id), node, inputs, outputs, channel_config);
                }
                ConnectNode {
                    from,
                    to,
                    output,
                    input,
                } => {
                    self.graph
                        .add_edge((NodeIndex(from), output), (NodeIndex(to), input));
                }
                DisconnectNode { from, to } => {
                    self.graph.remove_edge(NodeIndex(from), NodeIndex(to));
                }
                DisconnectAll { from } => {
                    self.graph.remove_edges_from(NodeIndex(from));
                }
                FreeWhenFinished { id } => {
                    self.graph.mark_free_when_finished(NodeIndex(id));
                }
                AudioParamEvent { to, event } => {
                    to.send(event).expect("Audioparam disappeared unexpectedly")
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
            let rendered = self.graph.render(&scope);

            buf.extend_alloc(rendered);
        }

        buf
    }

    // This code is not dead: false positive from clippy
    // due to the use of #[cfg(not(test))]
    #[allow(dead_code)]
    pub fn render<S: crate::Sample>(&mut self, mut buffer: &mut [S], output_latency: f64) {
        // update output latency, this value might change while running (e.g. sound card heat)
        self.output_latency.store(output_latency);

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

        // The audio graph is rendered in chunks of RENDER_QUANTUM_SIZE frames.  But some audio backends
        // may not be able to emit chunks of this size.
        let chunk_size = RENDER_QUANTUM_SIZE * self.number_of_channels;

        for data in buffer.chunks_mut(chunk_size) {
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
            let mut rendered = self.graph.render(&scope).clone();

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
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        log::info!("Audio render thread has been dropped");
    }
}
