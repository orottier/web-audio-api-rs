use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;

use crate::alloc::{Alloc, AudioBuffer};
use crate::buffer::{ChannelConfig, ChannelCountMode};
use crate::message::ControlMessage;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::SampleRate;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Graph,
    sample_rate: SampleRate,
    channels: usize,
    frames_played: AtomicU64,
    receiver: Receiver<ControlMessage>,
}

// todo, write safe Send wrapper for fresh initialized RenderThread
unsafe impl Send for RenderThread {}
//unsafe impl Sync for RenderThread { }

impl RenderThread {
    pub fn new(
        sample_rate: SampleRate,
        channels: usize,
        receiver: Receiver<ControlMessage>,
    ) -> Self {
        Self {
            graph: Graph::new(),
            sample_rate,
            channels,
            frames_played: AtomicU64::new(0),
            receiver,
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
            }
        }
    }

    pub fn render(&mut self, data: &mut [f32]) {
        // handle addition/removal of nodes/edges
        self.handle_control_messages();

        // update time
        let len = data.len() / self.channels as usize;
        let timestamp = self.frames_played.fetch_add(len as u64, Ordering::SeqCst) as f64
            / self.sample_rate.0 as f64;

        // render audio graph
        let rendered = self.graph.render(timestamp, self.sample_rate);

        // copy rendered audio into output slice
        for i in 0..self.channels {
            let output = data.iter_mut().skip(i).step_by(self.channels);
            let channel = rendered.channel_data(i).iter();
            for (sample, input) in output.zip(channel) {
                let value = cpal::Sample::from::<f32>(input);
                *sample = value;
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeIndex(pub u64);

/// Renderer Node in the Audio Graph
pub struct Node {
    /// Renderer: converts inputs to outputs
    processor: Box<dyn AudioProcessor>,
    /// Output buffers, consumed by subsequent Nodes in this graph
    buffers: Vec<AudioBuffer>,
    /// Number of inputs for the processor
    inputs: usize,
    /// Channel configuration: determines up/down-mixing of inputs
    channel_config: ChannelConfig,

    // lifecycle management flags:
    /// Indicates if the control thread has dropped this Node
    free_when_finished: bool,
    /// Indicates if the Node has had any inputs in the current render quantum
    has_inputs_connected: bool,
    /// Indicates if the output of this Node was consumed in the current render quantum
    has_outputs_connected: bool,
}

impl Node {
    /// Render an audio quantum
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) {
        self.processor.process(
            inputs,
            &mut self.buffers[..],
            params,
            timestamp,
            sample_rate,
        )
    }

    /// Determine if this node is done playing and can be removed from the audio graph
    fn can_free(&self) -> bool {
        // Only drop when the Control thread has dropped its handle.
        // Otherwise the node can be reconnected/restarted etc.
        if !self.free_when_finished {
            return false;
        }

        // Drop, if the node has no outputs connected
        if !self.has_outputs_connected {
            return true;
        }

        // Drop, when the node does not have any inputs connected,
        // and if the processor reports it won't yield output.
        if !self.has_inputs_connected && !self.processor.tail_time() {
            return true;
        }

        false
    }

    /// Get the current buffer for AudioParam values
    pub fn get_buffer(&self) -> &AudioBuffer {
        self.buffers.get(0).unwrap()
    }
}

pub(crate) struct Graph {
    nodes: HashMap<NodeIndex, Node>,

    // connections, from (node,output) to (node,input)
    edges: HashSet<((NodeIndex, u32), (NodeIndex, u32))>,

    marked: Vec<NodeIndex>,
    ordered: Vec<NodeIndex>,

    alloc: Alloc,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: HashSet::new(),
            ordered: vec![],
            marked: vec![],
            alloc: Alloc::with_capacity(64),
        }
    }

    pub fn add_node(
        &mut self,
        index: NodeIndex,
        processor: Box<dyn AudioProcessor>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfig,
    ) {
        let buffers = vec![AudioBuffer::new(self.alloc.allocate()); outputs];
        self.nodes.insert(
            index,
            Node {
                processor,
                buffers,
                inputs,
                channel_config,
                free_when_finished: false,
                has_inputs_connected: true,
                has_outputs_connected: true,
            },
        );
    }

    pub fn add_edge(&mut self, source: (NodeIndex, u32), dest: (NodeIndex, u32)) {
        self.edges.insert((source, dest));

        self.order_nodes();
    }

    pub fn remove_edge(&mut self, source: NodeIndex, dest: NodeIndex) {
        self.edges.retain(|&(s, d)| s.0 != source || d.0 != dest);

        self.order_nodes();
    }

    pub fn remove_edges_from(&mut self, source: NodeIndex) {
        self.edges.retain(|&(s, _d)| s.0 != source);

        self.order_nodes();
    }

    fn mark_free_when_finished(&mut self, index: NodeIndex) {
        self.nodes.get_mut(&index).unwrap().free_when_finished = true;
    }

    pub fn children(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.edges
            .iter()
            .filter(move |&(_s, d)| d.0 == node)
            .map(|&(s, _d)| s.0)
    }

    fn visit(&self, n: NodeIndex, marked: &mut Vec<NodeIndex>, ordered: &mut Vec<NodeIndex>) {
        if marked.contains(&n) {
            return;
        }
        marked.push(n);
        self.children(n)
            .for_each(|c| self.visit(c, marked, ordered));
        ordered.insert(0, n);
    }

    fn order_nodes(&mut self) {
        // empty ordered_nodes, and temporarily move out of self (no allocs)
        let mut ordered = std::mem::replace(&mut self.ordered, vec![]);
        ordered.resize(self.nodes.len(), NodeIndex(0));
        ordered.clear();

        // empty marked_nodes, and temporarily move out of self (no allocs)
        let mut marked = std::mem::replace(&mut self.marked, vec![]);
        marked.resize(self.nodes.len(), NodeIndex(0));
        marked.clear();

        // start by visiting the root node
        let start = NodeIndex(0);
        self.visit(start, &mut marked, &mut ordered);

        ordered.reverse();

        // re-instate vecs to prevent new allocs
        self.ordered = ordered;
        self.marked = marked;
    }

    pub fn render(&mut self, timestamp: f64, sample_rate: SampleRate) -> &AudioBuffer {
        // split (mut) borrows
        let ordered = &self.ordered;
        let edges = &self.edges;
        let silence = self.alloc.silence();
        let nodes = &mut self.nodes;

        // we will drop audio nodes if they are finished running
        let mut drop_nodes = vec![];

        ordered.iter().for_each(|index| {
            // remove node from map, re-insert later (for borrowck reasons)
            let mut node = nodes.remove(index).unwrap();
            // for lifecycle management, check if any inputs are present
            let mut has_inputs_connected = false;
            // mix all inputs together
            let mut input_bufs = vec![AudioBuffer::new(silence.clone()); node.inputs];
            edges
                .iter()
                .filter_map(move |(s, d)| {
                    // audio params are connected to the 'hidden' u32::MAX input, ignore them
                    if d.0 == *index && d.1 != u32::MAX {
                        Some((s, d.1))
                    } else {
                        None
                    }
                })
                .for_each(|(&(node_index, output), input)| {
                    let node = nodes.get(&node_index).unwrap();
                    let signal = &node.buffers[output as usize];

                    input_bufs[input as usize] = input_bufs[input as usize]
                        .add(signal, node.channel_config.interpretation());

                    has_inputs_connected = true;
                });

            let input_bufs: Vec<_> = input_bufs
                .iter_mut()
                .map(|input_buf| {
                    // up/down-mix to the desired channel count
                    let cur_channels = input_buf.number_of_channels();
                    let new_channels = match node.channel_config.count_mode() {
                        ChannelCountMode::Max => cur_channels,
                        ChannelCountMode::Explicit => node.channel_config.count(),
                        ChannelCountMode::ClampedMax => {
                            cur_channels.min(node.channel_config.count())
                        }
                    };
                    input_buf.mix(new_channels, node.channel_config.interpretation());

                    // return immutable refs
                    &*input_buf
                })
                .collect();

            let params = AudioParamValues::from(&*nodes);
            node.process(&input_bufs[..], params, timestamp, sample_rate);

            // check if the Node has reached end of lifecycle
            node.has_inputs_connected = has_inputs_connected;
            if node.can_free() {
                drop_nodes.push(*index);
            }

            // re-insert node in graph
            nodes.insert(*index, node);
        });

        for index in drop_nodes {
            self.remove_edges_from(index);
            self.nodes.remove(&index);
        }

        // return buffer of destination node
        // assume only 1 output (todo)
        &self.nodes.get(&NodeIndex(0)).unwrap().buffers[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestNode {}

    impl AudioProcessor for TestNode {
        fn process(
            &mut self,
            _: &[&AudioBuffer2],
            _: &mut [AudioBuffer2],
            _: AudioParamValues,
            _: f64,
            _: SampleRate,
        ) {
        }
        fn tail_time(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_add_remove() {
        let config: ChannelConfig = crate::buffer::ChannelConfigOptions {
            count: 2,
            mode: crate::buffer::ChannelCountMode::Explicit,
            interpretation: crate::buffer::ChannelInterpretation::Speakers,
        }
        .into();
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(NodeIndex(1), node.clone(), 1, 1, config.clone());
        graph.add_node(NodeIndex(2), node.clone(), 1, 1, config.clone());
        graph.add_node(NodeIndex(3), node.clone(), 1, 1, config.clone());

        graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(1), 0));
        graph.add_edge((NodeIndex(3), 0), (NodeIndex(0), 0));

        // sorting is not deterministic, can be either of these two
        if graph.ordered != &[NodeIndex(3), NodeIndex(2), NodeIndex(1), NodeIndex(0)] {
            assert_eq!(
                graph.ordered,
                vec![NodeIndex(2), NodeIndex(1), NodeIndex(3), NodeIndex(0)]
            );
        }

        graph.remove_edge(NodeIndex(1), NodeIndex(0));
        assert_eq!(graph.ordered, vec![NodeIndex(3), NodeIndex(0)]);
    }

    #[test]
    fn test_remove_all() {
        let config: ChannelConfig = crate::buffer::ChannelConfigOptions {
            count: 2,
            mode: crate::buffer::ChannelCountMode::Explicit,
            interpretation: crate::buffer::ChannelInterpretation::Speakers,
        }
        .into();
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(NodeIndex(1), node.clone(), 1, 1, config.clone());
        graph.add_node(NodeIndex(2), node.clone(), 1, 1, config.clone());

        graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(1), 0));

        assert_eq!(
            graph.ordered,
            vec![NodeIndex(2), NodeIndex(1), NodeIndex(0)]
        );

        graph.remove_edges_from(NodeIndex(2));

        assert_eq!(graph.ordered, vec![NodeIndex(1), NodeIndex(0)]);
    }
}
