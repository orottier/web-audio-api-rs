use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cpal::Sample;
use crossbeam_channel::Receiver;

use crate::alloc::{Alloc, AudioBuffer};
use crate::buffer::{AudioBufferOptions, ChannelConfig, ChannelCountMode};
use crate::message::ControlMessage;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::BUFFER_SIZE;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Graph,
    sample_rate: f32,
    channels: usize,
    frames_played: Arc<AtomicU64>,
    receiver: Receiver<ControlMessage>,
    buffer_offset: Option<(usize, AudioBuffer)>,
}

// SAFETY:
// The RenderThread is not Send since it contains AudioBuffers (which use Rc), but these are only
// accessed within the same thread (the render thread). Due to the cpal constraints we can neither
// move the RenderThread object into the render thread, nor can we initialize the Rc's in that
// thread.
unsafe impl Send for RenderThread {}

impl RenderThread {
    pub fn new(
        sample_rate: f32,
        channels: usize,
        receiver: Receiver<ControlMessage>,
        frames_played: Arc<AtomicU64>,
    ) -> Self {
        Self {
            graph: Graph::new(),
            sample_rate,
            channels,
            frames_played,
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

    pub fn render_audiobuffer(&mut self, length: usize) -> crate::buffer::AudioBuffer {
        // assert input was properly sized
        debug_assert_eq!(length % BUFFER_SIZE as usize, 0);

        let options = AudioBufferOptions {
            number_of_channels: Some(self.channels),
            length: 0,
            sample_rate: self.sample_rate,
        };
        let mut buf = crate::buffer::AudioBuffer::new(options);

        for _ in 0..length / BUFFER_SIZE as usize {
            // handle addition/removal of nodes/edges
            self.handle_control_messages();

            // update time
            let timestamp = self
                .frames_played
                .fetch_add(BUFFER_SIZE as u64, Ordering::SeqCst) as f64
                / self.sample_rate as f64;

            // render audio graph
            let rendered = self.graph.render(timestamp, self.sample_rate);

            buf.extend_alloc(rendered);
        }

        buf
    }

    // This code is not dead: false positive from clippy
    // due to the use of #[cfg(not(test))]
    #[allow(dead_code)]
    pub fn render<S: Sample>(&mut self, mut buffer: &mut [S]) {
        // There may be audio frames left over from the previous render call,
        // if the cpal buffer size did not align with our internal BUFFER_SIZE
        if let Some((offset, prev_rendered)) = self.buffer_offset.take() {
            let leftover_len = (BUFFER_SIZE as usize - offset) * self.channels;
            // split the leftover frames slice, to fit in `buffer`
            let (first, next) = buffer.split_at_mut(leftover_len.min(buffer.len()));

            // copy rendered audio into output slice
            for i in 0..self.channels {
                let output = first.iter_mut().skip(i).step_by(self.channels);
                let channel = prev_rendered.channel_data(i)[offset..].iter();
                for (sample, input) in output.zip(channel) {
                    let value = Sample::from::<f32>(input);
                    *sample = value;
                }
            }

            // exit early if we are done filling the buffer with the previously rendered data
            if next.is_empty() {
                self.buffer_offset = Some((offset + first.len() / self.channels, prev_rendered));
                return;
            }

            // if there's still space left in the buffer, continue rendering
            buffer = next;
        }

        // The audio graph is rendered in chunks of BUFFER_SIZE frames.  But some audio backends
        // may not be able to emit chunks of this size.
        let chunk_size = BUFFER_SIZE as usize * self.channels as usize;

        for data in buffer.chunks_mut(chunk_size) {
            // handle addition/removal of nodes/edges
            self.handle_control_messages();

            // update time
            let timestamp = self
                .frames_played
                .fetch_add(BUFFER_SIZE as u64, Ordering::SeqCst) as f64
                / self.sample_rate as f64;

            // render audio graph
            let rendered = self.graph.render(timestamp, self.sample_rate);

            // copy rendered audio into output slice
            for i in 0..self.channels {
                let output = data.iter_mut().skip(i).step_by(self.channels);
                let channel = rendered.channel_data(i).iter();
                for (sample, input) in output.zip(channel) {
                    let value = Sample::from::<f32>(input);
                    *sample = value;
                }
            }

            if data.len() != chunk_size {
                // this is the last chunk, and it contained less than BUFFER_SIZE samples
                let channel_offset = data.len() / self.channels;
                debug_assert!(channel_offset < BUFFER_SIZE as usize);
                self.buffer_offset = Some((channel_offset, rendered.clone()));
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
    /// Input buffers
    inputs: Vec<AudioBuffer>,
    /// Output buffers, consumed by subsequent Nodes in this graph
    outputs: Vec<AudioBuffer>,
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
    fn process(&mut self, params: AudioParamValues, timestamp: f64, sample_rate: f32) {
        self.processor.process(
            &self.inputs[..],
            &mut self.outputs[..],
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
        self.outputs.get(0).unwrap()
    }
}

pub(crate) struct Graph {
    // actual audio graph
    nodes: HashMap<NodeIndex, Node>,
    edges: HashSet<((NodeIndex, u32), (NodeIndex, u32))>, // (node,output) to (node,input)

    // topological sorting
    marked: Vec<NodeIndex>,
    marked_temp: Vec<NodeIndex>,
    ordered: Vec<NodeIndex>,
    in_cycle: Vec<NodeIndex>,

    // allocator for audio buffers
    alloc: Alloc,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: HashSet::new(),
            ordered: vec![],
            marked: vec![],
            marked_temp: vec![],
            in_cycle: vec![],
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
        // todo, allocate on control thread, make single alloc..?
        let inputs = vec![AudioBuffer::new(self.alloc.silence()); inputs];
        let outputs = vec![AudioBuffer::new(self.alloc.silence()); outputs];

        self.nodes.insert(
            index,
            Node {
                processor,
                inputs,
                outputs,
                channel_config,
                free_when_finished: false,
                has_inputs_connected: true,
                has_outputs_connected: true,
            },
        );
    }

    pub fn add_edge(&mut self, source: (NodeIndex, u32), dest: (NodeIndex, u32)) {
        self.edges.insert((source, dest));
        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edge(&mut self, source: NodeIndex, dest: NodeIndex) {
        self.edges.retain(|&(s, d)| s.0 != source || d.0 != dest);
        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edges_from(&mut self, source: NodeIndex) {
        self.edges.retain(|&(s, _d)| s.0 != source);
        self.ordered.clear(); // void current ordering
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

    /// Traverse node for topological sort
    fn visit(
        &self,
        n: NodeIndex,
        marked: &mut Vec<NodeIndex>,
        marked_temp: &mut Vec<NodeIndex>,
        ordered: &mut Vec<NodeIndex>,
        in_cycle: &mut Vec<NodeIndex>,
    ) {
        // detect cycles
        if let Some(pos) = marked_temp.iter().position(|&m| m == n) {
            in_cycle.extend_from_slice(&marked_temp[pos..]);
            return;
        }
        if marked.contains(&n) {
            return;
        }

        marked.push(n);
        marked_temp.push(n);

        self.children(n)
            .for_each(|c| self.visit(c, marked, marked_temp, ordered, in_cycle));

        marked_temp.retain(|marked| *marked != n);
        ordered.insert(0, n);
    }

    /// Perform a topological sort of the graph. Mute nodes that are in a cycle
    fn order_nodes(&mut self) {
        // For borrowck reasons, we need the `visit` call to be &self.
        // So move out the bookkeeping Vecs, and pass them around as &mut.
        let mut ordered = std::mem::take(&mut self.ordered);
        let mut marked = std::mem::take(&mut self.marked);
        let mut marked_temp = std::mem::take(&mut self.marked_temp);
        let mut in_cycle = std::mem::take(&mut self.in_cycle);

        // clear previous administration
        ordered.clear();
        marked.clear();
        marked_temp.clear();
        in_cycle.clear();

        // visit all registered nodes, depth first search
        self.nodes.keys().for_each(|&i| {
            self.visit(
                i,
                &mut marked,
                &mut marked_temp,
                &mut ordered,
                &mut in_cycle,
            );
        });

        // remove cycles from ordered nodes, leaving the ordering in place
        ordered.retain(|o| !in_cycle.contains(o));

        // mute the nodes inside cycles by clearing their output
        for key in in_cycle.iter() {
            self.nodes
                .get_mut(key)
                .unwrap()
                .outputs
                .iter_mut()
                .for_each(AudioBuffer::make_silent);
        }

        // depth first search yields reverse order
        ordered.reverse();

        // re-instate vecs
        self.ordered = ordered;
        self.marked = marked;
        self.marked_temp = marked_temp;
        self.in_cycle = in_cycle;
    }

    pub fn render(&mut self, timestamp: f64, sample_rate: f32) -> &AudioBuffer {
        if self.ordered.is_empty() {
            self.order_nodes();
        }

        // split (mut) borrows
        let ordered = &self.ordered;
        let edges = &self.edges;
        let nodes = &mut self.nodes;

        // we will drop audio nodes if they are finished running
        let mut drop_nodes = vec![];

        ordered.iter().for_each(|index| {
            // remove node from map, re-insert later (for borrowck reasons)
            let mut node = nodes.remove(index).unwrap();
            // for lifecycle management, check if any inputs are present
            let mut has_inputs_connected = false;
            // mix all inputs together
            node.inputs.iter_mut().for_each(|i| i.make_silent());

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
                    let input_node = nodes.get(&node_index).unwrap();
                    let signal = &input_node.outputs[output as usize];

                    node.inputs[input as usize].add(signal, node.channel_config.interpretation());

                    has_inputs_connected = true;
                });

            // up/down-mix to the desired channel count
            let mode = node.channel_config.count_mode();
            let count = node.channel_config.count();
            let interpretation = node.channel_config.interpretation();
            node.inputs.iter_mut().for_each(|input_buf| {
                let cur_channels = input_buf.number_of_channels();
                let new_channels = match mode {
                    ChannelCountMode::Max => cur_channels,
                    ChannelCountMode::Explicit => count,
                    ChannelCountMode::ClampedMax => cur_channels.min(count),
                };
                input_buf.mix(new_channels, interpretation);
            });

            let params = AudioParamValues::from(&*nodes);
            node.process(params, timestamp, sample_rate);

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
        &self.nodes.get(&NodeIndex(0)).unwrap().outputs[0]
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
            _inputs: &[AudioBuffer],
            _outputs: &mut [AudioBuffer],
            _params: AudioParamValues,
            _timestamp: f64,
            _sample_rate: f32,
        ) {
        }
        fn tail_time(&self) -> bool {
            false
        }
    }

    fn config() -> ChannelConfig {
        crate::buffer::ChannelConfigOptions {
            count: 2,
            mode: crate::buffer::ChannelCountMode::Explicit,
            interpretation: crate::buffer::ChannelInterpretation::Speakers,
        }
        .into()
    }

    #[test]
    fn test_add_remove() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(NodeIndex(0), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(1), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(2), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(3), node, 1, 1, config());

        graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(1), 0));
        graph.add_edge((NodeIndex(3), 0), (NodeIndex(0), 0));

        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        assert_eq!(graph.ordered[3], NodeIndex(0)); // root node comes last

        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2

        // Detach node 1 (and thus node 2) from the root node
        graph.remove_edge(NodeIndex(1), NodeIndex(0));
        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2
    }

    #[test]
    fn test_remove_all() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(NodeIndex(0), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(1), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(2), node, 1, 1, config());

        // link 1->0, 1->2 and 2->0
        graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(1), 0), (NodeIndex(2), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(0), 0));

        graph.order_nodes();

        assert_eq!(
            graph.ordered,
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(0)]
        );

        graph.remove_edges_from(NodeIndex(1));
        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 3); // all nodes present
        let pos0 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(0))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == NodeIndex(2))
            .unwrap();
        assert!(pos2 < pos0); // node 1 depends on node 0
    }

    #[test]
    fn test_cycle() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(NodeIndex(0), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(1), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(2), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(3), node.clone(), 1, 1, config());
        graph.add_node(NodeIndex(4), node, 1, 1, config());

        // link 4->2, 2->1, 1->0, 1->2, 3->0
        graph.add_edge((NodeIndex(4), 0), (NodeIndex(2), 0));
        graph.add_edge((NodeIndex(2), 0), (NodeIndex(1), 0));
        graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
        graph.add_edge((NodeIndex(1), 0), (NodeIndex(2), 0));
        graph.add_edge((NodeIndex(3), 0), (NodeIndex(0), 0));

        graph.order_nodes();

        let pos0 = graph.ordered.iter().position(|&n| n == NodeIndex(0));
        let pos1 = graph.ordered.iter().position(|&n| n == NodeIndex(1));
        let pos2 = graph.ordered.iter().position(|&n| n == NodeIndex(2));
        let pos3 = graph.ordered.iter().position(|&n| n == NodeIndex(3));
        let pos4 = graph.ordered.iter().position(|&n| n == NodeIndex(4));

        // cycle 1<>2 should be removed
        assert_eq!(pos1, None);
        assert_eq!(pos2, None);
        // detached leg from cycle will still be renderd
        assert!(pos4.is_some());
        // a-cyclic part should be present
        assert!(pos3.unwrap() < pos0.unwrap());
    }
}
