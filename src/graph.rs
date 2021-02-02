use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::mpsc::Receiver;

use crate::control::ControlMessage;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Graph,
    sample_rate: u32,
    channels: usize,
    timestamp: f64,
    receiver: Receiver<ControlMessage>,
}

impl RenderThread {
    pub fn new<N: Render + 'static>(
        root: N,
        sample_rate: u32,
        channels: usize,
        receiver: Receiver<ControlMessage>,
    ) -> Self {
        Self {
            graph: Graph::new(root),
            sample_rate,
            channels,
            timestamp: 0.,
            receiver,
        }
    }

    fn handle_control_messages(&mut self) {
        for msg in self.receiver.try_iter() {
            match msg {
                ControlMessage::RegisterNode { id, node, buffer } => {
                    self.graph.add_node(NodeIndex(id), node, buffer);
                }
                ControlMessage::ConnectNode { from, to, channel } => {
                    let conn = Connection { channel };
                    self.graph.add_edge(NodeIndex(from), NodeIndex(to), conn);
                }
            }
        }
    }

    pub fn render(&mut self, data: &mut [f32]) {
        self.handle_control_messages();

        // render audio data
        let len = data.len() / self.channels;
        let rendered = self.graph.render(self.timestamp, self.sample_rate, len);

        for (frame, value) in data.chunks_mut(self.channels).zip(rendered) {
            let value = cpal::Sample::from::<f32>(value);

            // for now, make stereo sound from mono
            for sample in frame.iter_mut() {
                *sample = value;
            }
        }

        // update time
        self.timestamp += len as f64 / self.sample_rate as f64;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeIndex(u64);

#[derive(Debug)]
pub struct Connection {
    pub channel: usize,
}

#[derive(Debug)]
struct Node {
    processor: Box<dyn Render>,
    buffer: Vec<f32>,
}

impl Node {
    fn process(&mut self, inputs: &[&[f32]], timestamp: f64, sample_rate: u32, len: usize) {
        self.processor
            .process(inputs, &mut self.buffer[0..len], timestamp, sample_rate)
    }
}

#[derive(Debug)]
pub(crate) struct Graph {
    nodes: HashMap<NodeIndex, Node>,
    edges: HashMap<(NodeIndex, NodeIndex), Connection>,

    marked: Vec<NodeIndex>,
    ordered: Vec<NodeIndex>,
}

pub trait Render: Debug + Send {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], timestamp: f64, sample_rate: u32);
}

impl Graph {
    pub fn new<N: Render + 'static>(root: N) -> Self {
        let root_index = NodeIndex(0);

        let mut graph = Graph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            ordered: vec![root_index],
            marked: vec![root_index],
        };

        // todo, size should be dependent on number of channels
        let buffer = vec![0.; crate::BUFFER_SIZE as usize];
        graph.add_node(root_index, Box::new(root), buffer);

        graph
    }

    pub fn add_node(&mut self, index: NodeIndex, processor: Box<dyn Render>, buffer: Vec<f32>) {
        self.nodes.insert(index, Node { processor, buffer });
    }

    pub fn add_edge(&mut self, source: NodeIndex, dest: NodeIndex, data: Connection) {
        self.edges.insert((source, dest), data);

        self.order_nodes();
    }

    pub fn children(&self, node: NodeIndex) -> impl Iterator<Item = (NodeIndex, &Connection)> {
        self.edges
            .iter()
            .filter(move |(&(_s, d), _e)| d == node)
            .map(|(&(s, _d), e)| (s, e))
    }

    fn visit(&self, n: NodeIndex, marked: &mut Vec<NodeIndex>, ordered: &mut Vec<NodeIndex>) {
        if marked.contains(&n) {
            return;
        }
        marked.push(n);
        self.children(n)
            .for_each(|c| self.visit(c.0, marked, ordered));
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

    pub fn render(&mut self, timestamp: f64, sample_rate: u32, len: usize) -> &[f32] {
        // split (mut) borrows
        let ordered = &self.ordered;
        let edges = &self.edges;
        let nodes = &mut self.nodes;

        ordered.iter().for_each(|index| {
            // remove node from map, re-insert later (for borrowck reasons)
            let mut node = nodes.remove(index).unwrap();

            let input_bufs: Vec<_> = edges
                .iter()
                .filter_map(
                    move |((s, d), e)| {
                        if d == index {
                            Some((s, e))
                        } else {
                            None
                        }
                    },
                )
                .map(|(input_index, _connection)| &nodes.get(input_index).unwrap().buffer[0..len])
                .collect();

            node.process(&input_bufs, timestamp, sample_rate, len);

            // re-insert node in graph
            nodes.insert(*index, node);
        });

        // return buffer of destination node
        &nodes.get(&NodeIndex(0)).unwrap().buffer[0..len]
    }
}
