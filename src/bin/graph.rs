use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeIndex(usize);

#[derive(Debug)]
pub struct Connection {
    pub channel: usize,
}

#[derive(Debug)]
struct Node {
    processor: Box<dyn AudioNode>,
    buffer: Vec<f32>,
}

impl Node {
    fn process(&mut self, inputs: &[&[f32]], timestamp: f64, sample_rate: u32) {
        self.processor
            .process(inputs, &mut self.buffer, timestamp, sample_rate)
    }
}

#[derive(Debug)]
pub struct Graph {
    increment: usize,
    nodes: HashMap<NodeIndex, Node>,
    edges: HashMap<(NodeIndex, NodeIndex), Connection>,

    marked: Vec<NodeIndex>,
    ordered: Vec<NodeIndex>,
}

#[derive(Debug)]
pub struct OscillatorNode {
    frequency: AtomicU32,
}

#[derive(Debug)]
pub struct DestinationNode {
    pub channels: usize,
}

pub trait AudioNode: Debug {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], timestamp: f64, sample_rate: u32);
}

impl AudioNode for OscillatorNode {
    fn process(
        &mut self,
        _inputs: &[&[f32]],
        output: &mut [f32],
        timestamp: f64,
        sample_rate: u32,
    ) {
        let freq = self.frequency.load(Ordering::SeqCst) as f64;
        (0..output.len())
            .map(move |i| timestamp + i as f64 / sample_rate as f64)
            .map(move |t| (2. * PI * freq * t).sin() as f32)
            .zip(output.iter_mut())
            .for_each(|(value, dest)| *dest = value);
    }
}

impl AudioNode for DestinationNode {
    fn process(
        &mut self,
        inputs: &[&[f32]],
        output: &mut [f32],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        // clear slice, it may be re-used
        for d in output.iter_mut() {
            *d = 0.;
        }

        // mix signal from all child nodes, prevent allocations
        for input in inputs.iter() {
            let frames = output.chunks_mut(self.channels);
            for (frame, v) in frames.zip(input.iter()) {
                for sample in frame.iter_mut() {
                    *sample += v;
                }
            }
        }
    }
}

impl Graph {
    pub fn new<N: AudioNode + 'static>(root: N) -> Self {
        let mut graph = Graph {
            increment: 0,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            ordered: vec![NodeIndex(0)],
            marked: vec![NodeIndex(0)],
        };

        graph.add_node(root);

        graph
    }

    pub fn root(&self) -> NodeIndex {
        NodeIndex(0)
    }

    pub fn add_node<N: AudioNode + 'static>(&mut self, node: N) -> NodeIndex {
        let index = NodeIndex(self.increment);
        self.increment += 1;

        let processor = Box::new(node);
        // todo, size should be dependent on number of channels
        let buffer = vec![0.; 128];

        self.nodes.insert(index, Node { processor, buffer });

        index
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

    pub fn children_mut(
        &mut self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, &mut Connection)> {
        self.edges
            .iter_mut()
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

    pub fn ordered_nodes(&self) -> &[NodeIndex] {
        &self.ordered
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

    pub fn render(&mut self, data: &mut [f32]) {
        // split (mut) borrows
        let ordered = &self.ordered;
        let edges = &self.edges;
        let nodes = &mut self.nodes;

        ordered.iter().for_each(|index| {
            dbg!(("iterate ordered", index));

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
                .map(|(input_index, _connection)| {
                    dbg!(("has connected", input_index));
                    nodes.get(input_index).unwrap().buffer.as_slice()
                })
                .collect();

            node.process(&input_bufs, 0., 44100);

            // re-insert node in graph
            nodes.insert(*index, node);
        });

        // copy destination node's buffer into output
        // todo, prevent this memcpy with some buffer ref magic
        data.copy_from_slice(&nodes.get(&NodeIndex(0)).unwrap().buffer);
    }
}

pub fn main() {
    let dest = DestinationNode { channels: 2 };
    let mut graph = Graph::new(dest);

    let osc = OscillatorNode {
        frequency: AtomicU32::new(440),
    };
    let s = graph.add_node(osc);
    graph.add_edge(s, graph.root(), Connection { channel: 1 });

    dbg!(graph.ordered_nodes());

    // todo, we actually need a 256 size buffer
    let mut data = vec![0.; 128];
    graph.render(&mut data);
    dbg!(data);
}
