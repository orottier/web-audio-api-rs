use std::collections::HashMap;

use crate::node::{ChannelConfig, ChannelCountMode};
use crate::SampleRate;

use super::{Alloc, AudioParamValues, AudioProcessor, AudioRenderQuantum, NodeIndex};

use smallvec::{smallvec, SmallVec};

struct Edge {
    self_index: u32,
    other_id: NodeIndex,
    other_index: u32,
}

/// Renderer Node in the Audio Graph
pub struct Node {
    /// Renderer: converts inputs to outputs
    processor: Box<dyn AudioProcessor>,
    /// Input buffers
    inputs: Vec<AudioRenderQuantum>,
    /// Output buffers, consumed by subsequent Nodes in this graph
    outputs: Vec<AudioRenderQuantum>,
    /// Channel configuration: determines up/down-mixing of inputs
    channel_config: ChannelConfig,

    /// Outgoing edges: tuple of outcoming node reference, our output index and their input index
    outgoing_edges: SmallVec<[Edge; 2]>,
    /// Incoming edges: tuple of incoming node reference and their output index, and our input index
    incoming_edges: SmallVec<[Edge; 2]>,

    /// Indicates if the control thread has dropped this Node
    free_when_finished: bool,
}

impl Node {
    /// Render an audio quantum
    fn process(
        &mut self,
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        self.processor.process(
            &self.inputs[..],
            &mut self.outputs[..],
            params,
            timestamp,
            sample_rate,
        )
    }

    /// Determine if this node is done playing and can be removed from the audio graph
    fn can_free(&self, tail_time: bool) -> bool {
        // Only drop when the Control thread has dropped its handle.
        // Otherwise the node can be reconnected/restarted etc.
        if !self.free_when_finished {
            return false;
        }

        // Drop, if the node has no outputs connected
        if self.outgoing_edges.is_empty() {
            return true;
        }

        // Drop, when the node does not have any inputs connected,
        // and if the processor reports it won't yield output.
        let has_incoming = self
            .incoming_edges
            .iter()
            .any(|edge| edge.self_index != u32::MAX);
        if !has_incoming && !tail_time {
            return true;
        }

        false
    }

    /// Get the current buffer for AudioParam values
    pub fn get_buffer(&self) -> &AudioRenderQuantum {
        self.outputs.get(0).unwrap()
    }
}

pub(crate) struct Graph {
    // actual audio graph
    nodes: HashMap<NodeIndex, Node>,

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
        let inputs = vec![AudioRenderQuantum::new(self.alloc.silence()); inputs];
        let outputs = vec![AudioRenderQuantum::new(self.alloc.silence()); outputs];

        self.nodes.insert(
            index,
            Node {
                processor,
                inputs,
                outputs,
                channel_config,
                incoming_edges: smallvec![],
                outgoing_edges: smallvec![],
                free_when_finished: false,
            },
        );
    }

    pub fn add_edge(&mut self, source: (NodeIndex, u32), dest: (NodeIndex, u32)) {
        self.nodes
            .get_mut(&source.0)
            .unwrap()
            .outgoing_edges
            .push(Edge {
                self_index: source.1,
                other_id: dest.0,
                other_index: dest.1,
            });

        self.nodes
            .get_mut(&dest.0)
            .unwrap()
            .incoming_edges
            .push(Edge {
                self_index: dest.1,
                other_id: source.0,
                other_index: source.1,
            });

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edge(&mut self, source: NodeIndex, dest: NodeIndex) {
        self.nodes
            .get_mut(&source)
            .unwrap()
            .outgoing_edges
            .retain(|edge| edge.other_id != dest);

        self.nodes
            .get_mut(&dest)
            .unwrap()
            .incoming_edges
            .retain(|edge| edge.other_id != source);

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edges_from(&mut self, source: NodeIndex) {
        let node = self.nodes.get_mut(&source).unwrap();
        node.outgoing_edges.clear();
        node.incoming_edges.clear();

        self.nodes.values_mut().for_each(|node| {
            node.outgoing_edges.retain(|edge| edge.other_id != source);
            node.incoming_edges.retain(|edge| edge.other_id != source);
        });

        self.ordered.clear(); // void current ordering
    }

    pub fn mark_free_when_finished(&mut self, index: NodeIndex) {
        self.nodes.get_mut(&index).unwrap().free_when_finished = true;
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

        // visit nodes connecting to this one
        self.nodes
            .get(&n)
            .unwrap()
            .incoming_edges
            .iter()
            .for_each(|edge| self.visit(edge.other_id, marked, marked_temp, ordered, in_cycle));

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
                .for_each(AudioRenderQuantum::make_silent);
        }

        // depth first search yields reverse order
        ordered.reverse();

        // re-instate vecs
        self.ordered = ordered;
        self.marked = marked;
        self.marked_temp = marked_temp;
        self.in_cycle = in_cycle;
    }

    pub fn render(&mut self, timestamp: f64, sample_rate: SampleRate) -> &AudioRenderQuantum {
        // if the audio graph was changed, determine the new ordering
        if self.ordered.is_empty() {
            self.order_nodes();
        }

        // keep track of end-of-lifecyle nodes
        let mut nodes_dropped = false;

        // split (mut) borrows
        let ordered = &self.ordered;
        let nodes = &mut self.nodes;

        // process every node, in topological sorted order
        ordered.iter().for_each(|index| {
            // remove node from graph, re-insert later (for borrowck reasons)
            let mut node = nodes.remove(index).unwrap();

            // up/down-mix the cumulative inputs to the desired channel count
            let channel_interpretation = node.channel_config.interpretation();
            let mode = node.channel_config.count_mode();
            let count = node.channel_config.count();
            node.inputs.iter_mut().for_each(|input_buf| {
                let cur_channels = input_buf.number_of_channels();
                let new_channels = match mode {
                    ChannelCountMode::Max => cur_channels,
                    ChannelCountMode::Explicit => count,
                    ChannelCountMode::ClampedMax => cur_channels.min(count),
                };
                input_buf.mix(new_channels, channel_interpretation);
            });

            // let the current node process
            let params = AudioParamValues::from(&*nodes);
            let tail_time = node.process(params, timestamp, sample_rate);

            // iterate all outgoing edges, lookup these nodes and add to their input
            node.outgoing_edges
                .iter()
                // audio params are connected to the 'hidden' u32::MAX output, ignore them here
                .filter(|edge| edge.other_index != u32::MAX)
                .for_each(|edge| {
                    let output_node = nodes.get_mut(&edge.other_id).unwrap();
                    let signal = &node.outputs[edge.self_index as usize];
                    output_node.inputs[edge.other_index as usize]
                        .add(signal, channel_interpretation);
                });

            // audio graph cleanup of decomissioned nodes
            if node.can_free(tail_time) {
                node.incoming_edges.iter().for_each(|edge| {
                    nodes
                        .get_mut(&edge.other_id)
                        .unwrap()
                        .outgoing_edges
                        .retain(|e| e.other_id != *index)
                });
                node.outgoing_edges.iter().for_each(|edge| {
                    nodes
                        .get_mut(&edge.other_id)
                        .unwrap()
                        .incoming_edges
                        .retain(|e| e.other_id != *index)
                });
                nodes_dropped = true;
            } else {
                // reset input buffers
                node.inputs
                    .iter_mut()
                    .for_each(AudioRenderQuantum::make_silent);

                // re-insert node in graph
                nodes.insert(*index, node);
            }
        });

        if nodes_dropped {
            self.ordered.clear();
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
            _inputs: &[AudioRenderQuantum],
            _outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues,
            _timestamp: f64,
            _sample_rate: SampleRate,
        ) -> bool {
            false
        }
    }

    fn config() -> ChannelConfig {
        crate::node::ChannelConfigOptions {
            count: 2,
            mode: crate::node::ChannelCountMode::Explicit,
            interpretation: crate::node::ChannelInterpretation::Speakers,
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
