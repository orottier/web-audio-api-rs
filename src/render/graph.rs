//! The audio graph topology and render algorithm

use std::collections::HashMap;

use super::{Alloc, AudioParamValues, AudioProcessor, AudioRenderQuantum, NodeIndex};
use crate::node::{ChannelConfig, ChannelCountMode};
use crate::render::RenderScope;

use smallvec::{smallvec, SmallVec};

/// Connection between two audio nodes
struct OutgoingEdge {
    /// index of the current Nodes output port
    self_index: usize,
    /// reference to the other Node
    other_id: NodeIndex,
    /// index of the other Nodes input port
    other_index: usize,
}

/// Renderer Node in the Audio Graph
pub struct Node {
    /// Renderer: converts inputs to outputs
    processor: Box<dyn AudioProcessor>,
    /// Reusable input buffers
    inputs: Vec<AudioRenderQuantum>,
    /// Reusable output buffers, consumed by subsequent Nodes in this graph
    outputs: Vec<AudioRenderQuantum>,
    /// Channel configuration: determines up/down-mixing of inputs
    channel_config: ChannelConfig,
    /// Outgoing edges: tuple of outcoming node reference, our output index and their input index
    outgoing_edges: SmallVec<[OutgoingEdge; 2]>,
    /// Indicates if the control thread has dropped this Node
    free_when_finished: bool,
    /// Indicates if the node has any incoming connections (for lifecycle management)
    has_inputs_connected: bool,
}

impl Node {
    /// Render an audio quantum
    fn process(&mut self, params: AudioParamValues, scope: &RenderScope) -> bool {
        self.processor
            .process(&self.inputs[..], &mut self.outputs[..], params, scope)
    }

    /// Determine if this node is done playing and can be removed from the audio graph
    fn can_free(&self, tail_time: bool) -> bool {
        // Only drop when the Control thread has dropped its handle.
        // Otherwise the node can be reconnected/restarted etc.
        if !self.free_when_finished {
            return false;
        }

        // Drop, when the node does not have any inputs connected,
        // and if the processor reports it won't yield output.
        if !self.has_inputs_connected && !tail_time {
            return true;
        }

        // Otherwise, do not drop the node.
        // (Even if it has no outputs connected, it may have side effects)
        false
    }

    /// Get the current buffer for AudioParam values
    pub fn get_buffer(&self) -> &AudioRenderQuantum {
        self.outputs.get(0).unwrap()
    }
}

/// The audio graph
pub(crate) struct Graph {
    /// Processing Nodes
    nodes: HashMap<NodeIndex, Node>,
    /// Allocator for audio buffers
    alloc: Alloc,

    /// Topological ordering of the nodes
    ordered: Vec<NodeIndex>,
    /// Topological sorting helper
    marked: Vec<NodeIndex>,
    /// Topological sorting helper
    marked_temp: Vec<NodeIndex>,
    /// Topological sorting helper
    in_cycle: Vec<NodeIndex>,
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
                outgoing_edges: smallvec![],
                free_when_finished: false,
                has_inputs_connected: false,
            },
        );
    }

    pub fn add_edge(&mut self, source: (NodeIndex, usize), dest: (NodeIndex, usize)) {
        self.nodes
            .get_mut(&source.0)
            .unwrap_or_else(|| panic!("cannot connect {:?} to {:?}", source, dest))
            .outgoing_edges
            .push(OutgoingEdge {
                self_index: source.1,
                other_id: dest.0,
                other_index: dest.1,
            });

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edge(&mut self, source: NodeIndex, dest: NodeIndex) {
        self.nodes
            .get_mut(&source)
            .unwrap_or_else(|| panic!("cannot remove the edge from {:?} to {:?}", source, dest))
            .outgoing_edges
            .retain(|edge| edge.other_id != dest);

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edges_from(&mut self, source: NodeIndex) {
        let node = self
            .nodes
            .get_mut(&source)
            .unwrap_or_else(|| panic!("cannot remove edges from {:?}", source));
        node.outgoing_edges.clear();

        self.nodes.values_mut().for_each(|node| {
            node.outgoing_edges.retain(|edge| edge.other_id != source);
        });

        self.ordered.clear(); // void current ordering
    }

    pub fn mark_free_when_finished(&mut self, index: NodeIndex) {
        // Issue #92, a race condition can occur for AudioParams. They may have already been
        // removed from the audio graph if the node they feed into was dropped.
        // Therefore, do not assume this node still exists:
        if let Some(node) = self.nodes.get_mut(&index) {
            node.free_when_finished = true;
        }
    }

    /// Helper function for `order_nodes` - traverse node and outgoing edges
    fn visit(
        &self,
        node_id: NodeIndex,
        marked: &mut Vec<NodeIndex>,
        marked_temp: &mut Vec<NodeIndex>,
        ordered: &mut Vec<NodeIndex>,
        in_cycle: &mut Vec<NodeIndex>,
    ) {
        // If this node is in the cycle detection list, it is part of a cycle!
        if let Some(pos) = marked_temp.iter().position(|&m| m == node_id) {
            // Mark all nodes in the cycle
            in_cycle.extend_from_slice(&marked_temp[pos..]);
            // Do not continue, as we already have visited all these nodes
            return;
        }

        // Do not visit nodes multiple times
        if marked.contains(&node_id) {
            return;
        }

        // Add node to the visited list
        marked.push(node_id);
        // Add node to the current cycle detection list
        marked_temp.push(node_id);

        // Visit outgoing nodes, and call `visit` on them recursively
        self.nodes
            .get(&node_id)
            .unwrap()
            .outgoing_edges
            .iter()
            .for_each(|edge| self.visit(edge.other_id, marked, marked_temp, ordered, in_cycle));

        // Then add this node to the ordered list
        ordered.push(node_id);

        // Finished visiting all nodes in this leg, clear the current cycle detection list
        marked_temp.retain(|marked| *marked != node_id);
    }

    /// Determine the order of the audio nodes for rendering
    ///
    /// By inspecting the audio node connections, we can determine which nodes should render before
    /// other nodes. For example, in a graph with an audio source, a gain node and the destination
    /// node, at every render quantum the source should render first and after that the gain node.
    ///
    /// Inspired by the spec recommendation at
    /// <https://webaudio.github.io/web-audio-api/#rendering-loop>
    ///
    /// The goals are:
    /// - Perform a topological sort of the graph
    /// - Mute nodes that are in a cycle
    /// - For performance: no new allocations (reuse Vecs)
    fn order_nodes(&mut self) {
        // For borrowck reasons, we need the `visit` call to be &self.
        // So move out the bookkeeping Vecs, and pass them around as &mut.
        let mut ordered = std::mem::take(&mut self.ordered);
        let mut marked = std::mem::take(&mut self.marked);
        let mut marked_temp = std::mem::take(&mut self.marked_temp);
        let mut in_cycle = std::mem::take(&mut self.in_cycle);

        // Clear previous administration
        ordered.clear();
        marked.clear();
        marked_temp.clear();
        in_cycle.clear();

        // Visit all registered nodes, and perform a depth first traversal.
        //
        // We cannot just start from the AudioDestinationNode and visit all nodes connecting to it,
        // since the audio graph could contain legs detached from the destination and those should
        // still be rendered.
        self.nodes.keys().for_each(|&node_id| {
            self.visit(
                node_id,
                &mut marked,
                &mut marked_temp,
                &mut ordered,
                &mut in_cycle,
            );
        });

        // Remove nodes from the ordering if they are part of a cycle. The spec mandates that their
        // outputs should be silenced, but with our rendering algorithm that is not necessary.
        // `retain` leaves the ordering in place
        ordered.retain(|o| !in_cycle.contains(o));

        // The `visit` function adds child nodes before their parent, so reverse the order
        ordered.reverse();

        // Re-instate Vecs
        self.ordered = ordered;
        self.marked = marked;
        self.marked_temp = marked_temp;
        self.in_cycle = in_cycle;
    }

    /// Render a single audio quantum by traversing the node list
    pub fn render(&mut self, scope: &RenderScope) -> &AudioRenderQuantum {
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
            let tail_time = node.process(params, scope);

            // iterate all outgoing edges, lookup these nodes and add to their input
            node.outgoing_edges
                .iter()
                // audio params are connected to the 'hidden' usize::MAX output, ignore them here
                .filter(|edge| edge.other_index != usize::MAX)
                .for_each(|edge| {
                    let output_node = nodes.get_mut(&edge.other_id).unwrap();
                    output_node.has_inputs_connected = true;
                    let signal = &node.outputs[edge.self_index];
                    output_node.inputs[edge.other_index].add(signal, channel_interpretation);
                });

            // Check if we can decommission this node (end of life)
            if node.can_free(tail_time) {
                // Node is dropped, we should perform a new topological sort of the audio graph
                nodes_dropped = true;

                // Nodes are only dropped when they do not have incoming connections.
                // But they may have AudioParams feeding into them, these can de dropped too.
                nodes.retain(|_id, n| !n.outgoing_edges.iter().any(|e| e.other_id == *index));
            } else {
                // Node is not dropped.
                // Reset input buffers as they will be summed up in the next render quantum.
                node.inputs
                    .iter_mut()
                    .for_each(AudioRenderQuantum::make_silent);

                // Reset input state
                node.has_inputs_connected = false;

                // Re-insert node in graph
                nodes.insert(*index, node);
            }
        });

        // If there were any nodes decomissioned, clear current graph order
        if nodes_dropped {
            self.ordered.clear();
        }

        // Return the output buffer of destination node
        &self.nodes.get(&NodeIndex(0)).unwrap().outputs[0]
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[derive(Debug, Clone)]
    struct TestNode {}

    impl AudioProcessor for TestNode {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            _outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues,
            _scope: &RenderScope,
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

    #[derive(Debug, Clone)]
    struct SpeakerNode {}

    impl AudioProcessor for SpeakerNode {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues,
            _scope: &RenderScope,
        ) -> bool {
            let output = &mut outputs[0];
            output.set_number_of_channels(2);
            output.modify_channels(|channel|
                channel.iter_mut().for_each(|value| *value = 1.)
            );
            true
        }
    }

    #[derive(Debug, Clone)]
    struct DiscreteNode {}

    impl AudioProcessor for DiscreteNode {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues,
            _scope: &RenderScope,
        ) -> bool {
            let output = &mut outputs[0];
            output.set_number_of_channels(1);
            output.modify_channels(|channel|
                channel.iter_mut().for_each(|value| *value = 1.)
            );
            true
        }
    }

    #[derive(Debug, Clone)]
    struct OutputNode {}

    impl AudioProcessor for OutputNode {
        fn process(
            &mut self,
            inputs: &[AudioRenderQuantum],
            outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues,
            _scope: &RenderScope,
        ) -> bool {
            let input = &inputs[0];
            let output = &mut outputs[0];
            *output = input.clone();
            true
        }
    }

    #[test]
    fn test_connection_mixing() {
        // run mutliple time as the issue should appear depending of graph ordering
        // - if discrete is run before speaker we should have two channels of [2.; 128]
        //   which is expected
        // - if discrete is run after speaker we should have [2.; 128] and [1.; 128]
        //   which is wrong
        for _ in 0..10 {
            let mut graph = Graph::new();

            let speaker_node = Box::new(SpeakerNode {});
            graph.add_node(NodeIndex(1), speaker_node, 1, 1, crate::node::ChannelConfigOptions {
                count: 2,
                mode: crate::node::ChannelCountMode::Explicit,
                interpretation: crate::node::ChannelInterpretation::Speakers,
            }.into());

            // this one should be present in both `output_node` channels, because it
            // should be mixed in input according to `output_node` channel config, i.e. Speaker
            let discrete_node = Box::new(DiscreteNode {});
            graph.add_node(NodeIndex(2), discrete_node, 1, 1, crate::node::ChannelConfigOptions {
                count: 1,
                mode: crate::node::ChannelCountMode::Explicit,
                interpretation: crate::node::ChannelInterpretation::Discrete,
            }.into());

            let output_node = Box::new(OutputNode {});
            graph.add_node(NodeIndex(0), output_node, 1, 1, crate::node::ChannelConfigOptions {
                count: 2,
                mode: crate::node::ChannelCountMode::Explicit,
                interpretation: crate::node::ChannelInterpretation::Speakers,
            }.into());

            // connect both nodes to output
            graph.add_edge((NodeIndex(1), 0), (NodeIndex(0), 0));
            graph.add_edge((NodeIndex(2), 0), (NodeIndex(0), 0));

            // connect both nodes to output
            let output = graph.render(&RenderScope {
                current_frame: 0,
                current_time: 0.,
                sample_rate: crate::SampleRate(1),
            });

            // println!("----------------------------------------------------");
            // println!("{:?}", &output.channel_data(0)[..]);
            // println!("{:?}", &output.channel_data(1)[..]);

            assert_float_eq!(&output.channel_data(0)[..], &[2.; 128][..], abs_all <= 0.);
            assert_float_eq!(&output.channel_data(1)[..], &[2.; 128][..], abs_all <= 0.);
        }
    }
}
