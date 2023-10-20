//! The audio graph topology and render algorithm
use std::any::Any;
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};

use crate::context::AudioNodeId;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};

use super::{Alloc, AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::node::ChannelConfig;
use crate::render::RenderScope;

/// Connection between two audio nodes
struct OutgoingEdge {
    /// index of the current Nodes output port
    self_index: usize,
    /// reference to the other Node
    other_id: AudioNodeId,
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
    /// Indicates if the node can act as a cycle breaker (only DelayNode for now)
    cycle_breaker: bool,
}

impl Node {
    /// Render an audio quantum
    fn process(&mut self, params: AudioParamValues<'_>, scope: &RenderScope) -> bool {
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
    nodes: FxHashMap<AudioNodeId, RefCell<Node>>,
    /// Allocator for audio buffers
    alloc: Alloc,

    /// Topological ordering of the nodes
    ordered: Vec<AudioNodeId>,
    /// Topological sorting helper
    marked: Vec<AudioNodeId>,
    /// Topological sorting helper
    marked_temp: Vec<AudioNodeId>,
    /// Topological sorting helper
    in_cycle: Vec<AudioNodeId>,
    /// Topological sorting helper
    cycle_breakers: Vec<AudioNodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: FxHashMap::default(),
            ordered: vec![],
            marked: vec![],
            marked_temp: vec![],
            in_cycle: vec![],
            cycle_breakers: vec![],
            alloc: Alloc::with_capacity(64),
        }
    }

    /// Check if the graph is fully initialized and can start rendering
    pub fn is_active(&self) -> bool {
        // currently we only require the destination node to be present
        !self.nodes.is_empty()
    }

    pub fn add_node(
        &mut self,
        index: AudioNodeId,
        processor: Box<dyn AudioProcessor>,
        number_of_inputs: usize,
        number_of_outputs: usize,
        channel_config: ChannelConfig,
    ) {
        // todo: pre-allocate the buffers on the control thread

        // set input and output buffers to single channel of silence, will be upmixed when
        // necessary
        let inputs = vec![AudioRenderQuantum::from(self.alloc.silence()); number_of_inputs];
        let outputs = vec![AudioRenderQuantum::from(self.alloc.silence()); number_of_outputs];

        self.nodes.insert(
            index,
            RefCell::new(Node {
                processor,
                inputs,
                outputs,
                channel_config,
                outgoing_edges: smallvec![],
                free_when_finished: false,
                has_inputs_connected: false,
                cycle_breaker: false,
            }),
        );
    }

    pub fn add_edge(&mut self, source: (AudioNodeId, usize), dest: (AudioNodeId, usize)) {
        self.nodes
            .get_mut(&source.0)
            .unwrap_or_else(|| panic!("cannot connect {:?} to {:?}", source, dest))
            .get_mut()
            .outgoing_edges
            .push(OutgoingEdge {
                self_index: source.1,
                other_id: dest.0,
                other_index: dest.1,
            });

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edge(&mut self, source: AudioNodeId, dest: AudioNodeId) {
        self.nodes
            .get_mut(&source)
            .unwrap_or_else(|| panic!("cannot remove the edge from {:?} to {:?}", source, dest))
            .get_mut()
            .outgoing_edges
            .retain(|edge| edge.other_id != dest);

        self.ordered.clear(); // void current ordering
    }

    pub fn remove_edges_from(&mut self, source: AudioNodeId) {
        self.nodes
            .get_mut(&source)
            .unwrap_or_else(|| panic!("cannot remove edges from {:?}", source))
            .get_mut()
            .outgoing_edges
            .clear();

        self.nodes.values_mut().for_each(|node| {
            node.get_mut()
                .outgoing_edges
                .retain(|edge| edge.other_id != source);
        });

        self.ordered.clear(); // void current ordering
    }

    pub fn mark_free_when_finished(&mut self, index: AudioNodeId) {
        // Issue #92, a race condition can occur for AudioParams. They may have already been
        // removed from the audio graph if the node they feed into was dropped.
        // Therefore, do not assume this node still exists:
        if let Some(node) = self.nodes.get_mut(&index) {
            node.get_mut().free_when_finished = true;
        }
    }

    pub fn mark_cycle_breaker(&mut self, index: AudioNodeId) {
        self.nodes.get_mut(&index).unwrap().get_mut().cycle_breaker = true;
    }

    pub fn route_message(&mut self, index: AudioNodeId, msg: &mut dyn Any) {
        self.nodes
            .get_mut(&index)
            .unwrap()
            .get_mut()
            .processor
            .onmessage(msg);
    }

    /// Helper function for `order_nodes` - traverse node and outgoing edges
    ///
    /// The return value indicates `cycle_breaker_applied`:
    /// - true: a cycle was found and a cycle breaker was applied, current ordering is invalidated
    /// - false: visiting this leg was successful and no topological changes were applied
    fn visit(
        &self,
        node_id: AudioNodeId,
        marked: &mut Vec<AudioNodeId>,
        marked_temp: &mut Vec<AudioNodeId>,
        ordered: &mut Vec<AudioNodeId>,
        in_cycle: &mut Vec<AudioNodeId>,
        cycle_breakers: &mut Vec<AudioNodeId>,
    ) -> bool {
        // If this node is in the cycle detection list, it is part of a cycle!
        if let Some(pos) = marked_temp.iter().position(|&m| m == node_id) {
            // check if we can find some node that can break the cycle
            let cycle_breaker_node = marked_temp
                .iter()
                .skip(pos)
                .find(|node_id| self.nodes.get(node_id).unwrap().borrow().cycle_breaker);

            match cycle_breaker_node {
                Some(&node_id) => {
                    // store node id to clear the node outgoing edges
                    cycle_breakers.push(node_id);

                    return true;
                }
                None => {
                    // Mark all nodes in the cycle
                    in_cycle.extend_from_slice(&marked_temp[pos..]);
                    // Do not continue, as we already have visited all these nodes
                    return false;
                }
            }
        }

        // Do not visit nodes multiple times
        if marked.contains(&node_id) {
            return false;
        }

        // Add node to the visited list
        marked.push(node_id);
        // Add node to the current cycle detection list
        marked_temp.push(node_id);

        // Visit outgoing nodes, and call `visit` on them recursively
        for edge in self
            .nodes
            .get(&node_id)
            .unwrap()
            .borrow()
            .outgoing_edges
            .iter()
        {
            let cycle_breaker_applied = self.visit(
                edge.other_id,
                marked,
                marked_temp,
                ordered,
                in_cycle,
                cycle_breakers,
            );
            if cycle_breaker_applied {
                return true;
            }
        }

        // Then add this node to the ordered list
        ordered.push(node_id);

        // Finished visiting all nodes in this leg, clear the current cycle detection list
        marked_temp.retain(|marked| *marked != node_id);

        false
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
    /// - Break cycles when possible (if there is a DelayNode present)
    /// - Mute nodes that are still in a cycle
    /// - For performance: no new allocations (reuse Vecs)
    fn order_nodes(&mut self) {
        // For borrowck reasons, we need the `visit` call to be &self.
        // So move out the bookkeeping Vecs, and pass them around as &mut.
        let mut ordered = std::mem::take(&mut self.ordered);
        let mut marked = std::mem::take(&mut self.marked);
        let mut marked_temp = std::mem::take(&mut self.marked_temp);
        let mut in_cycle = std::mem::take(&mut self.in_cycle);
        let mut cycle_breakers = std::mem::take(&mut self.cycle_breakers);

        // When a cycle breaker is applied, the graph topology changes and we need to run the
        // ordering again
        loop {
            // Clear previous administration
            ordered.clear();
            marked.clear();
            marked_temp.clear();
            in_cycle.clear();
            cycle_breakers.clear();

            // Visit all registered nodes, and perform a depth first traversal.
            //
            // We cannot just start from the AudioDestinationNode and visit all nodes connecting to it,
            // since the audio graph could contain legs detached from the destination and those should
            // still be rendered.
            let mut cycle_breaker_applied = false;
            for &node_id in self.nodes.keys() {
                cycle_breaker_applied = self.visit(
                    node_id,
                    &mut marked,
                    &mut marked_temp,
                    &mut ordered,
                    &mut in_cycle,
                    &mut cycle_breakers,
                );

                if cycle_breaker_applied {
                    break;
                }
            }

            if cycle_breaker_applied {
                // clear the outgoing edges of the nodes that have been recognized as cycle breaker
                let nodes = &mut self.nodes;
                cycle_breakers.iter().for_each(|node_id| {
                    let node = nodes.get_mut(node_id).unwrap();
                    node.get_mut().outgoing_edges.clear();
                });

                continue;
            }

            break;
        }

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
        self.cycle_breakers = cycle_breakers;
    }

    /// Render a single audio quantum by traversing the node list
    pub fn render(&mut self, scope: &RenderScope) -> &AudioRenderQuantum {
        // if the audio graph was changed, determine the new ordering
        if self.ordered.is_empty() {
            self.order_nodes();
        }

        // keep track of end-of-lifecyle nodes
        let mut nodes_dropped = false;

        // for borrow-checker reasons, move mutable borrow of nodes out of self
        let nodes = &mut self.nodes;

        // process every node, in topological sorted order
        self.ordered.iter().for_each(|index| {
            // acquire a mutable borrow of the current processing node
            let mut node = nodes.get(index).unwrap().borrow_mut();

            // make sure all input buffers have the correct number of channels, this might not be
            // the case if the node has no inputs connected or the channel count has just changed
            let interpretation = node.channel_config.interpretation();
            let count = node.channel_config.count();
            node.inputs
                .iter_mut()
                .for_each(|i| i.mix(count, interpretation));

            // let the current node process (catch any panics that may occur)
            let params = AudioParamValues::from(&*nodes);
            scope.node_id.set(*index);
            let (success, tail_time) = {
                // We are abusing AssertUnwindSafe here, we cannot guarantee it upholds.
                // This may lead to logic bugs later on, but it is the best that we can do.
                // The alternative is to crash and reboot the render thread.
                let catch_me = AssertUnwindSafe(|| node.process(params, scope));
                match panic::catch_unwind(catch_me) {
                    Ok(tail_time) => (true, tail_time),
                    Err(e) => {
                        node.outgoing_edges.clear();
                        scope.report_error(e);
                        (false, false)
                    }
                }
            };

            // iterate all outgoing edges, lookup these nodes and add to their input
            node.outgoing_edges
                .iter()
                // audio params are connected to the 'hidden' usize::MAX output, ignore them here
                .filter(|edge| edge.other_index != usize::MAX)
                .for_each(|edge| {
                    let mut output_node = nodes.get(&edge.other_id).unwrap().borrow_mut();
                    output_node.has_inputs_connected = true;
                    let signal = &node.outputs[edge.self_index];
                    let channel_config = &output_node.channel_config.clone();

                    output_node.inputs[edge.other_index].add(signal, channel_config);
                });

            let can_free = !success || node.can_free(tail_time);

            // Node is not dropped.
            if !can_free {
                // Reset input buffers as they will be summed up in the next render quantum.
                node.inputs
                    .iter_mut()
                    .for_each(AudioRenderQuantum::make_silent);

                // Reset input state
                node.has_inputs_connected = false;
            }

            drop(node); // release borrow of self.nodes

            // Check if we can decommission this node (end of life)
            if can_free {
                // Node is dropped, remove it from the node list
                nodes.remove(index);

                // And remove it from the ordering after we have processed all nodes
                nodes_dropped = true;

                // Nodes are only dropped when they do not have incoming connections.
                // But they may have AudioParams feeding into them, these can de dropped too.
                nodes.retain(|id, n| {
                    // Check if this node was connected to the dropped node. In that case, it is
                    // either an AudioParam (which can be dropped), or the AudioListener that feeds
                    // into a PannerNode (which can be disconnected).
                    let outgoing_edges = &mut n.borrow_mut().outgoing_edges;
                    let prev_len = outgoing_edges.len();
                    outgoing_edges.retain(|e| e.other_id != *index);
                    let was_connected = outgoing_edges.len() != prev_len;

                    let special = id.0 < 2; // never drop Listener and Destination node
                    special || !was_connected
                });
            }
        });

        // If there were any nodes decommissioned, remove from graph order
        if nodes_dropped {
            let mut i = 0;
            while i < self.ordered.len() {
                if !nodes.contains_key(&self.ordered[i]) {
                    self.ordered.remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Return the output buffer of destination node
        &self
            .nodes
            .get_mut(&AudioNodeId(0))
            .unwrap()
            .get_mut()
            .outputs[0]
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
            _params: AudioParamValues<'_>,
            _scope: &RenderScope,
        ) -> bool {
            false
        }
    }

    fn config() -> ChannelConfig {
        crate::node::ChannelConfigOptions {
            count: 2,
            count_mode: crate::node::ChannelCountMode::Explicit,
            interpretation: crate::node::ChannelInterpretation::Speakers,
        }
        .into()
    }

    #[test]
    fn test_add_remove() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(AudioNodeId(0), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(1), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(2), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(3), node, 1, 1, config());

        graph.add_edge((AudioNodeId(1), 0), (AudioNodeId(0), 0));
        graph.add_edge((AudioNodeId(2), 0), (AudioNodeId(1), 0));
        graph.add_edge((AudioNodeId(3), 0), (AudioNodeId(0), 0));

        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        assert_eq!(graph.ordered[3], AudioNodeId(0)); // root node comes last

        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2

        // Detach node 1 (and thus node 2) from the root node
        graph.remove_edge(AudioNodeId(1), AudioNodeId(0));
        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2
    }

    #[test]
    fn test_remove_all() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(AudioNodeId(0), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(1), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(2), node, 1, 1, config());

        // link 1->0, 1->2 and 2->0
        graph.add_edge((AudioNodeId(1), 0), (AudioNodeId(0), 0));
        graph.add_edge((AudioNodeId(1), 0), (AudioNodeId(2), 0));
        graph.add_edge((AudioNodeId(2), 0), (AudioNodeId(0), 0));

        graph.order_nodes();

        assert_eq!(
            graph.ordered,
            vec![AudioNodeId(1), AudioNodeId(2), AudioNodeId(0)]
        );

        graph.remove_edges_from(AudioNodeId(1));
        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 3); // all nodes present
        let pos0 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(0))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(2))
            .unwrap();
        assert!(pos2 < pos0); // node 1 depends on node 0
    }

    #[test]
    fn test_cycle() {
        let mut graph = Graph::new();

        let node = Box::new(TestNode {});
        graph.add_node(AudioNodeId(0), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(1), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(2), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(3), node.clone(), 1, 1, config());
        graph.add_node(AudioNodeId(4), node, 1, 1, config());

        // link 4->2, 2->1, 1->0, 1->2, 3->0
        graph.add_edge((AudioNodeId(4), 0), (AudioNodeId(2), 0));
        graph.add_edge((AudioNodeId(2), 0), (AudioNodeId(1), 0));
        graph.add_edge((AudioNodeId(1), 0), (AudioNodeId(0), 0));
        graph.add_edge((AudioNodeId(1), 0), (AudioNodeId(2), 0));
        graph.add_edge((AudioNodeId(3), 0), (AudioNodeId(0), 0));

        graph.order_nodes();

        let pos0 = graph.ordered.iter().position(|&n| n == AudioNodeId(0));
        let pos1 = graph.ordered.iter().position(|&n| n == AudioNodeId(1));
        let pos2 = graph.ordered.iter().position(|&n| n == AudioNodeId(2));
        let pos3 = graph.ordered.iter().position(|&n| n == AudioNodeId(3));
        let pos4 = graph.ordered.iter().position(|&n| n == AudioNodeId(4));

        // cycle 1<>2 should be removed
        assert_eq!(pos1, None);
        assert_eq!(pos2, None);
        // detached leg from cycle will still be rendered
        assert!(pos4.is_some());
        // a-cyclic part should be present
        assert!(pos3.unwrap() < pos0.unwrap());
    }
}
