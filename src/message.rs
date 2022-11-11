//! Message passing from control to render node

use crate::node::ChannelConfig;
use crate::param::AudioParamEvent;
use crate::render::graph::Graph;
use crate::render::AudioProcessor;

use crate::context::AudioNodeId;
use crossbeam_channel::Sender;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    /// Register a new node in the audio graph
    RegisterNode {
        id: AudioNodeId,
        node: Box<dyn AudioProcessor>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfig,
    },

    /// Connect a node to another in the audio graph
    ConnectNode {
        from: AudioNodeId,
        to: AudioNodeId,
        input: usize,
        output: usize,
    },

    /// Clear the connection between two given nodes in the audio graph
    DisconnectNode { from: AudioNodeId, to: AudioNodeId },

    /// Disconnect this node from the audio graph (drop all its connections)
    DisconnectAll { from: AudioNodeId },

    /// Notify the render thread this node is dropped in the control thread
    FreeWhenFinished { id: AudioNodeId },

    /// Pass an AudioParam AutomationEvent to the relevant node
    AudioParamEvent {
        to: Sender<AudioParamEvent>,
        event: AudioParamEvent,
    },

    /// Mark node as a cycle breaker (DelayNode only)
    MarkCycleBreaker { id: AudioNodeId },

    /// Shut down and recycle the audio graph
    Shutdown { sender: Sender<Graph> },

    /// Start rendering with given audio graph
    Startup { graph: Graph },
}
