//! Message passing from control to render node

use crate::node::ChannelConfig;
use crate::param::AudioParamEvent;
use crate::render::AudioProcessor;

use crossbeam_channel::Sender;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    /// Register a new node in the audio graph
    RegisterNode {
        id: u64,
        node: Box<dyn AudioProcessor>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfig,
    },

    /// Connect a node to another in the audio graph
    ConnectNode {
        from: u64,
        to: u64,
        input: usize,
        output: usize,
    },

    /// Clear the connection between two given nodes in the audio graph
    DisconnectNode {
        from: u64,
        to: u64,
    },

    /// Disconnect this node from the audio graph (drop all its connections)
    DisconnectAll {
        from: u64,
    },

    /// Notify the render thread this node is dropped in the control thread
    FreeWhenFinished {
        id: u64,
    },

    /// Pass an AudioParam AutomationEvent to the relevant node
    AudioParamEvent {
        to: Sender<AudioParamEvent>,
        event: AudioParamEvent,
    },

    MarkCycleBreaker {
        id: u64,
    },
}
