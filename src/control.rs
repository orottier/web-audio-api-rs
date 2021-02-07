use crate::buffer::AudioBuffer;
use crate::graph::Render;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    RegisterNode {
        id: u64,
        node: Box<dyn Render>,
        inputs: usize,
        outputs: usize,
        buffers: Vec<AudioBuffer>,
    },

    ConnectNode {
        from: u64,
        to: u64,
        input: u32,
        output: u32,
    },

    DisconnectNode {
        from: u64,
        to: u64,
    },

    DisconnectAll {
        from: u64,
    },
}
