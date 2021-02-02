use crate::graph::Render;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    RegisterNode {
        id: u64,
        node: Box<dyn Render>,
        buffer: Vec<f32>,
    },

    ConnectNode {
        from: u64,
        to: u64,
        channel: usize,
    },
}
