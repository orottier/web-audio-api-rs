use crate::buffer::ChannelConfig;
use crate::process::AudioProcessor2;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    RegisterNode {
        id: u64,
        node: Box<dyn AudioProcessor2>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfig,
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

    FreeWhenFinished {
        id: u64,
    },
}
