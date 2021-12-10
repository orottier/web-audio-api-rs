use crate::node::ChannelConfig;
use crate::param::AudioParamEvent;
use crate::renderer::AudioProcessor;

use crossbeam_channel::Sender;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    RegisterNode {
        id: u64,
        node: Box<dyn AudioProcessor>,
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

    AudioParamEvent {
        to: Sender<AudioParamEvent>,
        event: AudioParamEvent,
    },
}
