use crate::buffer::{AudioBuffer, ChannelConfig, ChannelConfigOptions, Resampler};
use crate::context::{AsBaseAudioContext, AudioContextRegistration};
use crate::control::{Controller, Scheduler};
use crate::media::MediaElement;
use crate::{SampleRate, BUFFER_SIZE};

use super::{
    AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode, MediaStreamRenderer,
};

/// Options for constructing a AudioBufferSourceNode
#[derive(Default)]
pub struct AudioBufferSourceNodeOptions {
    pub buffer: Option<AudioBuffer>,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from an in-memory audio asset in an AudioBuffer
///
/// Note: do not forget to `start()` the node.
pub struct AudioBufferSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl AudioScheduledSourceNode for AudioBufferSourceNode {
    fn scheduler(&self) -> &Scheduler {
        self.controller.scheduler()
    }
}
impl AudioControllableSourceNode for AudioBufferSourceNode {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl AudioNode for AudioBufferSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl AudioBufferSourceNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AudioBufferSourceNodeOptions) -> Self {
        context.base().register(move |registration| {
            // unwrap_or_default buffer
            let buffer = options
                .buffer
                .unwrap_or_else(|| AudioBuffer::new(1, BUFFER_SIZE, SampleRate(44_100)));

            // wrap input in resampler
            let resampler = Resampler::new(
                context.base().sample_rate(),
                BUFFER_SIZE,
                std::iter::once(Ok(buffer)),
            );

            // wrap resampler in media-element (for loop/play/pause)
            let media = MediaElement::new(resampler);
            let controller = media.controller().clone();
            let scheduler = controller.scheduler().clone();

            // setup user facing audio node
            let node = AudioBufferSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let render = MediaStreamRenderer::new(media, scheduler);

            (node, Box::new(render))
        })
    }
}
