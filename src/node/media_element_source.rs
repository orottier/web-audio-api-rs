use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::media::{MediaElement, Resampler};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, MediaStreamRenderer};

/// Options for constructing a [`MediaElementAudioSourceNode`]
pub struct MediaElementAudioSourceOptions<'a> {
    pub media_element: &'a mut MediaElement,
}

pub struct MediaElementAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaElementAudioSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        0
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl MediaElementAudioSourceNode {
    pub fn new<C: BaseAudioContext>(
        context: &C,
        options: MediaElementAudioSourceOptions<'_>,
    ) -> Self {
        context.register(move |registration| {
            let node = MediaElementAudioSourceNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            let stream = options
                .media_element
                .take_stream()
                .expect("stream already taken");

            let resampler = Resampler::new(context.sample_rate(), RENDER_QUANTUM_SIZE, stream);

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}
