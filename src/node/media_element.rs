use crate::buffer::Resampler;
use crate::context::{AudioContextRegistration, BaseAudioContext};
// use crate::control::{Controller, Scheduler};
use crate::media::MediaElement;
use crate::RENDER_QUANTUM_SIZE;

use super::{
    AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode, ChannelConfig,
    ChannelConfigOptions, MediaStreamRenderer,
};

/// Options for constructing a [`MediaElementAudioSourceNode`]
// dictionary MediaElementAudioSourceOptions {
//   required HTMLMediaElement mediaElement;
// };
pub struct MediaElementAudioSourceOptions {
    pub media_element: MediaElement,
}

/// An audio source from a [`MediaElement`] (e.g. .ogg, .wav, .mp3 files)
///
/// The media element will take care of buffering of the stream so the render thread never blocks.
/// This also allows for playback controls (pause, looping, playback rate, etc.)
///
/// Note: do not forget to `start()` the node.
pub struct MediaElementAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    // controller: Controller,
}

impl AudioNode for MediaElementAudioSourceNode {
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

impl MediaElementAudioSourceNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: MediaElementAudioSourceOptions) -> Self {
        context.base().register(move |registration| {
            // let controller = options.media_element.controller().clone();
            // let scheduler = controller.scheduler().clone();
            let channel_config = ChannelConfigOptions::default().into();

            let node = MediaElementAudioSourceNode {
                registration,
                channel_config,
                // controller,
            };

            let resampler = Resampler::new(
                context.sample_rate_raw(),
                RENDER_QUANTUM_SIZE,
                options.media_element,
            );

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}
