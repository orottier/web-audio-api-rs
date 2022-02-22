use crate::buffer::Resampler;
use crate::context::{AudioContextRegistration, BaseAudioContext};
// use crate::control::Scheduler;
use crate::media::MediaStream;

use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, MediaStreamRenderer};

/// Options for constructing a [`MediaStreamAudioSourceNode`]
// dictionary MediaStreamAudioSourceOptions {
//   required MediaStream mediaStream;
// };
pub struct MediaStreamAudioSourceOptions<M> {
    pub media_stream: M,
}

/// An audio source from a [`MediaStream`] (e.g. microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. A later version of the library will allow you to wrap the `MediaStream`
/// in a `MediaElement`, which buffers the stream on another thread so the render thread never
/// blocks. <https://github.com/orottier/web-audio-api-rs/issues/120>
pub struct MediaStreamAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaStreamAudioSourceNode {
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

impl MediaStreamAudioSourceNode {
    pub fn new<C: BaseAudioContext, M: MediaStream>(
        context: &C,
        options: MediaStreamAudioSourceOptions<M>,
    ) -> Self {
        context.base().register(move |registration| {
            let channel_config = ChannelConfigOptions::default().into();

            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config,
            };

            let resampler = Resampler::new(
                context.sample_rate_raw(),
                RENDER_QUANTUM_SIZE,
                options.media_stream,
            );

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}
