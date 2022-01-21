use crate::buffer::Resampler;
use crate::context::{Context, AudioContextRegistration};
use crate::control::Scheduler;
use crate::media::MediaStream;

use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, MediaStreamRenderer};

/// Options for constructing a MediaStreamAudioSourceNode
pub struct MediaStreamAudioSourceNodeOptions<M> {
    pub media: M,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from a [`MediaStream`] (e.g. microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. Consider wrapping the `MediaStream` in a `MediaElement`, which buffers the
/// stream on another thread so the render thread never blocks.
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
    pub fn new<C: Context, M: MediaStream>(
        context: &C,
        options: MediaStreamAudioSourceNodeOptions<M>,
    ) -> Self {
        context.register(move |registration| {
            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let resampler = Resampler::new(
                context.sample_rate_raw(),
                RENDER_QUANTUM_SIZE,
                options.media,
            );

            // setup void scheduler - always on
            let scheduler = Scheduler::new();
            scheduler.start_at(0.);

            let render = MediaStreamRenderer::new(resampler, scheduler);

            (node, Box::new(render))
        })
    }
}
