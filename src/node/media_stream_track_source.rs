use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::media::{MediaStreamTrack, Resampler};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, MediaStreamRenderer};

/// Options for constructing a [`MediaStreamTrackAudioSourceNode`]
// dictionary MediaStreamAudioSourceOptions {
//   required MediaStream mediaStream;
// };
pub struct MediaStreamTrackAudioSourceOptions {
    pub media_stream_track: MediaStreamTrack,
}

/// An audio source from a [`MediaStreamTrack`] (e.g. the audio track of the microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. Use a
/// [`MediaElementAudioSourceNode`](crate::node::MediaElementAudioSourceNode) for real time safe
/// media playback.
pub struct MediaStreamTrackAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaStreamTrackAudioSourceNode {
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

impl MediaStreamTrackAudioSourceNode {
    pub fn new<C: BaseAudioContext>(
        context: &C,
        options: MediaStreamTrackAudioSourceOptions,
    ) -> Self {
        context.register(move |registration| {
            let node = MediaStreamTrackAudioSourceNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            let resampler = Resampler::new(
                context.sample_rate(),
                RENDER_QUANTUM_SIZE,
                options.media_stream_track,
            );

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}
