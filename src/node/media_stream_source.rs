use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::media::{MediaStream, Resampler};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, MediaStreamRenderer};

/// Options for constructing a [`MediaStreamAudioSourceNode`]
// dictionary MediaStreamAudioSourceOptions {
//   required MediaStream mediaStream;
// };
pub struct MediaStreamAudioSourceOptions<'a> {
    pub media_stream: &'a MediaStream,
}

/// An audio source from a [`MediaStream`] (e.g. microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. Use a
/// [`MediaElementAudioSourceNode`](crate::node::MediaElementAudioSourceNode) for real time safe
/// media playback.
pub struct MediaStreamAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaStreamAudioSourceNode {
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

impl MediaStreamAudioSourceNode {
    /// Create a new `MediaStreamAudioSourceNode`
    ///
    /// # Panics
    ///
    /// This method will panic when the provided `MediaStream` does not contain any audio tracks.
    pub fn new<C: BaseAudioContext>(context: &C, options: MediaStreamAudioSourceOptions) -> Self {
        context.register(move |registration| {
            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            let resampler = Resampler::new(
                context.sample_rate(),
                RENDER_QUANTUM_SIZE,
                options.media_stream.get_tracks()[0].iter(),
            );

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}
