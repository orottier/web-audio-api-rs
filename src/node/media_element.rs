use crate::buffer::Resampler;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::control::Controller;
use crate::media::MediaElement;
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, MediaStreamRenderer};

/// Options for constructing a [`MediaElementAudioSourceNode`]
// dictionary MediaElementAudioSourceOptions {
//   required HTMLMediaElement mediaElement;
// };
pub struct MediaElementAudioSourceOptions {
    pub media_element: MediaElement,
}

/// An audio source for piping a [`MediaElement`] (e.g. .ogg, .wav, .mp3 files)
/// in a WebAudio graph.
///
/// The media element will take care of buffering of the stream so the render thread never blocks.
///
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/MediaElementAudioSourceNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#MediaElementAudioSourceNode>
/// - see also: [`BaseAudioContext::create_media_element_source`](crate::context::BaseAudioContext::create_media_element_source)
///
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::media::{MediaDecoder, MediaElement};
/// use web_audio_api::node::{AudioNode};
///
/// // build a decoded audio stream the decoder
/// let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
/// let stream = MediaDecoder::try_new(file).unwrap();
/// // wrap in a `MediaElement`
/// let media_element = MediaElement::new(stream);
/// // pipe the media element into the web audio graph
/// let context = AudioContext::new(None);
/// let node = context.create_media_element_source(media_element);
/// node.connect(&context.destination());
/// ```
/// # Examples
///
/// - `cargo run --release --example media_element`
///
pub struct MediaElementAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
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
            let channel_config = ChannelConfigOptions::default().into();
            let controller = options.media_element.controller().clone();
            let media_element = options.media_element;

            let node = MediaElementAudioSourceNode {
                controller,
                registration,
                channel_config,
            };

            let resampler = Resampler::new(
                context.sample_rate_raw(),
                RENDER_QUANTUM_SIZE,
                media_element,
            );

            let render = MediaStreamRenderer::new(resampler);

            (node, Box::new(render))
        })
    }

    pub fn seek(&self, when: f64) {
        self.controller.seek(when);
    }

    pub fn loop_(&self) -> bool {
        self.controller.loop_()
    }

    pub fn set_loop(&self, value: bool) {
        self.controller.set_loop(value);
    }

    pub fn loop_start(&self) -> f64 {
        self.controller.loop_start()
    }

    pub fn set_loop_start(&self, value: f64) {
        self.controller.set_loop_start(value);
    }

    pub fn loop_end(&self) -> f64 {
        self.controller.loop_end()
    }

    pub fn set_loop_end(&self, value: f64) {
        self.controller.set_loop_end(value);
    }
}
