use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Options for constructing a [`MediaStreamAudioDestinationNode`]
pub struct MediaStreamAudioDestinationOptions<F> {
    pub stream: F,
    pub channel_config: ChannelConfigOptions,
}

/// An audio stream destination (e.g. WebRTC sink)
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamAudioDestinationNode>
/// - specification: <https://www.w3.org/TR/webaudio/#mediastreamaudiodestinationnode>
/// - see also: [`BaseAudioContext::create_media_stream_destination`](crate::context::BaseAudioContext::create_media_stream_destination)
///
/// Since the w3c `MediaStream` interface is not part of this library, we cannot adhere to the
/// official specification. Instead, you can pass in any callback that handles audio buffers.
///
/// IMPORTANT: the media stream sink will run on the render thread so you must ensure the processor
/// never blocks. Consider to have the callback send the buffers over a message channel of let it
/// fill a fixed size ring buffer.
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // Create an audio context where all audio nodes lives
/// let context = AudioContext::new(None);
///
/// // Create an oscillator node with sine (default) type
/// let osc = context.create_oscillator();
///
/// // Create a media destination node that will ship the samples out of the audio graph
/// let (sender, receiver) = crossbeam_channel::unbounded();
/// let callback = move |buf| {
///     // this will run on the render thread so it should not block
///     sender.send(buf).unwrap();
/// };
/// let dest = context.create_media_stream_destination(callback);
/// osc.connect(&dest);
/// osc.start();
///
/// // Handle recorded buffers
/// println!("samples recorded:");
/// let mut samples_recorded = 0;
/// for buf in receiver.iter() {
///     // You could write the samples to a file here.
///
///     samples_recorded += buf.length();
///     print!("{}\r", samples_recorded);
/// }
/// ```
///
/// # Examples
///
/// - `cargo run --release --example recorder`

pub struct MediaStreamAudioDestinationNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaStreamAudioDestinationNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }

    fn number_of_outputs(&self) -> u32 {
        0
    }
}

impl MediaStreamAudioDestinationNode {
    pub fn new<C: BaseAudioContext, F: FnMut(AudioBuffer) + Send + 'static>(
        context: &C,
        options: MediaStreamAudioDestinationOptions<F>,
    ) -> Self {
        context.base().register(move |registration| {
            let node = MediaStreamAudioDestinationNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let render = DestinationRenderer {
                stream: options.stream,
            };

            (node, Box::new(render))
        })
    }
}

struct DestinationRenderer<F> {
    stream: F,
}

impl<F: FnMut(AudioBuffer) + Send + 'static> AudioProcessor for DestinationRenderer<F> {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single input, no output
        let input = &inputs[0];

        // convert AudioRenderQuantum to AudioBuffer
        let samples: Vec<_> = input
            .channels()
            .iter()
            .map(|c| c.as_slice().to_vec())
            .collect();
        let buffer = AudioBuffer::from(samples, sample_rate);

        // run destination callback on buffer
        (self.stream)(buffer);

        false
    }
}
