use std::error::Error;

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};

use super::{AudioNode, AudioNodeOptions, ChannelConfig};

use crate::media_streams::{MediaStream, MediaStreamTrack};
use crossbeam_channel::{self, Receiver, Sender};

/// An audio stream destination (e.g. WebRTC sink)
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamAudioDestinationNode>
/// - specification: <https://www.w3.org/TR/webaudio/#mediastreamaudiodestinationnode>
/// - see also: [`AudioContext::create_media_stream_destination`](crate::context::AudioContext::create_media_stream_destination)
///
/// Since the w3c `MediaStream` interface is not part of this library, we cannot adhere to the
/// official specification. Instead, you can pass in any callback that handles audio buffers.
///
/// IMPORTANT: you must consume the buffers faster than the render thread produces them, or you
/// will miss frames. Consider to spin up a dedicated thread to consume the buffers and cache them.
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // Create an audio context where all audio nodes lives
/// let context = AudioContext::default();
///
/// // Create an oscillator node with sine (default) type
/// let mut osc = context.create_oscillator();
///
/// // Create a media destination node
/// let dest = context.create_media_stream_destination();
/// osc.connect(&dest);
/// osc.start();
///
/// // Handle recorded buffers
/// println!("samples recorded:");
/// let mut samples_recorded = 0;
/// for item in dest.stream().get_tracks()[0].iter() {
///     let buffer = item.unwrap();
///
///     // You could write the samples to a file here.
///     samples_recorded += buffer.length();
///     print!("{}\r", samples_recorded);
/// }
/// ```
///
/// # Examples
///
/// - `cargo run --release --example recorder`
#[derive(Debug)]
pub struct MediaStreamAudioDestinationNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    stream: MediaStream,
}

impl AudioNode for MediaStreamAudioDestinationNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        0
    }
}

impl MediaStreamAudioDestinationNode {
    /// Create a new MediaStreamAudioDestinationNode
    pub fn new<C: BaseAudioContext>(context: &C, options: AudioNodeOptions) -> Self {
        let sample_rate = context.sample_rate();
        let silence_channels = options.channel_count;
        context.base().register(move |registration| {
            let (send, recv) = crossbeam_channel::bounded(1);

            let iter = AudioDestinationNodeStream {
                receiver: recv.clone(),
                sample_rate,
                channels: silence_channels,
            };
            let track = MediaStreamTrack::from_iter(iter);
            let stream = MediaStream::from_tracks(vec![track]);

            let node = MediaStreamAudioDestinationNode {
                registration,
                channel_config: options.into(),
                stream,
            };

            let render = DestinationRenderer { send, recv };

            (node, Box::new(render))
        })
    }

    /// A [`MediaStream`] producing audio buffers with the same number of channels as the node
    /// itself
    pub fn stream(&self) -> &MediaStream {
        &self.stream
    }
}

struct DestinationRenderer {
    send: Sender<AudioBuffer>,
    recv: Receiver<AudioBuffer>,
}

impl AudioProcessor for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input, no output
        let input = &inputs[0];

        // convert AudioRenderQuantum to AudioBuffer
        let samples: Vec<_> = input.channels().iter().map(|c| c.to_vec()).collect();
        let buffer = AudioBuffer::from(samples, scope.sample_rate);

        // clear previous entry if it was not consumed
        if self.recv.try_recv().is_ok() {
            log::warn!("MediaStreamDestination buffer dropped");
        }

        // ship out AudioBuffer
        let _ = self.send.send(buffer);

        false
    }
}

struct AudioDestinationNodeStream {
    receiver: Receiver<AudioBuffer>,
    sample_rate: f32,
    /// Channel count for underrun silence: starts at the node's channel_count
    /// and then follows the most recent real buffer. Consumers (the Resampler
    /// on cross-sample-rate paths) stitch buffers with AudioBuffer::extend,
    /// whose assert requires adjacent chunks to have equal channel counts - a
    /// drifting silence channel count panics the render thread.
    channels: usize,
}

impl Iterator for AudioDestinationNodeStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        // This iterator must never block. When the stream is consumed by a
        // MediaStreamAudioSourceNode living in the *same* context (a loopback:
        // dest.stream -> createMediaStreamSource), it is pulled from the render
        // thread itself. A blocking recv() then deadlocks the render thread in
        // any quantum where the source node is ordered before the destination
        // node - the thread waits for data that only it can produce, and
        // currentTime freezes. try_recv instead: on Empty emit one quantum of
        // silence (initial block / underrun; a source ordered first simply sees
        // one quantum of latency), on data pass it through, and a disconnected
        // channel still reports the error so the source node stops.
        match self.receiver.try_recv() {
            Ok(buf) => {
                self.channels = buf.number_of_channels();
                Some(Ok(buf))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => Some(Ok(AudioBuffer::from(
                vec![vec![0.; crate::RENDER_QUANTUM_SIZE]; self.channels.max(1)],
                self.sample_rate,
            ))),
            Err(e) => Some(Err(Box::new(e))),
        }
    }
}
