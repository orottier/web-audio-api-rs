//! The AudioNode interface and concrete types
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::context::{AudioContextRegistration, ConcreteBaseAudioContext};
use crate::events::Event;
use crate::media::MediaStream;
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use lazy_static::lazy_static;

mod analyser;
pub use analyser::*;
mod audio_buffer_source;
pub use audio_buffer_source::*;
mod biquad_filter;
pub use biquad_filter::*;
mod channel_merger;
pub use channel_merger::*;
mod channel_splitter;
pub use channel_splitter::*;
mod constant_source;
pub use constant_source::*;
mod convolver;
pub use convolver::*;
mod delay;
pub use delay::*;
mod destination;
pub use destination::*;
mod dynamics_compressor;
pub use dynamics_compressor::*;
mod gain;
pub use gain::*;
mod iir_filter;
pub use iir_filter::*;
mod media_element_source;
pub use media_element_source::*;
mod media_stream_destination;
pub use media_stream_destination::*;
mod media_stream_source;
pub use media_stream_source::*;
mod oscillator;
pub use oscillator::*;
mod panner;
pub use panner::*;
mod stereo_panner;
pub use stereo_panner::*;
mod waveshaper;
use crate::events::EventHandler;
pub use waveshaper::*;

pub(crate) const TABLE_LENGTH_USIZE: usize = 8192;
pub(crate) const TABLE_LENGTH_BY_4_USIZE: usize = TABLE_LENGTH_USIZE / 4;

pub(crate) const TABLE_LENGTH_F32: f32 = TABLE_LENGTH_USIZE as f32;
pub(crate) const TABLE_LENGTH_BY_4_F32: f32 = TABLE_LENGTH_BY_4_USIZE as f32;

// Compute one period sine wavetable of size TABLE_LENGTH
lazy_static! {
    pub(crate) static ref SINETABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| ((x as f32) * 2.0 * PI * (1. / (TABLE_LENGTH_F32))).sin())
            .collect();
        table
    };
}

/// How channels must be matched between the node's inputs and outputs.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChannelCountMode {
    /// `computedNumberOfChannels` is the maximum of the number of channels of all connections to an
    /// input. In this mode channelCount is ignored.
    Max,
    /// `computedNumberOfChannels` is determined as for "max" and then clamped to a maximum value of
    /// the given channelCount.
    ClampedMax,
    /// `computedNumberOfChannels` is the exact value as specified by the channelCount.
    Explicit,
}

impl From<u32> for ChannelCountMode {
    fn from(i: u32) -> Self {
        use ChannelCountMode::*;

        match i {
            0 => Max,
            1 => ClampedMax,
            2 => Explicit,
            _ => unreachable!(),
        }
    }
}

/// The meaning of the channels, defining how audio up-mixing and down-mixing will happen.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChannelInterpretation {
    Speakers,
    Discrete,
}

impl From<u32> for ChannelInterpretation {
    fn from(i: u32) -> Self {
        use ChannelInterpretation::*;

        match i {
            0 => Speakers,
            1 => Discrete,
            _ => unreachable!(),
        }
    }
}

/// Options that can be used in constructing all AudioNodes.
#[derive(Clone, Debug)]
pub struct ChannelConfigOptions {
    /// Desired number of channels for the [`AudioNode::channel_count`] attribute.
    pub count: usize,
    /// Desired mode for the [`AudioNode::channel_count_mode`] attribute.
    pub count_mode: ChannelCountMode,
    /// Desired mode for the [`AudioNode::channel_interpretation`] attribute.
    pub interpretation: ChannelInterpretation,
}

impl Default for ChannelConfigOptions {
    fn default() -> Self {
        Self {
            count: 2,
            count_mode: ChannelCountMode::Max,
            interpretation: ChannelInterpretation::Speakers,
        }
    }
}

/// Config for up/down-mixing of input channels for audio nodes
///
/// Only when implementing the [`AudioNode`] trait manually, is this struct of any concern. The
/// methods `set_channel_count`, `set_channel_count_mode` and `set_channel_interpretation` from the
/// audio node interface will use this struct to sync the required info to the render thread.
///
/// The only way to construct an instance is with [`ChannelConfigOptions`]
///
/// ```
/// use web_audio_api::node::{ChannelConfigOptions, ChannelConfig, ChannelInterpretation, ChannelCountMode};
///
/// let opts = ChannelConfigOptions {
///     count: 1,
///     count_mode: ChannelCountMode::Explicit,
///     interpretation: ChannelInterpretation::Discrete,
/// };
/// let _: ChannelConfig = opts.into();
#[derive(Clone, Debug)]
pub struct ChannelConfig {
    count: Arc<AtomicUsize>,
    count_mode: Arc<AtomicU32>,
    interpretation: Arc<AtomicU32>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        ChannelConfigOptions::default().into()
    }
}

// All methods on this struct are marked `pub(crate)` because we don't want outside users to be able to change the values directly.
// These methods are only accessible via the AudioNode interface, so AudioNode's that have channel count/mode constraints
// should be able to assert those.
impl ChannelConfig {
    /// Represents an enumerated value describing the way channels must be matched between the
    /// node's inputs and outputs.
    pub(crate) fn count_mode(&self) -> ChannelCountMode {
        self.count_mode.load(Ordering::SeqCst).into()
    }
    fn set_count_mode(&self, v: ChannelCountMode) {
        self.count_mode.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    pub(crate) fn interpretation(&self) -> ChannelInterpretation {
        self.interpretation.load(Ordering::SeqCst).into()
    }
    fn set_interpretation(&self, v: ChannelInterpretation) {
        self.interpretation.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    pub(crate) fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
    fn set_count(&self, v: usize) {
        crate::assert_valid_number_of_channels(v);
        self.count.store(v, Ordering::SeqCst)
    }
}

impl From<ChannelConfigOptions> for ChannelConfig {
    fn from(opts: ChannelConfigOptions) -> Self {
        Self {
            count: Arc::new(AtomicUsize::from(opts.count)),
            count_mode: Arc::new(AtomicU32::from(opts.count_mode as u32)),
            interpretation: Arc::new(AtomicU32::from(opts.interpretation as u32)),
        }
    }
}

/// This interface represents audio sources, the audio destination, and intermediate processing
/// modules.
///
/// These modules can be connected together to form processing graphs for rendering audio
/// to the audio hardware. Each node can have inputs and/or outputs.
///
/// Note that the AudioNode is typically constructed together with an [`AudioProcessor`]
/// (the object that lives the render thread). See [`BaseAudioContext::register`](crate::context::BaseAudioContext::register).
pub trait AudioNode {
    fn registration(&self) -> &AudioContextRegistration;

    fn channel_config(&self) -> &ChannelConfig;

    /// The [`BaseAudioContext`](crate::context::BaseAudioContext) concrete type which owns this
    /// AudioNode.
    fn context(&self) -> &ConcreteBaseAudioContext {
        self.registration().context()
    }

    /// Connect the output of this AudioNode to the input of another node.
    ///
    /// # Panics
    ///
    /// This function will panic when
    /// - the AudioContext of the source and destination does not match
    fn connect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        self.connect_at(dest, 0, 0)
    }

    /// Connect a specific output of this AudioNode to a specific input of another node.
    ///
    /// # Panics
    ///
    /// This function will panic when
    /// - the AudioContext of the source and destination does not match
    /// - if the input port is out of bounds for the destination node
    /// - if the output port is out of bounds for the source node
    fn connect_at<'a>(
        &self,
        dest: &'a dyn AudioNode,
        output: usize,
        input: usize,
    ) -> &'a dyn AudioNode {
        if self.context() != dest.context() {
            panic!("InvalidAccessError: Attempting to connect nodes from different contexts");
        }
        if self.number_of_outputs() <= output {
            panic!("IndexSizeError: output port {} is out of bounds", output);
        }
        if dest.number_of_inputs() <= input {
            panic!("IndexSizeError: input port {} is out of bounds", input);
        }

        self.context().connect(
            self.registration().id(),
            dest.registration().id(),
            output,
            input,
        );
        dest
    }

    /// Disconnects all outputs of the AudioNode that go to a specific destination AudioNode.
    fn disconnect_from<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if self.context() != dest.context() {
            panic!("attempting to disconnect nodes from different contexts");
        }

        self.context()
            .disconnect_from(self.registration().id(), dest.registration().id());

        dest
    }

    /// Disconnects all outgoing connections from the AudioNode.
    fn disconnect(&self) {
        self.context().disconnect(self.registration().id());
    }

    /// The number of inputs feeding into the AudioNode. For source nodes, this will be 0.
    fn number_of_inputs(&self) -> usize;

    /// The number of outputs coming out of the AudioNode.
    fn number_of_outputs(&self) -> usize;

    /// Represents an enumerated value describing the way channels must be matched between the
    /// node's inputs and outputs.
    fn channel_count_mode(&self) -> ChannelCountMode {
        self.channel_config().count_mode()
    }

    /// Update the `channel_count_mode` attribute
    fn set_channel_count_mode(&self, v: ChannelCountMode) {
        self.channel_config().set_count_mode(v)
    }

    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    fn channel_interpretation(&self) -> ChannelInterpretation {
        self.channel_config().interpretation()
    }

    /// Update the `channel_interpretation` attribute
    fn set_channel_interpretation(&self, v: ChannelInterpretation) {
        self.channel_config().set_interpretation(v)
    }
    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    fn channel_count(&self) -> usize {
        self.channel_config().count()
    }

    /// Update the `channel_count` attribute
    fn set_channel_count(&self, v: usize) {
        self.channel_config().set_count(v)
    }
}

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
pub trait AudioScheduledSourceNode: AudioNode {
    /// Play immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start(&self);

    /// Schedule playback start at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start_at(&self, when: f64);

    /// Stop immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was already stopped
    fn stop(&self);

    /// Schedule playback stop at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was already stopped
    fn stop_at(&self, when: f64);

    /// Register callback to run when the source node has stopped playing
    ///
    /// For all [`AudioScheduledSourceNode`]s, the ended event is dispatched when the stop time
    /// determined by stop() is reached. For an [`AudioBufferSourceNode`], the event is also
    /// dispatched because the duration has been reached or if the entire buffer has been played.
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    fn set_onended<F: FnOnce() + Send + 'static>(&self, callback: F) {
        let callback = move |_| callback();

        self.context().set_event_handler(
            Event::Ended(self.registration().id()),
            EventHandler::Once(Box::new(callback)),
        );
    }

    /// Unset the callback to run when the source node has stopped playing
    fn clear_onended(&self) {
        self.context()
            .clear_event_handler(Event::Ended(self.registration().id()));
    }
}

// `MediaStreamRenderer` is internally used by `MediaElementAudioSourceNode` and
// `MediaStreamAudioSourceNode`.
struct MediaStreamRenderer<R> {
    stream: R,
    finished: bool,
}

impl<R> MediaStreamRenderer<R> {
    fn new(stream: R) -> Self {
        Self {
            stream,
            // scheduler,
            finished: false,
        }
    }
}

impl<R: MediaStream> AudioProcessor for MediaStreamRenderer<R> {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // @note - maybe we need to disciminate between a paused and depleted term
        match self.stream.next() {
            Some(Ok(buffer)) => {
                let channels = buffer.number_of_channels();
                output.set_number_of_channels(channels);
                output
                    .channels_mut()
                    .iter_mut()
                    .zip(buffer.channels())
                    .for_each(|(o, i)| o.copy_from_slice(i.as_slice()));
            }
            Some(Err(e)) => {
                log::warn!("Error playing audio stream: {}", e);
                self.finished = true; // halt playback
                output.make_silent()
            }
            None => {
                if !self.finished {
                    log::debug!("Stream finished");
                    self.finished = true;
                }
                output.make_silent()
            }
        }

        !self.finished
    }
}
