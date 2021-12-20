//! The AudioNode interface and concrete types
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::context::{AudioContextRegistration, AudioNodeId, BaseAudioContext};
use crate::control::{Controller, ScheduledState, Scheduler};
use crate::media::MediaStream;
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{BufferDepletedError, SampleRate};

use lazy_static::lazy_static;

mod iir_filter;
pub use iir_filter::*;
mod biquad_filter;
pub use biquad_filter::*;
mod oscillator;
pub use oscillator::*;
mod destination;
pub use destination::*;
mod gain;
pub use gain::*;
mod delay;
pub use delay::*;
mod channel_splitter;
pub use channel_splitter::*;
mod channel_merger;
pub use channel_merger::*;
mod constant;
pub use constant::*;
mod panner;
pub use panner::*;
mod analyzer;
pub use analyzer::*;
mod audio_buffer_source;
pub use audio_buffer_source::*;
mod media_element;
pub use media_element::*;
mod media_stream;
pub use media_stream::*;
mod waveshaper;
pub use waveshaper::*;
mod stereo_panner;
pub use stereo_panner::*;

const TABLE_LENGTH_USIZE: usize = 2048;
const TABLE_LENGTH_BY_4_USIZE: usize = TABLE_LENGTH_USIZE / 4;
// 2048 casts without loss of precision cause its mantissa is 0b0
#[allow(clippy::cast_precision_loss)]
const TABLE_LENGTH_F32: f32 = TABLE_LENGTH_USIZE as f32;
const TABLE_LENGTH_BY_4_F32: f32 = TABLE_LENGTH_BY_4_USIZE as f32;

// Compute one period sine wavetable of size TABLE_LENGTH
lazy_static! {
    static ref SINETABLE: Vec<f32> = {
        #[allow(clippy::cast_precision_loss)]
        // 0 through 2048 are cast without loss of precision
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

/// Options for constructing ChannelConfig
#[derive(Clone, Debug)]
pub struct ChannelConfigOptions {
    pub count: usize,
    pub mode: ChannelCountMode,
    pub interpretation: ChannelInterpretation,
}

impl Default for ChannelConfigOptions {
    fn default() -> Self {
        Self {
            count: 2,
            mode: ChannelCountMode::Max,
            interpretation: ChannelInterpretation::Speakers,
        }
    }
}

/// Config for up/down-mixing of channels for audio nodes
#[derive(Clone, Debug)]
pub struct ChannelConfig {
    count: Arc<AtomicUsize>,
    mode: Arc<AtomicU32>,
    interpretation: Arc<AtomicU32>,
}

impl ChannelConfig {
    /// Represents an enumerated value describing the way channels must be matched between the
    /// node's inputs and outputs.
    pub fn count_mode(&self) -> ChannelCountMode {
        self.mode.load(Ordering::SeqCst).into()
    }
    pub fn set_count_mode(&self, v: ChannelCountMode) {
        self.mode.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    pub fn interpretation(&self) -> ChannelInterpretation {
        self.interpretation.load(Ordering::SeqCst).into()
    }
    pub fn set_interpretation(&self, v: ChannelInterpretation) {
        self.interpretation.store(v as u32, Ordering::SeqCst)
    }

    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
    pub fn set_count(&self, v: usize) {
        self.count.store(v, Ordering::SeqCst)
    }
}

impl From<ChannelConfigOptions> for ChannelConfig {
    fn from(opts: ChannelConfigOptions) -> Self {
        ChannelConfig {
            count: Arc::new(AtomicUsize::from(opts.count)),
            mode: Arc::new(AtomicU32::from(opts.mode as u32)),
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
/// (the object that lives the render thread). See [`BaseAudioContext::register`].
pub trait AudioNode {
    fn registration(&self) -> &AudioContextRegistration;

    fn id(&self) -> &AudioNodeId {
        self.registration().id()
    }
    fn channel_config_raw(&self) -> &ChannelConfig;
    fn channel_config_cloned(&self) -> ChannelConfig {
        self.channel_config_raw().clone()
    }

    /// The BaseAudioContext which owns this AudioNode.
    fn context(&self) -> &BaseAudioContext {
        self.registration().context()
    }

    /// Connect the output of this AudioNode to the input of another node.
    fn connect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        self.connect_at(dest, 0, 0).unwrap()
    }

    /// Connect a specific output of this AudioNode to a specific input of another node.
    fn connect_at<'a>(
        &self,
        dest: &'a dyn AudioNode,
        output: u32,
        input: u32,
    ) -> Result<&'a dyn AudioNode, crate::IndexSizeError> {
        if self.context() != dest.context() {
            panic!("attempting to connect nodes from different contexts");
        }

        if self.number_of_outputs() <= output || dest.number_of_inputs() <= input {
            return Err(crate::IndexSizeError {});
        }

        self.context().connect(self.id(), dest.id(), output, input);

        Ok(dest)
    }

    /// Disconnects all outputs of the AudioNode that go to a specific destination AudioNode.
    fn disconnect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if self.context() != dest.context() {
            panic!("attempting to disconnect nodes from different contexts");
        }

        self.context().disconnect(self.id(), dest.id());

        dest
    }

    /// Disconnects all outgoing connections from the AudioNode.
    fn disconnect_all(&self) {
        self.context().disconnect_all(self.id());
    }

    /// The number of inputs feeding into the AudioNode. For source nodes, this will be 0.
    fn number_of_inputs(&self) -> u32;
    /// The number of outputs coming out of the AudioNode.
    fn number_of_outputs(&self) -> u32;

    /// Represents an enumerated value describing the way channels must be matched between the
    /// node's inputs and outputs.
    fn channel_count_mode(&self) -> ChannelCountMode {
        self.channel_config_raw().count_mode()
    }
    fn set_channel_count_mode(&self, v: ChannelCountMode) {
        self.channel_config_raw().set_count_mode(v)
    }
    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    fn channel_interpretation(&self) -> ChannelInterpretation {
        self.channel_config_raw().interpretation()
    }
    fn set_channel_interpretation(&self, v: ChannelInterpretation) {
        self.channel_config_raw().set_interpretation(v)
    }
    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    fn channel_count(&self) -> usize {
        self.channel_config_raw().count()
    }
    fn set_channel_count(&self, v: usize) {
        self.channel_config_raw().set_count(v)
    }
}

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
pub trait AudioScheduledSourceNode {
    fn scheduler(&self) -> &Scheduler;

    /// Schedule playback start at this timestamp
    fn start_at(&self, start: f64) {
        self.scheduler().set_start(start)
    }

    /// Stop playback at this timestamp
    fn stop_at(&self, stop: f64) {
        self.scheduler().set_stop(stop)
    }

    /// Play immediately
    fn start(&self) {
        self.start_at(0.);
    }

    /// Stop immediately
    fn stop(&self) {
        self.stop_at(0.);
    }
}

/// Interface of source nodes, controlling pause/loop/offsets.
pub trait AudioControllableSourceNode {
    fn controller(&self) -> &Controller;

    fn loop_(&self) -> bool {
        self.controller().loop_()
    }

    fn set_loop(&self, loop_: bool) {
        self.controller().set_loop(loop_)
    }

    fn loop_start(&self) -> f64 {
        self.controller().loop_start()
    }

    fn set_loop_start(&self, loop_start: f64) {
        self.controller().set_loop_start(loop_start)
    }

    fn loop_end(&self) -> f64 {
        self.controller().loop_end()
    }

    fn set_loop_end(&self, loop_end: f64) {
        self.controller().set_loop_end(loop_end)
    }

    fn seek(&self, timestamp: f64) {
        self.controller().seek(timestamp)
    }
}

struct MediaStreamRenderer<R> {
    stream: R,
    scheduler: Scheduler,
    finished: bool,
}

impl<R> MediaStreamRenderer<R> {
    fn new(stream: R, scheduler: Scheduler) -> Self {
        Self {
            stream,
            scheduler,
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
        timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // todo, sub-quantum start/stop
        match self.scheduler.state(timestamp) {
            ScheduledState::Active => (),
            ScheduledState::NotStarted => {
                output.make_silent();
                return true; // will output in the future
            }
            ScheduledState::Ended => {
                output.make_silent();
                return false; // can clean up
            }
        }

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
            Some(Err(e)) if e.is::<BufferDepletedError>() => {
                log::debug!("media element buffer depleted");
                output.make_silent()
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
