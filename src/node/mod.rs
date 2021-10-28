//! The AudioNode interface and concrete types

use crate::buffer::{
    ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation, Resampler,
};
use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioNodeId, BaseAudioContext};
use crate::control::{Controller, Scheduler};
use crate::media::MediaStream;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::{BufferDepletedError, SampleRate, BUFFER_SIZE};

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
mod constant_src;
pub use constant_src::*;
mod panner;
pub use panner::*;
mod analyzer;
pub use analyzer::*;
mod audio_buffer_src;
pub use audio_buffer_src::*;
mod media_element_audio_src;
pub use media_element_audio_src::*;

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
        self.scheduler().start_at(start)
    }

    /// Stop playback at this timestamp
    fn stop_at(&self, stop: f64) {
        self.scheduler().stop_at(stop)
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

/// Options for constructing a MediaStreamAudioSourceNode
pub struct MediaStreamAudioSourceNodeOptions<M> {
    pub media: M,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from a [`MediaStream`] (e.g. microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. Consider wrapping the `MediaStream` in a [`MediaElement`], which buffers the
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
    pub fn new<C: AsBaseAudioContext, M: MediaStream>(
        context: &C,
        options: MediaStreamAudioSourceNodeOptions<M>,
    ) -> Self {
        context.base().register(move |registration| {
            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let resampler =
                Resampler::new(context.base().sample_rate(), BUFFER_SIZE, options.media);

            // setup void scheduler - always on
            let scheduler = Scheduler::new();
            scheduler.start_at(0.);

            let render = MediaStreamRenderer::new(resampler, scheduler);

            (node, Box::new(render))
        })
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
        _inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];

        // todo, sub-quantum start/stop
        if !self.scheduler.is_active(timestamp) {
            output.make_silent();
            return;
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
    }

    fn tail_time(&self) -> bool {
        !self.finished
    }
}
