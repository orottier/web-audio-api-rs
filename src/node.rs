//! The AudioNode interface and concrete types

use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::buffer::AudioBuffer;
use crate::context::{AsBaseAudioContext, AudioNodeId, BaseAudioContext};
use crate::graph::Render;

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

/// The meaning of the channels, defining how audio up-mixing and down-mixing will happen.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChannelInterpretation {
    Speakers,
    Discrete,
}

/// Config for up/down-mixing of channels for audio nodes
#[derive(Clone, Debug)]
pub struct ChannelConfig {
    pub count: usize,
    pub mode: ChannelCountMode,
    pub interpretation: ChannelInterpretation,
}

/// This interface represents audio sources, the audio destination, and intermediate processing
/// modules. These modules can be connected together to form processing graphs for rendering audio
/// to the audio hardware. Each node can have inputs and/or outputs.
pub trait AudioNode {
    fn id(&self) -> &AudioNodeId;
    fn to_render(&self) -> Box<dyn Render>;
    fn channel_config_raw(&self) -> &ChannelConfig;
    fn channel_config_raw_mut(&mut self) -> &mut ChannelConfig;

    /// The BaseAudioContext which owns this AudioNode.
    fn context(&self) -> &BaseAudioContext;

    /// Connect the output of this AudioNode to the input of another node.
    fn connect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if !std::ptr::eq(self.context(), dest.context()) {
            panic!("attempting to connect nodes from different contexts");
        }

        self.context().connect(self.id(), dest.id(), 0, 0);

        dest
    }

    /// Connect a specific output of this AudioNode to a specific input of another node.
    fn connect_at<'a>(
        &self,
        dest: &'a dyn AudioNode,
        output: u32,
        input: u32,
    ) -> Result<&'a dyn AudioNode, crate::IndexSizeError> {
        if !std::ptr::eq(self.context(), dest.context()) {
            panic!("attempting to connect nodes from different contexts");
        }

        if self.number_of_outputs() < output || dest.number_of_inputs() < input {
            return Err(crate::IndexSizeError {});
        }

        self.context().connect(self.id(), dest.id(), output, input);

        Ok(dest)
    }

    /// Disconnects all outputs of the AudioNode that go to a specific destination AudioNode.
    fn disconnect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if !std::ptr::eq(self.context(), dest.context()) {
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
        self.channel_config_raw().mode
    }
    /// Represents an enumerated value describing the meaning of the channels. This interpretation
    /// will define how audio up-mixing and down-mixing will happen.
    fn channel_interpretation(&self) -> ChannelInterpretation {
        self.channel_config_raw().interpretation
    }
    /// Represents an integer used to determine how many channels are used when up-mixing and
    /// down-mixing connections to any inputs to the node.
    fn channel_count(&self) -> usize {
        self.channel_config_raw().count
    }
}

/// Helper struct to start and stop audio streams
#[derive(Clone, Debug)]
pub struct Scheduler {
    start: Arc<AtomicU64>,
    stop: Arc<AtomicU64>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            start: Arc::new(AtomicU64::new(u64::MAX)),
            stop: Arc::new(AtomicU64::new(u64::MAX)),
        }
    }

    pub fn is_active(&self, frame: u64) -> bool {
        frame >= self.start.load(Ordering::SeqCst) && frame < self.stop.load(Ordering::SeqCst)
    }

    pub fn start(&self, start: u64) {
        self.start.store(start, Ordering::SeqCst);
    }

    pub fn stop(&self, stop: u64) {
        self.stop.store(stop, Ordering::SeqCst);
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have a Scheduler
pub trait Scheduled {
    fn scheduler(&self) -> &Scheduler;

    fn is_active(&self, frame: u64) -> bool {
        self.scheduler().is_active(frame)
    }

    fn start(&self, start: u64) {
        self.scheduler().start(start)
    }

    fn stop(&self, stop: u64) {
        self.scheduler().stop(stop)
    }
}

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
pub trait AudioScheduledSourceNode: AudioNode + Scheduled {
    /// Schedules a sound to playback at an exact time.
    fn start_at(&self, timestamp: f64) {
        let frame = (timestamp * self.context().sample_rate() as f64) as u64;
        self.scheduler().start(frame);
    }
    /// Schedules a sound to stop playback at an exact time.
    fn stop_at(&self, timestamp: f64) {
        let frame = (timestamp * self.context().sample_rate() as f64) as u64;
        self.scheduler().stop(frame);
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

/// Options for constructing an OscillatorNode
pub struct OscillatorOptions {
    pub type_: OscillatorType,
    pub frequency: u32,
    pub channel_config: ChannelConfig,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440,
            channel_config: ChannelConfig {
                count: 2,
                mode: ChannelCountMode::Max,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// Waveform of an oscillator
#[derive(Copy, Clone)]
pub enum OscillatorType {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Custom,
}

impl Default for OscillatorType {
    fn default() -> Self {
        OscillatorType::Sine
    }
}

impl From<u32> for OscillatorType {
    fn from(i: u32) -> Self {
        use OscillatorType::*;

        match i {
            0 => Sine,
            1 => Square,
            2 => Sawtooth,
            3 => Triangle,
            4 => Custom,
            _ => unreachable!(),
        }
    }
}

/// Audio source generating a periodic waveform
pub struct OscillatorNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
    pub(crate) frequency: Arc<AtomicU32>,
    pub(crate) type_: Arc<AtomicU32>,
    pub(crate) scheduler: Scheduler,
}

impl<'a> Scheduled for OscillatorNode<'a> {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl<'a> AudioScheduledSourceNode for OscillatorNode<'a> {}

impl<'a> AudioNode for OscillatorNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn to_render(&self) -> Box<dyn Render> {
        let render = OscillatorRenderer {
            frequency: self.frequency.clone(),
            type_: self.type_.clone(),
            scheduler: self.scheduler.clone(),
        };

        Box::new(render)
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn channel_config_raw_mut(&mut self) -> &mut ChannelConfig {
        &mut self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl<'a> OscillatorNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: OscillatorOptions) -> Self {
        context.base().register(move |id| {
            let frequency = Arc::new(AtomicU32::new(options.frequency));
            let type_ = Arc::new(AtomicU32::new(options.type_ as u32));
            let scheduler = Scheduler::new();

            OscillatorNode {
                context: context.base(),
                id,
                channel_config: options.channel_config,
                frequency,
                type_,
                scheduler,
            }
        })
    }

    pub fn frequency(&self) -> u32 {
        self.frequency.load(Ordering::SeqCst)
    }

    pub fn set_frequency(&self, freq: u32) {
        self.frequency.store(freq, Ordering::SeqCst);
    }

    pub fn type_(&self) -> OscillatorType {
        self.type_.load(Ordering::SeqCst).into()
    }

    pub fn set_type(&self, type_: OscillatorType) {
        self.type_.store(type_ as u32, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub(crate) struct OscillatorRenderer {
    pub frequency: Arc<AtomicU32>,
    pub type_: Arc<AtomicU32>,
    pub scheduler: Scheduler,
}

impl Scheduled for OscillatorRenderer {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl Render for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        timestamp: f64,
        sample_rate: u32,
    ) {
        // single output node
        let output = &mut outputs[0];

        // re-use previous buffer
        output.make_mono();
        let len = output.len();

        let frame = (timestamp * sample_rate as f64) as u64;

        // todo, sub-quantum start/stop
        if !self.is_active(frame) {
            output.make_silent();
            return;
        }

        let freq = self.frequency.load(Ordering::SeqCst) as f64;
        let type_ = self.type_.load(Ordering::SeqCst).into();

        output.modify_channels(|buffer| {
            let ts = (0..len).map(move |i| timestamp + i as f64 / sample_rate as f64);
            let io = ts.zip(buffer.iter_mut());

            use OscillatorType::*;

            match type_ {
                Sine => io.for_each(|(i, o)| *o = (2. * PI * freq * i).sin() as f32),
                Square => {
                    io.for_each(|(i, o)| *o = if (freq * i).fract() < 0.5 { 1. } else { -1. })
                }
                Sawtooth => io.for_each(|(i, o)| *o = 2. * ((freq * i).fract() - 0.5) as f32),
                _ => todo!(),
            }
        })
    }
}

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
}

#[derive(Debug)]
pub(crate) struct DestinationRenderer {}

impl Render for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        // todo, actually fill cpal buffer here
        *output = input.clone();
    }
}

impl<'a> AudioNode for DestinationNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn channel_config_raw_mut(&mut self) -> &mut ChannelConfig {
        &mut self.channel_config
    }

    fn to_render(&self) -> Box<dyn Render> {
        todo!()
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1 // todo, should be 0 actually, but we need it to copy into cpal for now
    }
}

/// Options for constructing a GainNode
pub struct GainOptions {
    pub gain: f32,
    pub channel_config: ChannelConfig,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self {
            gain: 1.,
            channel_config: ChannelConfig {
                count: 2,
                mode: ChannelCountMode::Max,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// AudioNode for volume control
pub struct GainNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
    pub(crate) gain: Arc<AtomicU32>,
}

impl<'a> AudioNode for GainNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn channel_config_raw_mut(&mut self) -> &mut ChannelConfig {
        &mut self.channel_config
    }

    fn to_render(&self) -> Box<dyn Render> {
        let render = GainRenderer {
            gain: self.gain.clone(),
        };
        Box::new(render)
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl<'a> GainNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: GainOptions) -> Self {
        context.base().register(move |id| {
            let gain = Arc::new(AtomicU32::new((options.gain * 100.) as u32));

            GainNode {
                context: context.base(),
                id,
                channel_config: options.channel_config,
                gain,
            }
        })
    }

    pub fn gain(&self) -> f32 {
        self.gain.load(Ordering::SeqCst) as f32 / 100.
    }

    pub fn set_gain(&self, gain: f32) {
        self.gain.store((gain * 100.) as u32, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub(crate) struct GainRenderer {
    pub gain: Arc<AtomicU32>,
}

impl Render for GainRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        let gain = self.gain.load(Ordering::SeqCst) as f32 / 100.;

        *output = input.clone();

        output.modify_channels(|channel| channel.iter_mut().for_each(|value| *value *= gain));
    }
}

/// Options for constructing a DelayNode
pub struct DelayOptions {
    // todo: actually delay by time
    pub render_quanta: u32,
    pub channel_config: ChannelConfig,
}

impl Default for DelayOptions {
    fn default() -> Self {
        Self {
            render_quanta: 0,
            channel_config: ChannelConfig {
                count: 2,
                mode: ChannelCountMode::Max,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// Node that delays the incoming audio signal by a certain amount
pub struct DelayNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) render_quanta: Arc<AtomicU32>,
    pub(crate) channel_config: ChannelConfig,
}

impl<'a> AudioNode for DelayNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn channel_config_raw_mut(&mut self) -> &mut ChannelConfig {
        &mut self.channel_config
    }

    fn to_render(&self) -> Box<dyn Render> {
        let render_quanta = self.render_quanta.load(Ordering::SeqCst);
        let cap = (render_quanta * crate::BUFFER_SIZE) as usize;
        let delay_buffer = Vec::with_capacity(cap);

        let render = DelayRenderer {
            render_quanta: self.render_quanta.clone(),
            delay_buffer,
            index: 0,
        };

        Box::new(render)
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl<'a> DelayNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: DelayOptions) -> Self {
        context.base().register(move |id| {
            let render_quanta = Arc::new(AtomicU32::new(options.render_quanta));

            DelayNode {
                context: context.base(),
                id,
                channel_config: options.channel_config,
                render_quanta,
            }
        })
    }

    pub fn render_quanta(&self) -> u32 {
        self.render_quanta.load(Ordering::SeqCst)
    }

    pub fn set_render_quanta(&self, render_quanta: u32) {
        self.render_quanta.store(render_quanta, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub(crate) struct DelayRenderer {
    pub render_quanta: Arc<AtomicU32>,
    pub delay_buffer: Vec<AudioBuffer>,
    pub index: usize,
}

impl Render for DelayRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        let quanta = self.render_quanta.load(Ordering::SeqCst) as usize;

        if quanta == 0 {
            // when no delay is set, simply copy input to output
            *output = input.clone();
        } else if self.delay_buffer.len() < quanta {
            // still filling buffer
            self.delay_buffer.push(input.clone());
            // clear output, it may have been re-used
            output.make_silent();
        } else {
            *output = std::mem::replace(&mut self.delay_buffer[self.index], input.clone());
            // progress index
            self.index = (self.index + 1) % quanta;
        }
    }
}
