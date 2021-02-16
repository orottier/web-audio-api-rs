//! The AudioNode interface and concrete types

use std::f32::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::buffer::{
    AudioBuffer, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelData,
    ChannelInterpretation,
};
use crate::context::{AsBaseAudioContext, AudioNodeId, BaseAudioContext};
use crate::graph::Render;
use crate::media::MediaElement;
use crate::param::{AudioParam, AudioParamOptions, AudioParamRenderer};

/// This interface represents audio sources, the audio destination, and intermediate processing
/// modules. These modules can be connected together to form processing graphs for rendering audio
/// to the audio hardware. Each node can have inputs and/or outputs.
pub trait AudioNode {
    fn id(&self) -> &AudioNodeId;
    fn channel_config_raw(&self) -> &ChannelConfig;

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
    pub frequency: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440.,
            channel_config: ChannelConfigOptions {
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
    pub(crate) frequency: AudioParam,
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

    fn id(&self) -> &AudioNodeId {
        &self.id
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

impl<'a> OscillatorNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: OscillatorOptions) -> Self {
        context.base().register(move |id| {
            let nyquist = context.base().sample_rate() as f32 / 2.;
            let param_opts = AudioParamOptions {
                min_value: -nyquist,
                max_value: nyquist,
                default_value: 440.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_render) = crate::param::audio_param_pair(param_opts);
            f_param.set_value(options.frequency);

            let type_ = Arc::new(AtomicU32::new(options.type_ as u32));
            let scheduler = Scheduler::new();

            let render = OscillatorRenderer {
                frequency: f_render,
                type_: type_.clone(),
                scheduler: scheduler.clone(),
            };
            let node = OscillatorNode {
                context: context.base(),
                id,
                channel_config: options.channel_config.into(),
                frequency: f_param,
                type_,
                scheduler,
            };

            (node, Box::new(render))
        })
    }

    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
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
    pub frequency: AudioParamRenderer,
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
        let len = output.sample_len();

        let frame = (timestamp * sample_rate as f64) as u64;

        // todo, sub-quantum start/stop
        if !self.is_active(frame) {
            output.make_silent();
            return;
        }

        let dt = 1. / sample_rate as f64;
        let freq_values = self.frequency.tick(timestamp, dt, len);
        let freq = freq_values[0]; // force a-rate processing

        let type_ = self.type_.load(Ordering::SeqCst).into();

        output.modify_channels(|buffer| {
            let ts = (0..len).map(move |i| timestamp as f32 + i as f32 / sample_rate as f32);
            let io = ts.zip(buffer.iter_mut());

            use OscillatorType::*;

            match type_ {
                Sine => io.for_each(|(i, o)| *o = (2. * PI * freq * i).sin()),
                Square => {
                    io.for_each(|(i, o)| *o = if (freq * i).fract() < 0.5 { 1. } else { -1. })
                }
                Sawtooth => io.for_each(|(i, o)| *o = 2. * ((freq * i).fract() - 0.5)),
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
    pub channel_config: ChannelConfigOptions,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self {
            gain: 1.,
            channel_config: ChannelConfigOptions {
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
    pub(crate) gain: AudioParam,
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
            let context = context.base();

            let param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, render) = crate::param::audio_param_pair(param_opts);

            param.set_value_at_time(options.gain, 0.);

            let render = GainRenderer { gain: render };

            let node = GainNode {
                context,
                id,
                channel_config: options.channel_config.into(),
                gain: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }
}

#[derive(Debug)]
pub(crate) struct GainRenderer {
    pub gain: AudioParamRenderer,
}

impl Render for GainRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        timestamp: f64,
        sample_rate: u32,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        let dt = 1. / sample_rate as f64;
        let gain_values = self.gain.tick(timestamp, dt, input.sample_len());

        *output = input.clone();

        output.modify_channels(|channel| {
            channel
                .iter_mut()
                .zip(gain_values.iter())
                .for_each(|(value, g)| *value *= g)
        });
    }
}

/// Options for constructing a DelayNode
pub struct DelayOptions {
    // todo: actually delay by time
    pub render_quanta: u32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for DelayOptions {
    fn default() -> Self {
        Self {
            render_quanta: 0,
            channel_config: ChannelConfigOptions {
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

            let cap = (options.render_quanta * crate::BUFFER_SIZE) as usize;
            let delay_buffer = Vec::with_capacity(cap);

            let render = DelayRenderer {
                render_quanta: render_quanta.clone(),
                delay_buffer,
                index: 0,
            };

            let node = DelayNode {
                context: context.base(),
                id,
                channel_config: options.channel_config.into(),
                render_quanta,
            };

            (node, Box::new(render))
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

/// Options for constructing a ChannelSplitterNode
pub struct ChannelSplitterOptions {
    pub number_of_outputs: u32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelSplitterOptions {
    fn default() -> Self {
        Self {
            number_of_outputs: 6,
            channel_config: ChannelConfigOptions {
                count: 6, // must be same as number_of_outputs
                mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Discrete,
            },
        }
    }
}

/// AudioNode for accessing the individual channels of an audio stream in the routing graph
pub struct ChannelSplitterNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
}

impl<'a> AudioNode for ChannelSplitterNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn set_channel_count(&self, _v: usize) {
        panic!("Cannot edit channel count of ChannelSplitterNode")
    }
    fn set_channel_count_mode(&self, _v: ChannelCountMode) {
        panic!("Cannot edit channel count mode of ChannelSplitterNode")
    }
    fn set_channel_interpretation(&self, _v: ChannelInterpretation) {
        panic!("Cannot edit channel interpretation of ChannelSplitterNode")
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        self.channel_count() as _
    }
}

impl<'a> ChannelSplitterNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, mut options: ChannelSplitterOptions) -> Self {
        context.base().register(move |id| {
            options.channel_config.count = options.number_of_outputs as _;

            let node = ChannelSplitterNode {
                context: context.base(),
                id,
                channel_config: options.channel_config.into(),
            };

            let render = ChannelSplitterRenderer {
                number_of_outputs: node.channel_count() as _,
            };

            (node, Box::new(render))
        })
    }
}

#[derive(Debug)]
pub(crate) struct ChannelSplitterRenderer {
    pub number_of_outputs: usize,
}

impl Render for ChannelSplitterRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        sample_rate: u32,
    ) {
        // single input node
        let input = inputs[0];

        // assert number of outputs was correctly set by renderer
        assert_eq!(self.number_of_outputs, outputs.len());

        for (i, output) in outputs.iter_mut().enumerate() {
            if i < input.number_of_channels() {
                if let Some(channel_data) = input.channel_data(i) {
                    *output = AudioBuffer::from_mono(channel_data.clone(), sample_rate);
                } else {
                    *output = AudioBuffer::new(input.sample_len(), 1, sample_rate);
                }
            } else {
                *output = AudioBuffer::new(input.sample_len(), 1, sample_rate);
            }
        }
    }
}

/// Options for constructing a ChannelMergerNode
pub struct ChannelMergerOptions {
    pub number_of_inputs: u32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ChannelMergerOptions {
    fn default() -> Self {
        Self {
            number_of_inputs: 6,
            channel_config: ChannelConfigOptions {
                count: 1,
                mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// AudioNode for combining channels from multiple audio streams into a single audio stream.
pub struct ChannelMergerNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
}

impl<'a> AudioNode for ChannelMergerNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }

    fn id(&self) -> &AudioNodeId {
        &self.id
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn set_channel_count(&self, _v: usize) {
        panic!("Cannot edit channel count of ChannelMergerNode")
    }
    fn set_channel_count_mode(&self, _v: ChannelCountMode) {
        panic!("Cannot edit channel count mode of ChannelMergerNode")
    }
    fn set_channel_interpretation(&self, _v: ChannelInterpretation) {
        panic!("Cannot edit channel interpretation of ChannelMergerNode")
    }

    fn number_of_inputs(&self) -> u32 {
        self.channel_count() as _
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl<'a> ChannelMergerNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, mut options: ChannelMergerOptions) -> Self {
        context.base().register(move |id| {
            options.channel_config.count = options.number_of_inputs as _;

            let node = ChannelMergerNode {
                context: context.base(),
                id,
                channel_config: options.channel_config.into(),
            };

            let render = ChannelMergerRenderer {
                number_of_inputs: node.channel_config.count(),
            };

            (node, Box::new(render))
        })
    }
}

#[derive(Debug)]
pub(crate) struct ChannelMergerRenderer {
    pub number_of_inputs: usize,
}

impl Render for ChannelMergerRenderer {
    fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        sample_rate: u32,
    ) {
        // single output node
        let output = &mut outputs[0];

        let silence = ChannelData::new(output.sample_len());
        let mut channels = vec![silence; self.number_of_inputs];

        for (input, channel) in inputs.iter().zip(channels.iter_mut()) {
            if let Some(channel_data) = input.channel_data(0) {
                *channel = channel_data.clone();
            }
        }

        *output = AudioBuffer::from_channels(channels, sample_rate);
    }
}

/// Options for constructing a MediaElementAudioSourceNode
pub struct MediaElementAudioSourceNodeOptions<MediaElement> {
    pub media: MediaElement,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from external media files (.ogg, .wav, .mp3)
pub struct MediaElementAudioSourceNode<'a> {
    pub(crate) context: &'a BaseAudioContext,
    pub(crate) id: AudioNodeId,
    pub(crate) channel_config: ChannelConfig,
}

impl<'a> AudioNode for MediaElementAudioSourceNode<'a> {
    fn context(&self) -> &BaseAudioContext {
        self.context
    }
    fn id(&self) -> &AudioNodeId {
        &self.id
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

impl<'a> MediaElementAudioSourceNode<'a> {
    pub fn new<C: AsBaseAudioContext, M: MediaElement>(
        context: &'a C,
        mut options: MediaElementAudioSourceNodeOptions<M>,
    ) -> Self {
        context.base().register(move |id| {
            let node = MediaElementAudioSourceNode {
                context: context.base(),
                id,
                channel_config: options.channel_config.into(),
            };

            // todo, stream audio instead of buffering fully
            let mut buffers = vec![];
            while let Ok(Some(buffer)) = options.media.stream_chunk() {
                buffers.push(buffer);
            }

            // todo, proper resampling
            let buffers = buffers
                .into_iter()
                .collect::<AudioBuffer>() // concat all chunks
                .split(crate::BUFFER_SIZE); // split full buffer into right sized chunks

            let render = MediaElementAudioSourceRenderer { buffers };

            (node, Box::new(render))
        })
    }
}

#[derive(Debug)]
pub(crate) struct MediaElementAudioSourceRenderer {
    pub buffers: Vec<AudioBuffer>,
}

impl Render for MediaElementAudioSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[&AudioBuffer],
        outputs: &mut [AudioBuffer],
        _timestamp: f64,
        sample_rate: u32,
    ) {
        // single output node
        let output = &mut outputs[0];

        if self.buffers.is_empty() {
            *output = AudioBuffer::new(
                output.number_of_channels(),
                output.sample_len(),
                sample_rate,
            );
        } else {
            *output = self.buffers.remove(0);
        }
    }
}
