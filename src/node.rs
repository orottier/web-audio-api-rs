//! The AudioNode interface and concrete types

use std::f32::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::buffer::{
    AudioBuffer, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelData,
    ChannelInterpretation, Resampler,
};
use crate::context::{
    AsBaseAudioContext, AudioContextRegistration, AudioNodeId, AudioParamId, BaseAudioContext,
};
use crate::control::{Controller, Scheduler};
use crate::media::{MediaElement, MediaStream};
use crate::param::{AudioParam, AudioParamOptions};
use crate::process::{AudioParamValues, AudioProcessor, AudioProcessor2};
use crate::SampleRate;

/// This interface represents audio sources, the audio destination, and intermediate processing
/// modules.
///
/// These modules can be connected together to form processing graphs for rendering audio
/// to the audio hardware. Each node can have inputs and/or outputs.
///
/// Note that the AudioNode is typically constructed together with an `[crate::process::AudioProcessor]`
/// (the object that lives the render thread). See `[BaseAudioContext::register]`.
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
        if !std::ptr::eq(self.context(), dest.context()) {
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
            channel_config: ChannelConfigOptions::default(),
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
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    frequency: AudioParam<'a>,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
}

impl<'a> AudioScheduledSourceNode for OscillatorNode<'a> {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl<'a> AudioNode for OscillatorNode<'a> {
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

impl<'a> OscillatorNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: OscillatorOptions) -> Self {
        context.base().register(move |registration| {
            let nyquist = context.base().sample_rate().0 as f32 / 2.;
            let param_opts = AudioParamOptions {
                min_value: -nyquist,
                max_value: nyquist,
                default_value: 440.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());
            f_param.set_value(options.frequency);

            let type_ = Arc::new(AtomicU32::new(options.type_ as u32));
            let scheduler = Scheduler::new();

            let render = OscillatorRenderer {
                frequency: f_proc,
                type_: type_.clone(),
                scheduler: scheduler.clone(),
            };
            let node = OscillatorNode {
                registration,
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

struct OscillatorRenderer {
    frequency: AudioParamId,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
}

impl AudioProcessor2 for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];

        // re-use previous buffer
        output.force_mono();

        // todo, sub-quantum start/stop
        if !self.scheduler.is_active(timestamp) {
            output.make_silent();
            return;
        }

        let freq_values = params.get(&self.frequency);
        let freq = freq_values[0]; // force a-rate processing

        let type_ = self.type_.load(Ordering::SeqCst).into();

        let buffer = output.channel_data_mut(0);
        let io = buffer
            .iter_mut()
            .enumerate()
            .map(move |(i, v)| (timestamp as f32 + i as f32 / sample_rate.0 as f32, v));

        use OscillatorType::*;

        match type_ {
            Sine => io.for_each(|(t, o)| *o = (2. * PI * freq * t).sin()),
            Square => io.for_each(|(t, o)| *o = if (freq * t).fract() < 0.5 { 1. } else { -1. }),
            Sawtooth => io.for_each(|(t, o)| *o = 2. * ((freq * t).fract() - 0.5)),
            _ => todo!(),
        }
    }

    fn tail_time(&self) -> bool {
        true
    }
}

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode<'a> {
    pub(crate) registration: AudioContextRegistration<'a>,
    pub(crate) channel_count: usize,
}

struct DestinationRenderer {}

impl AudioProcessor2 for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        // todo, actually fill cpal buffer here
        *output = input.clone();
    }

    fn tail_time(&self) -> bool {
        unreachable!() // will never drop in control thread
    }
}

impl<'a> AudioNode for DestinationNode<'a> {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        unreachable!()
    }

    fn channel_config_cloned(&self) -> ChannelConfig {
        ChannelConfigOptions {
            count: self.channel_count,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Speakers,
        }
        .into()
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1 // todo, should be 0 actually, but we need it to copy into cpal for now
    }

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::Explicit
    }

    fn channel_interpretation(&self) -> ChannelInterpretation {
        ChannelInterpretation::Speakers
    }

    fn channel_count(&self) -> usize {
        self.channel_count
    }
}

impl<'a> DestinationNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, channel_count: usize) -> Self {
        context.base().register(move |registration| {
            let node = Self {
                registration,
                channel_count,
            };
            let proc = DestinationRenderer {};

            (node, Box::new(proc))
        })
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
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// AudioNode for volume control
pub struct GainNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    gain: AudioParam<'a>,
}

impl<'a> AudioNode for GainNode<'a> {
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
        1
    }
}

impl<'a> GainNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: GainOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());

            param.set_value_at_time(options.gain, 0.);

            let render = GainRenderer { gain: proc };

            let node = GainNode {
                registration,
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

struct GainRenderer {
    gain: AudioParamId,
}

impl AudioProcessor2 for GainRenderer {
    fn process<'a>(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = inputs[0];
        let output = &mut outputs[0];

        let gain_values = params.get(&self.gain);

        *output = input.clone();

        output.modify_channels(|channel| {
            channel
                .iter_mut()
                .zip(gain_values.iter())
                .for_each(|(value, g)| *value *= g)
        });
    }

    fn tail_time(&self) -> bool {
        false
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
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Node that delays the incoming audio signal by a certain amount
pub struct DelayNode<'a> {
    registration: AudioContextRegistration<'a>,
    render_quanta: Arc<AtomicU32>,
    channel_config: ChannelConfig,
}

impl<'a> AudioNode for DelayNode<'a> {
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
        1
    }
}

impl<'a> DelayNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: DelayOptions) -> Self {
        context.base().register(move |registration| {
            let render_quanta = Arc::new(AtomicU32::new(options.render_quanta));

            let cap = (options.render_quanta * crate::BUFFER_SIZE) as usize;
            let delay_buffer = Vec::with_capacity(cap);

            let render = DelayRenderer {
                render_quanta: render_quanta.clone(),
                delay_buffer,
                index: 0,
            };

            let node = DelayNode {
                registration,
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

struct DelayRenderer {
    render_quanta: Arc<AtomicU32>,
    delay_buffer: Vec<crate::buffer2::AudioBuffer>,
    index: usize,
}
unsafe impl Send for DelayRenderer {}

impl AudioProcessor2 for DelayRenderer {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
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

    fn tail_time(&self) -> bool {
        // todo: return false when all inputs disconnected and buffer exhausted
        true
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
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
}

impl<'a> AudioNode for ChannelSplitterNode<'a> {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
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
        context.base().register(move |registration| {
            options.channel_config.count = options.number_of_outputs as _;

            let node = ChannelSplitterNode {
                registration,
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
struct ChannelSplitterRenderer {
    pub number_of_outputs: usize,
}

impl AudioProcessor2 for ChannelSplitterRenderer {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input node
        let input = inputs[0];

        // assert number of outputs was correctly set by renderer
        assert_eq!(self.number_of_outputs, outputs.len());

        for (i, output) in outputs.iter_mut().enumerate() {
            output.force_mono();
            if i < input.number_of_channels() {
                *output.channel_data_mut(0) = input.channel_data(i).clone();
            } else {
                // input does not have this channel filled, emit silence
                output.make_silent();
            }
        }
    }

    fn tail_time(&self) -> bool {
        false
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
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
}

impl<'a> AudioNode for ChannelMergerNode<'a> {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
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
        context.base().register(move |registration| {
            options.channel_config.count = options.number_of_inputs as _;

            let node = ChannelMergerNode {
                registration,
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
struct ChannelMergerRenderer {
    number_of_inputs: usize,
}

impl AudioProcessor2 for ChannelMergerRenderer {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];
        output.set_number_of_channels(inputs.len());

        inputs.iter().enumerate().for_each(|(i, input)| {
            *output.channel_data_mut(i) = input.channel_data(0).clone();
        });
    }

    fn tail_time(&self) -> bool {
        false
    }
}

/// Options for constructing a MediaStreamAudioSourceNode
pub struct MediaStreamAudioSourceNodeOptions<M> {
    pub media: M,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from external media files (.ogg, .wav, .mp3)
pub struct MediaStreamAudioSourceNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
}

impl<'a> AudioNode for MediaStreamAudioSourceNode<'a> {
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

impl<'a> MediaStreamAudioSourceNode<'a> {
    pub fn new<C: AsBaseAudioContext, M: MediaStream>(
        context: &'a C,
        options: MediaStreamAudioSourceNodeOptions<M>,
    ) -> Self {
        context.base().register(move |registration| {
            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let resampler = Resampler::new(
                context.base().sample_rate(),
                crate::BUFFER_SIZE,
                options.media,
            );
            let render = AudioBufferRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}

/// Options for constructing a MediaElementAudioSourceNode
pub struct MediaElementAudioSourceNodeOptions<M> {
    pub media: MediaElement<M>,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from external media files (.ogg, .wav, .mp3)
pub struct MediaElementAudioSourceNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl<'a> AudioScheduledSourceNode for MediaElementAudioSourceNode<'a> {
    fn scheduler(&self) -> &Scheduler {
        &self.controller.scheduler()
    }
}
impl<'a> AudioControllableSourceNode for MediaElementAudioSourceNode<'a> {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl<'a> AudioNode for MediaElementAudioSourceNode<'a> {
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

impl<'a> MediaElementAudioSourceNode<'a> {
    pub fn new<C: AsBaseAudioContext, M: MediaStream>(
        context: &'a C,
        options: MediaElementAudioSourceNodeOptions<M>,
    ) -> Self {
        context.base().register(move |registration| {
            let controller = options.media.controller().clone();

            // wrap media input in resampler
            let resampler = Resampler::new(
                context.base().sample_rate(),
                crate::BUFFER_SIZE,
                options.media,
            );

            // setup user facing audio node
            let node = MediaElementAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let render = AudioBufferRenderer::new(resampler);

            (node, Box::new(render))
        })
    }
}

struct AudioBufferRenderer<R> {
    stream: R,
    finished: bool,
}

impl<R> AudioBufferRenderer<R> {
    fn new(stream: R) -> Self {
        Self {
            stream,
            finished: false,
        }
    }
}

impl<R: MediaStream> AudioProcessor2 for AudioBufferRenderer<R> {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];

        if let Some(Ok(buffer)) = self.stream.next() {
            let channels = buffer.number_of_channels();
            output.set_number_of_channels(channels);
            for i in 0..channels {
                match buffer.channel_data(i) {
                    None => *output.channel_data_mut(i) = output.channel_data(i).silence(),
                    Some(c) => output.channel_data_mut(i).copy_from_slice(c.as_slice()),
                }
            }
        } else {
            self.finished = true;
            output.make_silent()
        }
    }

    fn tail_time(&self) -> bool {
        !self.finished
    }
}

/// Options for constructing a AudioBufferSourceNode
pub struct AudioBufferSourceNodeOptions {
    pub buffer: Option<AudioBuffer>,
    pub channel_config: ChannelConfigOptions,
}

impl Default for AudioBufferSourceNodeOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// An audio source from an in-memory audio asset in an AudioBuffer
pub struct AudioBufferSourceNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl<'a> AudioScheduledSourceNode for AudioBufferSourceNode<'a> {
    fn scheduler(&self) -> &Scheduler {
        &self.controller.scheduler()
    }
}
impl<'a> AudioControllableSourceNode for AudioBufferSourceNode<'a> {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl<'a> AudioNode for AudioBufferSourceNode<'a> {
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

impl<'a> AudioBufferSourceNode<'a> {
    pub fn new<C: AsBaseAudioContext>(
        context: &'a C,
        options: AudioBufferSourceNodeOptions,
    ) -> Self {
        context.base().register(move |registration| {
            // unwrap_or_default buffer
            let buffer = options.buffer.unwrap_or_else(|| {
                AudioBuffer::new(1, crate::BUFFER_SIZE as usize, SampleRate(44_100))
            });

            // wrap input in resampler
            let resampler = Resampler::new(
                context.base().sample_rate(),
                crate::BUFFER_SIZE,
                std::iter::once(Ok(buffer)),
            );

            // wrap resampler in media-element (for loop/play/pause)
            let media = MediaElement::new(resampler);

            // setup user facing audio node
            let node = AudioBufferSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller: media.controller().clone(),
            };

            let render = AudioBufferRenderer::new(media);

            (node, Box::new(render))
        })
    }
}

/// Options for constructing an ConstantSourceNode
pub struct ConstantSourceOptions {
    pub offset: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for ConstantSourceOptions {
    fn default() -> Self {
        Self {
            offset: 1.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Audio source whose output is nominally a constant value
pub struct ConstantSourceNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    offset: AudioParam<'a>,
}

impl<'a> AudioNode for ConstantSourceNode<'a> {
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

impl<'a> ConstantSourceNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: ConstantSourceOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());
            param.set_value(options.offset);

            let render = ConstantSourceRenderer { offset: proc };
            let node = ConstantSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                offset: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn offset(&self) -> &AudioParam {
        &self.offset
    }
}

struct ConstantSourceRenderer {
    pub offset: AudioParamId,
}

impl AudioProcessor2 for ConstantSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single output node
        let output = &mut outputs[0];

        let offset_values = params.get(&self.offset);

        output.force_mono();
        output.channel_data_mut(0).copy_from_slice(offset_values);
    }

    fn tail_time(&self) -> bool {
        true
    }
}

/// Options for constructing a PannerNode
#[derive(Default)]
pub struct PannerOptions {
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub forward_x: f32,
    pub forward_y: f32,
    pub forward_z: f32,
    pub up_x: f32,
    pub up_y: f32,
    pub up_z: f32,
}

/// Positions / spatializes an incoming audio stream in three-dimensional space.
pub struct PannerNode<'a> {
    registration: AudioContextRegistration<'a>,
    channel_config: ChannelConfig,
    position_x: AudioParam<'a>,
    position_y: AudioParam<'a>,
    position_z: AudioParam<'a>,
}

impl<'a> AudioNode for PannerNode<'a> {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1 + 9 // todo, user should not be able to see these ports
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl<'a> PannerNode<'a> {
    pub fn new<C: AsBaseAudioContext>(context: &'a C, options: PannerOptions) -> Self {
        context.base().register(move |registration| {
            use crate::spatial::PARAM_OPTS;
            let id = registration.id();
            let (position_x, render_px) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_y, render_py) = context.base().create_audio_param(PARAM_OPTS, id);
            let (position_z, render_pz) = context.base().create_audio_param(PARAM_OPTS, id);

            position_x.set_value_at_time(options.position_x, 0.);
            position_y.set_value_at_time(options.position_y, 0.);
            position_z.set_value_at_time(options.position_z, 0.);

            let render = PannerRenderer {
                position_x: render_px,
                position_y: render_py,
                position_z: render_pz,
            };

            let node = PannerNode {
                registration,
                channel_config: ChannelConfigOptions {
                    count: 2,
                    mode: ChannelCountMode::ClampedMax,
                    interpretation: ChannelInterpretation::Speakers,
                }
                .into(),
                position_x,
                position_y,
                position_z,
            };

            context.base().connect_listener_to_panner(&node.id());

            (node, Box::new(render))
        })
    }

    pub fn position_x(&self) -> &AudioParam {
        &self.position_x
    }

    pub fn position_y(&self) -> &AudioParam {
        &self.position_y
    }

    pub fn position_z(&self) -> &AudioParam {
        &self.position_z
    }
}

struct PannerRenderer {
    position_x: AudioParamId,
    position_y: AudioParamId,
    position_z: AudioParamId,
}

impl AudioProcessor2 for PannerRenderer {
    fn process(
        &mut self,
        inputs: &[&crate::buffer2::AudioBuffer],
        outputs: &mut [crate::buffer2::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input node, assume mono, not silent
        let input = inputs[0].channel_data(0);
        // single output node
        let output = &mut outputs[0];

        // a-rate processing for now

        // source parameters (Panner)
        let source_position_x = params.get(&self.position_x)[0];
        let source_position_y = params.get(&self.position_y)[0];
        let source_position_z = params.get(&self.position_z)[0];

        // listener parameters (AudioListener)
        let l_position_x = inputs[1].channel_data(0)[0];
        let l_position_y = inputs[2].channel_data(0)[0];
        let l_position_z = inputs[3].channel_data(0)[0];
        let l_forward_x = inputs[4].channel_data(0)[0];
        let l_forward_y = inputs[5].channel_data(0)[0];
        let l_forward_z = inputs[6].channel_data(0)[0];
        let l_up_x = inputs[7].channel_data(0)[0];
        let l_up_y = inputs[8].channel_data(0)[0];
        let l_up_z = inputs[9].channel_data(0)[0];

        let (mut azimuth, _elevation) = crate::spatial::azimuth_and_elevation(
            [source_position_x, source_position_y, source_position_z],
            [l_position_x, l_position_y, l_position_z],
            [l_forward_x, l_forward_y, l_forward_z],
            [l_up_x, l_up_y, l_up_z],
        );

        // First, clamp azimuth to allowed range of [-180, 180].
        azimuth = azimuth.max(-180.);
        azimuth = azimuth.min(180.);
        // Then wrap to range [-90, 90].
        if azimuth < -90. {
            azimuth = -180. - azimuth;
        } else if azimuth > 90. {
            azimuth = 180. - azimuth;
        }

        let x = (azimuth + 90.) / 180.;
        let gain_l = (x * PI / 2.).cos();
        let gain_r = (x * PI / 2.).sin();

        let distance = crate::spatial::distance(
            [source_position_x, source_position_y, source_position_z],
            [l_position_x, l_position_y, l_position_z],
        );
        let dist_gain = 1. / distance;

        let left = input.iter().map(|&v| v * gain_l * dist_gain);
        let right = input.iter().map(|&v| v * gain_r * dist_gain);

        output.set_number_of_channels(2);
        output
            .channel_data_mut(0)
            .iter_mut()
            .zip(left)
            .for_each(|(o, i)| *o = i);
        output
            .channel_data_mut(1)
            .iter_mut()
            .zip(right)
            .for_each(|(o, i)| *o = i);
    }

    fn tail_time(&self) -> bool {
        false // only for panning model HRTF
    }
}
