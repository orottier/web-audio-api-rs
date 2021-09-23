//! The AudioNode interface and concrete types

use std::f32::consts::{PI, TAU};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::analysis::Analyser;
use crate::buffer::{
    AudioBuffer, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
    Resampler,
};
use crate::context::{
    AsBaseAudioContext, AudioContextRegistration, AudioNodeId, AudioParamId, BaseAudioContext,
};
use crate::control::{Controller, Scheduler};
use crate::media::{MediaElement, MediaStream};
use crate::param::{AudioParam, AudioParamOptions};
use crate::process::{AudioParamValues, AudioProcessor};
use crate::{BufferDepletedError, SampleRate, BUFFER_SIZE};

use crossbeam_channel::{self, Receiver, Sender};

use lazy_static::lazy_static;

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

const TABLE_LENGTH_USIZE: usize = 2048;
const TABLE_LENGTH_F32: f32 = TABLE_LENGTH_USIZE as f32;

// Compute one period sine wavetable of size TABLE_LENGTH
lazy_static! {
    static ref SINETABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| ((x as f32) * TAU * (1. / (TABLE_LENGTH_F32))).sin())
            .collect();
        table
    };
    static ref SAWTABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| {
                let norm_phase = x as f32 / TABLE_LENGTH_F32;
                (2.0 * norm_phase) - 1.0
            })
            .collect();
        table
    };
    static ref SQUARETABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| {
                let norm_phase = x as f32 / TABLE_LENGTH_F32;
                if norm_phase <= 0.5 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        table
    };
    static ref TRIANGLETABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| {
                let norm_phase = x as f32 / TABLE_LENGTH_F32;
                1. - ((norm_phase - 0.5).abs() * 4.)
            })
            .collect();
        table
    };
}

/// Options for constructing a periodic wave
pub struct PeriodicWaveConstraints {
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    disable_normalization: bool,
}

/// PeriodicWave is a setup struct required to build
/// custom periodic waveform oscillator type.
#[derive(Clone)]
pub struct PeriodicWave {
    /// The real parameter represents an array of cosine terms of Fourrier series.
    ///
    /// The first element (index 0) represents the DC-offset.
    /// This offset has to be given but will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    real: Vec<f32>,
    /// The imag parameter represents an array of sine terms of Fourrier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    imag: Vec<f32>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    disable_normalization: bool,
}

impl PeriodicWave {
    /// Returns a PeriodicWave
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourrier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourrier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the PeriodicWave
    pub(crate) fn new(
        real: Vec<f32>,
        imag: Vec<f32>,
        constraints: Option<PeriodicWaveConstraints>,
    ) -> Self {
        if let Some(c) = constraints {
            Self {
                real,
                imag,
                disable_normalization: c.disable_normalization,
            }
        } else {
            Self {
                real,
                imag,
                disable_normalization: false,
            }
        }
    }
}

/// Options for constructing an OscillatorNode
pub struct OscillatorOptions {
    pub type_: OscillatorType,
    pub frequency: f32,
    pub channel_config: ChannelConfigOptions,
    pub periodic_wave: Option<PeriodicWave>,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440.,
            channel_config: ChannelConfigOptions::default(),
            periodic_wave: None,
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
pub struct OscillatorNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    frequency: AudioParam,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
}

impl AudioScheduledSourceNode for OscillatorNode {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl AudioNode for OscillatorNode {
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

impl OscillatorNode {
    /// Returns an OscillatorNode
    ///
    /// # Arguments:
    ///
    /// * `context` - The AudioContext
    /// * `options` - The Oscillatoroptions
    pub fn new<C: AsBaseAudioContext>(context: &C, options: OscillatorOptions) -> Self {
        context.base().register(move |registration| {
            let sample_rate = context.base().sample_rate().0 as f32;
            let nyquist = sample_rate / 2.;
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

            let sine_renderer = SineRenderer::new(options.frequency, sample_rate);
            let sawtooth_renderer = SawRenderer::new(options.frequency, sample_rate);
            let triangle_renderer = TriangleRenderer::new(options.frequency, sample_rate);
            let square_renderer = SquareRenderer::new(options.frequency, sample_rate);
            let custom_renderer = CustomRenderer::new(
                options.frequency,
                sample_rate,
                options.periodic_wave.clone(),
            );

            let render = OscillatorRenderer {
                frequency: f_proc,
                type_: type_.clone(),
                scheduler: scheduler.clone(),
                sine_renderer,
                sawtooth_renderer,
                triangle_renderer,
                square_renderer,
                custom_renderer,
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

    /// Returns the oscillator frequency audio parameter
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the oscillator type
    pub fn type_(&self) -> OscillatorType {
        self.type_.load(Ordering::SeqCst).into()
    }

    /// set the oscillator type
    pub fn set_type(&self, type_: OscillatorType) {
        self.type_.store(type_ as u32, Ordering::SeqCst);
    }

    /// set the oscillator type to custom. The oscillator will generate
    /// a perdioc waveform following the PeriodicWave characteristics
    //
    //  TODO: The current implementation doesn't communicate its state
    //  to the OscillatorRenderer, and so has no effect on the rendering.
    //  This function should send the updated periodics waveform characteristics
    //  to the OscillatorRenderer and more specifically to the CustomRenderer
    pub fn set_periodic_wave(&mut self, _periodic_wave: PeriodicWave) {
        self.set_type(OscillatorType::Custom);
        todo!();
    }
}

struct OscillatorRenderer {
    frequency: AudioParamId,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
    sine_renderer: SineRenderer,
    sawtooth_renderer: SawRenderer,
    triangle_renderer: TriangleRenderer,
    square_renderer: SquareRenderer,
    custom_renderer: CustomRenderer,
}

impl AudioProcessor for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        timestamp: f64,
        _sample_rate: SampleRate,
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

        use OscillatorType::*;

        match type_ {
            Sine => {
                // K-rate
                self.sine_renderer.set_frequency(freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.sine_renderer.tick());
            }
            Square => {
                // K-rate
                self.square_renderer.set_frequency(freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.square_renderer.tick());
            }
            Sawtooth => {
                // K-rate
                self.sawtooth_renderer.set_frequency(freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.sawtooth_renderer.tick());
            }
            Triangle => {
                // K-rate
                self.triangle_renderer.set_frequency(freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.triangle_renderer.tick())
            }
            Custom => {
                // K-rate
                self.custom_renderer.set_frequency(freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.custom_renderer.tick())
            }
        }
    }

    fn tail_time(&self) -> bool {
        true
    }
}

trait PolyBlep {
    fn poly_blep(&self, mut t: f32) -> f32 {
        let dt = self.incr_phase() / TAU;
        if t < dt {
            t /= dt;
            t + t - t * t - 1.0
        } else if t > 1.0 - dt {
            t = (t - 1.0) / dt;
            t * t + t + t + 1.0
        } else {
            0.0
        }
    }

    fn incr_phase(&self) -> f32;
}

trait Ticker {
    fn tick(&mut self) -> f32;
}

#[derive(Debug)]
struct SineRenderer {
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    incr_phase: f32,
    mu: f32,
}

impl SineRenderer {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        let incr_phase = 2. * PI * (frequency / sample_rate);
        let mu = (incr_phase - incr_phase.round()).abs();
        Self {
            frequency,
            sample_rate,
            phase: 0.0,
            incr_phase,
            mu,
        }
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.incr_phase = TABLE_LENGTH_F32 * frequency / self.sample_rate;
        self.mu = (self.incr_phase - self.incr_phase.round()).abs();
        self.frequency = frequency;
    }
}

impl Ticker for SineRenderer {
    fn tick(&mut self) -> f32 {
        let idx = (self.phase + self.incr_phase) as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
        let sample = SINETABLE[inf_idx] * (1. - self.mu) + SINETABLE[sup_idx] * self.mu;
        self.phase = (self.phase + self.incr_phase) % TABLE_LENGTH_F32;
        sample
    }
}

#[derive(Debug)]
struct SawRenderer {
    frequency: f32,
    sample_rate: f32,
    incr_phase: f32,
    mu: f32,
    phase: f32,
}

impl SawRenderer {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        let incr_phase = (TABLE_LENGTH_F32 / sample_rate) * frequency;
        let mu = (incr_phase - incr_phase.round()).abs();
        Self {
            frequency,
            sample_rate,
            phase: 0.0,
            incr_phase,
            mu,
        }
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.incr_phase = TABLE_LENGTH_F32 * frequency / self.sample_rate;
        self.mu = (self.incr_phase - self.incr_phase.round()).abs();
        self.frequency = frequency;
    }
}

impl PolyBlep for SawRenderer {
    fn incr_phase(&self) -> f32 {
        self.incr_phase
    }
}

impl Ticker for SawRenderer {
    fn tick(&mut self) -> f32 {
        let idx = (self.phase + self.incr_phase) as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;

        let mut sample = SAWTABLE[inf_idx] * (1. - self.mu) + SAWTABLE[sup_idx] * self.mu;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);

        self.phase = (self.phase + self.incr_phase) % TABLE_LENGTH_F32;
        sample
    }
}

#[derive(Debug)]
struct TriangleRenderer {
    frequency: f32,
    sample_rate: f32,
    incr_phase: f32,
    mu: f32,
    phase: f32,
}

impl TriangleRenderer {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        let incr_phase = TABLE_LENGTH_F32 * frequency / sample_rate;
        let mu = (incr_phase - incr_phase.round()).abs();
        Self {
            frequency,
            sample_rate,
            phase: 0.0,
            incr_phase,
            mu,
        }
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.incr_phase = TABLE_LENGTH_F32 * frequency / self.sample_rate;
        self.mu = (self.incr_phase - self.incr_phase.round()).abs();
        self.frequency = frequency;
    }
}

impl PolyBlep for TriangleRenderer {
    fn incr_phase(&self) -> f32 {
        self.incr_phase
    }
}

impl Ticker for TriangleRenderer {
    fn tick(&mut self) -> f32 {
        let idx = (self.phase + self.incr_phase) as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;

        let mut sample = TRIANGLETABLE[inf_idx] * (1. - self.mu) + TRIANGLETABLE[sup_idx] * self.mu;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);
        self.phase = (self.phase + self.incr_phase) % TABLE_LENGTH_F32;

        sample
    }
}

#[derive(Debug)]
struct SquareRenderer {
    frequency: f32,
    sample_rate: f32,
    incr_phase: f32,
    mu: f32,
    phase: f32,
}

impl SquareRenderer {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        let incr_phase = (TABLE_LENGTH_F32 / sample_rate) * frequency;
        let mu = (incr_phase - incr_phase.round()).abs();
        Self {
            frequency,
            sample_rate,
            phase: 0.0,
            incr_phase,
            mu,
        }
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.incr_phase = (TABLE_LENGTH_F32 / self.sample_rate) * frequency;
        self.mu = (self.incr_phase - self.incr_phase.round()).abs();
        self.frequency = frequency;
    }
}

impl PolyBlep for SquareRenderer {
    fn incr_phase(&self) -> f32 {
        self.incr_phase
    }
}

impl Ticker for SquareRenderer {
    fn tick(&mut self) -> f32 {
        let idx = (self.phase + self.incr_phase) as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;

        let mut sample = SQUARETABLE[inf_idx] * (1. - self.mu) + SQUARETABLE[sup_idx] * self.mu;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);
        self.phase = (self.phase + self.incr_phase) % TABLE_LENGTH_F32;
        sample
    }
}

struct CustomRenderer {
    frequency: f32,
    sample_rate: f32,
    cplxs: Vec<(f32, f32)>,
    norms: Vec<f32>,
    phases: Vec<f32>,
    incr_phases: Vec<f32>,
    mus: Vec<f32>,
    normalizer: Option<f32>,
}

impl CustomRenderer {
    fn new(frequency: f32, sample_rate: f32, periodic_wave: Option<PeriodicWave>) -> Self {
        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = if let Some(p_w) = periodic_wave {
            p_w
        } else {
            PeriodicWave::new(vec![0., 1.0], vec![0., 0.], None)
        };

        let cplxs: Vec<(f32, f32)> = real.iter().zip(&imag).map(|(&r, &i)| (r, i)).collect();

        let norms: Vec<f32> = cplxs
            .iter()
            .map(|(r, i)| (f32::powi(*r, 2i32) + f32::powi(*i, 2i32)).sqrt())
            .collect();

        let phases: Vec<f32> = cplxs
            .iter()
            .map(|(r, i)| {
                let phase = f32::atan2(*i, *r);
                if phase < 0. {
                    (phase + 2. * PI) * (TABLE_LENGTH_F32 / (2.0 * PI))
                } else {
                    phase * (TABLE_LENGTH_F32 / 2.0 * PI)
                }
            })
            .collect();

        let incr_phases: Vec<f32> = cplxs
            .iter()
            .enumerate()
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (frequency / sample_rate))
            .collect();

        let mus: Vec<f32> = incr_phases
            .iter()
            .map(|incr_phase| (incr_phase - incr_phase.round()).abs())
            .collect();

        let normalizer = if !disable_normalization {
            Some(Self::get_normalizer(
                phases.clone(),
                incr_phases.clone(),
                mus.clone(),
                norms.clone(),
            ))
        } else {
            None
        };

        Self {
            frequency,
            sample_rate,
            cplxs,
            norms,
            phases,
            incr_phases,
            mus,
            normalizer,
        }
    }

    fn get_normalizer(
        mut phases: Vec<f32>,
        incr_phases: Vec<f32>,
        mus: Vec<f32>,
        norms: Vec<f32>,
    ) -> f32 {
        let mut samples: Vec<f32> = Vec::new();

        while phases[1] <= TABLE_LENGTH_F32 {
            let mut sample = 0.0;
            for i in 1..phases.len() {
                let gain = norms[i];
                let phase = phases[i];
                let incr_phase = incr_phases[i];
                let mu = mus[i];
                let idx = (phase + incr_phase) as usize;
                let inf_idx = idx % TABLE_LENGTH_USIZE;
                let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
                sample += (SINETABLE[inf_idx] * (1. - mu) + SINETABLE[sup_idx] * mu) * gain;
                phases[i] = phase + incr_phase;
            }
            samples.push(sample);
        }

        1. / samples
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("Maximum value not found")
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
        self.incr_phases = self
            .cplxs
            .iter()
            .enumerate()
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (self.frequency / self.sample_rate))
            .collect();

        self.mus = self
            .incr_phases
            .iter()
            .map(|incr_phase| (incr_phase - incr_phase.round()).abs())
            .collect();
    }
}

impl Ticker for CustomRenderer {
    fn tick(&mut self) -> f32 {
        let mut sample = 0.;
        for i in 1..self.phases.len() {
            let gain = self.norms[i];
            let phase = self.phases[i];
            let incr_phase = self.incr_phases[i];
            let mu = self.mus[i];
            let idx = (phase + incr_phase) as usize;
            let inf_idx = idx % TABLE_LENGTH_USIZE;
            let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
            sample += (SINETABLE[inf_idx] * (1. - mu) + SINETABLE[sup_idx] * mu)
                * gain
                * self.normalizer.unwrap_or(1.);
            self.phases[i] = (phase + incr_phase) % TABLE_LENGTH_F32;
        }
        sample
    }
}

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode {
    pub(crate) registration: AudioContextRegistration,
    pub(crate) channel_count: usize,
}

struct DestinationRenderer {}

impl AudioProcessor for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // todo, actually fill cpal buffer here
        *output = input.clone();
    }

    fn tail_time(&self) -> bool {
        unreachable!() // will never drop in control thread
    }
}

impl AudioNode for DestinationNode {
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
        1
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

impl DestinationNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, channel_count: usize) -> Self {
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
pub struct GainNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    gain: AudioParam,
}

impl AudioNode for GainNode {
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

impl GainNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: GainOptions) -> Self {
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

impl AudioProcessor for GainRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
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
    pub max_delay_time: f32,
    pub delay_time: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for DelayOptions {
    fn default() -> Self {
        Self {
            max_delay_time: 1.,
            delay_time: 0.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Node that delays the incoming audio signal by a certain amount
pub struct DelayNode {
    registration: AudioContextRegistration,
    delay_time: AudioParam,
    channel_config: ChannelConfig,
}

impl AudioNode for DelayNode {
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

impl DelayNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: DelayOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: 0.,
                max_value: options.max_delay_time,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());

            param.set_value_at_time(options.delay_time, 0.);

            // allocate large enough buffer to store all delayed samples
            let max_samples = options.max_delay_time * context.base().sample_rate().0 as f32;
            let max_quanta = (max_samples.ceil() as u32 + BUFFER_SIZE - 1) / BUFFER_SIZE;
            let delay_buffer = Vec::with_capacity(max_quanta as usize);

            let render = DelayRenderer {
                delay_time: proc,
                delay_buffer,
                index: 0,
            };

            let node = DelayNode {
                registration,
                channel_config: options.channel_config.into(),
                delay_time: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn delay_time(&self) -> &AudioParam {
        &self.delay_time
    }
}

struct DelayRenderer {
    delay_time: AudioParamId,
    delay_buffer: Vec<crate::alloc::AudioBuffer>,
    index: usize,
}

// SAFETY:
// AudioBuffers are not Send but we promise the `delay_buffer` Vec is emtpy before we ship it to
// the render thread.
unsafe impl Send for DelayRenderer {}

impl AudioProcessor for DelayRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // todo: a-rate processing
        let delay = params.get(&self.delay_time)[0];

        // calculate the delay in chunks of BUFFER_SIZE (todo: sub quantum delays)
        let quanta = (delay * sample_rate.0 as f32) as usize / BUFFER_SIZE as usize;

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
pub struct ChannelSplitterNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for ChannelSplitterNode {
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

impl ChannelSplitterNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, mut options: ChannelSplitterOptions) -> Self {
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

impl AudioProcessor for ChannelSplitterRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input node
        let input = &inputs[0];

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
pub struct ChannelMergerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for ChannelMergerNode {
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

impl ChannelMergerNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, mut options: ChannelMergerOptions) -> Self {
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

impl AudioProcessor for ChannelMergerRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
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
            let render = MediaStreamRenderer::new(resampler, Scheduler::new());

            (node, Box::new(render))
        })
    }
}

/// Options for constructing a MediaElementAudioSourceNode
pub struct MediaElementAudioSourceNodeOptions {
    pub media: MediaElement,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from a [`MediaElement`] (e.g. .ogg, .wav, .mp3 files)
///
/// The media element will take care of buffering of the stream so the render thread never blocks.
/// This also allows for playback controls (pause, looping, playback rate, etc.)
///
/// Note: do not forget to `start()` the node.
pub struct MediaElementAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl AudioScheduledSourceNode for MediaElementAudioSourceNode {
    fn scheduler(&self) -> &Scheduler {
        self.controller.scheduler()
    }
}
impl AudioControllableSourceNode for MediaElementAudioSourceNode {
    fn controller(&self) -> &Controller {
        &self.controller
    }
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
    pub fn new<C: AsBaseAudioContext>(
        context: &C,
        options: MediaElementAudioSourceNodeOptions,
    ) -> Self {
        context.base().register(move |registration| {
            let controller = options.media.controller().clone();
            let scheduler = controller.scheduler().clone();

            let node = MediaElementAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let resampler =
                Resampler::new(context.base().sample_rate(), BUFFER_SIZE, options.media);
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
///
/// Note: do not forget to `start()` the node.
pub struct AudioBufferSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl AudioScheduledSourceNode for AudioBufferSourceNode {
    fn scheduler(&self) -> &Scheduler {
        self.controller.scheduler()
    }
}
impl AudioControllableSourceNode for AudioBufferSourceNode {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl AudioNode for AudioBufferSourceNode {
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

impl AudioBufferSourceNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AudioBufferSourceNodeOptions) -> Self {
        context.base().register(move |registration| {
            // unwrap_or_default buffer
            let buffer = options
                .buffer
                .unwrap_or_else(|| AudioBuffer::new(1, BUFFER_SIZE as usize, SampleRate(44_100)));

            // wrap input in resampler
            let resampler = Resampler::new(
                context.base().sample_rate(),
                BUFFER_SIZE,
                std::iter::once(Ok(buffer)),
            );

            // wrap resampler in media-element (for loop/play/pause)
            let media = MediaElement::new(resampler);
            let controller = media.controller().clone();
            let scheduler = controller.scheduler().clone();

            // setup user facing audio node
            let node = AudioBufferSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let render = MediaStreamRenderer::new(media, scheduler);

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
pub struct ConstantSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    offset: AudioParam,
}

impl AudioNode for ConstantSourceNode {
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

impl ConstantSourceNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: ConstantSourceOptions) -> Self {
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

impl AudioProcessor for ConstantSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
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
pub struct PannerNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    position_x: AudioParam,
    position_y: AudioParam,
    position_z: AudioParam,
}

impl AudioNode for PannerNode {
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

impl PannerNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: PannerOptions) -> Self {
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

            context.base().connect_listener_to_panner(node.id());

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

impl AudioProcessor for PannerRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
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

/// Options for constructing an AnalyserNode
pub struct AnalyserOptions {
    pub fft_size: usize,
    pub smoothing_time_constant: f32,
    /*
    pub max_decibels: f32,
    pub min_decibels: f32,
    */
    pub channel_config: ChannelConfigOptions,
}

impl Default for AnalyserOptions {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            smoothing_time_constant: 0.8,
            /*
            max_decibels: -30.,
            min_decibels: 100.,
            */
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

enum AnalyserRequest {
    FloatTime {
        sender: Sender<Vec<f32>>,
        buffer: Vec<f32>,
    },
    FloatFrequency {
        sender: Sender<Vec<f32>>,
        buffer: Vec<f32>,
    },
}

/// Provides real-time frequency and time-domain analysis information
pub struct AnalyserNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    fft_size: Arc<AtomicUsize>,
    smoothing_time_constant: Arc<AtomicU32>,
    sender: Sender<AnalyserRequest>,
    /*
    max_decibels: f32,
    min_decibels: f32,
    */
}

impl AudioNode for AnalyserNode {
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

impl AnalyserNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AnalyserOptions) -> Self {
        context.base().register(move |registration| {
            let fft_size = Arc::new(AtomicUsize::new(options.fft_size));
            let smoothing_time_constant = Arc::new(AtomicU32::new(
                (options.smoothing_time_constant * 100.) as u32,
            ));

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let render = AnalyserRenderer {
                analyser: Analyser::new(options.fft_size),
                fft_size: fft_size.clone(),
                smoothing_time_constant: smoothing_time_constant.clone(),
                receiver,
            };

            let node = AnalyserNode {
                registration,
                channel_config: options.channel_config.into(),
                fft_size,
                smoothing_time_constant,
                sender,
            };

            (node, Box::new(render))
        })
    }

    /// Half the FFT size
    pub fn frequency_bin_count(&self) -> usize {
        self.fft_size.load(Ordering::SeqCst) / 2
    }

    /// The size of the FFT used for frequency-domain analysis (in sample-frames)
    pub fn fft_size(&self) -> usize {
        self.fft_size.load(Ordering::SeqCst)
    }

    /// This MUST be a power of two in the range 32 to 32768
    pub fn set_fft_size(&self, fft_size: usize) {
        // todo assert size
        self.fft_size.store(fft_size, Ordering::SeqCst);
    }

    /// Time averaging parameter with the last analysis frame.
    pub fn smoothing_time_constant(&self) -> f32 {
        self.smoothing_time_constant.load(Ordering::SeqCst) as f32 / 100.
    }

    /// Set smoothing time constant, this MUST be a value between 0 and 1
    pub fn set_smoothing_time_constant(&self, v: f32) {
        // todo assert range
        self.smoothing_time_constant
            .store((v * 100.) as u32, Ordering::SeqCst);
    }

    /// Copies the current time domain data (waveform data) into the provided buffer
    pub fn get_float_time_domain_data(&self, buffer: Vec<f32>) -> Vec<f32> {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let request = AnalyserRequest::FloatTime { sender, buffer };
        self.sender.send(request).unwrap();
        receiver.recv().unwrap()
    }

    /// Copies the current frequency data into the provided buffer
    pub fn get_float_frequency_data(&self, buffer: Vec<f32>) -> Vec<f32> {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let request = AnalyserRequest::FloatFrequency { sender, buffer };
        self.sender.send(request).unwrap();
        receiver.recv().unwrap()
    }
}

struct AnalyserRenderer {
    pub analyser: Analyser,
    pub fft_size: Arc<AtomicUsize>,
    pub smoothing_time_constant: Arc<AtomicU32>,
    pub receiver: Receiver<AnalyserRequest>,
}

// SAFETY:
// AudioBuffer is not Send, but the buffer Vec is empty when we move it to the render thread.
unsafe impl Send for AnalyserRenderer {}

impl AudioProcessor for AnalyserRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // pass through input
        *output = input.clone();

        // add current input to ring buffer
        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);
        let mono_data = mono.channel_data(0).clone();
        self.analyser.add_data(mono_data);

        // calculate frequency domain every `fft_size` samples
        let fft_size = self.fft_size.load(Ordering::Relaxed);
        let resized = self.analyser.current_fft_size() != fft_size;
        let complete_cycle = self.analyser.check_complete_cycle(fft_size);
        if resized || complete_cycle {
            let smoothing_time_constant =
                self.smoothing_time_constant.load(Ordering::Relaxed) as f32 / 100.;
            self.analyser
                .calculate_float_frequency(fft_size, smoothing_time_constant);
        }

        // check if any information was requested from the control thread
        if let Ok(request) = self.receiver.try_recv() {
            match request {
                AnalyserRequest::FloatTime { sender, mut buffer } => {
                    self.analyser.get_float_time(&mut buffer[..], fft_size);

                    // allow to fail when receiver is disconnected
                    let _ = sender.send(buffer);
                }
                AnalyserRequest::FloatFrequency { sender, mut buffer } => {
                    self.analyser.get_float_frequency(&mut buffer[..]);

                    // allow to fail when receiver is disconnected
                    let _ = sender.send(buffer);
                }
            }
        }
    }

    fn tail_time(&self) -> bool {
        false
    }
}
