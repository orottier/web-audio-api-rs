//! The oscillator control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]
use crossbeam_channel::{self, Receiver, Sender};
use std::f32::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::control::Scheduler;
use crate::param::{AudioParam, AudioParamOptions, AutomationRate};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{
    AudioNode, AudioScheduledSourceNode, ChannelConfig, ChannelConfigOptions, SINETABLE,
    TABLE_LENGTH_F32, TABLE_LENGTH_USIZE,
};

/// Options for constructing a periodic wave
pub struct PeriodicWaveOptions {
    /// The real parameter represents an array of cosine terms of Fourrier series.
    ///
    /// The first element (index 0) represents the DC-offset.
    /// This offset has to be given but will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    pub real: Option<Vec<f32>>,
    /// The imag parameter represents an array of sine terms of Fourrier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    pub imag: Option<Vec<f32>>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    pub disable_normalization: Option<bool>,
}

/// `PeriodicWave` is a setup struct required to build
/// custom periodic waveform oscillator type.
#[derive(Debug, Clone)]
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
    /// Returns a `PeriodicWave`
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourrier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourrier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the `PeriodicWave`
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * `real` is defined and its length is less than 2
    /// * `imag` is defined and its length is less than 2
    /// * `real` and `imag` are defined and theirs lengths are not equal
    /// * `PeriodicWave` is more than 8192 components
    ///
    /// # Example
    ///
    /// ```no_run
    ///    use web_audio_api::context::{AudioContext, AsBaseAudioContext};
    ///    use web_audio_api::node::{PeriodicWave, PeriodicWaveOptions};
    ///
    ///    let context = AudioContext::new(None);
    ///
    ///    let options = PeriodicWaveOptions {
    ///    real: Some(vec![0.,1.,1.]),
    ///    imag: Some(vec![0.,0.,0.]),
    ///    disable_normalization: Some(false),
    ///    };
    ///
    ///    let periodic_wave = PeriodicWave::new(&context, Some(options));
    /// ```
    ///
    pub fn new<C: AsBaseAudioContext>(_context: &C, options: Option<PeriodicWaveOptions>) -> Self {
        if let Some(PeriodicWaveOptions {
            real,
            imag,
            disable_normalization,
        }) = options
        {
            let (real, imag) = match (real, imag) {
                (Some(r), Some(i)) => {
                    assert!(
                        r.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        i.len() >= 2,
                        "RangeError: Imag field length should be at least 2",
                    );
                    assert!(
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        r.len() <= 8192,
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    assert!(
                        r.len() == i.len(),
                        "RangeError: Imag and real field length should be equal"
                    );
                    (r, i)
                }
                (Some(r), None) => {
                    assert!(
                        r.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        r.len() <= 8192,
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    let r_len = r.len();
                    (r, vec![0.; r_len])
                }
                (None, Some(i)) => {
                    assert!(
                        i.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        i.len() <= 8192,
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    let i_len = i.len();
                    (vec![0.; i_len], i)
                }
                _ => (vec![0.0, 1.0], vec![0., 0.]),
            };

            Self {
                real,
                imag,
                disable_normalization: disable_normalization.unwrap_or(false),
            }
        } else {
            Self {
                real: vec![0., 1.],
                imag: vec![0., 0.],
                disable_normalization: false,
            }
        }
    }
}

/// Generate the wavetable
///
/// # Arguments
///
/// * `wavetable` - placeholder for generated wavetable data
/// * `phases` - the phase of each harmonics
/// * `norms` - the norm of each harmonics
/// * `incr_phases` - the phase to increment of each harmonics
/// * `interpol_ratios` - the interpolation ratio of each harmonics used by linear interpolatio
#[inline]
fn generate_wavetable(
    wavetable: &mut Vec<f32>,
    phases: &mut [f32],
    norms: &[f32],
    incr_phases: &[f32],
    interpol_ratios: &[f32],
) {
    wavetable.clear();

    while phases[1] <= TABLE_LENGTH_F32 {
        let mut sample = 0.0;
        for i in 1..phases.len() { // the phase offset problem might come from...
            let gain = norms[i];
            let phase = phases[i];
            let incr_phase = incr_phases[i];
            let mu = interpol_ratios[i];
            // truncation is desired
            #[allow(clippy::cast_possible_truncation)]
            // phase + incr_phase is always positive
            #[allow(clippy::cast_sign_loss)]
            let idx = (phase + incr_phase) as usize;
            let inf_idx = idx % TABLE_LENGTH_USIZE;
            let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
            // Linear interpolation
            sample += SINETABLE[inf_idx].mul_add(1. - mu, SINETABLE[sup_idx] * mu) * gain;
            phases[i] = phase + incr_phase;
        }

        wavetable.push(sample);
    }

    println!("length: {:?}", wavetable.len());
    println!("content: {:?}", wavetable);
}

pub(crate) fn generate_wavetable_test(reals: &[f32], imags: &[f32], size: usize, normalize: bool) {
    let mut output = Vec::with_capacity(size);
    let pi_2 = 2. * PI;

    for i in 0..size {
        let mut sample = 0.;
        let phase = pi_2 * i as f32 / size as f32;

        for j in 1..reals.len() {
            let freq = j as f32;
            let real = reals[j];
            let imag = imags[j];
            let rad = phase * freq;
            let contrib = real * rad.cos() + imag * rad.sin();
            sample += contrib;
        }

        output.push(sample);
    }

    println!("-------------------------------------------------");
    println!("{:?}", output);
    println!("-------------------------------------------------");

    if normalize {
        let norm = norm_factor(&output);

        for sample in output.iter_mut() {
            *sample *= norm;
        }
    }

    println!("-------------------------------------------------");
    println!("{:?}", output);
    println!("-------------------------------------------------");
}

/// Compute the normalization factor
///
/// The normalization factor is applied as a gain to the periodic wave
/// to normalize the signal peak amplitude in the interval [-1.0,1.0].
///
/// # Arguments
///
/// * `buffer` - the wavetable generated from periodic wave charateristics
#[inline]
fn norm_factor(buffer: &[f32]) -> f32 {
    let mut max = 0.;

    for sample in buffer.iter() {
        let abs = sample.abs();
        if abs > max {
            max = abs;
        }
    }

    1. / max
}

/// Options for constructing an `OscillatorNode`
#[derive(Debug)]
pub struct OscillatorOptions {
    /// The shape of the periodic waveform
    pub type_: Option<OscillatorType>,
    /// The frequency of the fundamental frequency.
    pub frequency: Option<f32>,
    /// A detuning value (in cents) which will offset the frequency by the given amount.
    pub detune: Option<f32>,
    /// channel config options
    pub channel_config: Option<ChannelConfigOptions>,
    /// The PeriodicWave for the OscillatorNode
    /// If this is specified, then any valid value for type is ignored;
    /// it is treated as if "custom" were specified.
    pub periodic_wave: Option<PeriodicWave>,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: Some(OscillatorType::default()),
            frequency: Some(440.),
            detune: Some(0.),
            channel_config: Some(ChannelConfigOptions::default()),
            periodic_wave: None,
        }
    }
}

/// Waveform of an oscillator
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum OscillatorType {
    /// Sine wave
    Sine,
    /// Square wave
    Square,
    /// Sawtooth wave
    Sawtooth,
    /// Triangle wave
    Triangle,
    /// type used when periodic_wave is specified
    Custom,
}

impl Default for OscillatorType {
    fn default() -> Self {
        Self::Sine
    }
}

impl From<u32> for OscillatorType {
    fn from(i: u32) -> Self {
        use OscillatorType::{Custom, Sawtooth, Sine, Square, Triangle};

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

/// Message types used to communicate between [`OscillatorNode`] and [`OscillatorRenderer`]
enum OscMsg {
    /// represents all data required to build a periodic wave processing
    PeriodicWaveMsg {
        /// `computed_freq` is computed from `frequency` and `detune`
        computed_freq: f32,
        /// wavetable computed at runtime and following periodic wave charateristics
        dyn_wavetable: Vec<f32>,
        /// Peak normalization factor to apply to output
        /// if `disable_normalization` is set to false
        norm_factor: f32,
        /// if set to false, normalization factor is applied
        disable_normalization: bool,
    },
}

/// Audio source generating a periodic waveform
pub struct OscillatorNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// The frequency of the fundamental frequency.
    frequency: AudioParam,
    /// A detuning value (in cents) which will offset the frequency by the given amount.
    detune: AudioParam,
    /// Waveform of an oscillator
    type_: Arc<AtomicU32>,
    /// starts and stops Oscillator audio streams
    scheduler: Scheduler,
    /// channel between control and renderer parts (sender part)
    sender: Sender<OscMsg>,
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

    /// `OscillatorNode` is a source node. A source node is by definition with no input
    fn number_of_inputs(&self) -> u32 {
        0
    }

    /// `OscillatorNode` is a mono source node.
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl OscillatorNode {
    /// Returns an `OscillatorNode`
    ///
    /// # Arguments:
    ///
    /// * `context` - The `AudioContext`
    /// * `options` - The OscillatorOptions
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<OscillatorOptions>) -> Self {
        context.base().register(move |registration| {
            let sample_rate = context.sample_rate();
            let nyquist = sample_rate / 2.;
            let default_freq = 440.;
            let default_det = 0.;

            let OscillatorOptions {
                type_,
                frequency,
                detune,
                channel_config,
                periodic_wave,
            } = options.unwrap_or_default();

            // frequency audio parameter
            let freq_param_opts = AudioParamOptions {
                min_value: -nyquist,
                max_value: nyquist,
                default_value: default_freq,
                automation_rate: AutomationRate::A,
            };
            let (f_param, f_proc) = context
                .base()
                .create_audio_param(freq_param_opts, registration.id());
            f_param.set_value(frequency.unwrap_or(default_freq));

            // detune audio parameter
            let det_param_opts = AudioParamOptions {
                min_value: -153_600.,
                max_value: 153_600.,
                default_value: default_det,
                automation_rate: AutomationRate::A,
            };
            let (det_param, det_proc) = context
                .base()
                .create_audio_param(det_param_opts, registration.id());
            det_param.set_value(detune.unwrap_or(default_det));

            // if Periodic wave is defined, the oscillator type is custom
            // and options.type is ignored (following the specs)
            let type_ = if periodic_wave.is_some() {
                Arc::new(AtomicU32::new(OscillatorType::Custom as u32))
            } else {
                Arc::new(AtomicU32::new(type_.unwrap_or(OscillatorType::Sine) as u32))
            };

            let scheduler = Scheduler::new();
            let (sender, receiver) = crossbeam_channel::bounded(1);
            let computed_freq = default_freq * (default_det / 1200.).exp2();

            let config = OscRendererConfig {
                type_: type_.clone(),
                frequency: f_proc,
                detune: det_proc,
                scheduler: scheduler.clone(),
                receiver,
                computed_freq,
                sample_rate: context.sample_rate(),
                periodic_wave,
            };
            let renderer = OscillatorRenderer::new(config);

            let node = Self {
                registration,
                channel_config: channel_config.unwrap_or_default().into(),
                frequency: f_param,
                detune: det_param,
                type_,
                scheduler,
                sender,
            };

            (node, Box::new(renderer))
        })
    }

    /// Returns the frequency audio parameter
    /// The oscillator frequency is calculated as follow:
    /// frequency * 2^(detune/1200)
    #[must_use]
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the detune audio parameter. detune unity is cents.
    /// The oscillator frequency is calculated as follow:
    /// frequency * 2^(detune/1200)
    #[must_use]
    pub fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Returns the oscillator type
    #[must_use]
    pub fn type_(&self) -> OscillatorType {
        self.type_.load(Ordering::SeqCst).into()
    }

    /// set the oscillator type
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,square,triangle,sawtooth, and custom)
    ///
    /// # Panics
    ///
    /// if `type_` is `OscillatorType::Custom`
    pub fn set_type(&self, type_: OscillatorType) {
        assert_ne!(
            type_,
            OscillatorType::Custom,
            "InvalidStateError: Custom type cannot be set manually"
        );

        // if periodic wave is specified, type_ changes are should be ignored
        // by specs definition
        if self.type_.load(Ordering::SeqCst) == OscillatorType::Custom as u32 {
            return;
        }

        self.type_.store(type_ as u32, Ordering::SeqCst);
    }

    /// set the oscillator type to custom and generate
    /// a periodic waveform following the `PeriodicWave` characteristics
    pub fn set_periodic_wave(&self, periodic_wave: PeriodicWave) {
        self.type_
            .store(OscillatorType::Custom as u32, Ordering::SeqCst);

        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = periodic_wave;

        let mut cplxs = Vec::with_capacity(real.len());
        let mut norms = Vec::with_capacity(real.len());
        let mut phases = Vec::with_capacity(real.len());
        let mut incr_phases = Vec::with_capacity(real.len());
        let mut interpol_ratios = Vec::with_capacity(real.len());

        // let result = generate_wavetable_test(&real, &imag, 2048);

        // update cplxs
        for cplx in real.into_iter().zip(imag) {
            cplxs.push(cplx);
        }

        let frequency = self.frequency().value();
        let detune = self.detune().value();
        let computed_freq = frequency * (detune / 1200.).exp2();

        for (idx, (real, img)) in cplxs.iter().enumerate() {
            // update norms
            norms.push((f32::powi(*real, 2_i32) + f32::powi(*img, 2_i32)).sqrt());

            // update phases
            let phase = f32::atan2(*img, *real);
            if phase < 0. {
                phases.push(2.0_f32.mul_add(PI, phase) * (TABLE_LENGTH_F32 / (2.0 * PI)));
            } else {
                phases.push(phase * (TABLE_LENGTH_F32 / (2.0 * PI)));
            }

            // update incr_phases
            // 0 through max value 8192 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            incr_phases.push(
                TABLE_LENGTH_F32 * idx as f32 * (computed_freq / self.context().sample_rate()),
            );
        }

        // update interpol_ratios
        for incr_phase in &incr_phases {
            interpol_ratios.push((incr_phase - incr_phase.round()).abs());
        }

        // generate the wavetable following periodic wave characteristics
        let mut dyn_wavetable = Vec::with_capacity(TABLE_LENGTH_USIZE);
        generate_wavetable(
            &mut dyn_wavetable,
            &mut phases,
            &norms,
            &incr_phases,
            &interpol_ratios,
        );

        // update norm_factor
        let norm_factor = norm_factor(&dyn_wavetable);

        self.sender
            .send(OscMsg::PeriodicWaveMsg {
                computed_freq,
                dyn_wavetable,
                norm_factor,
                disable_normalization,
            })
            .expect("Sending periodic wave to the node renderer failed");
    }
}

/// Helper struct which regroups all parameters
/// required to build `OscillatorRenderer`
struct OscRendererConfig {
    /// The shape of the periodic waveform
    type_: Arc<AtomicU32>,
    /// The frequency of the fundamental frequency.
    frequency: AudioParamId,
    /// A detuning value (in cents) which will offset the frequency by the given amount.
    detune: AudioParamId,
    /// starts and stops oscillator audio streams
    scheduler: Scheduler,
    /// channel between control and renderer parts (receiver part)
    receiver: Receiver<OscMsg>,
    /// `computed_freq` is precomputed from `frequency` and `detune`
    computed_freq: f32,
    /// sample rate at which the processor should render
    sample_rate: f32,
    /// The PeriodicWave for the OscillatorNode
    /// If this is specified, then any valid value for type is ignored;
    /// it is treated as if "custom" were specified.
    periodic_wave: Option<PeriodicWave>,
}

/// States relative to Sine `OscillatorType`
// struct SineState {
//     /// linear interpolation ratio
//     interpol_ratio: f32,
//     /// if set to true, requires sine parameters to be initialized
//     needs_init: bool,
// }

/// States relative to Triangle `OscillatorType`
// struct TriangleState {
//     /// this memory state is used in the leaky integrator
//     last_output: f32,
// }

/// States relative to Custom `OscillatorType`
struct PeriodicState {
    /// incremental phase for each harmonics
    incr_phases: Vec<f32>,
    /// linear interpolation ratio for each harmonics
    interpol_ratios: Vec<f32>,
    /// Peak normalization factor
    norm_factor: Option<f32>,
    /// if set to false, apply `norm_factor`
    disable_normalization: bool,
    /// States required to generate the periodic wavetable
    wavetable: WavetableState,
}

/// States required to generate the periodic wavetable
struct WavetableState {
    /// dynamic wavetable
    /// computes each time periodic wave paramaters change
    dyn_table: Vec<f32>,
    /// current phase of the oscillator
    phase: f32,
    /// phase amount to add to the oscillator phase
    /// which generates an output at `computed_freq` frequency
    incr_phase: f32,
    /// frequency for which `WavetableState` has been computed
    ref_freq: f32,
}

/// Rendering component of the oscillator node
struct OscillatorRenderer {
    /// The shape of the periodic waveform
    type_: Arc<AtomicU32>,
    /// The frequency of the fundamental frequency.
    frequency: AudioParamId,
    /// A detuning value (in cents) which will offset the frequency by the given amount.
    detune: AudioParamId,
    /// buffer for holding compound param
    // computed_freqs: Vec<f32>,
    /// starts and stops oscillator audio streams
    scheduler: Scheduler,
    /// channel between control and renderer parts (receiver part)
    receiver: Receiver<OscMsg>,
    /// sample rate at which the processor should render
    sample_rate: f32,
    /// `computed_freq` is precomputed from `frequency` and `detune`
    // computed_freq: f32,
    /// current phase of the oscillator
    phase: f64, // phase concerns time, we must be accurate here
    /// phase amount to add to phase at each tick
    // incr_phase: f64,
    started: bool,
    /// states required to build a sine wave
    // sine: SineState,
    /// states required to build a triangle wave
    // triangle: TriangleState,
    /// states required to build a custom oscillator
    periodic: PeriodicState,
}

impl AudioProcessor for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];
        // 1 channel output
        output.set_number_of_channels(1);

        // check if any message was send from the control thread
        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                OscMsg::PeriodicWaveMsg {
                    computed_freq,
                    dyn_wavetable,
                    norm_factor,
                    disable_normalization,
                } => {
                    self.periodic.wavetable.ref_freq = computed_freq;
                    self.periodic.wavetable.dyn_table = dyn_wavetable;
                    self.periodic.norm_factor = Some(norm_factor);
                    self.periodic.disable_normalization = disable_normalization;
                }
            }
        }

        let dt = 1. / sample_rate.0 as f64;
        let num_frames = RENDER_QUANTUM_SIZE;
        let next_block_time = timestamp + dt * num_frames as f64;

        let start_time = self.scheduler.get_start_at();
        let stop_time = self.scheduler.get_stop_at();

        if start_time >= next_block_time {
            output.make_silent();
            return true;
        } else if stop_time < timestamp {
            output.make_silent();
            return false;
        }

        let type_ = self.type_.load(Ordering::SeqCst).into();
        let channel_data = output.channel_data_mut(0);
        let frequency_values = params.get(&self.frequency);
        let detune_values = params.get(&self.detune);
        let sample_rate = sample_rate.0 as f32;

        let mut current_time = timestamp;

        for (index, output_sample) in channel_data.iter_mut().enumerate() {
            if current_time < start_time || current_time >= stop_time {
                *output_sample = 0.;
                current_time += dt;

                continue;
            }

            let frequency = frequency_values[index];
            let detune = detune_values[index];
            let computed_frequency = frequency * (detune / 1200.).exp2();

            // first sample
            if !self.started {
                // if sstart time was between last frame and current frame
                // we need to adjust the phase
                if current_time > start_time {
                    let offset = (current_time - start_time) / dt;
                    let phase_incr = computed_frequency as f64 / sample_rate as f64;
                    self.phase = offset * phase_incr as f64;
                }

                self.started = true;
            }

            let phase_incr = computed_frequency as f64 / sample_rate as f64;

            // none of this is used in the generation...
            // if (self.type_ === OscillatorType::Custom) {
            //     for incr_phase in &mut self.periodic.incr_phases {
            //         *incr_phase *= new_comp_freq / self.computed_freq;
            //     }

            //     for (r, incr_ph) in self
            //         .periodic
            //         .interpol_ratios
            //         .iter_mut()
            //         .zip(self.periodic.incr_phases.iter())
            //     {
            //         *r = incr_ph - incr_ph.floor();
            //     }

            //     self.periodic.wavetable.incr_phase =
            //         computed_frequency / self.periodic.wavetable.ref_freq;
            // }

            *output_sample = match type_ {
                OscillatorType::Sine => self.generate_sine(),
                OscillatorType::Sawtooth => self.generate_sawtooth(phase_incr),
                OscillatorType::Square => self.generate_square(phase_incr),
                OscillatorType::Triangle => self.generate_triangle(),
                OscillatorType::Custom => self.generate_custom(),
                _ => 0.,
            };

            current_time += dt;
            self.phase = Self::unroll_phase(self.phase + phase_incr);
        }

        true
    }
}

impl OscillatorRenderer {
    /// Creates an `OscillatorRenderer`
    ///
    /// # Arguments
    ///
    /// * `context` - Audio context in which the node will live
    /// * `options` - node options
    fn new(config: OscRendererConfig) -> Self {
        let OscRendererConfig {
            type_,
            frequency,
            detune,
            scheduler,
            receiver,
            computed_freq,
            sample_rate,
            periodic_wave,
        } = config;

        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = periodic_wave.map_or_else(
            || PeriodicWave {
                real: vec![0., 1.], // this is a cosine
                imag: vec![0., 0.], // [0., 1.] should be there
                disable_normalization: false,
            },
            |p_w| p_w,
        );

        let cplxs: Vec<(f32, f32)> = real.iter().zip(&imag).map(|(&r, &i)| (r, i)).collect();

        let norms: Vec<f32> = cplxs
            .iter()
            .map(|(r, i)| (f32::powi(*r, 2_i32) + f32::powi(*i, 2_i32)).sqrt())
            .collect();

        let mut phases: Vec<f32> = cplxs
            .iter()
            .map(|(r, i)| {
                let phase = f32::atan2(*i, *r);
                if phase < 0. {
                    2.0_f32.mul_add(PI, phase) * (TABLE_LENGTH_F32 / (2.0 * PI))
                } else {
                    phase * (TABLE_LENGTH_F32 / (2.0 * PI))
                }
            })
            .collect();

        // 0 through max value 8192 casts without loss of precision
        #[allow(clippy::cast_precision_loss)]
        let incr_phases: Vec<f32> = cplxs
            .iter()
            .enumerate()
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (computed_freq / sample_rate))
            .collect();

        let interpol_ratios: Vec<f32> = incr_phases
            .iter()
            .map(|incr_phase| incr_phase - incr_phase.floor())
            .collect();

        let mut periodic_wavetable = Vec::with_capacity(TABLE_LENGTH_USIZE);
        generate_wavetable(
            &mut periodic_wavetable,
            &mut phases,
            &norms,
            &incr_phases,
            &interpol_ratios,
        );

        let norm_factor = if disable_normalization {
            None
        } else {
            let norm_factor = norm_factor(&periodic_wavetable);
            Some(norm_factor)
        };

        let mut computed_freqs = Vec::with_capacity(RENDER_QUANTUM_SIZE);
        computed_freqs.resize(RENDER_QUANTUM_SIZE, 0.);

        Self {
            type_,
            frequency,
            detune,
            // computed_freqs,
            scheduler,
            receiver,
            // computed_freq,
            sample_rate,
            phase: 0.,
            // incr_phase: 0.,
            started: false,
            // sine: SineState {
            //     interpol_ratio, // @todo - remove
            //     needs_init: true,
            // },
            // triangle: TriangleState { last_output: 0.0 },
            periodic: PeriodicState {
                incr_phases,
                interpol_ratios,
                norm_factor,
                disable_normalization,
                wavetable: WavetableState {
                    dyn_table: periodic_wavetable,
                    phase: 0.,
                    incr_phase: 1.,
                    ref_freq: computed_freq,
                },
            },
        }
    }

    #[inline]
    fn generate_sine(&mut self) -> f32 {
        let position = self.phase * TABLE_LENGTH_USIZE as f64;
        let floored = position.floor();

        let prev_index = floored as usize;
        let mut next_index = prev_index + 1;
        if next_index == TABLE_LENGTH_USIZE {
            next_index = 0;
        }

        let k = (position - floored) as f32;
        // linear interpolation
        let sample = SINETABLE[prev_index].mul_add(
            1. - k,
            SINETABLE[next_index] * k,
        );

        sample
    }

    #[inline]
    fn generate_sawtooth(&mut self, phase_incr: f64) -> f32 {
        // offset phase to start at 0. (not -1.)
        let phase = Self::unroll_phase(self.phase + 0.5);
        let mut sample = 2.0 * phase - 1.0;
        sample -= Self::poly_blep(phase, phase_incr);

        sample as f32
    }

    #[inline]
    fn generate_square(&mut self, phase_incr: f64) -> f32 {
        let mut sample = if self.phase < 0.5 { 1.0 } else { -1.0 };
        sample += Self::poly_blep(self.phase, phase_incr);

        let shift_phase = Self::unroll_phase(self.phase + 0.5);
        sample -= Self::poly_blep(shift_phase, phase_incr);

        sample as f32
    }

    #[inline]
    fn generate_triangle(&mut self) -> f32 {
        let mut sample = -4. * self.phase + 2.;

        if sample > 1. {
            sample = 2. - sample;
        } else if sample < -1. {
            sample = -2. - sample;
        }

        sample as f32
    }

    /// generate the audio data when oscillator is of type custom
    /// buffer is filled with the periodic waveform audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_custom(&mut self) -> f32 {
        // TABLE_LENGTH_F32
        // TABLE_LENGTH_USIZE
        // // that's basically a table lookup ??
        //     let phase = self.periodic.wavetable.phase;
        //     let incr_phase = self.periodic.wavetable.incr_phase;
        //     let table_len = self.periodic.wavetable.dyn_table.len();

        //     // 2048 casts without loss of precision
        //     #[allow(clippy::cast_precision_loss)]
        //     let table_len_f32 = table_len as f32;
        //     let buffer = &self.periodic.wavetable.dyn_table;
        //     // truncation is desired
        //     #[allow(clippy::cast_possible_truncation)]
        //     // phase is always positive
        //     #[allow(clippy::cast_sign_loss)]
        //     let inf_idx = phase as usize;
        //     let sup_idx = (inf_idx + 1) % table_len;
        //     let interpol_ratio = phase - phase.trunc();

        //     *o = (1.0 - interpol_ratio).mul_add(buffer[inf_idx], interpol_ratio * buffer[sup_idx]);

        //     // Update phase with optimized float modulo op
        //     self.periodic.wavetable.phase = if phase + incr_phase >= table_len_f32 {
        //         (phase + incr_phase) - table_len_f32
        //     } else {
        //         phase + incr_phase
        //     };

        // println!("> {:?}", );
        let len = self.periodic.wavetable.dyn_table.len();

        let position = self.phase * len as f64;
        let floored = position.floor();

        let prev_index = floored as usize;
        let mut next_index = prev_index + 1;
        if next_index == len {
            next_index = 0;
        }

        let k = (position - floored) as f32;
        let lookup_table = &self.periodic.wavetable.dyn_table;
        let sample = lookup_table[prev_index].mul_add(
            1. - k,
            lookup_table[prev_index] * k
        );

        sample
    }


    #[inline]
    fn unroll_phase(mut phase: f64) -> f64 {
        if phase >= 1. {
            phase -= 1.
        }

        phase
    }

    // computes the `polyBLEP` corrections to apply to aliasing signal
    // `polyBLEP` stands for `polyBandLimitedstEP`
    // This basically soften the sharp edges in square and sawtooth signals
    // to avoid infinite frequencies impulses (jumps from -1 to 1 or inverse).
    // cf. http://www.martin-finke.de/blog/articles/audio-plugins-018-polyblep-oscillator/
    //
    // @note: do not apply in tests so we can avoid relying on snapshots
    #[inline]
    fn poly_blep(mut t: f64, dt: f64) -> f64 {
        if cfg!(test) {
            0.
        } else {
            let res = if t < dt {
                t /= dt;
                t + t - t * t - 1.0
            } else if t > 1.0 - dt {
                t = (t - 1.0) / dt;
                t.mul_add(t, t) + t + 1.0
            } else {
                0.0
            };

            res
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::f64::consts::PI;

    use crate::context::{AsBaseAudioContext, AudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode};
    use crate::{snapshot, SampleRate};

    use super::{OscillatorNode, OscillatorOptions, OscillatorType, PeriodicWave, PeriodicWaveOptions};

    // keep that around this is usefull to write data into files
    // use std::fs::File;
    // use std::io::Write;

    // let mut file = File::create("_signal-expected.txt").unwrap();
    // for i in expected.iter() {
    //     let mut tmp = String::from(i.to_string());
    //     tmp += ",\n";
    //     file.write(tmp.to_string().as_bytes()).unwrap();
    // }

    // let mut file = File::create("_signal-result.txt").unwrap();
    // for i in result.iter() {
    //     let mut tmp = String::from(i.to_string());
    //     tmp += ",\n";
    //     file.write(tmp.to_string().as_bytes()).unwrap();
    // }

    const LENGTH: usize = 555;

    #[test]
    #[should_panic]
    fn fails_to_build_when_real_is_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: Some(vec![0., 0., 0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_real_is_defined_and_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: None,
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_is_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_imag_is_defined_and_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_not_equal_length() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0., 0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_are_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.; 8193]),
            imag: Some(vec![0.; 8193]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_real_is_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.; 8193]),
            imag: None,
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_is_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.; 8193]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    fn assert_default_periodic_options() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: None,
            imag: None,
            disable_normalization: None,
        };

        let periodic_wave = PeriodicWave::new(&context, Some(options));

        // the default has to be a sine signal
        assert_float_eq!(periodic_wave.real, vec![0., 1.], abs_all <= 0.);
        assert_float_eq!(periodic_wave.imag, vec![0., 0.], abs_all <= 0.);
        assert!(!periodic_wave.disable_normalization);
    }

    #[test]
    fn assert_osc_default_build_with_factory_func() {
        let default_freq = 440.;
        let default_det = 0.;
        let default_type = OscillatorType::Sine;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let osc = context.create_oscillator();

        let freq = osc.frequency.value();
        assert_float_eq!(freq, default_freq, abs_all <= 0.);

        let det = osc.detune.value();
        assert_float_eq!(det, default_det, abs_all <= 0.);

        let type_ = osc.type_.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(type_, default_type as u32);
    }

    #[test]
    fn assert_osc_default_build() {
        let default_freq = 440.;
        let default_det = 0.;
        let default_type = OscillatorType::Sine;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let osc = OscillatorNode::new(&context, None);

        let freq = osc.frequency.value();
        assert_float_eq!(freq, default_freq, abs_all <= 0.);

        let det = osc.detune.value();
        assert_float_eq!(det, default_det, abs_all <= 0.);

        let type_ = osc.type_.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(type_, default_type as u32);
    }

    #[test]
    #[should_panic]
    fn set_type_to_custom_should_panic() {
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Custom);
    }

    #[test]
    fn type_is_custom_when_periodic_wave_is_some() {
        let expected_type = OscillatorType::Custom;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let periodic_opt = PeriodicWaveOptions {
            real: None,
            imag: None,
            disable_normalization: None,
        };

        let periodic_wave = PeriodicWave::new(&context, Some(periodic_opt));

        let options = OscillatorOptions {
            periodic_wave: Some(periodic_wave),
            ..OscillatorOptions::default()
        };

        let osc = OscillatorNode::new(&context, Some(options));

        let type_ = osc.type_.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(type_, expected_type as u32);
    }

    #[test]
    fn set_type_is_ignored_when_periodic_wave_is_some() {
        let expected_type = OscillatorType::Custom;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));

        let periodic_opt = PeriodicWaveOptions {
            real: None,
            imag: None,
            disable_normalization: None,
        };

        let periodic_wave = PeriodicWave::new(&context, Some(periodic_opt));

        let options = OscillatorOptions {
            periodic_wave: Some(periodic_wave),
            ..OscillatorOptions::default()
        };

        let osc = OscillatorNode::new(&context, Some(options));

        osc.set_type(OscillatorType::Sine);

        let type_ = osc.type_.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(type_, expected_type as u32);
    }

    #[test]
    fn periodic_wave_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/periodic_2f.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let options = Some(PeriodicWaveOptions {
            real: Some(vec![0., 0.5, 0.5]),
            imag: Some(vec![0., 0., 0.]), //
            disable_normalization: Some(false),
        });

        // Create a custom periodic wave
        let periodic_wave = context.create_periodic_wave(options);

        let options = OscillatorOptions {
            periodic_wave: Some(periodic_wave),
            ..OscillatorOptions::default()
        };

        let osc = OscillatorNode::new(&context, Some(options));

        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.get_channel_data(0)[..],
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
        assert_float_eq!(
            output.get_channel_data(1)[..],
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
    }

    #[test]
    fn default_periodic_wave_rendering_should_match_snapshot() {
        let ref_sine = snapshot::read("./snapshots/default_periodic.json")
            .expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
        let options = Some(PeriodicWaveOptions {
            real: None,
            imag: None,
            disable_normalization: Some(false),
        });

        // Create a custom periodic wave
        let periodic_wave = context.create_periodic_wave(options);

        let options = OscillatorOptions {
            periodic_wave: Some(periodic_wave),
            ..OscillatorOptions::default()
        };

        let osc = OscillatorNode::new(&context, Some(options));

        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.get_channel_data(0)[..],
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
        assert_float_eq!(
            output.get_channel_data(1)[..],
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
    }

    #[test]
    fn sine_raw() {
         // 1, 10, 100, 1_000, 10_000 Hz
        for i in 0..5 {
            let freq = 10_f32.powf(i as f32);
            let sample_rate = 44_100;

            let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));

            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.frequency().set_value(freq);
            osc.start_at(0.);

            let output = context.start_rendering();
            let result = output.get_channel_data(0);

            let mut expected = Vec::<f32>::with_capacity(sample_rate);
            let mut phase: f64 = 0.;
            let phase_incr = freq as f64 / sample_rate as f64;

            for _i in 0..sample_rate {
                let sample = (phase * 2. * PI).sin();

                expected.push(sample as f32);

                phase += phase_incr;
                if phase >= 1. { phase -= 1.; }
            }

            assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
        }
    }

     #[test]
    fn sine_raw_exact_phase() {
        // 1, 10, 100, 1_000, 10_000 Hz
        for i in 0..5 {
            let freq = 10_f32.powf(i as f32);
            let sample_rate = 44_100;

            let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));

            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.frequency().set_value(freq);
            osc.start_at(0.);

            let output = context.start_rendering();
            let result = output.get_channel_data(0);
            let mut expected = Vec::<f32>::with_capacity(sample_rate);

            for i in 0..sample_rate {
                let phase =  freq as f64 * i as f64 / sample_rate as f64;
                let sample = (phase * 2. * PI).sin();
                // phase += phase_incr;
                expected.push(sample as f32);
            }

            assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
        }
    }

    // ## Notes:
    //
    // - PolyBlep is not applied on `square` and `triangle` for tests, so we can
    //   compared according to a crude synthesis
    // - for `square`, `triangle` and `sawtooth` the tests may appear a bit
    //   tautological (and they are) as the code from the test is the mostly as same
    //   as in the renderer, just written in a more compact way. However they
    //   should help to prevent regression, and/or allow testing against some trusted
    //   and simple implementation in case of future changes in the renderer
    //   (for performance improvements or whatever).
    #[test]
    fn square_raw() {
        // 1, 10, 100, 1_000, 10_000 Hz
        for i in 0..5 {
            let freq = 10_f32.powf(i as f32);
            let sample_rate = 44100;

            let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));

            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.frequency().set_value(freq);
            osc.set_type(OscillatorType::Square);
            osc.start_at(0.);

            let output = context.start_rendering();
            let result = output.get_channel_data(0);

            let mut expected = Vec::<f32>::with_capacity(sample_rate);
            let mut phase: f64 = 0.;
            let phase_incr = freq as f64 / sample_rate as f64;

            for _i in 0..sample_rate {
                // 0.5 belongs to the second half of the waveform
                let sample = if phase < 0.5 { 1. } else { -1. };

                expected.push(sample as f32);

                phase += phase_incr;
                if phase >= 1. { phase -= 1.; }
            }

            assert_float_eq!(result[..], expected[..], abs_all <= 1e-10);
        }
    }

    #[test]
    fn triangle_raw() {
        // 1, 10, 100, 1_000, 10_000 Hz
        for i in 0..5 {
            let freq = 10_f32.powf(i as f32);
            let sample_rate = 44_100;

            let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));

            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.frequency().set_value(freq);
            osc.set_type(OscillatorType::Triangle);
            osc.start_at(0.);

            let output = context.start_rendering();
            let result = output.get_channel_data(0);

            let mut expected = Vec::<f32>::with_capacity(sample_rate);
            let mut phase: f64 = 0.;
            let phase_incr = freq as f64 / sample_rate as f64;

            for _i in 0..sample_rate {
                // triangle starts a 0.
                // [0., 1.]  between [0, 0.25]
                // [1., -1.] between [0.25, 0.75]
                // [-1., 0.] between [0.75, 1]
                let mut sample = -4. * phase + 2.;

                if sample > 1. {
                    sample = 2. - sample;
                } else if sample < -1. {
                    sample = -2. - sample;
                }

                expected.push(sample as f32);

                phase += phase_incr;
                if phase >= 1. { phase -= 1.; }
            }

            assert_float_eq!(result[..], expected[..], abs_all <= 1e-10);
        }
    }

    #[test]
    fn sawtooth_raw() {
        // 1, 10, 100, 1_000, 10_000 Hz
        for i in 0..1 {
            let freq = 10_f32.powf(i as f32);
            let sample_rate = 128;

            let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));

            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.frequency().set_value(freq);
            osc.set_type(OscillatorType::Sawtooth);
            osc.start_at(0.);

            let output = context.start_rendering();
            let result = output.get_channel_data(0);

            let mut expected = Vec::<f32>::with_capacity(sample_rate);
            let mut phase: f64 = 0.;
            let phase_incr = freq as f64 / sample_rate as f64;

            for _i in 0..sample_rate {
                // triangle starts a 0.
                // [0, 1] between [0, 0.5]
                // [-1, 0] between [0.5, 1]
                let mut offset_phase = phase + 0.5;
                if offset_phase >= 1. { offset_phase -= 1.; }
                let sample = 2. * offset_phase - 1.;

                expected.push(sample as f32);

                phase += phase_incr;
                if phase >= 1. { phase -= 1.; }
            }

            assert_float_eq!(result[..], expected[..], abs_all <= 1e-10);
        }
    }

    #[test]
    fn osc_sub_quantum_start() {
        let freq = 1.25;
        let sample_rate = 44_100;

        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.frequency().set_value(freq);
        osc.start_at(2. / sample_rate as f64);

        let output = context.start_rendering();
        let result = output.get_channel_data(0);

        let mut expected = Vec::<f32>::with_capacity(sample_rate);
        let mut phase: f64 = 0.;
        let phase_incr = freq as f64 / sample_rate as f64;

        expected.push(0.);
        expected.push(0.);

        for _i in 2..sample_rate {
            let sample = (phase * 2. * PI).sin();
            phase += phase_incr;
            expected.push(sample as f32);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
    }

    #[test]
    fn osc_sub_sample_start() {
        let freq = 444.;
        let sample_rate = 44_100;

        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.frequency().set_value(freq);
        // start between second and third sample
        osc.start_at(1.3 / sample_rate as f64);

        let output = context.start_rendering();
        let result = output.get_channel_data(0);

        let mut expected = Vec::<f32>::with_capacity(sample_rate);
        let phase_incr = freq as f64 / sample_rate as f64;
        // on first computed sample, phase is 0.7 (e.g. 2. - 1.3)
        let mut phase: f64 = 0.7 * phase_incr;

        expected.push(0.);
        expected.push(0.);

        for _i in 2..sample_rate {
            let sample = (phase * 2. * PI).sin();
            phase += phase_incr;
            expected.push(sample as f32);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
    }

    #[test]
    fn osc_sub_quantum_stop() {
        let freq = 2345.6;
        let sample_rate = 44_100;

        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.frequency().set_value(freq);
        osc.start_at(0.);
        osc.stop_at(6. / sample_rate as f64);

        let output = context.start_rendering();
        let result = output.get_channel_data(0);

        let mut expected = Vec::<f32>::with_capacity(sample_rate);
        let mut phase: f64 = 0.;
        let phase_incr = freq as f64 / sample_rate as f64;

        for i in 0..sample_rate {
            if i < 6 {
                let sample = (phase * 2. * PI).sin();
                phase += phase_incr;
                expected.push(sample as f32);
            } else {
                expected.push(0.);
            }
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
    }

    #[test]
    fn osc_sub_sample_stop() {
        let freq = 8910.1;
        let sample_rate = 44_100;

        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, SampleRate(sample_rate as u32));
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.frequency().set_value(freq);
        osc.start_at(0.);
        osc.stop_at(19.4 / sample_rate as f64);

        let output = context.start_rendering();
        let result = output.get_channel_data(0);

        let mut expected = Vec::<f32>::with_capacity(sample_rate);
        let mut phase: f64 = 0.;
        let phase_incr = freq as f64 / sample_rate as f64;

        for i in 0..sample_rate {
            if i < 20 {
                let sample = (phase * 2. * PI).sin();
                phase += phase_incr;
                expected.push(sample as f32);
            } else {
                expected.push(0.);
            }
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-5);
    }

    use super::generate_wavetable_test;

    #[test]
    fn test_periodic_wave() {
        // this is what should be the 2f file
        let reals = [0., 0., 0., 0., 0., 0., 0.];
        let imags = [0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        generate_wavetable_test(&reals, &imags, 128, true);
    }
}
