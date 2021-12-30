//! The oscillator control and renderer parts
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::perf,
    clippy::missing_docs_in_private_items
)]
use std::f32::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::control::{ScheduledState, Scheduler};
use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioRenderQuantumChannel,
};
use crate::SampleRate;

use crossbeam_channel::{self, Receiver, Sender};

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

/// Options for constructing an `OscillatorNode`
#[derive(Debug)]
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
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
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
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
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct OscillatorNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// Sample rate (equals to audio context sample rate)
    sample_rate: f32,
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
    /// * `options` - The Oscillatoroptions
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<OscillatorOptions>) -> Self {
        context.base().register(move |registration| {
            // Cannot guarantee that the cast is safe for all possible sample rate
            // cast without loss of precision for usual sample rates
            #[allow(clippy::cast_precision_loss)]
            let sample_rate = context.base().sample_rate().0 as f32;
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
                automation_rate: crate::param::AutomationRate::A,
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
                automation_rate: crate::param::AutomationRate::A,
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

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let computed_freq = default_freq * (default_det / 1200.).exp2();

            let config = OscRendererConfig {
                type_: type_.clone(),
                frequency: f_proc,
                detune: det_proc,
                scheduler: scheduler.clone(),
                receiver,
                computed_freq,
                sample_rate,
                periodic_wave,
            };
            let renderer = OscillatorRenderer::new(config);

            let node = Self {
                registration,
                channel_config: channel_config.unwrap_or_default().into(),
                sample_rate,
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
    pub const fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    /// Returns the detune audio parameter. detune unity is cents.
    /// The oscillator frequency is calculated as follow:
    /// frequency * 2^(detune/1200)
    #[must_use]
    pub const fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Returns the `computedOscFrequency` which is the oscillator frequency.
    fn computed_freq(&self) -> f32 {
        let frequency = self.frequency().value();
        let detune = self.detune().value();
        frequency * (detune / 1200.).exp2()
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
    /// Will panic if:
    ///
    /// * `type_` is `OscillatorType::Custom`
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

    /// set the oscillator type to any `OscillatorType` variant.
    /// This private function is used internally. To modify `OscillatorNode` type,
    /// you should use the public function `set_type`
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,square,triangle,sawtooth, and custom)
    fn change_type(&self, type_: OscillatorType) {
        self.type_.store(type_ as u32, Ordering::SeqCst);
    }

    /// set the oscillator type to custom and generate
    /// a perdioc waveform following the `PeriodicWave` characteristics
    pub fn set_periodic_wave(&mut self, periodic_wave: PeriodicWave) {
        // The oscillator type is set to custom following the spec
        self.change_type(OscillatorType::Custom);

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

        // update cplxs
        for cplx in real.into_iter().zip(imag) {
            cplxs.push(cplx);
        }

        let computed_freq = self.computed_freq();

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
            incr_phases.push(TABLE_LENGTH_F32 * idx as f32 * (computed_freq / self.sample_rate));
        }

        // update interpol_ratios
        for incr_phase in &incr_phases {
            interpol_ratios.push((incr_phase - incr_phase.round()).abs());
        }

        // generate the wavetable following periodic wave characteristics
        let dyn_wavetable =
            Self::generate_wavetable(&norms, &mut phases, &incr_phases, &interpol_ratios);

        // update norm_factor
        let norm_factor = Self::norm_factor(&dyn_wavetable);

        self.sender
            .send(OscMsg::PeriodicWaveMsg {
                computed_freq,
                dyn_wavetable,
                norm_factor,
                disable_normalization,
            })
            .expect("Sending periodic wave to the node renderer failed");
    }

    /// Generate the wavetable
    ///
    /// # Arguments
    ///
    /// * `norms` - the norm of each harmonics
    /// * `phases` - the phase of each harmonics
    /// * `incr_phases` - the phase to increment of each harmonics
    /// * `interpol_ratios` - the interpolation ratio of each harmonics used by linear interpolation
    fn generate_wavetable(
        norms: &[f32],
        phases: &mut [f32],
        incr_phases: &[f32],
        interpol_ratios: &[f32],
    ) -> Vec<f32> {
        let mut buffer = Vec::new();

        while phases[1] <= TABLE_LENGTH_F32 {
            let mut sample = 0.0;
            for i in 1..phases.len() {
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

            buffer.push(sample);
        }

        buffer
    }

    /// Compute the normalization factor
    ///
    /// The normalization factor is applied as a gain to the periodic wave
    /// to normalize the signal peak amplitude in the interval [-1.0,1.0].
    ///
    /// # Arguments
    ///
    /// * `buffer` - the wavetable generated from periodic wave charateristics
    fn norm_factor(buffer: &[f32]) -> f32 {
        1. / buffer
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("Maximum value not found")
    }
}

/// States relative to Sine `OscillatorType`
struct SineState {
    /// linear interpolation ratio
    interpol_ratio: f32,
    /// if set to true, requires sine parameters to be initialized  
    needs_init: bool,
}

/// States relative to Triangle `OscillatorType`
struct TriangleState {
    /// this memory state is used in the leaky integrator
    last_output: f32,
}

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
    /// starts and stops oscillator audio streams
    scheduler: Scheduler,
    /// channel between control and renderer parts (receiver part)
    receiver: Receiver<OscMsg>,
    /// `computed_freq` is precomputed from `frequency` and `detune`
    computed_freq: f32,
    /// channel between control and renderer parts (sender part)
    sample_rate: f32,
    /// current phase of the oscillator
    phase: f32,
    /// phase amount to add to phase at each tick
    incr_phase: f32,
    /// states required to build a sine wave
    sine: SineState,
    /// states required to build a triangle wave
    triangle: TriangleState,
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
        _sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // re-use previous buffer
        output.force_mono();

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

        let freq_values = params.get(&self.frequency);

        let det_values = params.get(&self.detune);

        let mut computed_freqs: [f32; 128] = [0.; 128];

        if det_values
            .windows(2)
            .all(|w| (w[0] - w[1]).abs() < 0.000_001)
        {
            let d = (det_values[0] / 1200.).exp2();
            for (i, f) in freq_values.iter().enumerate() {
                computed_freqs[i] = f * d;
            }
        } else {
            for (i, (f, d)) in freq_values.iter().zip(det_values).enumerate() {
                computed_freqs[i] = f * (d / 1200.).exp2();
            }
        }

        let type_ = self.type_.load(Ordering::SeqCst).into();

        let buffer = output.channel_data_mut(0);

        // check if any message was send from the control thread
        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                OscMsg::PeriodicWaveMsg {
                    computed_freq,
                    dyn_wavetable,
                    norm_factor,
                    disable_normalization,
                } => self.set_periodic_wave(
                    computed_freq,
                    dyn_wavetable,
                    norm_factor,
                    disable_normalization,
                ),
            }
        }

        self.generate_output(type_, buffer, &computed_freqs[..]);

        true // do not clean up source nodes
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
    /// channel between control and renderer parts (sender part)
    sample_rate: f32,
    /// The PeriodicWave for the OscillatorNode
    /// If this is specified, then any valid value for type is ignored;
    /// it is treated as if "custom" were specified.
    periodic_wave: Option<PeriodicWave>,
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
        let incr_phase = computed_freq / sample_rate;
        let interpol_ratio = (incr_phase - incr_phase.floor()) * TABLE_LENGTH_F32;

        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = periodic_wave.map_or_else(
            || PeriodicWave {
                real: vec![0., 1.],
                imag: vec![0., 0.],
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

        let mut periodic_wavetable = Vec::with_capacity(2048);

        Self::generate_wavetable(
            &norms,
            &mut phases,
            &incr_phases,
            &interpol_ratios,
            &mut periodic_wavetable,
        );

        let norm_factor = if disable_normalization {
            None
        } else {
            let norm_factor = Self::norm_factor(&periodic_wavetable);
            Some(norm_factor)
        };

        Self {
            type_,
            frequency,
            detune,
            scheduler,
            receiver,
            computed_freq,
            sample_rate,
            phase: 0.0,
            incr_phase,
            sine: SineState {
                interpol_ratio,
                needs_init: true,
            },
            triangle: TriangleState { last_output: 0.0 },
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

    /// set periodic states
    ///
    /// # Arguments
    ///
    /// * `ref_freq` - the `computedOscFrequency` used to build the wavetable
    /// * `dyn_wavetable` - wavetable following periodic wave characteristics
    /// * `norm_factor` - normalization factor applied when `disable_normalization` is false
    /// * `disable_normalization` - disable normalization. If false, the peak amplitude signal is 1.0
    fn set_periodic_wave(
        &mut self,
        ref_freq: f32,
        dyn_wavetable: Vec<f32>,
        norm_factor: f32,
        disable_normalization: bool,
    ) {
        self.periodic.wavetable.ref_freq = ref_freq;
        self.periodic.wavetable.dyn_table = dyn_wavetable;
        self.periodic.norm_factor = Some(norm_factor);
        self.periodic.disable_normalization = disable_normalization;
    }

    /// Compute params at each audio sample for the following oscillator type:
    /// * sine
    /// * sawtooth
    /// * triangle
    /// * and square
    #[inline]
    fn arate_params(&mut self, type_: OscillatorType, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if type_ == OscillatorType::Sine {
            if self.sine.needs_init {
                self.sine.needs_init = false;
                self.incr_phase = computed_freq / self.sample_rate * TABLE_LENGTH_F32;
            }
            if (self.computed_freq - computed_freq).abs() < 0.01 {
                return;
            }
            self.computed_freq = computed_freq;
            self.incr_phase = computed_freq / self.sample_rate * TABLE_LENGTH_F32;
        }
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.computed_freq = computed_freq;
        self.incr_phase = computed_freq / self.sample_rate;
    }

    /// Compute params at each audio sample for the custom oscillator type
    #[inline]
    fn arate_periodic_params(&mut self, new_comp_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - new_comp_freq).abs() < 0.01 {
            return;
        }

        for incr_phase in &mut self.periodic.incr_phases {
            *incr_phase *= new_comp_freq / self.computed_freq;
        }

        for (r, incr_ph) in self
            .periodic
            .interpol_ratios
            .iter_mut()
            .zip(self.periodic.incr_phases.iter())
        {
            *r = incr_ph - incr_ph.floor();
        }

        self.periodic.wavetable.incr_phase = new_comp_freq / self.periodic.wavetable.ref_freq;
        self.computed_freq = new_comp_freq;
    }

    /// generate the audio data according to the oscillator type and frequency parameters
    /// buffer is filled with the generated audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_output(
        &mut self,
        type_: OscillatorType,
        buffer: &mut AudioRenderQuantumChannel,
        freq_values: &[f32],
    ) {
        match type_ {
            OscillatorType::Sine => self.generate_sine(type_, buffer, freq_values),
            OscillatorType::Square => self.generate_square(type_, buffer, freq_values),
            OscillatorType::Sawtooth => self.generate_sawtooth(type_, buffer, freq_values),
            OscillatorType::Triangle => self.generate_triangle(type_, buffer, freq_values),
            OscillatorType::Custom => self.generate_custom(buffer, freq_values),
        }
    }

    /// generate the audio data when oscillator is of type sine
    /// buffer is filled with the sine audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_sine(
        &mut self,
        type_: OscillatorType,
        buffer: &mut AudioRenderQuantumChannel,
        freq_values: &[f32],
    ) {
        for (o, &computed_freq) in buffer.iter_mut().zip(freq_values) {
            self.arate_params(type_, computed_freq);
            // truncation is desired
            #[allow(clippy::cast_possible_truncation)]
            // phase is always positive
            #[allow(clippy::cast_sign_loss)]
            let inf_idx = self.phase as usize;
            let sup_idx = (inf_idx + 1) % TABLE_LENGTH_USIZE;

            // Linear interpolation
            *o = SINETABLE[inf_idx].mul_add(
                1. - self.sine.interpol_ratio,
                SINETABLE[sup_idx] * self.sine.interpol_ratio,
            );

            // Optimized float modulo op
            self.phase = if self.phase + self.incr_phase >= TABLE_LENGTH_F32 {
                (self.phase + self.incr_phase) - TABLE_LENGTH_F32
            } else {
                self.phase + self.incr_phase
            };
        }
    }

    /// generate the audio data when oscillator is of type sawtooth
    /// buffer is filled with the sawtooth audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_sawtooth(
        &mut self,
        type_: OscillatorType,
        buffer: &mut AudioRenderQuantumChannel,
        freq_values: &[f32],
    ) {
        for (o, &computed_freq) in buffer.iter_mut().zip(freq_values) {
            self.arate_params(type_, computed_freq);
            let mut sample = (2.0 * self.phase) - 1.0;
            sample -= self.poly_blep(self.phase);

            // Optimized float modulo op
            self.phase += self.incr_phase;
            while self.phase >= 1. {
                self.phase -= 1.;
            }

            *o = sample;
        }
    }

    /// generate the audio data when oscillator is of type square
    /// buffer is filled with the square audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_square(
        &mut self,
        type_: OscillatorType,
        buffer: &mut AudioRenderQuantumChannel,
        freq_values: &[f32],
    ) {
        for (o, &computed_freq) in buffer.iter_mut().zip(freq_values) {
            self.arate_params(type_, computed_freq);
            let mut sample = if self.phase <= 0.5 { 1.0 } else { -1.0 };

            sample += self.poly_blep(self.phase);

            // Optimized float modulo op
            let mut shift_phase = self.phase + 0.5;
            while shift_phase >= 1. {
                shift_phase -= 1.;
            }
            sample -= self.poly_blep(shift_phase);

            // Optimized float modulo op
            self.phase += self.incr_phase;
            while self.phase >= 1. {
                self.phase -= 1.;
            }
            *o = sample;
        }
    }

    /// generate the audio data when oscillator is of type triangle
    /// buffer is filled with the triangle audio data.
    ///
    /// # Arguments
    ///
    /// * `type_` - oscillator type (sine,sawtooth,triangle,square, or custom)
    /// * `buffer` - audio output buffer
    /// * `freq_values` - frequencies at which each sample should be generated
    #[inline]
    fn generate_triangle(
        &mut self,
        type_: OscillatorType,
        buffer: &mut AudioRenderQuantumChannel,
        freq_values: &[f32],
    ) {
        for (o, &computed_freq) in buffer.iter_mut().zip(freq_values) {
            self.arate_params(type_, computed_freq);
            let mut sample = if self.phase <= 0.5 { 1.0 } else { -1.0 };

            sample += self.poly_blep(self.phase);

            // Optimized float modulo op
            let mut shift_phase = self.phase + 0.5;
            while shift_phase >= 1. {
                shift_phase -= 1.;
            }
            sample -= self.poly_blep(shift_phase);

            // Optimized float modulo op
            self.phase += self.incr_phase;
            while self.phase >= 1. {
                self.phase -= 1.;
            }

            // Leaky integrator: y[n] = A * x[n] + (1 - A) * y[n-1]
            // Classic integration cannot be used due to float errors accumulation over execution time
            sample = self
                .incr_phase
                .mul_add(sample, (1.0 - self.incr_phase) * self.triangle.last_output);
            self.triangle.last_output = sample;

            // Normalized amplitude into intervall [-1.0,1.0]
            *o = sample * 4.;
        }
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
    fn generate_custom(&mut self, output: &mut AudioRenderQuantumChannel, freq_values: &[f32]) {
        for (o, &computed_freq) in output.iter_mut().zip(freq_values) {
            self.arate_periodic_params(computed_freq);

            let phase = self.periodic.wavetable.phase;
            let incr_phase = self.periodic.wavetable.incr_phase;
            let table_len = self.periodic.wavetable.dyn_table.len();

            // 2048 casts without loss of precision
            #[allow(clippy::cast_precision_loss)]
            let table_len_f32 = table_len as f32;
            let buffer = &self.periodic.wavetable.dyn_table;
            // truncation is desired
            #[allow(clippy::cast_possible_truncation)]
            // phase is always positive
            #[allow(clippy::cast_sign_loss)]
            let inf_idx = phase as usize;
            let sup_idx = (inf_idx + 1) % table_len;
            let interpol_ratio = phase - phase.trunc();

            *o = (1.0 - interpol_ratio).mul_add(buffer[inf_idx], interpol_ratio * buffer[sup_idx]);

            // Update phase with optimized float modulo op
            self.periodic.wavetable.phase = if phase + incr_phase >= table_len_f32 {
                (phase + incr_phase) - table_len_f32
            } else {
                phase + incr_phase
            };
        }
    }

    /// Generate the wavetable
    ///
    /// # Arguments
    ///
    /// * `norms` - the norm of each harmonics
    /// * `phases` - the phase of each harmonics
    /// * `incr_phases` - the phase to increment of each harmonics
    /// * `interpol_ratios` - the interpolation ratio of each harmonics used by linear interpolatio
    /// * `buffer` - the buffer is filled with generated wavetable data (avoid allocation)
    #[inline]
    fn generate_wavetable(
        norms: &[f32],
        phases: &mut [f32],
        incr_phases: &[f32],
        interpol_ratios: &[f32],
        buffer: &mut Vec<f32>,
    ) {
        buffer.clear();

        while phases[1] <= TABLE_LENGTH_F32 {
            let mut sample = 0.0;
            for i in 1..phases.len() {
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

            buffer.push(sample);
        }
    }

    /// normalizes the given buffer
    fn norm_factor(buffer: &[f32]) -> f32 {
        1. / buffer
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("Maximum value not found")
    }

    /// computes the `polyBLEP` corrections to apply to aliasing signal
    ///
    /// `polyBLEP` stands for `polyBandLimitedstEP`
    fn poly_blep(&self, mut t: f32) -> f32 {
        let dt = self.incr_phase;
        if t < dt {
            t /= dt;
            t + t - t * t - 1.0
        } else if t > 1.0 - dt {
            t = (t - 1.0) / dt;
            t.mul_add(t, t) + t + 1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {

    use float_eq::assert_float_eq;

    use super::{PeriodicWave, PeriodicWaveOptions};
    use crate::{
        context::{AsBaseAudioContext, AudioContext, OfflineAudioContext},
        node::{
            AudioNode, AudioScheduledSourceNode, OscillatorNode, OscillatorOptions, OscillatorType,
        },
        snapshot, SampleRate,
    };

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

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));

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

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));

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
        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));

        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Custom);
    }

    #[test]
    fn type_is_custom_when_periodic_wave_is_some() {
        let expected_type = OscillatorType::Custom;

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));

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

        let context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));

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
    fn silence_rendering_if_osc_is_not_started() {
        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Sine);
        osc.connect(&context.destination());

        let output = context.start_rendering();

        assert_float_eq!(
            output.channel_data(0).as_slice(),
            &[0.; LENGTH][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &[0.; LENGTH][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn default_sine_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/sine.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Sine);
        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-4
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-4
        );
    }

    #[test]
    fn default_square_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/square.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Square);
        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
    }

    #[test]
    fn default_triangle_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/triangle.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Triangle);
        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
    }

    #[test]
    fn default_sawtooth_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/sawtooth.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let osc = OscillatorNode::new(&context, None);

        osc.set_type(OscillatorType::Sawtooth);
        osc.connect(&context.destination());
        osc.start();

        let output = context.start_rendering();

        assert_float_eq!(
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-10
        );
    }

    #[test]
    fn periodic_wave_rendering_should_match_snapshot() {
        let ref_sine =
            snapshot::read("./snapshots/periodic_2f.json").expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
        let options = Some(PeriodicWaveOptions {
            real: Some(vec![0., 0.5, 0.5]),
            imag: Some(vec![0., 0., 0.]),
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
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
    }

    #[test]
    fn default_periodic_wave_rendering_should_match_snapshot() {
        let ref_sine = snapshot::read("./snapshots/default_periodic.json")
            .expect("Reading snapshot file failed");

        let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100.));
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
            output.channel_data(0).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
        assert_float_eq!(
            output.channel_data(1).as_slice(),
            &ref_sine.data[..],
            abs_all <= 1.0e-6
        );
    }
}
