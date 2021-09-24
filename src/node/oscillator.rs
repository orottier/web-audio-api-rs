use std::f32::consts::{PI, TAU};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::buffer::{ChannelConfig, ChannelConfigOptions};
use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::control::Scheduler;
use crate::param::{AudioParam, AudioParamOptions};
use crate::process::{AudioParamValues, AudioProcessor};
use crate::SampleRate;

use lazy_static::lazy_static;

use super::{AudioNode, AudioScheduledSourceNode};

const TABLE_LENGTH_USIZE: usize = 2048;
const TABLE_LENGTH_F32: f32 = TABLE_LENGTH_USIZE as f32;

// Compute one period sine wavetable of size TABLE_LENGTH
lazy_static! {
    static ref SINETABLE: Vec<f32> = {
        let table: Vec<f32> = (0..TABLE_LENGTH_USIZE)
            .map(|x| ((x as f32) * 2.0 * PI * (1. / (TABLE_LENGTH_F32))).sin())
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
pub struct PeriodicWaveOptions {
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
    pub fn new<C: AsBaseAudioContext>(context: &C, options: Option<PeriodicWaveOptions>) -> Self {
        if let Some(PeriodicWaveOptions {
            real,
            imag,
            disable_normalization,
        }) = options
        {
            assert!(
                real.len() >= 2,
                "RangeError: Real field length should be at least 2"
            );
            assert!(
                imag.len() >= 2,
                "RangeError: Imag field length should be at least 2",
            );
            assert!(
                real.len() == imag.len(),
                "RangeError: Imag and real field length should be equal"
            );
            Self {
                real,
                imag,
                disable_normalization,
            }
        } else {
            Self {
                real: vec![0., 1.0],
                imag: vec![0., 0.],
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
            let norm = Self::get_normalizer(
                phases.clone(),
                incr_phases.clone(),
                mus.clone(),
                norms.clone(),
                frequency,
            );
            Some(norm)
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
        frequency: f32,
    ) -> f32 {
        let mut samples: Vec<f32> = Vec::new();

        if frequency == 0. {
            return 1.;
        }

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
