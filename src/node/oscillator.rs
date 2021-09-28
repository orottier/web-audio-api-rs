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

use crossbeam_channel::{self, Receiver, Sender};
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
    pub real: Vec<f32>,
    /// The imag parameter represents an array of sine terms of Fourrier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    pub imag: Vec<f32>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    pub disable_normalization: bool,
}

/// PeriodicWave is a setup struct required to build
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
    /// Returns a PeriodicWave
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourrier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourrier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the PeriodicWave
    ///
    /// # Example
    ///
    /// ```no_run
    ///    use web_audio_api::context::{AudioContext, AsBaseAudioContext};
    ///    use web_audio_api::node::{PeriodicWave, PeriodicWaveOptions};
    ///
    ///    let context = AudioContext::new();
    ///
    ///    let options = PeriodicWaveOptions {
    ///    real: vec![0.,1.,1.],
    ///    imag: vec![0.,0.,0.],
    ///    disable_normalization: false,
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
            // Todo: assertions not complete. Missing assertions when
            // only real field is specified, anly imag field is specified,
            // and neither real or imag is specified
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
                real: vec![0., 0.],
                imag: vec![0., 1.],
                disable_normalization: false,
            }
        }
    }
}

/// Options for constructing an OscillatorNode
pub struct OscillatorOptions {
    pub type_: OscillatorType,
    pub frequency: f32,
    pub detune: f32,
    pub channel_config: ChannelConfigOptions,
    pub periodic_wave: Option<PeriodicWave>,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440.,
            detune: 0.,
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

enum OscMsg {
    PeriodicWaveMsg(PeriodicWave),
}

/// Audio source generating a periodic waveform
pub struct OscillatorNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    frequency: AudioParam,
    detune: AudioParam,
    type_: Arc<AtomicU32>,
    scheduler: Scheduler,
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

            // frequency audio parameter
            let freq_param_opts = AudioParamOptions {
                min_value: -nyquist,
                max_value: nyquist,
                default_value: 440.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (f_param, f_proc) = context
                .base()
                .create_audio_param(freq_param_opts, registration.id());
            f_param.set_value(options.frequency);

            // detune audio parameter
            let det_param_opts = AudioParamOptions {
                min_value: -153600.,
                max_value: 153600.,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (det_param, det_proc) = context
                .base()
                .create_audio_param(det_param_opts, registration.id());
            det_param.set_value(options.detune);

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

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let render = OscillatorRenderer {
                type_: type_.clone(),
                frequency: f_proc,
                detune: det_proc,
                scheduler: scheduler.clone(),
                sine_renderer,
                sawtooth_renderer,
                triangle_renderer,
                square_renderer,
                custom_renderer,
                receiver,
            };
            let node = OscillatorNode {
                registration,
                channel_config: options.channel_config.into(),
                frequency: f_param,
                detune: det_param,
                type_,
                scheduler,
                sender,
            };

            (node, Box::new(render))
        })
    }

    /// Returns the oscillator frequency audio parameter
    pub fn frequency(&self) -> &AudioParam {
        &self.frequency
    }

    pub fn detune(&self) -> &AudioParam {
        &self.detune
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
    pub fn set_periodic_wave(&mut self, periodic_wave: PeriodicWave) {
        self.set_type(OscillatorType::Custom);
        self.sender
            .send(OscMsg::PeriodicWaveMsg(periodic_wave))
            .expect("Sending periodic wave to the node renderer failed");
    }
}

struct OscillatorRenderer {
    type_: Arc<AtomicU32>,
    frequency: AudioParamId,
    detune: AudioParamId,
    scheduler: Scheduler,
    sine_renderer: SineRenderer,
    sawtooth_renderer: SawRenderer,
    triangle_renderer: TriangleRenderer,
    square_renderer: SquareRenderer,
    custom_renderer: CustomRenderer,
    receiver: Receiver<OscMsg>,
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

        let det_values = params.get(&self.detune);
        let detune = det_values[0]; // force a-rate processing

        let computed_freq = freq + 2f32.powf(detune / 1200.);

        let type_ = self.type_.load(Ordering::SeqCst).into();

        let buffer = output.channel_data_mut(0);

        use OscillatorType::*;

        // check if any message was send from the control thread
        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                OscMsg::PeriodicWaveMsg(p_w) => self.custom_renderer.set_periodic_wave(p_w),
            }
        }

        match type_ {
            Sine => {
                // K-rate
                self.sine_renderer.compute_params(computed_freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.sine_renderer.tick());
            }
            Square => {
                // K-rate
                self.square_renderer.compute_params(computed_freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.square_renderer.tick());
            }
            Sawtooth => {
                // K-rate
                self.sawtooth_renderer.compute_params(computed_freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.sawtooth_renderer.tick());
            }
            Triangle => {
                // K-rate
                self.triangle_renderer.compute_params(computed_freq);
                buffer
                    .iter_mut()
                    .for_each(|o| *o = self.triangle_renderer.tick())
            }
            Custom => {
                // K-rate
                self.custom_renderer.compute_params(computed_freq);
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
    computed_freq: f32,
    sample_rate: f32,
    phase: f32,
    incr_phase: f32,
    interpol_ratio: f32,
}

impl SineRenderer {
    fn new(computed_freq: f32, sample_rate: f32) -> Self {
        let incr_phase = 2. * PI * (computed_freq / sample_rate);
        let interpol_ratio = incr_phase - incr_phase.floor();
        Self {
            computed_freq,
            sample_rate,
            phase: 0.0,
            incr_phase,
            interpol_ratio,
        }
    }

    fn compute_params(&mut self, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.incr_phase = TABLE_LENGTH_F32 * computed_freq / self.sample_rate;
        self.interpol_ratio = self.incr_phase - self.incr_phase.floor();
        self.computed_freq = computed_freq;
    }
}

impl Ticker for SineRenderer {
    fn tick(&mut self) -> f32 {
        let idx = self.phase as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
        let sample = SINETABLE[inf_idx] * (1. - self.interpol_ratio)
            + SINETABLE[sup_idx] * self.interpol_ratio;

        // Optimized float modulo op
        self.phase = if self.phase + self.incr_phase >= TABLE_LENGTH_F32 {
            (self.phase + self.incr_phase) - TABLE_LENGTH_F32
        } else {
            self.phase + self.incr_phase
        };
        sample
    }
}

#[derive(Debug)]
struct SawRenderer {
    computed_freq: f32,
    sample_rate: f32,
    incr_phase: f32,
    phase: f32,
    interpol_ratio: f32,
}

impl SawRenderer {
    fn new(computed_freq: f32, sample_rate: f32) -> Self {
        let incr_phase = (TABLE_LENGTH_F32 / sample_rate) * computed_freq;
        let interpol_ratio = incr_phase - incr_phase.floor();
        Self {
            computed_freq,
            sample_rate,
            phase: 0.0,
            incr_phase,
            interpol_ratio,
        }
    }

    fn compute_params(&mut self, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.incr_phase = TABLE_LENGTH_F32 * computed_freq / self.sample_rate;
        self.interpol_ratio = self.incr_phase - self.incr_phase.floor();
        self.computed_freq = computed_freq;
    }
}

impl PolyBlep for SawRenderer {
    fn incr_phase(&self) -> f32 {
        self.incr_phase
    }
}

impl Ticker for SawRenderer {
    fn tick(&mut self) -> f32 {
        let idx = self.phase as usize;
        let inf_idx = idx % TABLE_LENGTH_USIZE;
        let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;

        let mut sample = SAWTABLE[inf_idx] * (1. - self.interpol_ratio)
            + SAWTABLE[sup_idx] * self.interpol_ratio;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);

        // Optimized float modulo op
        self.phase = if self.phase + self.incr_phase >= TABLE_LENGTH_F32 {
            (self.phase + self.incr_phase) - TABLE_LENGTH_F32
        } else {
            self.phase + self.incr_phase
        };
        sample
    }
}

#[derive(Debug)]
struct TriangleRenderer {
    computed_freq: f32,
    sample_rate: f32,
    phase: f32,
    incr_phase: f32,
    interpol_ratio: f32,
}

impl TriangleRenderer {
    fn new(computed_freq: f32, sample_rate: f32) -> Self {
        let incr_phase = TABLE_LENGTH_F32 * computed_freq / sample_rate;
        let interpol_ratio = incr_phase - incr_phase.floor();
        Self {
            computed_freq,
            sample_rate,
            phase: 0.0,
            incr_phase,
            interpol_ratio,
        }
    }

    fn compute_params(&mut self, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.incr_phase = TABLE_LENGTH_F32 * computed_freq / self.sample_rate;
        self.interpol_ratio = self.incr_phase - self.incr_phase.floor();
        self.computed_freq = computed_freq;
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

        let mut sample = TRIANGLETABLE[inf_idx] * (1. - self.interpol_ratio)
            + TRIANGLETABLE[sup_idx] * self.interpol_ratio;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);

        // Optimized float modulo op
        self.phase = if self.phase + self.incr_phase >= TABLE_LENGTH_F32 {
            (self.phase + self.incr_phase) - TABLE_LENGTH_F32
        } else {
            self.phase + self.incr_phase
        };

        sample
    }
}

#[derive(Debug)]
struct SquareRenderer {
    computed_freq: f32,
    sample_rate: f32,
    phase: f32,
    incr_phase: f32,
    interpol_ratio: f32,
}

impl SquareRenderer {
    fn new(computed_freq: f32, sample_rate: f32) -> Self {
        let incr_phase = (TABLE_LENGTH_F32 / sample_rate) * computed_freq;
        let interpol_ratio = incr_phase - incr_phase.floor();
        Self {
            computed_freq,
            sample_rate,
            phase: 0.0,
            incr_phase,
            interpol_ratio,
        }
    }

    fn compute_params(&mut self, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.incr_phase = (TABLE_LENGTH_F32 / self.sample_rate) * computed_freq;
        self.interpol_ratio = self.incr_phase - self.incr_phase.floor();
        self.computed_freq = computed_freq;
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

        let mut sample = SQUARETABLE[inf_idx] * (1. - self.interpol_ratio)
            + SQUARETABLE[sup_idx] * self.interpol_ratio;

        let norm_phase = self.phase / TABLE_LENGTH_F32;
        sample -= self.poly_blep(norm_phase);

        // Optimized float modulo op
        self.phase = if self.phase + self.incr_phase >= TABLE_LENGTH_F32 {
            (self.phase + self.incr_phase) - TABLE_LENGTH_F32
        } else {
            self.phase + self.incr_phase
        };
        sample
    }
}

struct CustomRenderer {
    computed_freq: f32,
    sample_rate: f32,
    cplxs: Vec<(f32, f32)>,
    norms: Vec<f32>,
    phases: Vec<f32>,
    incr_phases: Vec<f32>,
    interpol_ratios: Vec<f32>,
    normalizer: Option<f32>,
}

impl CustomRenderer {
    fn new(computed_freq: f32, sample_rate: f32, periodic_wave: Option<PeriodicWave>) -> Self {
        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = if let Some(p_w) = periodic_wave {
            p_w
        } else {
            PeriodicWave {
                real: vec![0., 1.],
                imag: vec![0., 0.],
                disable_normalization: false,
            }
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
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (computed_freq / sample_rate))
            .collect();

        let interpol_ratios: Vec<f32> = incr_phases
            .iter()
            .map(|incr_phase| incr_phase - incr_phase.floor())
            .collect();
        let normalizer = if !disable_normalization {
            let norm = Self::get_normalizer(
                phases.clone(),
                incr_phases.clone(),
                interpol_ratios.clone(),
                norms.clone(),
                computed_freq,
            );
            Some(norm)
        } else {
            None
        };

        Self {
            computed_freq,
            sample_rate,
            cplxs,
            norms,
            phases,
            incr_phases,
            interpol_ratios,
            normalizer,
        }
    }

    fn get_normalizer(
        mut phases: Vec<f32>,
        incr_phases: Vec<f32>,
        interpol_ratios: Vec<f32>,
        norms: Vec<f32>,
        computed_freq: f32,
    ) -> f32 {
        let mut samples: Vec<f32> = Vec::new();

        if computed_freq == 0. {
            return 1.;
        }

        while phases[1] <= TABLE_LENGTH_F32 {
            let mut sample = 0.0;
            for i in 1..phases.len() {
                let gain = norms[i];
                let phase = phases[i];
                let incr_phase = incr_phases[i];
                let mu = interpol_ratios[i];
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

    fn compute_params(&mut self, computed_freq: f32) {
        // No need to compute if frequency has not changed
        if (self.computed_freq - computed_freq).abs() < 0.01 {
            return;
        }
        self.computed_freq = computed_freq;
        self.incr_phases = self
            .cplxs
            .iter()
            .enumerate()
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (self.computed_freq / self.sample_rate))
            .collect();

        self.interpol_ratios = self
            .incr_phases
            .iter()
            .map(|incr_phase| incr_phase - incr_phase.floor())
            .collect();
    }

    fn set_periodic_wave(&mut self, periodic_wave: PeriodicWave) {
        let PeriodicWave {
            real,
            imag,
            disable_normalization,
        } = periodic_wave;

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
            .map(|(idx, _)| TABLE_LENGTH_F32 * idx as f32 * (self.computed_freq / self.sample_rate))
            .collect();

        let interpol_ratios: Vec<f32> = incr_phases
            .iter()
            .map(|incr_phase| (incr_phase - incr_phase.round()).abs())
            .collect();

        let normalizer = if !disable_normalization {
            let norm = Self::get_normalizer(
                phases.clone(),
                incr_phases.clone(),
                interpol_ratios.clone(),
                norms.clone(),
                self.computed_freq,
            );
            Some(norm)
        } else {
            None
        };

        self.cplxs = cplxs;
        self.phases = phases;
        self.incr_phases = incr_phases;
        self.interpol_ratios = interpol_ratios;
        self.norms = norms;
        self.normalizer = normalizer;
    }
}

impl Ticker for CustomRenderer {
    fn tick(&mut self) -> f32 {
        let mut sample = 0.;
        for i in 1..self.phases.len() {
            let gain = self.norms[i];
            let phase = self.phases[i];
            let incr_phase = self.incr_phases[i];
            let mu = self.interpol_ratios[i];
            let idx = (phase + incr_phase) as usize;
            let inf_idx = idx % TABLE_LENGTH_USIZE;
            let sup_idx = (idx + 1) % TABLE_LENGTH_USIZE;
            sample += (SINETABLE[inf_idx] * (1. - mu) + SINETABLE[sup_idx] * mu)
                * gain
                * self.normalizer.unwrap_or(1.);

            // Optimized float modulo op
            self.phases[i] = if phase + incr_phase >= TABLE_LENGTH_F32 {
                (phase + incr_phase) - TABLE_LENGTH_F32
            } else {
                phase + incr_phase
            };
        }
        sample
    }
}

#[cfg(test)]

mod tests {
    use super::{PeriodicWave, PeriodicWaveOptions};
    use crate::context::AudioContext;

    #[test]
    #[should_panic]
    fn fails_to_build_when_real_is_too_short() {
        let context = AudioContext::new();

        let options = PeriodicWaveOptions {
            real: vec![0.],
            imag: vec![0., 0., 0.],
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_is_too_short() {
        let context = AudioContext::new();

        let options = PeriodicWaveOptions {
            real: vec![0., 0., 0.],
            imag: vec![0.],
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_not_equal_length() {
        let context = AudioContext::new();

        let options = PeriodicWaveOptions {
            real: vec![0., 0., 0.],
            imag: vec![0., 0.],
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }
}
