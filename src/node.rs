use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::context::AudioContext;
use crate::graph::Render;

pub trait AudioNode {
    fn id(&self) -> u64;
    fn context(&self) -> &AudioContext;

    fn connect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if !std::ptr::eq(self.context(), dest.context()) {
            panic!("attempting to connect nodes from different contexts");
        }

        self.context().connect(self.id(), dest.id(), 0);

        dest
    }
}

pub struct OscillatorOptions {
    pub type_: OscillatorType,
    pub frequency: u32,
}

impl Default for OscillatorOptions {
    fn default() -> Self {
        Self {
            type_: OscillatorType::default(),
            frequency: 440,
        }
    }
}

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
        match i {
            0 => OscillatorType::Sine,
            1 => OscillatorType::Square,
            2 => OscillatorType::Sawtooth,
            3 => OscillatorType::Triangle,
            4 => OscillatorType::Custom,
            _ => unreachable!(),
        }
    }
}

/// Audio source generating a periodic waveform
pub struct OscillatorNode<'a> {
    pub(crate) context: &'a AudioContext,
    pub(crate) id: u64,
    pub(crate) frequency: Arc<AtomicU32>,
    pub(crate) type_: Arc<AtomicU32>,
}

impl<'a> AudioNode for OscillatorNode<'a> {
    fn context(&self) -> &AudioContext {
        self.context
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl<'a> OscillatorNode<'a> {
    pub fn new(context: &'a AudioContext, options: OscillatorOptions) -> Self {
        context.create_oscillator_with(options)
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
}

impl Render for OscillatorRenderer {
    fn process(
        &mut self,
        _inputs: &[&[f32]],
        output: &mut [f32],
        timestamp: f64,
        sample_rate: u32,
    ) {
        let freq = self.frequency.load(Ordering::SeqCst) as f64;
        let type_ = self.type_.load(Ordering::SeqCst).into();

        let ts = (0..output.len()).map(move |i| timestamp + i as f64 / sample_rate as f64);
        let io = ts.zip(output.iter_mut());

        match type_ {
            OscillatorType::Sine => io.for_each(|(i, o)| *o = (2. * PI * freq * i).sin() as f32),
            OscillatorType::Square => {
                io.for_each(|(i, o)| *o = if (freq * i).fract() < 0.5 { 1. } else { -1. })
            }
            OscillatorType::Sawtooth => {
                io.for_each(|(i, o)| *o = 2. * ((freq * i).fract() - 0.5) as f32)
            }
            _ => todo!(),
        }
    }
}

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode<'a> {
    pub(crate) context: &'a AudioContext,
    pub(crate) id: u64,
    pub(crate) channels: usize,
}

#[derive(Debug)]
pub(crate) struct DestinationRenderer {
    pub channels: usize,
}

impl Render for DestinationRenderer {
    fn process(
        &mut self,
        inputs: &[&[f32]],
        output: &mut [f32],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        // clear slice, it may be re-used
        for d in output.iter_mut() {
            *d = 0.;
        }

        // sum signal from all child nodes
        for input in inputs.iter() {
            for (i, o) in input.iter().zip(output.iter_mut()) {
                *o += i;
            }
        }
    }
}

impl<'a> AudioNode for DestinationNode<'a> {
    fn context(&self) -> &AudioContext {
        self.context
    }

    fn id(&self) -> u64 {
        self.id
    }
}

pub struct GainOptions {
    pub gain: f32,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self { gain: 1. }
    }
}

/// AudioNode for volume control
pub struct GainNode<'a> {
    pub(crate) context: &'a AudioContext,
    pub(crate) id: u64,
    pub(crate) gain: Arc<AtomicU32>,
}

impl<'a> AudioNode for GainNode<'a> {
    fn context(&self) -> &AudioContext {
        self.context
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl<'a> GainNode<'a> {
    pub fn new(context: &'a AudioContext, options: GainOptions) -> Self {
        context.create_gain_with(options)
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
        inputs: &[&[f32]],
        output: &mut [f32],
        _timestamp: f64,
        _sample_rate: u32,
    ) {
        let gain = self.gain.load(Ordering::SeqCst) as f32 / 100.;
        inputs[0]
            .iter()
            .zip(output.iter_mut())
            .for_each(|(value, dest)| *dest = value * gain);
    }
}
