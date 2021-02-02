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

/// Audio source generating a periodic waveform
pub struct OscillatorNode<'a> {
    pub context: &'a AudioContext,
    pub id: u64,
    pub frequency: Arc<AtomicU32>,
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
    pub fn frequency(&self) -> u32 {
        self.frequency.load(Ordering::SeqCst)
    }

    pub fn set_frequency(&self, freq: u32) {
        self.frequency.store(freq, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub(crate) struct OscillatorRenderer {
    pub frequency: Arc<AtomicU32>,
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
        (0..output.len())
            .map(move |i| timestamp + i as f64 / sample_rate as f64)
            .map(move |t| (2. * PI * freq * t).sin() as f32)
            .zip(output.iter_mut())
            .for_each(|(value, dest)| *dest = value);
    }
}

/// Representing the final audio destination and is what the user will ultimately hear.
pub struct DestinationNode<'a> {
    pub context: &'a AudioContext,
    pub id: u64,
    pub channels: usize,
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

/// AudioNode for volume control
pub struct GainNode<'a> {
    pub context: &'a AudioContext,
    pub id: u64,
    pub gain: Arc<AtomicU32>,
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
