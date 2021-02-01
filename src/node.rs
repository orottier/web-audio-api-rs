use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::graph::AudioNode;

#[derive(Debug)]
pub struct OscillatorNode {
    frequency: AtomicU32,
}

impl AudioNode for OscillatorNode {
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

#[derive(Debug)]
pub struct DestinationNode {
    pub channels: usize,
}

impl AudioNode for DestinationNode {
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

        // mix signal from all child nodes, prevent allocations
        for input in inputs.iter() {
            let frames = output.chunks_mut(self.channels);
            for (frame, v) in frames.zip(input.iter()) {
                for sample in frame.iter_mut() {
                    *sample += v;
                }
            }
        }
    }
}
