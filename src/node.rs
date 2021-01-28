use std::f64::consts::PI;
use std::sync::Arc;
use std::sync::Mutex;

use crate::context::AudioGraph;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// The OscillatorNode interface represents a periodic waveform, such as a sine wave. It is an
/// AudioScheduledSourceNode audio-processing module that causes a specified frequency of a given
/// wave to be created—in effect, a constant tone.
pub struct OscillatorNode {
    inner: Arc<OscillatorNodeInner>,
}

// todo, pub(crate) after abstractions
pub struct OscillatorNodeInner {
    frequency: AtomicU32,
    active: AtomicBool,
}

impl OscillatorNode {
    pub(crate) fn new(frequency: u32) -> Self {
        let inner = OscillatorNodeInner {
            frequency: AtomicU32::new(frequency),
            active: AtomicBool::new(false),
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn connect(&self, dest: &Mutex<AudioGraph>) {
        let children = &mut dest.lock().unwrap().children;
        children.push(self.inner.clone());
    }

    pub fn start(&self) {
        self.inner.active.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.inner.active.store(false, Ordering::SeqCst);
    }

    pub fn frequency(&self) -> u32 {
        self.inner.frequency.load(Ordering::SeqCst)
    }

    pub fn set_frequency(&self, freq: u32) {
        self.inner.frequency.store(freq, Ordering::SeqCst)
    }
}

impl OscillatorNodeInner {
    pub(crate) fn signal(
        &self,
        ts: f64,
        samples: usize,
        sample_rate: u32,
    ) -> impl Iterator<Item = f32> {
        let freq = self.frequency.load(Ordering::SeqCst) as f64;
        (0..samples)
            .map(move |i| ts + i as f64 / sample_rate as f64)
            .map(move |t| (2. * PI * freq * t).sin() as f32)
    }
}
