//! AudioParam interface

use std::collections::BinaryHeap;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Precision of value calculation per render quantum
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AutomationRate {
    /// sampled for each sample-frame of the block
    A,
    /// sampled at the time of the very first sample-frame, then used for the entire block
    K,
}

/// Options for constructing an AudioParam
pub struct AudioParamOptions {
    pub automation_rate: AutomationRate,
    pub default_value: f32,
    pub min_value: f32,
    pub max_value: f32,
}

#[derive(Debug)]
enum AutomationEvent {
    SetValueAtTime { v: f32, start: f64 },
    LinearRampToValueAtTime { v: f32, start: f64, end: f64 },
}

use AutomationEvent::*;

impl AutomationEvent {
    fn time(&self) -> f64 {
        match &self {
            SetValueAtTime { start, .. } => *start,
            LinearRampToValueAtTime { end, .. } => *end,
        }
    }

    fn value(&self) -> f32 {
        match &self {
            SetValueAtTime { v, .. } => *v,
            LinearRampToValueAtTime { v, .. } => *v,
        }
    }
}

impl PartialEq for AutomationEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time().eq(&other.time())
    }
}
impl Eq for AutomationEvent {}

impl std::cmp::PartialOrd for AutomationEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // reverse ordering
        other.time().partial_cmp(&self.time())
    }
}

impl std::cmp::Ord for AutomationEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time().partial_cmp(&other.time()).unwrap()
    }
}

/// AudioParam controls an individual aspect of an AudioNode's functionality, such as volume.
#[derive(Debug)]
pub struct AudioParam {
    value: Arc<AtomicU32>,
    sender: Sender<AutomationEvent>,
}
#[derive(Debug)]
pub struct AudioParamRenderer {
    value: f32,
    shared_value: Arc<AtomicU32>,
    receiver: Receiver<AutomationEvent>,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    events: BinaryHeap<AutomationEvent>,
}

pub fn audio_param_pair(opts: AudioParamOptions) -> (AudioParam, AudioParamRenderer) {
    let (sender, receiver) = mpsc::channel();
    let value_as_int = u32::from_be(opts.default_value.to_bits());
    let shared_value = Arc::new(AtomicU32::new(value_as_int));

    let param = AudioParam {
        value: shared_value.clone(),
        sender,
    };

    let render = AudioParamRenderer {
        value: opts.default_value,
        shared_value,
        receiver,
        automation_rate: opts.automation_rate,
        default_value: opts.default_value,
        min_value: opts.min_value,
        max_value: opts.max_value,
        events: BinaryHeap::new(),
    };

    (param, render)
}

impl AudioParam {
    pub fn set_value(&self, v: f32) {
        self.sender.send(SetValueAtTime { v, start: 0. }).unwrap()
    }

    pub fn set_value_at_time(&self, v: f32, start: f64) {
        self.sender.send(SetValueAtTime { v, start }).unwrap()
    }

    pub fn linear_ramp_to_value_at_time(&self, v: f32, end: f64) {
        self.sender
            .send(LinearRampToValueAtTime { v, start: 0., end })
            .unwrap()
    }
}

impl AudioParamRenderer {
    pub fn value(&self) -> f32 {
        if self.value.is_nan() {
            self.default_value
        } else {
            self.value.clamp(self.min_value, self.max_value)
        }
    }

    pub fn set_value(&mut self, v: f32) {
        self.value = v;
        // todo, `set_value_at_time(v, context.current_time())`
    }

    pub fn tick(&mut self, ts: f64, dt: f64, count: usize) -> Vec<f32> {
        for event in self.receiver.try_iter() {
            self.events.push(event);
        }

        let next = match self.events.peek() {
            None => return vec![self.value(); count],
            Some(event) => event,
        };

        let max_ts = ts + dt * count as f64;
        if next.time() > max_ts {
            return vec![self.value(); count];
        }

        let mut result = match self.automation_rate {
            AutomationRate::A => Vec::with_capacity(count),
            AutomationRate::K => {
                // by filling the vec already, no expensive calculations are performed later
                vec![self.value(); count]
            }
        };

        loop {
            let next = self.events.pop().unwrap();

            // we should have processed all earlier events in the previous call,
            // but new events could have been added, clamp it to `ts`
            let time = next.time().max(ts);

            let end_index = ((time - ts) / dt) as usize;
            let end_index = end_index.min(count); // never trust floats

            for _ in result.len()..end_index {
                // todo, actual value calculation
                result.push(self.value());
            }
            self.value = next.value(); // todo

            if !matches!(self.events.peek(), Some(e) if e.time() <= max_ts) {
                for _ in result.len()..count {
                    // todo, actual value calculation (when ramp-to-value)
                    result.push(self.value());
                }
                break;
            }
        }

        assert_eq!(result.len(), count);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steps_a_rate() {
        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts);

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should clamp
        param.set_value_at_time(8., 10.0); // should not occur 1st run

        let vs = render.tick(0., 1., 10);
        assert_eq!(vs, vec![0., 0., 5., 5., 5., 5., 5., 5., 10., 10.]);

        let vs = render.tick(10., 1., 10);
        assert_eq!(vs, vec![8.; 10]);
    }

    #[test]
    fn test_steps_k_rate() {
        let opts = AudioParamOptions {
            automation_rate: AutomationRate::K,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts);

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should clamp
        param.set_value_at_time(8., 10.0); // should not occur 1st run

        let vs = render.tick(0., 1., 10);
        assert_eq!(vs, vec![0.; 10]);

        let vs = render.tick(10., 1., 10);
        assert_eq!(vs, vec![8.; 10]);
    }
}
