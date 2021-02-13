//! AudioParam interface

use std::collections::BinaryHeap;

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

enum AutomationEvent {
    SetValueAtTime(f32, f64),
}

use AutomationEvent::*;

impl AutomationEvent {
    fn time(&self) -> f64 {
        match &self {
            SetValueAtTime(_, t) => *t,
        }
    }

    fn value(&self) -> f32 {
        match &self {
            SetValueAtTime(v, _) => *v,
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
pub struct AudioParam {
    value: f32,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    events: BinaryHeap<AutomationEvent>,
}

impl From<AudioParamOptions> for AudioParam {
    fn from(opts: AudioParamOptions) -> Self {
        AudioParam {
            value: opts.default_value,
            automation_rate: opts.automation_rate,
            default_value: opts.default_value,
            min_value: opts.min_value,
            max_value: opts.max_value,
            events: BinaryHeap::new(),
        }
    }
}

impl AudioParam {
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

    pub fn set_value_at_time(&mut self, v: f32, t: f64) {
        self.events.push(SetValueAtTime(v, t))
    }

    pub fn tick(&mut self, ts: f64, dt: f64, count: usize) -> Vec<f32> {
        let next = match self.events.peek() {
            None => return vec![self.value(); count],
            Some(event) => event,
        };

        // we should have processed all earlier events in the previous call
        assert!(next.time() > ts);

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
            let end_index = ((next.time() - ts) / dt) as usize;

            // never trust floats
            let end_index = end_index.min(count);

            for _ in result.len()..end_index {
                result.push(self.value());
            }

            self.value = next.value();

            if !matches!(self.events.peek(), Some(e) if e.time() <= max_ts) {
                for _ in result.len()..count {
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
        let mut param: AudioParam = opts.into();

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should clamp
        param.set_value_at_time(8., 10.0); // should not occur 1st run

        let vs = param.tick(0., 1., 10);
        assert_eq!(vs, vec![0., 0., 5., 5., 5., 5., 5., 5., 10., 10.]);

        let vs = param.tick(10., 1., 10);
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
        let mut param: AudioParam = opts.into();

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should clamp
        param.set_value_at_time(8., 10.0); // should not occur 1st run

        let vs = param.tick(0., 1., 10);
        assert_eq!(vs, vec![0.; 10]);

        let vs = param.tick(10., 1., 10);
        assert_eq!(vs, vec![8.; 10]);
    }
}
