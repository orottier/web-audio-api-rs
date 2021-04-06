//! AudioParam interface

use std::collections::BinaryHeap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use AutomationEvent::*;

use crate::buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::context::AudioContextRegistration;
use crate::node::AudioNode;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::{AtomicF64, SampleRate};

/// Precision of value calculation per render quantum
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AutomationRate {
    /// sampled for each sample-frame of the block
    A,
    /// sampled at the time of the very first sample-frame, then used for the entire block
    K,
}

/// Options for constructing an [`AudioParam`]
pub struct AudioParamOptions {
    pub automation_rate: AutomationRate,
    pub default_value: f32,
    pub min_value: f32,
    pub max_value: f32,
}

#[derive(Debug)]
pub(crate) enum AutomationEvent {
    SetValueAtTime { v: f32, start: f64 },
    LinearRampToValueAtTime { v: f32, start: f64, end: f64 },
}

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
pub struct AudioParam<'a> {
    registration: AudioContextRegistration<'a>,
    value: Arc<AtomicF64>,
    sender: Sender<AutomationEvent>,
}

impl<'a> AudioNode for AudioParam<'a> {
    fn registration(&self) -> &AudioContextRegistration<'a> {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        unreachable!()
    }

    fn channel_config_cloned(&self) -> ChannelConfig {
        ChannelConfigOptions {
            count: 1,
            mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Discrete,
        }
        .into()
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }

    fn number_of_outputs(&self) -> u32 {
        1
    }

    fn channel_count_mode(&self) -> ChannelCountMode {
        ChannelCountMode::Explicit
    }

    fn channel_interpretation(&self) -> ChannelInterpretation {
        ChannelInterpretation::Discrete
    }

    fn channel_count(&self) -> usize {
        1
    }
}

#[derive(Debug)]
pub(crate) struct AudioParamProcessor {
    value: f32,
    shared_value: Arc<AtomicF64>,
    receiver: Receiver<AutomationEvent>,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    events: BinaryHeap<AutomationEvent>,
}

impl AudioProcessor for AudioParamProcessor {
    fn process<'a>(
        &mut self,
        inputs: &[&crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) {
        let input = inputs[0]; // single input mode

        let intrinsic = self.tick(
            timestamp,
            1. / sample_rate.0 as f64,
            crate::BUFFER_SIZE as _,
        );
        let mut buffer = inputs[0].clone(); // get new buf
        buffer.force_mono();
        buffer
            .channel_data_mut(0)
            .copy_from_slice(intrinsic.as_slice());

        outputs[0] = input.add(&buffer, ChannelInterpretation::Discrete);
    }

    fn tail_time(&self) -> bool {
        true // has intrinsic value
    }
}

pub(crate) fn audio_param_pair(
    opts: AudioParamOptions,
    registration: AudioContextRegistration<'_>,
) -> (AudioParam<'_>, AudioParamProcessor) {
    let (sender, receiver) = mpsc::channel();
    let shared_value = Arc::new(AtomicF64::new(opts.default_value as f64));

    let param = AudioParam {
        registration,
        value: shared_value.clone(),
        sender,
    };

    let render = AudioParamProcessor {
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

impl<'a> AudioParam<'a> {
    pub fn value(&self) -> f32 {
        self.value.load() as _
    }
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

    // helper function to detach from context (for borrow reasons)
    pub(crate) fn into_raw_parts(self) -> (Arc<AtomicF64>, Sender<AutomationEvent>) {
        (self.value, self.sender)
    }

    // helper function to attach to context (for borrow reasons)
    pub(crate) fn from_raw_parts(
        registration: AudioContextRegistration<'a>,
        parts: (Arc<AtomicF64>, Sender<AutomationEvent>),
    ) -> Self {
        Self {
            registration,
            value: parts.0,
            sender: parts.1,
        }
    }
}

impl AudioParamProcessor {
    pub fn value(&self) -> f32 {
        if self.value.is_nan() {
            self.default_value
        } else {
            self.value.clamp(self.min_value, self.max_value)
        }
    }

    fn tick(&mut self, ts: f64, dt: f64, count: usize) -> Vec<f32> {
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

        self.shared_value.store(self.value() as f64);

        assert_eq!(result.len(), count);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{AsBaseAudioContext, OfflineAudioContext};

    use super::*;

    #[test]
    fn test_steps_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

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
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        let opts = AudioParamOptions {
            automation_rate: AutomationRate::K,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should clamp
        param.set_value_at_time(8., 10.0); // should not occur 1st run

        let vs = render.tick(0., 1., 10);
        assert_eq!(vs, vec![0.; 10]);

        let vs = render.tick(10., 1., 10);
        assert_eq!(vs, vec![8.; 10]);
    }
}
