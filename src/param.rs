//! AudioParam interface

use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::alloc::AudioBuffer;
use crate::buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::context::AudioContextRegistration;
use crate::node::AudioNode;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::{AtomicF64, SampleRate};

use crossbeam_channel::{Receiver, Sender};

use AutomationEvent::*;

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
    LinearRampToValueAtTime { v: f32, end: f64 },
}

impl AutomationEvent {
    fn time(&self) -> f64 {
        match &self {
            SetValueAtTime { start, .. } => *start,
            LinearRampToValueAtTime { end, .. } => *end,
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
    registration: AudioContextRegistration,
    value: Arc<AtomicF64>,
    sender: Sender<AutomationEvent>,
}

impl AudioNode for AudioParam {
    fn registration(&self) -> &AudioContextRegistration {
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
    fn process(
        &mut self,
        inputs: &[AudioBuffer],
        outputs: &mut [AudioBuffer],
        _params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) {
        let input = &inputs[0]; // single input mode

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

        buffer.add(input, ChannelInterpretation::Discrete);

        outputs[0] = buffer;
    }

    fn tail_time(&self) -> bool {
        true // has intrinsic value
    }
}

pub(crate) fn audio_param_pair(
    opts: AudioParamOptions,
    registration: AudioContextRegistration,
) -> (AudioParam, AudioParamProcessor) {
    let (sender, receiver) = crossbeam_channel::unbounded();
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

impl AudioParam {
    pub fn value(&self) -> f32 {
        self.value.load() as _
    }

    pub fn set_value(&self, v: f32) {
        let event = SetValueAtTime { v, start: 0. };
        self.context().pass_audio_param_event(&self.sender, event);
    }

    pub fn set_value_at_time(&self, v: f32, start: f64) {
        let event = SetValueAtTime { v, start };
        self.context().pass_audio_param_event(&self.sender, event);
    }

    pub fn linear_ramp_to_value_at_time(&self, v: f32, end: f64) {
        let event = LinearRampToValueAtTime { v, end };
        self.context().pass_audio_param_event(&self.sender, event);
    }

    // helper function to detach from context (for borrow reasons)
    pub(crate) fn into_raw_parts(self) -> (Arc<AtomicF64>, Sender<AutomationEvent>) {
        (self.value, self.sender)
    }

    // helper function to attach to context (for borrow reasons)
    pub(crate) fn from_raw_parts(
        registration: AudioContextRegistration,
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
        // store incoming automation events in sorted queue
        for event in self.receiver.try_iter() {
            self.events.push(event);
        }

        // setup return value buffer
        let a_rate = self.automation_rate == AutomationRate::A;
        let mut result = if a_rate {
            // empty buffer
            Vec::with_capacity(count)
        } else {
            // filling the vec already, no expensive calculations are performed later
            vec![self.value(); count]
        };

        // end of the render quantum
        let max_ts = ts + dt * count as f64;

        loop {
            match self.events.peek() {
                None => {
                    // fill remaining buffer for K-rate processing
                    for _ in result.len()..count {
                        result.push(self.value());
                    }
                    break;
                }
                Some(SetValueAtTime { v, start }) => {
                    let end_index = ((start - ts).max(0.) / dt) as usize;
                    let end_index = end_index.min(count);

                    // fill remaining buffer for K-rate processing
                    for _ in result.len()..end_index {
                        result.push(self.value());
                    }

                    // if start time is outside this render quantum, return
                    if *start > max_ts {
                        break;
                    }

                    self.value = *v;
                }
                Some(LinearRampToValueAtTime { v, end }) => {
                    let end_index = ((end - ts).max(0.) / dt) as usize;
                    if a_rate && end_index > result.len() {
                        let start_index = result.len();

                        let dv = v - self.value;
                        let dt = end_index - start_index;
                        let slope = dv / dt as f32;

                        let end_index_clipped = end_index.min(count);
                        let n_values = end_index_clipped - start_index;

                        for i in 0..n_values {
                            let val = self.value + i as f32 * slope;
                            result.push(val.clamp(self.min_value, self.max_value));
                        }
                    }

                    // if end time is outside this render quantum, return
                    if *end > max_ts {
                        break;
                    }

                    self.value = *v;
                }
            }

            // previous event was handled
            self.events.pop();
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

    // Bypass AudioContext enveloping of control messages for simpler testing
    impl AudioParam {
        pub fn set_value_direct(&self, v: f32) {
            let event = SetValueAtTime { v, start: 0. };
            self.sender.send(event).unwrap()
        }
        pub fn set_value_at_time_direct(&self, v: f32, start: f64) {
            let event = SetValueAtTime { v, start };
            self.sender.send(event).unwrap()
        }
        pub fn linear_ramp_to_value_at_time_direct(&self, v: f32, end: f64) {
            let event = LinearRampToValueAtTime { v, end };
            self.sender.send(event).unwrap()
        }
    }

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

        param.set_value_at_time_direct(5., 2.0);
        param.set_value_at_time_direct(12., 8.0); // should clamp
        param.set_value_at_time_direct(8., 10.0); // should not occur 1st run

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

        param.set_value_at_time_direct(5., 2.0);
        param.set_value_at_time_direct(12., 8.0); // should clamp
        param.set_value_at_time_direct(8., 10.0); // should not occur 1st run

        let vs = render.tick(0., 1., 10);
        assert_eq!(vs, vec![0.; 10]);

        let vs = render.tick(10., 1., 10);
        assert_eq!(vs, vec![8.; 10]);
    }

    #[test]
    fn test_linear_ramp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 5 at t = 2
        param.set_value_at_time_direct(5., 2.0);
        // ramp to 8 from t = 2 to t = 5
        param.linear_ramp_to_value_at_time_direct(8.0, 5.0);
        // ramp to 0 from t = 5 to t = 13
        param.linear_ramp_to_value_at_time_direct(0., 13.0);

        let vs = render.tick(0., 1., 10);
        assert_eq!(vs, vec![0., 0., 5., 6., 7., 8., 7., 6., 5., 4.]);
    }

    #[test]
    fn test_linear_ramp_clamp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -1.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.linear_ramp_to_value_at_time_direct(5.0, 5.0);
        param.linear_ramp_to_value_at_time_direct(0., 10.0);

        let vs = render.tick(0., 1., 10);
        // Todo last value should actually be zero, but it rounds not nicely
        // I guess this will not be a problem in practise.
        assert_eq!(vs, vec![0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]);
    }
}
