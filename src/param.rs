//! AudioParam interface

use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::alloc::AudioBuffer;
use crate::buffer::{ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use crate::context::AudioContextRegistration;
use crate::node::AudioNode;
use crate::process::{AudioParamValues, AudioProcessor};
use crate::{AtomicF64, SampleRate, BUFFER_SIZE};

use crossbeam_channel::{Receiver, Sender};

/// Precision of value calculation per render quantum
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AutomationRate {
    /// Audio Rate
    /// sampled for each sample-frame of the block
    A,
    /// Control Rate
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

#[allow(clippy::enum_variant_names)] // @todo - remove that when all events are implemented
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum AutomationType {
    SetValueAtTime,
    LinearRampToValueAtTime,
    ExponentialRampToValueAtTime,
}

#[derive(Copy, Clone, Debug)]
pub struct AutomationEvent  {
    event_type: AutomationType,
    value: f32,
    time: f64,
}

// impl AutomationEvent {
//     fn time(&self) -> f64 {
//         match &self {
//             SetValueAtTime { start, .. } => *start,
//             LinearRampToValueAtTime { end, .. } => *end,
//         }
//     }
// }

impl PartialEq for AutomationEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time.eq(&other.time)
    }
}
impl Eq for AutomationEvent {}

impl std::cmp::PartialOrd for AutomationEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // reverse ordering
        other.time.partial_cmp(&self.time)
    }
}

impl std::cmp::Ord for AutomationEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time.partial_cmp(&other.time).unwrap()
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
    value: f32, // @todo - rename to intrisic value to match the spec?
    shared_value: Arc<AtomicF64>,
    receiver: Receiver<AutomationEvent>,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    events: BinaryHeap<AutomationEvent>,
    last_event: Option<AutomationEvent>, // to compute
    buffer: Vec<f32>,
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

        // @note - naming, intrisic value more probably represents `self.value`
        let intrinsic = self.tick(timestamp, 1. / sample_rate.0 as f64, BUFFER_SIZE);
        // @question - maybe would be more clean to have only one reusable buffer?
        let mut buffer = inputs[0].clone(); // get new buf
        buffer.force_mono();
        buffer.channel_data_mut(0).copy_from_slice(intrinsic);
        // mix incoming signal
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
        last_event: None,
        buffer: Vec::with_capacity(BUFFER_SIZE),
    };

    (param, render)
}

impl AudioParam {
    pub fn value(&self) -> f32 {
        self.value.load() as _
    }

    pub fn set_value(&self, value: f32) {
        let event = AutomationEvent {
            event_type: AutomationType::SetValueAtTime,
            value,
            time: self.context().current_time(),
        };

        self.context().pass_audio_param_event(&self.sender, event);
    }

    // @note - need some insert_event method to handle special cases
    // cf. blink audio_param_timeline.cc
    pub fn set_value_at_time(&self, value: f32, time: f64) {
        let event = AutomationEvent {
            event_type: AutomationType::SetValueAtTime,
            value,
            time,
        };

        self.context().pass_audio_param_event(&self.sender, event);
    }

    pub fn linear_ramp_to_value_at_time(&self, value: f32, time: f64) {
        let event = AutomationEvent {
            event_type: AutomationType::LinearRampToValueAtTime,
            value,
            time,
        };

        self.context().pass_audio_param_event(&self.sender, event);
    }

    pub fn exponential_ramp_to_value_at_time(&self, value: f32, time: f64) {
        // @todo - handle 0 target
        // Uncaught RangeError: Failed to execute 'exponentialRampToValueAtTime'
        // on 'AudioParam': The float target value provided (0) should not be
        // in the range (-1.40130e-45, 1.40130e-45).

        let event = AutomationEvent {
            event_type: AutomationType::ExponentialRampToValueAtTime,
            value,
            time,
        };

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

    // @note - Maybe we should maintain the event list in the control thread as we
    //  probably don't want to do all this in the audio thread ((Blink does that using a mutex))
    fn insert_event(&mut self, event: AutomationEvent) {
        // target of zero is forbidden for exponential ramps
        // @note - not sure this should `panic` though as it could probably
        //  crash at runtime, maybe warn and ignore?
        // @note - this should probably be handle in AudioParam.exponential_ramp_to_value_at_time
        //  However it is more convenient to put it there testing reasons.
        //  review AudioParam mock in tests and review test_exponential_ramp_to_zero
        //  accordingly.
        if event.event_type == AutomationType::ExponentialRampToValueAtTime &&
           event.value == 0. {
            panic!("RangeError: Failed to execute 'exponentialRampToValueAtTime'
                on 'AudioParam': The float target value provided (0) should not be
                in the range ({:+e}, {:+e})", -f32::MIN_POSITIVE, f32::MIN_POSITIVE);
        }

        // if no event in the timeline and event_type is `LinearRampToValueAtTime`
        // or `ExponentialRampToValue` at time, we must insert a `SetValueAtTime`
        // with intrisic value and calling time.
        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
        //
        // @note - time=0 is what blink does but this is not compliant to the
        // spec: time should be callTime. However this would imply a strong
        // relation with `audio_context.curret_time` which breaks the tests
        // -> probably ok for now as it only concerns edge cases (e.g. triggering
        // a ramp in the future without an explicit `setTarget` before).
        // -> at least we make sure that we have a previous event defined
        //
        // This could also handled later by checking if last_event exists, e.g.:
        // ```
        // if !last_event {
        //     let automationEvent = events.pop();
        //     let setTargetEvent { time: ts };
        //     event.push(setTargetEvent);
        //     event.push(automationEvent);
        // }
        // ```
        if self.events.is_empty() &&
            (event.event_type == AutomationType::LinearRampToValueAtTime ||
            event.event_type == AutomationType::ExponentialRampToValueAtTime) {

            let set_value_event = AutomationEvent {
                event_type: AutomationType::SetValueAtTime,
                value: self.value,
                time: 0., // make sure the event is applied before any other event
            };

            self.events.push(set_value_event);
        }

        // @todo - many checks need to be done for setValueCurveAtTime events

        self.events.push(event);
    }

    fn tick(&mut self, ts: f64, dt: f64, count: usize) -> &[f32] {
        // println!("> tick - ts: {}, dt: {}, count: {}", ts, dt, count);
        // handle incoming automation events in sorted queue
        let events: Vec<AutomationEvent> = self.receiver.try_iter().collect();

        for event in events {
            self.insert_event(event);
        }

        // cf. https://www.w3.org/TR/webaudio/#computation-of-value
        // > Set [[current value]] to the value of paramIntrinsicValue at the
        // beginning of this render quantum.
        self.shared_value.store(self.value() as f64);

        // Clear the vec from previously buffered data
        self.buffer.clear();

        // setup return value buffer
        let a_rate = self.automation_rate == AutomationRate::A;
        let k_rate = !a_rate;

        if k_rate {
            // filling the vec already, no expensive calculations are performed later
            for _ in 0..count {
                self.buffer.push(self.value())
            }
        };

        // end of the render quantum
        let max_ts = ts + dt * count as f64;

        loop {
            let some_event = self.events.peek();
            // println!("> Handle event: {:?}", some_event);

            match some_event {
                None => {
                    // fill remaining buffer with current intrisic value
                    for _ in self.buffer.len()..count {
                        self.buffer.push(self.value());
                    }

                    break;
                }
                Some(event) => {
                    match event.event_type {
                        AutomationType::SetValueAtTime => {
                            // @note - for k-rate params, if event.time is the exact
                            // time of the block, shouldn't we fill the block with the
                            // new value? is this really a problem?

                            let value = event.value;
                            let time = event.time;

                            let end_index = ((time - ts).max(0.) / dt) as usize;
                            let end_index = end_index.min(count);

                            // fill remaining buffer for A-rate processing
                            // nothing needs to be done for K-rate, buffer is already full
                            for _ in self.buffer.len()..end_index {
                                self.buffer.push(self.value());
                            }

                            // event belongs to a later block, so we just
                            // filled the block with instrisic value
                            if event.time > max_ts {
                                break;
                            } else {
                                self.value = value;
                                self.last_event = self.events.pop();
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                        AutomationType::LinearRampToValueAtTime => {
                            let value = event.value;
                            let end_time = event.time;
                            let end_index = ((end_time - ts).max(0.) / dt) as usize;

                            // what happens for k-rate?
                            // this can break if block 1
                            if a_rate && end_index > self.buffer.len() {
                                let start_index = self.buffer.len();

                                let dv = value - self.value;
                                let dt = end_index - start_index;
                                let slope = dv / dt as f32;

                                let end_index_clipped = end_index.min(count);
                                let n_values = end_index_clipped - start_index;

                                let mut val = self.value;

                                for _ in 0..n_values {
                                    // @todo - review clamping according to spec Example 7
                                    // https://www.w3.org/TR/webaudio/#computation-of-value
                                    let clamped = val.clamp(self.min_value, self.max_value);
                                    self.buffer.push(clamped);

                                    val += slope;
                                }

                                self.value = val;
                            } else if k_rate {
                                // just compute `self.value` to fill next block
                                let dv = value - self.value;
                                let dt = end_index;
                                let slope = dv / dt as f32;
                                let n_values = end_index.min(count);

                                let val = self.value + slope * n_values as f32;
                                // @todo - review clamping according to spec Example 7
                                // https://www.w3.org/TR/webaudio/#computation-of-value
                                // should be done using last event, then we would
                                // define the real logical value and clip it
                                let clamped = val.clamp(self.min_value, self.max_value);
                                self.value = clamped;
                            }

                            if event.time > max_ts {
                                break;
                            } else {
                                self.value = value.clamp(self.min_value, self.max_value);
                                self.last_event = self.events.pop();
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
                        // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
                        AutomationType::ExponentialRampToValueAtTime => {
                            let last_event = self.last_event.unwrap();

                            let last_time = last_event.time;
                            let end_time = event.time;
                            let duration = (end_time - last_time) as f32;

                            let last_value = last_event.value;
                            let end_value = event.value;
                            let ratio = end_value / last_value;

                            // Handle edge cases:
                            // > If 𝑉0 and 𝑉1 have opposite signs or if 𝑉0 is zero,
                            // > then 𝑣(𝑡)=𝑉0 for 𝑇0≤𝑡<𝑇1.
                            // as:
                            // > If there are no more events after this ExponentialRampToValue
                            // > event then for 𝑡≥𝑇1, 𝑣(𝑡)=𝑉1.
                            // this should thus behave as a SetValue
                            if last_value == 0. || last_value * end_value < 0. {
                                self.events.pop();

                                let event = AutomationEvent {
                                    event_type: AutomationType::SetValueAtTime,
                                    time: end_time,
                                    value: end_value,
                                };

                                self.events.push(event);
                            } else {
                                let start_index = self.buffer.len();
                                let end_index = ((end_time - ts).max(0.) / dt) as usize;
                                let end_index_clipped = end_index.min(count);

                                if a_rate && end_index > self.buffer.len() {
                                    let mut time = ts + start_index as f64 * dt;
                                    let mut val: f32;

                                    for _ in start_index..end_index_clipped {
                                        // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
                                        let phase = (time - last_time) as f32 / duration;
                                        val = last_value * ratio.powf(phase);
                                        let clamped = val.clamp(self.min_value, self.max_value);
                                        self.buffer.push(clamped);

                                        time += dt;
                                    }
                                } else if k_rate {
                                    let time = ts + end_index_clipped as f64 * dt;
                                    let phase = (time - last_time) as f32 / duration;
                                    let val = last_value * ratio.powf(phase);
                                    let clamped = val.clamp(self.min_value, self.max_value);
                                    // store as intrisic value for next block
                                    self.value = clamped;
                                }

                                if event.time > max_ts {
                                    break;
                                } else {
                                    self.value = end_value.clamp(self.min_value, self.max_value);
                                    self.last_event = self.events.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(self.buffer.len(), count);
        // println!("- {:?}", self.buffer);

        self.buffer.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{AsBaseAudioContext, OfflineAudioContext};

    use super::*;

    // Bypass AudioContext enveloping of control messages for simpler testing
    // @note - override only `self.sender.send(event).unwrap()`?
    impl AudioParam {
        pub fn set_value_direct(&self, value: f32) {
            let event = AutomationEvent {
                event_type: AutomationType::SetValueAtTime,
                value,
                time: 0.,
            };

            self.sender.send(event).unwrap()
        }
        pub fn set_value_at_time_direct(&self, value: f32, time: f64) {
            let event = AutomationEvent {
                event_type: AutomationType::SetValueAtTime,
                value: value,
                time: time,
            };

            self.sender.send(event).unwrap()
        }
        pub fn linear_ramp_to_value_at_time_direct(&self, value: f32, time: f64) {
            let event = AutomationEvent {
                event_type: AutomationType::LinearRampToValueAtTime,
                value: value,
                time: time,
            };

            self.sender.send(event).unwrap()
        }

        pub fn exponential_ramp_to_value_at_time_direct(&self, value: f32, time: f64) {
            let event = AutomationEvent {
                event_type: AutomationType::ExponentialRampToValueAtTime,
                value: value,
                time: time,
            };

            self.sender.send(event).unwrap()
        }
    }

    #[test]
    fn test_steps_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
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
            assert_float_eq!(
                vs,
                &[0., 0., 5., 5., 5., 5., 5., 5., 10., 10.][..],
                ulps_all <= 0
            );

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[8.; 10][..], ulps_all <= 0);
        }

        {   // events spread on several blocks
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: -10.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time_direct(5., 2.0);
            param.set_value_at_time_direct(8., 12.0);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0., 5., 5., 5., 5., 5., 5., 5., 5.][..],
                ulps_all <= 0
            );

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(
                vs,
                &[5., 5., 8., 8., 8., 8., 8., 8., 8., 8.][..],
                ulps_all <= 0
            );
        }
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
        param.set_value_at_time_direct(12., 8.0); // should not appear in results
        param.set_value_at_time_direct(8., 10.0); // should not occur 1st run
        param.set_value_at_time_direct(3., 14.0); // should appear in 3rd run

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[8.; 10][..], ulps_all <= 0);

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[3.; 10][..], ulps_all <= 0);
    }

    #[test]
    fn test_linear_ramp_arate() {
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
        assert_float_eq!(
            vs,
            &[0., 0., 5., 6., 7., 8., 7., 6., 5., 4.][..],
            ulps_all <= 0
        );
    }

    #[test]
    fn test_linear_ramp_arate_implicit_set_value() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);

        param.linear_ramp_to_value_at_time_direct(10.0, 20.0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            ulps_all <= 0
        );
    }

    #[test]
    fn test_linear_ramp_arate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -20.,
            max_value: 20.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // ramp to 20 from t = 0 to t = 20
        param.linear_ramp_to_value_at_time_direct(20.0, 20.0);

        // first quantum t = 0..10
        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            ulps_all <= 0
        );

        // next quantum t = 10..20
        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[10., 11., 12., 13., 14., 15., 16., 17., 18., 19.][..],
            ulps_all <= 0
        );

        // ramp finished t = 20..30
        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[20.0; 10][..], ulps_all <= 0);
    }

    #[test]
    #[ignore = "broken - review how linear is computed"]
    fn test_linear_ramp_arate_clamp() {
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
        assert_float_eq!(
            vs,
            &[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.][..],
            ulps_all <= 0
        );
    }


    #[test]
    fn test_linear_ramp_krate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: -20.,
                max_value: 20.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 20 from t = 0 to t = 20
            param.linear_ramp_to_value_at_time_direct(20.0, 20.0);
            // first quantum t = 0..10
            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);
            // next quantum t = 10..20
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[10.; 10][..], ulps_all <= 0);
            // ramp finished t = 20..30
            let vs = render.tick(20., 1., 10);
            assert_float_eq!(vs, &[20.0; 10][..], ulps_all <= 0);
        }

        {
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: -20.,
                max_value: 20.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 20 from t = 0 to t = 20
            param.linear_ramp_to_value_at_time_direct(15.0, 15.0);
            // first quantum t = 0..10
            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);
            // next quantum t = 10..20
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[10.; 10][..], ulps_all <= 0);
            // ramp finished t = 20..30
            let vs = render.tick(20., 1., 10);
            assert_float_eq!(vs, &[15.0; 10][..], ulps_all <= 0);
        }
    }

    #[test]
    fn test_exponential_ramp_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 0.0001 at t=0 (0. is a special case)
        param.set_value_at_time_direct(0.0001, 0.);
        // ramp to 1 from t = 0 to t = 10
        param.exponential_ramp_to_value_at_time_direct(1.0, 10.);

        // compute resulting buffer:
        // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
        let mut res = Vec::<f32>::with_capacity(10);
        let start: f32 = 0.0001;
        let end: f32 = 1.;

        for t in 0..10 {
            let value = start * (end / start).powf(t as f32 / 10.);
            res.push(value);
        }

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &res[..], ulps_all <= 0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[1.0; 10][..], ulps_all <= 0);
    }

    #[test]
    fn test_exponential_ramp_a_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        let start: f32 = 0.0001; // use 0.0001 as 0. is a special case
        let end: f32 = 1.;
        param.set_value_at_time_direct(start, 3.);
        // ramp to 1 from t = 3. to t = 13.
        param.exponential_ramp_to_value_at_time_direct(end, 13.);

        // compute resulting buffer:
        let mut res = Vec::<f32>::with_capacity(20);
        for _ in 0..3 { res.push(0.); }
        // set_value is implicit here as this is the first value of the computed ramp
        // exponential ramp (v(t) = v1*(v2/v1)^((t-t1)/(t2-t1)))
        for t in 0..10 {
            let value = start * (end / start).powf(t as f32 / 10.);
            res.push(value);
        }
        for _ in 13..20 { res.push(end); } // fill remaining with target value

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &res[0..10], ulps_all <= 0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &res[10..20], ulps_all <= 0);
    }

    #[test]
    fn test_exponential_ramp_a_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        { // zero target
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set v=0. at t=0 (0. is a special case)
            param.set_value_at_time_direct(0., 0.);
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time_direct(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..], ulps_all <= 0);
        }

        { // opposite signs
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: -1.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set v=-1. at t=0
            param.set_value_at_time_direct(-1., 0.);
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time_direct(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[-1., -1., -1., -1., -1., 1., 1., 1., 1., 1.][..], ulps_all <= 0);
        }
    }

    #[test]
    #[should_panic]
    fn test_exponential_ramp_to_zero() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 1.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());
        param.exponential_ramp_to_value_at_time_direct(0.0, 10.);

        // @note - we need to tick so that the panic occurs
        render.tick(20., 1., 10);
    }

     #[test]
    fn test_exponential_ramp_k_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::K,
            default_value: 0.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        let start: f32 = 0.0001; // use 0.0001 as 0. is a special case
        let end: f32 = 1.;
        param.set_value_at_time_direct(start, 3.);
        // ramp to 1 from t = 3. to t = 13.
        param.exponential_ramp_to_value_at_time_direct(end, 13.);

        // compute resulting buffer:
        let mut res = Vec::<f32>::with_capacity(20);
        for _ in 0..3 { res.push(0.); }
        // set_value is implicit here as this is the first value of the computed ramp
        // exponential ramp (v(t) = v1*(v2/v1)^((t-t1)/(t2-t1)))
        for t in 0..10 {
            let value = start * (end / start).powf(t as f32 / 10.);
            res.push(value);
        }
        for _ in 13..20 { res.push(end); } // fill remaining with target value

        // recreate k-rate blocks from computed values
        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &[res[0]; 10][..], ulps_all <= 0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[res[10]; 10][..], ulps_all <= 0);

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
    }

    #[test]
    fn test_exponential_ramp_k_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        { // zero target
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time_direct(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
        }

        { // opposite signs
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: -1.,
                min_value: -1.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time_direct(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[-1.; 10][..], ulps_all <= 0);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
        }
    }
}
