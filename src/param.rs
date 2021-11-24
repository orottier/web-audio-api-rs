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
    SetValue,
    SetValueAtTime,
    LinearRampToValueAtTime,
    ExponentialRampToValueAtTime,
}

#[derive(Copy, Clone, Debug)]
pub struct AutomationEvent {
    event_type: AutomationType,
    value: f32,
    time: f64,
}

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
    automation_rate: AutomationRate, // // treat as readonly for now
    default_value: f32,              // readonly
    min_value: f32,                  // readonly
    max_value: f32,                  // readonly
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

impl AudioParam {
    pub fn automation_rate(&self) -> AutomationRate {
        self.automation_rate
    }

    pub fn default_value(&self) -> f32 {
        self.default_value
    }

    pub fn min_value(&self) -> f32 {
        self.min_value
    }

    pub fn max_value(&self) -> f32 {
        self.max_value
    }

    // @note - we need to test more how this behaves
    pub fn value(&self) -> f32 {
        self.value.load() as _
    }

    // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
    // Setting this attribute has the effect of assigning the requested value
    // to the [[current value]] slot, and calling the setValueAtTime() method
    // with the current AudioContext's currentTime and [[current value]].
    // Any exceptions that would be thrown by setValueAtTime() will also be
    // thrown by setting this attribute.
    pub fn set_value(&self, value: f32) {
        let clamped = value.clamp(self.min_value, self.max_value);
        self.value.store(clamped as f64);

        // this event is ment to update param intrisic value before any calculation is done
        let event = AutomationEvent {
            event_type: AutomationType::SetValue,
            value: clamped,
            time: 0.,
        };

        self.send_event(event);
    }

    pub fn set_value_at_time(&self, value: f32, time: f64) {
        let event = AutomationEvent {
            event_type: AutomationType::SetValueAtTime,
            value,
            time,
        };

        self.send_event(event);
    }

    pub fn linear_ramp_to_value_at_time(&self, value: f32, time: f64) {
        let event = AutomationEvent {
            event_type: AutomationType::LinearRampToValueAtTime,
            value,
            time,
        };

        self.send_event(event);
    }

    pub fn exponential_ramp_to_value_at_time(&self, value: f32, time: f64) {
        // @note - this should probably `panic` as this could
        //  crash at runtime.
        // cf. Error pattern in `iir_filter.rs`
        if value == 0. {
            panic!(
                "RangeError: Failed to execute 'exponentialRampToValueAtTime'
                on 'AudioParam': The float target value provided (0) should not be
                in the range ({:+e}, {:+e})",
                -f32::MIN_POSITIVE,
                f32::MIN_POSITIVE
            )
        }

        let event = AutomationEvent {
            event_type: AutomationType::ExponentialRampToValueAtTime,
            value,
            time,
        };

        self.send_event(event);
    }

    // helper function to detach from context (for borrow reasons)
    pub(crate) fn into_raw_parts(
        self,
    ) -> (
        AutomationRate,
        f32,
        f32,
        f32,
        Arc<AtomicF64>,
        Sender<AutomationEvent>,
    ) {
        (
            self.automation_rate,
            self.default_value,
            self.min_value,
            self.max_value,
            self.value,
            self.sender,
        )
    }

    // helper function to attach to context (for borrow reasons)
    pub(crate) fn from_raw_parts(
        registration: AudioContextRegistration,
        parts: (
            AutomationRate,
            f32,
            f32,
            f32,
            Arc<AtomicF64>,
            Sender<AutomationEvent>,
        ),
    ) -> Self {
        Self {
            registration,
            automation_rate: parts.0,
            default_value: parts.1,
            min_value: parts.2,
            max_value: parts.3,
            value: parts.4,
            sender: parts.5,
        }
    }

    fn send_event(&self, event: AutomationEvent) {
        if cfg!(test) {
            // bypass audiocontext enveloping of control messages for simpler testing
            self.sender.send(event).unwrap();
        } else {
            self.context().pass_audio_param_event(&self.sender, event);
        }
    }
}

#[derive(Debug)]
pub(crate) struct AudioParamProcessor {
    value: f32, // @todo - rename to `intrisic_value` to match the spec
    shared_value: Arc<AtomicF64>,
    receiver: Receiver<AutomationEvent>,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    events: BinaryHeap<AutomationEvent>,
    last_event: Option<AutomationEvent>,
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

        let period = 1. / sample_rate.0 as f64;
        let param_intrisic_value = self.tick(timestamp, period, BUFFER_SIZE);

        let mut param_computed_value = inputs[0].clone(); // get new buf
        param_computed_value.force_mono();
        param_computed_value
            .channel_data_mut(0)
            .copy_from_slice(param_intrisic_value);
        // mix incoming signal
        param_computed_value.add(input, ChannelInterpretation::Discrete);

        outputs[0] = param_computed_value;
    }

    fn tail_time(&self) -> bool {
        true // has intrinsic value
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

    fn tick(&mut self, ts: f64, dt: f64, count: usize) -> &[f32] {
        // println!("> tick - ts: {}, dt: {}, count: {}", ts, dt, count);
        // handle incoming automation events in sorted queue
        // let events: Vec<AutomationEvent> = self.receiver.try_iter().collect();

        // cf. https://www.w3.org/TR/webaudio/#computation-of-value
        // 1. paramIntrinsicValue will be calculated at each time, which is either the
        // value set directly to the value attribute, or, if there are any automation
        // events with times before or at this time, the value as calculated from
        // these events. If automation events are removed from a given time range,
        // then the paramIntrinsicValue value will remain unchanged and stay at its
        // previous value until either the value attribute is directly set, or
        // automation events are added for the time range.
        for event in self.receiver.try_iter() {
            // param intrisic value must be updated from the set_value call
            // but we don't want to insert these events in the queue
            if event.event_type == AutomationType::SetValue {
                let current_value = self.shared_value.load() as f32;
                self.value = current_value;
            }

            // @note - should probably live in its own method just for clarity but
            // can't get rid of this error:
            //    for event in self.receiver.try_iter() {
            //                 ------------------------
            //                 |
            //                 immutable borrow occurs here
            //                 immutable borrow later used here
            //
            //        self.insert_event(event);
            //            ^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here
            //
            // if no event in the timeline and event_type is `LinearRampToValueAtTime`
            // or `ExponentialRampToValue` at time, we must insert a `SetValueAtTime`
            // with intrisic value and calling time.
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
            if self.events.is_empty()
                && (event.event_type == AutomationType::LinearRampToValueAtTime
                    || event.event_type == AutomationType::ExponentialRampToValueAtTime)
            {
                let set_value_event = AutomationEvent {
                    event_type: AutomationType::SetValueAtTime,
                    value: self.value,
                    // make sure the event is applied before any other event
                    // time will be replaced by block timestamp durin event processing
                    time: 0.,
                };

                self.events.push(set_value_event);
            }

            // @todo - many checks need to be done for setValueCurveAtTime events

            self.events.push(event);
        }

        // 2. Set [[current value]] to the value of paramIntrinsicValue at the
        // beginning of this render quantum.
        //
        // in case of set value
        self.shared_value.store(self.value() as f64);

        // Clear the vec from previously buffered data
        self.buffer.clear();

        let is_a_rate = self.automation_rate == AutomationRate::A;
        let is_k_rate = !is_a_rate;

        if is_k_rate {
            // filling the vec already, no expensive calculations are performed later
            for _ in 0..count {
                self.buffer.push(self.value())
            }
        };

        // time at the beginning of the next render quantum
        let max_ts = ts + dt * count as f64;

        loop {
            let some_event = self.events.peek();
            // println!("> Handle event: {:?}, ts: {:?}", some_event, ts);

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
                        AutomationType::SetValue | AutomationType::SetValueAtTime => {
                            let value = event.value;
                            let mut time = event.time;

                            // `set_value` calls and implicitely inserted events
                            // are inserted with a `time = 0.` to make sure
                            // they are processed first, replacing w/ ts allow
                            // to conform to the spec:
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime

                            if time == 0. {
                                time = ts;
                            }

                            let end_index = ((time - ts).max(0.) / dt) as usize;
                            let end_index_clipped = end_index.min(count);

                            // fill remaining buffer for A-rate processing with
                            // intrisic value until with reach event.time
                            // nothing is done here for K-rate, buffer is already full
                            for _ in self.buffer.len()..end_index_clipped {
                                self.buffer.push(self.value());
                            }

                            if time > max_ts {
                                break;
                            } else {
                                self.value = value;

                                // no computation has been done here, it's a
                                // strict unequality check
                                #[allow(clippy::float_cmp)]
                                if time != event.time {
                                    // store as last event with the applied time
                                    let mut event = self.events.pop().unwrap();
                                    event.time = time;
                                    self.last_event = Some(event);
                                } else {
                                    self.last_event = self.events.pop();
                                }
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                        // 𝑣(𝑡) = 𝑉0 + (𝑉1−𝑉0) * ((𝑡−𝑇0) / (𝑇1−𝑇0))
                        AutomationType::LinearRampToValueAtTime => {
                            let last_event = self.last_event.unwrap();

                            let last_time = last_event.time;
                            let end_time = event.time;
                            let duration = (end_time - last_time) as f32;

                            let last_value = last_event.value;
                            let end_value = event.value;
                            let dv = end_value - last_value;

                            let start_index = self.buffer.len();
                            let end_index = ((end_time - ts).max(0.) / dt) as usize;
                            let end_index_clipped = end_index.min(count);

                            // compute "real" value according to `t` then clamp it
                            // cf. Example 7 https://www.w3.org/TR/webaudio/#computation-of-value
                            if is_a_rate && end_index_clipped > self.buffer.len() {
                                let mut time = ts + start_index as f64 * dt;
                                let mut clamped: f32;

                                for _ in start_index..end_index_clipped {
                                    let value =
                                        last_value + dv * (time - last_time) as f32 / duration;
                                    clamped = value.clamp(self.min_value, self.max_value);
                                    self.buffer.push(clamped);

                                    time += dt;
                                    self.value = clamped;
                                }
                            } else if is_k_rate {
                                let time = ts + end_index_clipped as f64 * dt;
                                let value = last_value + dv * (time - last_time) as f32 / duration;
                                let clamped = value.clamp(self.min_value, self.max_value);

                                self.value = clamped;
                            }

                            if end_time > max_ts {
                                break;
                            } else {
                                // next intrisic value
                                self.value = end_value.clamp(self.min_value, self.max_value);
                                self.last_event = self.events.pop();
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
                        // v(t) = v1 * (v2/v1)^((t-t1) / (t2-t1))
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

                                if is_a_rate && end_index_clipped > self.buffer.len() {
                                    let mut time = ts + start_index as f64 * dt;
                                    let mut clamped: f32;

                                    for _ in start_index..end_index_clipped {
                                        // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
                                        let phase = (time - last_time) as f32 / duration;
                                        let val = last_value * ratio.powf(phase);
                                        clamped = val.clamp(self.min_value, self.max_value);
                                        self.buffer.push(clamped);

                                        time += dt;
                                        self.value = clamped;
                                    }
                                } else if is_k_rate {
                                    let time = ts + end_index_clipped as f64 * dt;
                                    let phase = (time - last_time) as f32 / duration;
                                    let val = last_value * ratio.powf(phase);
                                    let clamped = val.clamp(self.min_value, self.max_value);
                                    // store as intrisic value for next block
                                    self.value = clamped;
                                }

                                if end_time > max_ts {
                                    break;
                                } else {
                                    // next intrisic value
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

pub(crate) fn audio_param_pair(
    opts: AudioParamOptions,
    registration: AudioContextRegistration,
) -> (AudioParam, AudioParamProcessor) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let shared_value = Arc::new(AtomicF64::new(opts.default_value as f64));

    let param = AudioParam {
        registration,
        automation_rate: opts.automation_rate,
        default_value: opts.default_value,
        min_value: opts.min_value,
        max_value: opts.max_value,
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

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{AsBaseAudioContext, OfflineAudioContext};

    use super::*;

    #[test]
    fn test_default_and_accessors() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, _render) = audio_param_pair(opts, context.mock_registration());

        assert_eq!(param.automation_rate(), AutomationRate::A);
        assert_float_eq!(param.default_value(), 0., ulps_all <= 0);
        assert_float_eq!(param.min_value(), -10., ulps_all <= 0);
        assert_float_eq!(param.max_value(), 10., ulps_all <= 0);
        assert_float_eq!(param.value(), 0., ulps_all <= 0);
    }

    #[test]
    fn test_set_value() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value(2.);
        assert_float_eq!(param.value(), 2., ulps_all <= 0);

        let vs = render.tick(0., 1., 10);

        // current_value should not be overriden by intrisic value
        assert_float_eq!(param.value(), 2., ulps_all <= 0);
        assert_float_eq!(vs, &[2.; 10][..], ulps_all <= 0);
    }

    #[test]
    fn test_set_value_clamped() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -1.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value(2.);
        assert_float_eq!(param.value(), 1., ulps_all <= 0);

        let vs = render.tick(0., 1., 10);

        // current_value should not be overriden by intrisic value
        assert_float_eq!(param.value(), 1., ulps_all <= 0);
        assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
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

            param.set_value_at_time(5., 2.0);
            param.set_value_at_time(12., 8.0); // should clamp
            param.set_value_at_time(8., 10.0); // should not occur 1st run

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0., 5., 5., 5., 5., 5., 5., 10., 10.][..],
                ulps_all <= 0
            );

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[8.; 10][..], ulps_all <= 0);
        }

        {
            // events spread on several blocks
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: -10.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(5., 2.0);
            param.set_value_at_time(8., 12.0);

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

        param.set_value_at_time(5., 2.0);
        param.set_value_at_time(12., 8.0); // should not appear in results
        param.set_value_at_time(8., 10.0); // should not occur 1st run
        param.set_value_at_time(3., 14.0); // should appear in 3rd run

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
        param.set_value_at_time(5., 2.0);
        // ramp to 8 from t = 2 to t = 5
        param.linear_ramp_to_value_at_time(8.0, 5.0);
        // ramp to 0 from t = 5 to t = 13
        param.linear_ramp_to_value_at_time(0., 13.0);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0., 5., 6., 7., 8., 7., 6., 5., 4.][..],
            ulps_all <= 0
        );
    }

    #[test]
    fn test_linear_ramp_arate_end_of_block() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 0 at t = 0
        param.set_value_at_time(0., 0.);
        // ramp to 9 from t = 0 to t = 9
        param.linear_ramp_to_value_at_time(9.0, 9.0);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
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

        // mimic a ramp inserted after start
        // i.e. setTimeout(() => param.linearRampToValueAtTime(10, now + 10)), 10 * 1000);
        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);

        param.linear_ramp_to_value_at_time(10.0, 20.0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            ulps_all <= 0
        );

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[10.; 10][..], ulps_all <= 0);
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
        param.linear_ramp_to_value_at_time(20.0, 20.0);

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
    fn test_linear_ramp_arate_clamp() {
        // must be compliant with ex.7 cf. https://www.w3.org/TR/webaudio/#computation-of-value
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamOptions {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 3.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.linear_ramp_to_value_at_time(5.0, 5.0);
        param.linear_ramp_to_value_at_time(0., 10.0);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 3., 3., 3., 3., 2., 1.][..],
            ulps_all <= 0
        );

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);
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
            param.linear_ramp_to_value_at_time(20.0, 20.0);
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
            param.linear_ramp_to_value_at_time(15.0, 15.0);
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
        param.set_value_at_time(0.0001, 0.);
        // ramp to 1 from t = 0 to t = 10
        param.exponential_ramp_to_value_at_time(1.0, 10.);

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
        param.set_value_at_time(start, 3.);
        // ramp to 1 from t = 3. to t = 13.
        param.exponential_ramp_to_value_at_time(end, 13.);

        // compute resulting buffer:
        let mut res = vec![0.; 3];
        // set_value is implicit here as this is the first value of the computed ramp
        // exponential ramp (v(t) = v1*(v2/v1)^((t-t1)/(t2-t1)))
        for t in 0..10 {
            let value = start * (end / start).powf(t as f32 / 10.);
            res.push(value);
        }
        // fill remaining with target value
        res.append(&mut vec![1.; 7]);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &res[0..10], ulps_all <= 0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &res[10..20], ulps_all <= 0);
    }

    #[test]
    fn test_exponential_ramp_a_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            // zero target
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set v=0. at t=0 (0. is a special case)
            param.set_value_at_time(0., 0.);
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
                ulps_all <= 0
            );
        }

        {
            // opposite signs
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: -1.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set v=-1. at t=0
            param.set_value_at_time(-1., 0.);
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[-1., -1., -1., -1., -1., 1., 1., 1., 1., 1.][..],
                ulps_all <= 0
            );
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
        let (param, mut _render) = audio_param_pair(opts, context.mock_registration());
        param.exponential_ramp_to_value_at_time(0.0, 10.);
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
        param.set_value_at_time(start, 3.);
        // ramp to 1 from t = 3. to t = 13.
        param.exponential_ramp_to_value_at_time(end, 13.);

        // compute resulting buffer:
        let mut res = vec![0.; 3];
        // set_value is implicit here as this is the first value of the computed ramp
        // exponential ramp (v(t) = v1*(v2/v1)^((t-t1)/(t2-t1)))
        for t in 0..10 {
            let value = start * (end / start).powf(t as f32 / 10.);
            res.push(value);
        }
        // fill remaining with target value
        res.append(&mut vec![1.; 7]);

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

        {
            // zero target
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], ulps_all <= 0);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
        }

        {
            // opposite signs
            let opts = AudioParamOptions {
                automation_rate: AutomationRate::K,
                default_value: -1.,
                min_value: -1.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[-1.; 10][..], ulps_all <= 0);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], ulps_all <= 0);
        }
    }
}
