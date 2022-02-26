//! AudioParam interface
use std::slice::{Iter, IterMut};
use std::sync::Arc;

use crate::context::AudioContextRegistration;
use crate::node::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{AtomicF32, SampleRate, RENDER_QUANTUM_SIZE};

use crossbeam_channel::{Receiver, Sender};

/// Precision of value calculation per render quantum
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AutomationRate {
    /// Audio Rate - sampled for each sample-frame of the block
    A,
    /// Control Rate - sampled at the time of the very first sample-frame,
    /// then used for the entire block
    K,
}

/// Options for constructing an [`AudioParam`]
pub struct AudioParamDescriptor {
    pub automation_rate: AutomationRate,
    pub default_value: f32,
    pub min_value: f32,
    pub max_value: f32,
}

#[derive(PartialEq, Eq, Debug)]
enum AudioParamEventType {
    SetValue,
    SetValueAtTime,
    LinearRampToValueAtTime,
    ExponentialRampToValueAtTime,
    CancelScheduledValues,
    SetTargetAtTime,
    CancelAndHoldAtTime,
    SetValueCurveAtTime,
}

#[derive(Debug)]
pub(crate) struct AudioParamEvent {
    event_type: AudioParamEventType,
    value: f32,
    time: f64,
    time_constant: Option<f64>, // populated by `SetTargetAtTime` events
    cancel_time: Option<f64>,   // populated by `CancelAndHoldAtTime` events
    duration: Option<f64>,      // populated by `SetValueCurveAtTime` events
    values: Option<Box<[f32]>>, // populated by `SetValueCurveAtTime` events
}

// Event queue that contains `AudioParamEvent`s, most of the time, events must be
// ordered (using stable sort), some operation may break this ordering (e.g. `push`)
// in which cases `sort` must be called explicitely.
// In the current implementation of the param rendering, `sort` is called once after
// all events have been inserted in the queue at each `tick` (with the exception
// `CancelAndHoldAtTime` which needs a clean queue to find its neighbors, but this
// occurs during the insertion of events)
// After this point, the queue should be considered sorted and no operations that
// breaks the ordering should be done.
#[derive(Debug)]
struct AudioParamEventTimeline {
    inner: Vec<AudioParamEvent>,
    dirty: bool,
}

impl AudioParamEventTimeline {
    fn new() -> Self {
        Self {
            inner: Vec::new(),
            dirty: false,
        }
    }

    fn push(&mut self, item: AudioParamEvent) {
        self.dirty = true;
        self.inner.push(item);
    }

    // `pop` and `retain` preserve order so they don't make the queue dirty
    fn pop(&mut self) -> Option<AudioParamEvent> {
        if !self.inner.is_empty() {
            Some(self.inner.remove(0))
        } else {
            None
        }
    }

    fn retain<F>(&mut self, func: F)
    where
        F: Fn(&AudioParamEvent) -> bool,
    {
        self.inner.retain(func);
    }

    // Only used to handle special cases in `ExponentialRampToValueAtTime`:
    // as the replaced item has the same time, order is preserved.
    // If the method turned out to be used elsewhere, this could maybe
    // become wrong, be careful here.
    fn replace_peek(&mut self, item: AudioParamEvent) {
        self.inner[0] = item;
    }

    fn is_empty(&mut self) -> bool {
        self.inner.is_empty()
    }

    fn unsorted_peek(&self) -> Option<&AudioParamEvent> {
        self.inner.get(0)
    }

    // panic if dirty, we are doing something wrong here
    fn peek(&self) -> Option<&AudioParamEvent> {
        if self.dirty {
            panic!("`AudioParamEventTimeline`: Invalid `.peek()` call, the queue is dirty");
        }
        self.inner.get(0)
    }

    fn next(&self) -> Option<&AudioParamEvent> {
        if self.dirty {
            panic!("`AudioParamEventTimeline`: Invalid `.next()` call, the queue is dirty");
        }
        self.inner.get(1)
    }

    fn sort(&mut self) {
        self.inner
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        self.dirty = false;
    }

    fn iter(&mut self) -> Iter<'_, AudioParamEvent> {
        self.inner.iter()
    }

    fn iter_mut(&mut self) -> IterMut<'_, AudioParamEvent> {
        self.inner.iter_mut()
    }
}

/// AudioParam controls an individual aspect of an AudioNode's functionality, such as volume.
pub struct AudioParam {
    registration: AudioContextRegistration,
    automation_rate: AutomationRate, // treat as readonly for now
    default_value: f32,              // readonly
    min_value: f32,                  // readonly
    max_value: f32,                  // readonly
    current_value: Arc<AtomicF32>,
    sender: Sender<AudioParamEvent>,
}

// helper struct to attach / detach to context (for borrow reasons)
#[derive(Clone)]
pub(crate) struct AudioParamRaw {
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    current_value: Arc<AtomicF32>,
    sender: Sender<AudioParamEvent>,
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

    // the choice here is to have this coherent with the first sample of
    // the last rendered block, which means `intrisic_value` must be calculated
    // for next_block_time at each tick.
    // @note - maybe check with spec editors that it is correct
    //
    // see. test_linear_ramp_arate_multiple_blocks
    //      test_linear_ramp_krate_multiple_blocks
    //      test_exponential_ramp_a_rate_multiple_blocks
    //      test_exponential_ramp_k_rate_multiple_blocks
    pub fn value(&self) -> f32 {
        self.current_value.load()
    }

    // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
    // Setting this attribute has the effect of assigning the requested value
    // to the [[current value]] slot, and calling the setValueAtTime() method
    // with the current AudioContext's currentTime and [[current value]].
    // Any exceptions that would be thrown by setValueAtTime() will also be
    // thrown by setting this attribute.
    pub fn set_value(&self, value: f32) -> &Self {
        let clamped = value.clamp(self.min_value, self.max_value);
        self.current_value.store(clamped);

        // this event is meant to update param intrisic value before any calculation
        // is done, will behave as SetValueAtTime with `time == block_timestamp`
        let event = AudioParamEvent {
            event_type: AudioParamEventType::SetValue,
            value: clamped,
            time: 0.,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn set_value_at_time(&self, value: f32, start_time: f64) -> &Self {
        let event = AudioParamEvent {
            event_type: AudioParamEventType::SetValueAtTime,
            value,
            time: start_time,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn linear_ramp_to_value_at_time(&self, value: f32, end_time: f64) -> &Self {
        let event = AudioParamEvent {
            event_type: AudioParamEventType::LinearRampToValueAtTime,
            value,
            time: end_time,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn exponential_ramp_to_value_at_time(&self, value: f32, end_time: f64) -> &Self {
        // @note - this should probably not `panic` as this could crash at runtime.
        // cf. Error pattern in `iir_filter.rs`
        // @todo - implement clean Errors
        if value == 0. {
            panic!(
                "RangeError: Failed to execute 'exponentialRampToValueAtTime'
                on 'AudioParam': The float target value provided (0) should not be
                in the range ({:+e}, {:+e})",
                -f32::MIN_POSITIVE,
                f32::MIN_POSITIVE
            )
        }

        let event = AudioParamEvent {
            event_type: AudioParamEventType::ExponentialRampToValueAtTime,
            value,
            time: end_time,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn set_target_at_time(&self, value: f32, start_time: f64, time_constant: f64) -> &Self {
        if time_constant <= 0. {
            panic!(
                "RangeError: Failed to execute 'setTargetAtTime' on 'AudioParam':
                 Time constant must be a finite positive number: {:?}",
                time_constant
            )
        }

        // what about time_constant = 0. which leads to a division by zero
        // both Chrome and Firefox accept that...
        let event = AudioParamEvent {
            event_type: AudioParamEventType::SetTargetAtTime,
            value,
            time: start_time,
            time_constant: Some(time_constant),
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn cancel_scheduled_values(&self, cancel_time: f64) -> &Self {
        let event = AudioParamEvent {
            event_type: AudioParamEventType::CancelScheduledValues,
            value: 0., // no value
            time: cancel_time,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn cancel_and_hold_at_time(&self, cancel_time: f64) -> &Self {
        let event = AudioParamEvent {
            event_type: AudioParamEventType::CancelAndHoldAtTime,
            value: 0., // value will be defined by cancel event
            time: cancel_time,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    pub fn set_value_curve_at_time(&self, values: &[f32], start_time: f64, duration: f64) -> &Self {
        // @todo - An InvalidStateError MUST be thrown if this attribute is a
        // sequence<float> object that has a length less than 2.

        // When this method is called, an internal copy of the curve is
        // created for automation purposes.
        let copy = values.to_vec();
        let boxed_copy = copy.into_boxed_slice();

        let event = AudioParamEvent {
            event_type: AudioParamEventType::SetValueCurveAtTime,
            value: 0., // value will be defined at the end of the event
            time: start_time,
            time_constant: None,
            cancel_time: None,
            duration: Some(duration),
            values: Some(boxed_copy),
        };

        self.send_event(event);

        self
    }

    // helper function to detach from context (for borrow reasons)
    pub(crate) fn into_raw_parts(self) -> AudioParamRaw {
        AudioParamRaw {
            automation_rate: self.automation_rate,
            default_value: self.default_value,
            min_value: self.min_value,
            max_value: self.max_value,
            current_value: self.current_value,
            sender: self.sender,
        }
    }

    // helper function to attach to context (for borrow reasons)
    pub(crate) fn from_raw_parts(
        registration: AudioContextRegistration,
        parts: AudioParamRaw,
    ) -> Self {
        Self {
            registration,
            automation_rate: parts.automation_rate,
            default_value: parts.default_value,
            min_value: parts.min_value,
            max_value: parts.max_value,
            current_value: parts.current_value,
            sender: parts.sender,
        }
    }

    fn send_event(&self, event: AudioParamEvent) {
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
    intrisic_value: f32,
    current_value: Arc<AtomicF32>,
    receiver: Receiver<AudioParamEvent>,
    automation_rate: AutomationRate,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    event_timeline: AudioParamEventTimeline,
    last_event: Option<AudioParamEvent>,
    buffer: Vec<f32>,
}

impl AudioProcessor for AudioParamProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        let period = 1. / sample_rate.0 as f64;
        let param_intrisic_values = self.tick(timestamp, period, RENDER_QUANTUM_SIZE);

        let input = &inputs[0]; // single input mode
        let param_computed_values = &mut outputs[0];

        param_computed_values
            .channel_data_mut(0)
            .copy_from_slice(param_intrisic_values);

        param_computed_values.add(input, ChannelInterpretation::Discrete);

        true // has intrinsic value
    }
}

impl AudioParamProcessor {
    pub fn intrisic_value(&self) -> f32 {
        if self.intrisic_value.is_nan() {
            self.default_value
        } else {
            self.intrisic_value.clamp(self.min_value, self.max_value)
        }
    }

    // 𝑣(𝑡) = 𝑉0 + (𝑉1−𝑉0) * ((𝑡−𝑇0) / (𝑇1−𝑇0))
    #[inline(always)]
    fn compute_linear_ramp_sample(
        &self,
        start_time: f64,
        duration: f64,
        start_value: f32,
        diff: f32, // end_value - start_value
        time: f64,
    ) -> f32 {
        let phase = (time - start_time) / duration;
        let value = start_value + diff * phase as f32;
        value.clamp(self.min_value, self.max_value)
    }

    // v(t) = v1 * (v2/v1)^((t-t1) / (t2-t1))
    #[inline(always)]
    fn compute_exponential_ramp_sample(
        &self,
        start_time: f64,
        duration: f64,
        start_value: f32,
        ratio: f32, // end_value / start_value
        time: f64,
    ) -> f32 {
        let phase = (time - start_time) / duration;
        let value = start_value * ratio.powf(phase as f32);
        value.clamp(self.min_value, self.max_value)
    }

    // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
    #[inline(always)]
    fn compute_set_target_sample(
        &self,
        start_time: f64,
        time_constant: f64,
        end_value: f32,
        diff: f32, // start_value - end_value
        time: f64,
    ) -> f32 {
        let exponent = -1. * ((time - start_time) / time_constant);
        let value = end_value + diff * exponent.exp() as f32;
        value.clamp(self.min_value, self.max_value)
    }

    // 𝑘=⌊𝑁−1 / 𝑇𝐷 * (𝑡−𝑇0)⌋
    // Then 𝑣(𝑡) is computed by linearly interpolating between 𝑉[𝑘] and 𝑉[𝑘+1],
    #[inline(always)]
    fn compute_set_value_curve_sample(
        &self,
        start_time: f64,
        duration: f64,
        values: &[f32],
        time: f64,
    ) -> f32 {
        if time - start_time >= duration {
            let value = values[values.len() - 1];
            value.clamp(self.min_value, self.max_value)
        } else {
            let position = (values.len() - 1) as f64 * (time - start_time) / duration;
            let k = position as usize;
            let phase = (position - position.floor()) as f32;
            let value = (values[k + 1] - values[k]) * phase + values[k];
            value.clamp(self.min_value, self.max_value)
        }
    }

    fn tick(&mut self, block_time: f64, dt: f64, count: usize) -> &[f32] {
        // handle incoming automation events in sorted queue
        //
        // cf. https://www.w3.org/TR/webaudio/#computation-of-value
        // 1. paramIntrinsicValue will be calculated at each time, which is either the
        // value set directly to the value attribute, or, if there are any automation
        // events with times before or at this time, the value as calculated from
        // these events. If automation events are removed from a given time range,
        // then the paramIntrinsicValue value will remain unchanged and stay at its
        // previous value until either the value attribute is directly set, or
        // automation events are added for the time range.
        let mut events_received = false;

        for event in self.receiver.try_iter() {
            events_received = true;

            // @note - the following could live in its own method just for clarity
            // but can't get rid of this error:
            //    for event in self.receiver.try_iter() {
            //                 ------------------------
            //                 |
            //                 immutable borrow occurs here
            //                 immutable borrow later used here
            //
            //        self.insert_event(event);
            //            ^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here

            // handle CancelScheduledValues events
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-cancelscheduledvalues
            if event.event_type == AudioParamEventType::CancelScheduledValues {
                // peek current event before inserting new events, and possibly sort
                // the queue, we need that for checking that we are (or not) in the middle
                // of a ramp when handling `CancelScheduledValues`
                // @note - probably not robust enough in some edge cases where the
                // event is not the first received at this tick (`SetValueCurveAtTime`
                // and `CancelAndHold` need to sort the queue)
                let some_current_event = self.event_timeline.unsorted_peek();

                match some_current_event {
                    None => (),
                    Some(current_event) => {
                        // @note - The spec is not particularly clear about what to do
                        // with on-going SetTarget events, but both Chrome and Firefox
                        // ignore the Cancel event if at same time nor if startTime is
                        // equal to cancelTime, this can makes sens as the event time
                        // (which is a `startTime` for `SetTarget`) is before the `cancelTime`
                        // (maybe this should be clarified w/ spec editors)

                        match current_event.event_type {
                            AudioParamEventType::LinearRampToValueAtTime
                            | AudioParamEventType::ExponentialRampToValueAtTime => {
                                // we are in the middle of a ramp
                                //
                                // @note - Firefox and Chrome behave differently
                                // on this: Firefox actually restore intrisic_value
                                // from the value at the beginning of the vent, while
                                // Chrome just keeps the current intrisic_value
                                // The spec is not very clear there, but Firefox
                                // seems to be the more compliant:
                                // "Any active automations whose automation event
                                // time is less than cancelTime are also cancelled,
                                // and such cancellations may cause discontinuities
                                // because the original value (**from before such
                                // automation**) is restored immediately."
                                //
                                // @note - last_event cannot be `None` here, because
                                // linear or exponential ramps are always preceded
                                // by another event (even a set_value_at_time
                                // inserted implicitly), so if the ramp is the next
                                // event that means that at least one event has
                                // already been processed.
                                if current_event.time >= event.time {
                                    let last_event = self.last_event.as_ref().unwrap();
                                    self.intrisic_value = last_event.value;
                                }
                            }
                            _ => (),
                        }
                    }
                }

                // remove all event in queue where cancel_time >= event.time
                self.event_timeline
                    .retain(|queued| queued.time < event.time);
                continue; // no need to insert cancel_values events in queue
            }

            if event.event_type == AudioParamEventType::CancelAndHoldAtTime {
                // 1. Let 𝐸1 be the event (if any) at time 𝑡1 where 𝑡1 is the
                // largest number satisfying 𝑡1 ≤ 𝑡𝑐.
                // 2. Let 𝐸2 be the event (if any) at time 𝑡2 where 𝑡2 is the
                // smallest number satisfying 𝑡𝑐 < 𝑡2.
                let mut e1: Option<&mut AudioParamEvent> = None;
                let mut e2: Option<&mut AudioParamEvent> = None;
                let mut t1 = f64::MIN;
                let mut t2 = f64::MAX;
                // we need a sorted timeline here to find siblings
                self.event_timeline.sort();

                for queued in self.event_timeline.iter_mut() {
                    // closest before cancel time: if several events at same time,
                    // we want the last one
                    if queued.time >= t1 && queued.time <= event.time {
                        t1 = queued.time;
                        e1 = Some(queued);
                    // closest after cancel time: if several events at same time,
                    // we want the first one
                    } else if queued.time < t2 && queued.time > event.time {
                        t2 = queued.time;
                        e2 = Some(queued);
                    }
                }

                // If 𝐸2 exists:
                if let Some(matched) = e2 {
                    // If 𝐸2 is a linear or exponential ramp,
                    // Effectively rewrite 𝐸2 to be the same kind of ramp ending
                    // at time 𝑡𝑐 with an end value that would be the value of the
                    // original ramp at time 𝑡𝑐.
                    // @note - this is done during the actual computation of the
                    //  ramp using the cancel_time
                    if matched.event_type == AudioParamEventType::LinearRampToValueAtTime
                        || matched.event_type == AudioParamEventType::ExponentialRampToValueAtTime
                    {
                        matched.cancel_time = Some(event.time);
                    }
                } else if let Some(matched) = e1 {
                    if matched.event_type == AudioParamEventType::SetTargetAtTime {
                        // Implicitly insert a setValueAtTime event at time 𝑡𝑐 with
                        // the value that the setTarget would
                        // @note - same strategy as for ramps
                        matched.cancel_time = Some(event.time);
                    } else if matched.event_type == AudioParamEventType::SetValueCurveAtTime {
                        // If 𝐸1 is a setValueCurve with a start time of 𝑡3 and a duration of 𝑑
                        // If 𝑡𝑐 <= 𝑡3 + 𝑑 :
                        // Effectively replace this event with a setValueCurve event
                        // with a start time of 𝑡3 and a new duration of 𝑡𝑐−𝑡3. However,
                        // this is not a true replacement; this automation MUST take
                        // care to produce the same output as the original, and not
                        // one computed using a different duration. (That would cause
                        // sampling of the value curve in a slightly different way,
                        // producing different results.)
                        let start_time = matched.time;
                        let duration = matched.duration.unwrap();

                        if event.time <= start_time + duration {
                            matched.cancel_time = Some(event.time);
                        }
                    }
                }

                // [spec] Remove all events with time greater than 𝑡𝑐.
                self.event_timeline.retain(|queued| {
                    let mut time = queued.time;
                    // if the event has a `cancel_time` we use it instead of `time`
                    if let Some(cancel_time) = queued.cancel_time {
                        time = cancel_time;
                    }

                    time <= event.time
                });
                continue; // cancel_and_hold events are not inserted timeline
            }

            // handle SetValueCurveAtTime
            // @note - These rules argue in favor of having events inserted in
            // the control thread, let's panic for now
            //
            // [spec] If setValueCurveAtTime() is called for time 𝑇 and duration 𝐷
            // and there are any events having a time strictly greater than 𝑇, but
            // strictly less than 𝑇+𝐷, then a NotSupportedError exception MUST be thrown.
            // In other words, it’s not ok to schedule a value curve during a time period
            // containing other events, but it’s ok to schedule a value curve exactly
            // at the time of another event.
            if event.event_type == AudioParamEventType::SetValueCurveAtTime {
                // check if we don't try to insert at the time of another event
                let start_time = event.time;
                let end_time = start_time + event.duration.unwrap();

                for queued in self.event_timeline.iter() {
                    if queued.time > start_time && queued.time < end_time {
                        panic!(
                            "NotSupportedError: scheduling SetValueCurveAtTime ({:?}) at
                            time of another automation event ({:?})",
                            event, queued,
                        );
                    }
                }
            }
            // // [spec] Similarly a NotSupportedError exception MUST be thrown if any
            // // automation method is called at a time which is contained in [𝑇,𝑇+𝐷), 𝑇
            // // being the time of the curve and 𝐷 its duration.
            // // @note - Cancel methods are not automation methods
            if event.event_type == AudioParamEventType::SetValueAtTime
                || event.event_type == AudioParamEventType::SetValue
                || event.event_type == AudioParamEventType::LinearRampToValueAtTime
                || event.event_type == AudioParamEventType::ExponentialRampToValueAtTime
                || event.event_type == AudioParamEventType::SetTargetAtTime
            {
                for queued in self.event_timeline.iter() {
                    if queued.event_type == AudioParamEventType::SetValueCurveAtTime {
                        let start_time = queued.time;
                        let end_time = start_time + queued.duration.unwrap();

                        if event.time > start_time && event.time < end_time {
                            panic!(
                                "NotSupportedError: scheduling automation event ({:?})
                                during SetValueCurveAtTime ({:?})",
                                event, queued,
                            );
                        }
                    }
                }
            }

            // handle SetValue - param intrisic value must be updated from event value
            if event.event_type == AudioParamEventType::SetValue {
                self.intrisic_value = event.value;
            }

            // If no event in the timeline and event_type is `LinearRampToValueAtTime`
            // or `ExponentialRampToValue` at time, we must insert a `SetValueAtTime`
            // with intrisic value and calling time.
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
            //
            // for SetTarget, this behavior is not per se specified, but it allows
            // to make sure we have a stable start_value available without having
            // to store it elsewhere.
            if self.event_timeline.is_empty()
                && (event.event_type == AudioParamEventType::LinearRampToValueAtTime
                    || event.event_type == AudioParamEventType::ExponentialRampToValueAtTime
                    || event.event_type == AudioParamEventType::SetTargetAtTime)
            {
                let set_value_event = AudioParamEvent {
                    event_type: AudioParamEventType::SetValue,
                    value: self.intrisic_value,
                    // make sure the event is applied before any other event, time
                    // will be replaced by the block timestamp during event processing
                    time: 0.,
                    time_constant: None,
                    cancel_time: None,
                    duration: None,
                    values: None,
                };

                self.event_timeline.push(set_value_event);
            }

            self.event_timeline.push(event);
        }

        if events_received {
            self.event_timeline.sort();
        }

        // 2. Set [[current value]] to the value of paramIntrinsicValue at the
        // beginning of this render quantum.
        self.current_value.store(self.intrisic_value());
        self.buffer.clear();

        let next_block_time = block_time + dt * count as f64;
        let is_a_rate = self.automation_rate == AutomationRate::A;
        let is_k_rate = !is_a_rate;

        if is_k_rate {
            self.buffer.resize(count, self.intrisic_value());
        }

        loop {
            let some_event = self.event_timeline.peek();

            match some_event {
                None => {
                    if is_a_rate {
                        self.buffer.resize(count, self.intrisic_value());
                    }
                    break;
                }
                Some(event) => {
                    match event.event_type {
                        AudioParamEventType::SetValue | AudioParamEventType::SetValueAtTime => {
                            let value = event.value;
                            let mut time = event.time;

                            // `set_value` calls and implicitely inserted events
                            // are inserted with a `time = 0.` to make sure
                            // they are processed first, replacing w/ block_time
                            // allows to conform to the spec:
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
                            if time == 0. {
                                time = block_time;
                            }

                            // fill buffer with current intrisic value until `event.time`
                            if is_a_rate {
                                let end_index = ((time - block_time).max(0.) / dt) as usize;
                                let end_index_clipped = end_index.min(count);

                                for _ in self.buffer.len()..end_index_clipped {
                                    self.buffer.push(self.intrisic_value());
                                }
                            }

                            if time > next_block_time {
                                break;
                            } else {
                                self.intrisic_value = value;

                                // no computation has been done on `time`
                                #[allow(clippy::float_cmp)]
                                if time != event.time {
                                    // store as last event with the applied time
                                    let mut event = self.event_timeline.pop().unwrap();
                                    event.time = time;
                                    self.last_event = Some(event);
                                } else {
                                    self.last_event = self.event_timeline.pop();
                                }
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                        // 𝑣(𝑡) = 𝑉0 + (𝑉1−𝑉0) * ((𝑡−𝑇0) / (𝑇1−𝑇0))
                        AudioParamEventType::LinearRampToValueAtTime => {
                            let last_event = self.last_event.as_ref().unwrap();

                            let start_time = last_event.time;
                            let mut end_time = event.time;
                            // compute duration before clapping `end_time` to
                            // `cancel_time` to keep declared slope of the ramp
                            let duration = end_time - start_time;
                            if let Some(cancel_time) = event.cancel_time {
                                end_time = cancel_time;
                            }

                            let start_value = last_event.value;
                            let end_value = event.value;
                            let diff = end_value - start_value;

                            if is_a_rate {
                                let start_index = self.buffer.len();
                                // we need to `ceil()` because if `end_time` is between two samples
                                // we actually want the sample before `end_time` to be computed
                                let end_index =
                                    ((end_time - block_time).max(0.) / dt).ceil() as usize;
                                let end_index_clipped = end_index.min(count);

                                // compute "real" value according to `t` then clamp it
                                // cf. Example 7 https://www.w3.org/TR/webaudio/#computation-of-value
                                if end_index_clipped > self.buffer.len() {
                                    let mut time = block_time + start_index as f64 * dt;

                                    for _ in start_index..end_index_clipped {
                                        let value = self.compute_linear_ramp_sample(
                                            start_time,
                                            duration,
                                            start_value,
                                            diff,
                                            time,
                                        );
                                        self.buffer.push(value);

                                        time += dt;
                                        self.intrisic_value = value;
                                    }
                                }
                            }

                            // Event will continue in next tick:
                            // compute value for `next_block_time` so that `param.value()`
                            // stays coherent, also allows to properly fill k-rate
                            // within next block too
                            if end_time >= next_block_time {
                                let value = self.compute_linear_ramp_sample(
                                    start_time,
                                    duration,
                                    start_value,
                                    diff,
                                    next_block_time,
                                );
                                self.intrisic_value = value;
                                break;

                            // Event cancelled during this block
                            } else if event.cancel_time.is_some() {
                                let value = self.compute_linear_ramp_sample(
                                    start_time,
                                    duration,
                                    start_value,
                                    diff,
                                    end_time,
                                );

                                self.intrisic_value = value;

                                let mut last_event = self.event_timeline.pop().unwrap();
                                last_event.time = end_time;
                                last_event.value = value;
                                self.last_event = Some(last_event);

                            // Event ended during this block
                            } else {
                                self.intrisic_value =
                                    end_value.clamp(self.min_value, self.max_value);
                                self.last_event = self.event_timeline.pop();
                            }
                        }
                        // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
                        // v(t) = v1 * (v2/v1)^((t-t1) / (t2-t1))
                        AudioParamEventType::ExponentialRampToValueAtTime => {
                            let last_event = self.last_event.as_ref().unwrap();

                            let start_time = last_event.time;
                            let mut end_time = event.time;
                            // compute duration before clapping `end_time` to
                            // `cancel_time` to keep declared slope of the ramp
                            let duration = end_time - start_time;
                            if let Some(cancel_time) = event.cancel_time {
                                end_time = cancel_time;
                            }

                            let start_value = last_event.value;
                            let end_value = event.value;
                            let ratio = end_value / start_value;

                            // Handle edge cases:
                            // > If 𝑉0 and 𝑉1 have opposite signs or if 𝑉0 is zero,
                            // > then 𝑣(𝑡)=𝑉0 for 𝑇0≤𝑡<𝑇1.
                            // as:
                            // > If there are no more events after this ExponentialRampToValue
                            // > event then for 𝑡≥𝑇1, 𝑣(𝑡)=𝑉1.
                            // this should thus behave as a SetValue
                            if start_value == 0. || start_value * end_value < 0. {
                                let event = AudioParamEvent {
                                    event_type: AudioParamEventType::SetValueAtTime,
                                    time: end_time,
                                    value: end_value,
                                    time_constant: None,
                                    cancel_time: None,
                                    duration: None,
                                    values: None,
                                };

                                self.event_timeline.replace_peek(event);
                            } else {
                                if is_a_rate {
                                    let start_index = self.buffer.len();
                                    // we need to `ceil()` because if `end_time` is between two samples
                                    // we actually want the sample before `end_time` to be computed
                                    // @todo - more tests
                                    let end_index =
                                        ((end_time - block_time).max(0.) / dt).ceil() as usize;
                                    let end_index_clipped = end_index.min(count);

                                    if end_index_clipped > self.buffer.len() {
                                        let mut time = block_time + start_index as f64 * dt;

                                        for _ in start_index..end_index_clipped {
                                            let value = self.compute_exponential_ramp_sample(
                                                start_time,
                                                duration,
                                                start_value,
                                                ratio,
                                                time,
                                            );

                                            self.buffer.push(value);
                                            self.intrisic_value = value;

                                            time += dt;
                                        }
                                    }
                                }

                                // Event will continue in next tick:
                                // compute value for `next_block_time` so that `param.value()`
                                // stays coherent, also allows to properly fill k-rate
                                // within next block too
                                if end_time >= next_block_time {
                                    let value = self.compute_exponential_ramp_sample(
                                        start_time,
                                        duration,
                                        start_value,
                                        ratio,
                                        next_block_time,
                                    );
                                    self.intrisic_value = value;
                                    break;

                                // Event cancelled during this block
                                } else if event.cancel_time.is_some() {
                                    let value = self.compute_exponential_ramp_sample(
                                        start_time,
                                        duration,
                                        start_value,
                                        ratio,
                                        end_time,
                                    );

                                    self.intrisic_value = value;

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value;
                                    self.last_event = Some(last_event);

                                // Event ended during this block
                                } else {
                                    self.intrisic_value =
                                        end_value.clamp(self.min_value, self.max_value);
                                    self.last_event = self.event_timeline.pop();
                                }
                            }
                        }
                        // https://webaudio.github.io/web-audio-api/#dom-audioparam-settargetattime
                        // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
                        //
                        // @todo - as SetTarget never resolves on an end value, some
                        // strategy could be implemented here so that when the value
                        // is close enough to the target a SetValue event could be
                        // inserted in the timeline. This could be done at k-rate.
                        // Note that Chrome has such strategy, cf. `HasSetTargetConverged`
                        AudioParamEventType::SetTargetAtTime => {
                            let mut end_time = next_block_time;
                            let mut ended = false;

                            // handle next event stop SetTarget if any
                            let some_next_event = self.event_timeline.next();

                            if let Some(next_event) = some_next_event {
                                match next_event.event_type {
                                    AudioParamEventType::LinearRampToValueAtTime
                                    | AudioParamEventType::ExponentialRampToValueAtTime => {
                                        // [spec] If the preceding event is a SetTarget
                                        // event, 𝑇0 and 𝑉0 are chosen from the current
                                        // time and value of SetTarget automation. That
                                        // is, if the SetTarget event has not started,
                                        // 𝑇0 is the start time of the event, and 𝑉0
                                        // is the value just before the SetTarget event
                                        // starts. In this case, the LinearRampToValue
                                        // event effectively replaces the SetTarget event.
                                        // If the SetTarget event has already started,
                                        // 𝑇0 is the current context time, and 𝑉0 is
                                        // the current SetTarget automation value at time 𝑇0.
                                        // In both cases, the automation curve is continuous.
                                        end_time = block_time;
                                        ended = true;
                                    }
                                    _ => {
                                        // For all other events, the SetTarget
                                        // event ends at the time of the next event.
                                        if next_event.time < next_block_time {
                                            end_time = next_event.time;
                                            ended = true;
                                        }
                                    }
                                }
                            }

                            // handle CancelAndHoldAtTime
                            if let Some(cancel_time) = event.cancel_time {
                                if cancel_time < next_block_time {
                                    end_time = cancel_time;
                                    ended = true;
                                }
                            }

                            let start_time = event.time;
                            // if SetTarget is the first event registered, we implicitely
                            // insert a SetValue event just before just as for Ramps.
                            // Therefore we are sure last_event exists
                            let start_value = self.last_event.as_ref().unwrap().value;
                            let end_value = event.value;
                            let diff = start_value - end_value;
                            let time_constant = event.time_constant.unwrap();

                            if is_a_rate {
                                let start_index = self.buffer.len();
                                // we need to `ceil()` because if `end_time` is between two samples
                                // we actually want the sample before `end_time` to be computed
                                // @todo - more tests
                                let end_index =
                                    ((end_time - block_time).max(0.) / dt).ceil() as usize;
                                let end_index_clipped = end_index.min(count);

                                if end_index_clipped > self.buffer.len() {
                                    let mut time = block_time + start_index as f64 * dt;

                                    for _ in start_index..end_index_clipped {
                                        let value = self.compute_set_target_sample(
                                            start_time,
                                            time_constant,
                                            end_value,
                                            diff,
                                            time,
                                        );

                                        self.buffer.push(value);
                                        self.intrisic_value = value;

                                        time += dt;
                                    }
                                }
                            }

                            if !ended {
                                // compute value for `next_block_time` so that `param.value()`
                                // stays coherent (see. comment in `AudioParam`)
                                // allows to properly fill k-rate within next block too
                                let value = self.compute_set_target_sample(
                                    start_time,
                                    time_constant,
                                    end_value,
                                    diff,
                                    next_block_time,
                                );
                                self.intrisic_value = value;
                                break;
                            } else {
                                // setTarget has no "real" end value, compute according
                                // to next event start time
                                let value = self.compute_set_target_sample(
                                    start_time,
                                    time_constant,
                                    end_value,
                                    diff,
                                    end_time,
                                );

                                self.intrisic_value = value;
                                // end_value and end_time must be stored for use
                                // as start time by next event
                                let mut event = self.event_timeline.pop().unwrap();
                                event.time = end_time;
                                event.value = value;
                                self.last_event = Some(event);
                            }
                        }
                        AudioParamEventType::SetValueCurveAtTime => {
                            let start_time = event.time;
                            let duration = event.duration.unwrap();
                            let values = event.values.as_ref().unwrap();
                            let mut end_time = start_time + duration;

                            // we must check for the cancel event after we have
                            // the "real" duration computed to not change the
                            // slope of the ramp
                            if let Some(cancel_time) = event.cancel_time {
                                end_time = cancel_time;
                            }

                            if is_a_rate {
                                let start_index = self.buffer.len();
                                // we need to `ceil()` because if `end_time` is between two samples
                                // we actually want the sample before `end_time` to be computed
                                // @todo - more tests
                                let end_index =
                                    ((end_time - block_time).max(0.) / dt).ceil() as usize;
                                let end_index_clipped = end_index.min(count);

                                if end_index_clipped > self.buffer.len() {
                                    let mut time = block_time + start_index as f64 * dt;

                                    for _ in start_index..end_index_clipped {
                                        let value = self.compute_set_value_curve_sample(
                                            start_time, duration, values, time,
                                        );

                                        self.buffer.push(value);
                                        self.intrisic_value = value;

                                        time += dt;
                                    }
                                }
                            }

                            // event will continue in next tick
                            if end_time >= next_block_time {
                                // compute value for `next_block_time` so that `param.value()`
                                // stays coherent (see. comment in `AudioParam`)
                                // allows to properly fill k-rate within next block too
                                let value = self.compute_set_value_curve_sample(
                                    start_time,
                                    duration,
                                    values,
                                    next_block_time,
                                );
                                self.intrisic_value = value;
                                break;

                            // handle end of event during this block
                            } else {
                                // event has been cancelled
                                if event.cancel_time.is_some() {
                                    let value = self.compute_set_value_curve_sample(
                                        start_time, duration, values, end_time,
                                    );

                                    self.intrisic_value = value;

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value;
                                    self.last_event = Some(last_event);
                                // event has ended
                                } else {
                                    let value = values[values.len() - 1];
                                    let value_clamped = value.clamp(self.min_value, self.max_value);

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value_clamped;

                                    self.intrisic_value = value_clamped;
                                    self.last_event = Some(last_event);
                                }
                            }
                        }
                        _ => panic!(
                            "AudioParamEvent {:?} should not appear in AudioParamEventTimeline",
                            event.event_type
                        ),
                    }
                }
            }
        }

        if cfg!(test) {
            assert_eq!(self.buffer.len(), count);
        }

        self.buffer.as_slice()
    }
}

pub(crate) fn audio_param_pair(
    opts: AudioParamDescriptor,
    registration: AudioContextRegistration,
) -> (AudioParam, AudioParamProcessor) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let current_value = Arc::new(AtomicF32::new(opts.default_value));

    let param = AudioParam {
        registration,
        automation_rate: opts.automation_rate,
        default_value: opts.default_value,
        min_value: opts.min_value,
        max_value: opts.max_value,
        current_value: current_value.clone(),
        sender,
    };

    let render = AudioParamProcessor {
        intrisic_value: opts.default_value,
        current_value,
        receiver,
        automation_rate: opts.automation_rate,
        default_value: opts.default_value,
        min_value: opts.min_value,
        max_value: opts.max_value,
        event_timeline: AudioParamEventTimeline::new(),
        last_event: None,
        buffer: Vec::with_capacity(RENDER_QUANTUM_SIZE),
    };

    (param, render)
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{BaseAudioContext, OfflineAudioContext};

    use super::*;

    #[test]
    fn test_default_and_accessors() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, _render) = audio_param_pair(opts, context.mock_registration());

        assert_eq!(param.automation_rate(), AutomationRate::A);
        assert_float_eq!(param.default_value(), 0., abs_all <= 0.);
        assert_float_eq!(param.min_value(), -10., abs_all <= 0.);
        assert_float_eq!(param.max_value(), 10., abs_all <= 0.);
        assert_float_eq!(param.value(), 0., abs_all <= 0.);
    }

    #[test]
    fn test_set_value() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value(2.);
        assert_float_eq!(param.value(), 2., abs_all <= 0.);

        let vs = render.tick(0., 1., 10);

        // current_value should not be overriden by intrisic value
        assert_float_eq!(param.value(), 2., abs_all <= 0.);
        assert_float_eq!(vs, &[2.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_set_value_clamped() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -1.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value(2.);
        assert_float_eq!(param.value(), 1., abs_all <= 0.);

        let vs = render.tick(0., 1., 10);

        // current_value should not be overriden by intrisic value
        assert_float_eq!(param.value(), 1., abs_all <= 0.);
        assert_float_eq!(vs, &[1.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_steps_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
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
                abs_all <= 0.
            );

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[8.; 10][..], abs_all <= 0.);
        }

        {
            // events spread on several blocks
            let opts = AudioParamDescriptor {
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
                abs_all <= 0.
            );

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(
                vs,
                &[5., 5., 8., 8., 8., 8., 8., 8., 8., 8.][..],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_steps_k_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        let opts = AudioParamDescriptor {
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
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[8.; 10][..], abs_all <= 0.);

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[3.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_linear_ramp_arate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
            abs_all <= 0.
        );
    }

    #[test]
    fn test_linear_ramp_arate_end_of_block() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
            abs_all <= 0.
        );
    }

    #[test]
    fn test_linear_ramp_arate_implicit_set_value() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // mimic a ramp inserted after start
        // i.e. setTimeout(() => param.linearRampToValueAtTime(10, now + 10)), 10 * 1000);
        let vs = render.tick(0., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);

        param.linear_ramp_to_value_at_time(10.0, 20.0);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            abs_all <= 0.
        );

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[10.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_linear_ramp_arate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
            abs_all <= 0.
        );
        assert_float_eq!(param.value(), 0., abs <= 0.);

        // next quantum t = 10..20
        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[10., 11., 12., 13., 14., 15., 16., 17., 18., 19.][..],
            abs_all <= 0.
        );
        assert_float_eq!(param.value(), 10., abs <= 0.);

        // ramp finished t = 20..30
        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[20.0; 10][..], abs_all <= 0.);
        assert_float_eq!(param.value(), 20., abs <= 0.);
    }

    #[test]
    fn test_linear_ramp_arate_clamp() {
        // must be compliant with ex.7 cf. https://www.w3.org/TR/webaudio/#computation-of-value
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
            abs_all <= 0.
        );

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_linear_ramp_krate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
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
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 0., abs <= 0.);
            // next quantum t = 10..20
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[10.; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 10., abs <= 0.);
            // ramp finished t = 20..30
            let vs = render.tick(20., 1., 10);
            assert_float_eq!(vs, &[20.0; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 20., abs <= 0.);
        }

        {
            // finish in the middle of a block
            let opts = AudioParamDescriptor {
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
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 0., abs <= 0.);
            // next quantum t = 10..20
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[10.; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 10., abs <= 0.);
            // ramp finished t = 20..30
            let vs = render.tick(20., 1., 10);
            assert_float_eq!(vs, &[15.0; 10][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 15., abs <= 0.);
        }
    }

    #[test]
    fn test_exponential_ramp_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
        assert_float_eq!(vs, &res[..], abs_all <= 0.);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[1.0; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_exponential_ramp_a_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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
        assert_float_eq!(vs, &res[0..10], abs_all <= 0.);
        assert_float_eq!(param.value(), res[0], abs <= 0.);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        assert_float_eq!(param.value(), res[10], abs <= 0.);
    }

    #[test]
    fn test_exponential_ramp_a_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            // zero target
            let opts = AudioParamDescriptor {
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
                abs_all <= 0.
            );
        }

        {
            // opposite signs
            let opts = AudioParamDescriptor {
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
                abs_all <= 0.
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_exponential_ramp_to_zero() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
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

        let opts = AudioParamDescriptor {
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
        assert_float_eq!(vs, &[res[0]; 10][..], abs_all <= 0.);

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[res[10]; 10][..], abs_all <= 0.);

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[1.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_exponential_ramp_k_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            // zero target
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], abs_all <= 0.);
        }

        {
            // opposite signs
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::K,
                default_value: -1.,
                min_value: -1.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.exponential_ramp_to_value_at_time(1.0, 5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[-1.; 10][..], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[1.; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 1.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;

            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);
            let vs = render.tick(0., 1., 10);

            let mut res = Vec::<f32>::with_capacity(10);
            for t in 0..10 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }

        {
            // implicit SetValue if SetTarget is first event
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.; // will be implicitly set in param (see default_value)
            let v1: f32 = 1.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;

            param.set_target_at_time(v1, t0, time_constant);
            let vs = render.tick(0., 1., 10);

            let mut res = Vec::<f32>::with_capacity(10);
            for t in 0..10 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }

        {
            // start later in block with arbitrary values
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 100.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 1.;
            let v1: f32 = 42.;
            let t0: f64 = 1.;
            let time_constant: f64 = 2.1;

            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);

            let mut res = Vec::<f32>::with_capacity(10);
            for t in 0..10 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }
            // ramp start at 1 so
            res[0] = 0.;

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 2.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);

            let mut res = Vec::<f32>::with_capacity(20);
            for t in 0..20 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate_followed_by_set_value() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 2.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;

            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);
            param.set_value_at_time(0.5, 15.);

            let mut res = Vec::<f32>::with_capacity(20);

            for t in 0..15 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            res.resize(20, 0.5);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate_followed_by_ramp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 2.;
            let t0: f64 = 0.;
            let time_constant: f64 = 10.;

            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);

            let mut res = Vec::<f32>::with_capacity(20);

            for t in 0..11 {
                // we compute the 10th elements as it will be the start value of the ramp
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            // ramp
            let v0 = res.pop().unwrap(); // v0 is defined by the SetTarget
            let v1 = 10.;
            let t0 = 10.;
            let t1 = 20.;

            param.linear_ramp_to_value_at_time(v1, t1);

            for t in 10..20 {
                let time = t as f64;
                let value = v0 + (v1 - v0) * (time - t0) as f32 / (t1 - t0) as f32;
                res.push(value);
            }

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 1.0e-6);
            // ramp ended
            let vs = render.tick(20., 1., 10);
            assert_float_eq!(vs, &[v1; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_k_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::K,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 2.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;
            // ramp to 1 from t=0 to t=5 -> should behave as a set target at t=5
            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);

            let mut res = Vec::<f32>::with_capacity(20);
            for t in 0..20 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[res[0]; 10][..], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[res[10]; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_scheduled_values() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());
        for t in 0..10 {
            param.set_value_at_time(t as f32, t as f64);
        }

        param.cancel_scheduled_values(5.);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 4., 4., 4., 4., 4.][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_cancel_scheduled_values_ramp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(0., 0.);
            param.linear_ramp_to_value_at_time(10., 10.);
            param.cancel_scheduled_values(10.); // cancels everything

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
        }

        // ramp already started, go back to previous value
        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 20.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(0., 0.);
            param.linear_ramp_to_value_at_time(20., 20.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
                abs_all <= 0.
            );

            param.cancel_scheduled_values(10.);
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
        }

        // make sure we can't go into a situation where next_event is a ramp
        // and last_event is not defined
        // @see - note in CancelScheduledValues insertion in timeline
        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.linear_ramp_to_value_at_time(10., 10.);
            param.cancel_scheduled_values(10.); // cancels the ramp

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
        }

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 20.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.linear_ramp_to_value_at_time(20., 20.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
                abs_all <= 0.
            );

            param.cancel_scheduled_values(10.);
            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));
        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(1., 1.);
            param.set_value_at_time(2., 2.);
            param.set_value_at_time(3., 3.);
            param.set_value_at_time(4., 4.);
            param.cancel_and_hold_at_time(2.5);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 2., 2., 2., 2., 2., 2., 2.][0..10],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_cancel_and_hold_during_set_target() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
            let v0: f32 = 0.;
            let v1: f32 = 2.;
            let t0: f64 = 0.;
            let time_constant: f64 = 1.;

            param.set_value_at_time(v0, t0);
            param.set_target_at_time(v1, t0, time_constant);
            param.cancel_and_hold_at_time(15.);

            let mut res = Vec::<f32>::with_capacity(20);

            // compute index 15 to have hold_value
            for t in 0..16 {
                let val = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
                res.push(val);
            }

            let hold_value = res.pop().unwrap();
            res.resize(20, hold_value);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.tick(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold_during_linear_ramp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.linear_ramp_to_value_at_time(10., 10.);
            param.cancel_and_hold_at_time(5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 5., 5., 5., 5., 5.][0..10],
                abs_all <= 0.
            );
        }

        {
            // cancel between two samples
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.linear_ramp_to_value_at_time(10., 10.);
            param.cancel_and_hold_at_time(4.5);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 4.5, 4.5, 4.5, 4.5, 4.5][0..10],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_cancel_and_hold_during_exponential_ramp() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set to 0.0001 at t=0 (0. is a special case)
            param.set_value_at_time(0.0001, 0.);
            param.exponential_ramp_to_value_at_time(1.0, 10.);
            param.cancel_and_hold_at_time(5.);

            // compute resulting buffer:
            // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
            let mut res = Vec::<f32>::with_capacity(10);
            let start: f32 = 0.0001;
            let end: f32 = 1.;

            for t in 0..6 {
                let value = start * (end / start).powf(t as f32 / 10.);
                res.push(value);
            }

            let hold_value = res.pop().unwrap();
            res.resize(10, hold_value);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }

        {
            // cancel between 2 samples
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            // set to 0.0001 at t=0 (0. is a special case)
            param.set_value_at_time(0.0001, 0.);
            param.exponential_ramp_to_value_at_time(1.0, 10.);
            param.cancel_and_hold_at_time(4.5);

            // compute resulting buffer:
            // v(t) = v1*(v2/v1)^((t-t1)/(t2-t1))
            let mut res = Vec::<f32>::with_capacity(10);
            let start: f32 = 0.0001;
            let end: f32 = 1.;

            for t in 0..5 {
                let value = start * (end / start).powf(t as f32 / 10.);
                res.push(value);
            }

            let hold_value = start * (end / start).powf(4.5 / 10.);
            res.resize(10, hold_value);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold_during_set_value_curve() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        {
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            let curve = [0., 0.5, 1., 0.5, 0.];
            param.set_value_curve_at_time(&curve[..], 0., 10.);
            param.cancel_and_hold_at_time(5.);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0.2, 0.4, 0.6, 0.8, 1., 1., 1., 1., 1.][..],
                abs_all <= 1e-7
            );
        }

        {
            // sub-sample
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 2.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            let curve = [0., 0.5, 1., 0.5, 0.];
            param.set_value_curve_at_time(&curve[..], 0., 10.);
            param.cancel_and_hold_at_time(4.5);

            let vs = render.tick(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0.2, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9][..],
                abs_all <= 1e-7
            );
        }
    }

    #[test]
    fn test_set_value_curve_at_time_a_rate() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 0.0001 at t=0 (0. is a special case)
        let curve = [0., 0.5, 1., 0.5, 0.];
        param.set_value_curve_at_time(&curve[..], 0., 10.);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2][..],
            abs_all <= 1e-7
        );

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_set_value_curve_at_time_a_rate_multiple_frames() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 0.0001 at t=0 (0. is a special case)
        let curve = [0., 0.5, 1., 0.5, 0.];
        param.set_value_curve_at_time(&curve[..], 0., 20.);

        let vs = render.tick(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][..],
            abs_all <= 1e-7
        );

        let vs = render.tick(10., 1., 10);
        assert_float_eq!(
            vs,
            &[1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][..],
            abs_all <= 1e-7
        );

        let vs = render.tick(20., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    #[should_panic]
    fn test_set_value_curve_at_time_insert_while_another_event() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 1.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(0.0, 5.);

        let curve = [0., 0.5, 1., 0.5, 0.];
        param.set_value_curve_at_time(&curve[..], 0., 10.);
        // this is necessary as the panic is triggered in the audio thread
        // @note - argues in favor of maintaining the queue in control thread
        let _vs = render.tick(0., 1., 10);
    }

    #[test]
    #[should_panic]
    fn test_set_value_curve_at_time_insert_another_event_inside() {
        let context = OfflineAudioContext::new(1, 0, SampleRate(0));

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 1.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        let curve = [0., 0.5, 1., 0.5, 0.];
        param.set_value_curve_at_time(&curve[..], 0., 10.);
        param.set_value_at_time(0.0, 5.);
        // this is necessary as the panic is triggered in the audio thread
        // @note - argues in favor of maintaining the queue in control thread
        let _vs = render.tick(0., 1., 10);
    }
}
