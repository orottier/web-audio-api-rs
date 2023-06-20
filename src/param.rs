//! AudioParam interface
use std::slice::{Iter, IterMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::context::AudioContextRegistration;
use crate::node::{
    AudioNode, ChannelConfig, ChannelConfigOptions, ChannelCountMode, ChannelInterpretation,
};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::{AtomicF32, RENDER_QUANTUM_SIZE};

use crossbeam_channel::{Receiver, Sender};
use lazy_static::lazy_static;

/// For SetTargetAtTime event, that theoretically cannot end, if the diff between
/// the current value and the target is below this threshold, the value is set
/// to target value and the event is considered ended.
const SNAP_TO_TARGET: f32 = 1e-10;

// arguments sanity check functions for automation methods
#[track_caller]
fn assert_non_negative(value: f64) {
    if value < 0. {
        panic!(
            "RangeError - timing value ({:?}) should not be negative",
            value
        );
    }
}

#[track_caller]
fn assert_strictly_positive(value: f64) {
    if value <= 0. {
        panic!(
            "RangeError - duration ({:?}) should be strictly positive",
            value
        );
    }
}

#[track_caller]
fn assert_not_zero(value: f32) {
    if value == 0. {
        panic!(
            "RangeError - value ({:?}) should not be equal to zero",
            value,
        )
    }
}

#[track_caller]
fn assert_sequence_length(values: &[f32]) {
    if values.len() < 2 {
        panic!(
            "InvalidStateError - sequence length ({:?}) should not be less than 2",
            values.len()
        )
    }
}

/// Precision of AudioParam value calculation per render quantum
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AutomationRate {
    /// Audio Rate - sampled for each sample-frame of the block
    A,
    /// Control Rate - sampled at the time of the very first sample-frame,
    /// then used for the entire block
    K,
}

/// Options for constructing an [`AudioParam`]
#[derive(Clone, Debug)]
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
// in which cases `sort` must be called explicitly.
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

    fn is_empty(&self) -> bool {
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
    is_a_rate: Arc<AtomicBool>,
    automation_rate_constrained: bool,
    default_value: f32, // readonly
    min_value: f32,     // readonly
    max_value: f32,     // readonly
    current_value: Arc<AtomicF32>,
    sender: Sender<AudioParamEvent>,
}

// helper struct to attach / detach to context (for borrow reasons)
#[derive(Clone)]
pub(crate) struct AudioParamRaw {
    is_a_rate: Arc<AtomicBool>,
    automation_rate_constrained: bool,
    default_value: f32,
    min_value: f32,
    max_value: f32,
    current_value: Arc<AtomicF32>,
    sender: Sender<AudioParamEvent>,
}

lazy_static! {
    static ref AUDIO_PARAM_CHANNEL_CONFIG: ChannelConfig = ChannelConfigOptions {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    }
    .into();
}

impl AudioNode for AudioParam {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &AUDIO_PARAM_CHANNEL_CONFIG
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }

    fn set_channel_count(&self, _v: usize) {
        panic!("AudioParam has channel count constraints");
    }
    fn set_channel_count_mode(&self, _v: ChannelCountMode) {
        panic!("AudioParam has channel count mode constraints");
    }
    fn set_channel_interpretation(&self, _v: ChannelInterpretation) {
        panic!("AudioParam has channel interpretation constraints");
    }
}

impl AudioParam {
    /// Current value of the automation rate of the AudioParam
    pub fn automation_rate(&self) -> AutomationRate {
        if self.is_a_rate.load(Ordering::SeqCst) {
            AutomationRate::A
        } else {
            AutomationRate::K
        }
    }

    /// Update the current value of the automation rate of the AudioParam
    ///
    /// # Panics
    ///
    /// Some nodes have automation rate constraints and may panic when updating the value
    pub fn set_automation_rate(&self, value: AutomationRate) {
        if self.automation_rate_constrained && value != self.automation_rate() {
            panic!("InvalidStateError: automation rate cannot be changed for this param");
        }

        let is_a_rate = value == AutomationRate::A;
        self.is_a_rate.store(is_a_rate, Ordering::SeqCst);
    }

    pub(crate) fn set_automation_rate_constrained(&mut self, value: bool) {
        self.automation_rate_constrained = value;
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

    /// Retrieve the current value of the `AudioParam`.
    //
    // @note: the choice here is to have this coherent with the first sample of
    // the last rendered block, which means `intrinsic_value` must be calculated
    // for next_block_time at each tick.
    // @note - maybe check with spec editors that it is correct
    //
    // see. test_linear_ramp_arate_multiple_blocks
    //      test_linear_ramp_krate_multiple_blocks
    //      test_exponential_ramp_a_rate_multiple_blocks
    //      test_exponential_ramp_k_rate_multiple_blocks
    pub fn value(&self) -> f32 {
        self.current_value.load(Ordering::SeqCst)
    }

    /// Set the value of the `AudioParam`.
    ///
    /// Is equivalent to calling the `set_value_at_time` method with the current
    /// AudioContext's currentTime
    //
    // @note: Setting this attribute has the effect of assigning the requested value
    // to the [[current value]] slot, and calling the setValueAtTime() method
    // with the current AudioContext's currentTime and [[current value]].
    // Any exceptions that would be thrown by setValueAtTime() will also be
    // thrown by setting this attribute.
    // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
    pub fn set_value(&self, value: f32) -> &Self {
        // current_value should always be clamped
        let clamped = value.clamp(self.min_value, self.max_value);
        self.current_value.store(clamped, Ordering::SeqCst);

        // this event is meant to update param intrinsic value before any calculation
        // is done, will behave as SetValueAtTime with `time == block_timestamp`
        let event = AudioParamEvent {
            event_type: AudioParamEventType::SetValue,
            value,
            time: 0.,
            time_constant: None,
            cancel_time: None,
            duration: None,
            values: None,
        };

        self.send_event(event);

        self
    }

    /// Schedules a parameter value change at the given time.
    ///
    /// # Panics
    ///
    /// Will panic if `start_time` is negative
    pub fn set_value_at_time(&self, value: f32, start_time: f64) -> &Self {
        assert_non_negative(start_time);

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

    /// Schedules a linear continuous change in parameter value from the
    /// previous scheduled parameter value to the given value.
    ///
    /// # Panics
    ///
    /// Will panic if `end_time` is negative
    pub fn linear_ramp_to_value_at_time(&self, value: f32, end_time: f64) -> &Self {
        assert_non_negative(end_time);

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

    /// Schedules an exponential continuous change in parameter value from the
    /// previous scheduled parameter value to the given value.
    ///
    /// # Panics
    ///
    /// Will panic if:
    /// - `value` is zero
    /// - `end_time` is negative
    pub fn exponential_ramp_to_value_at_time(&self, value: f32, end_time: f64) -> &Self {
        assert_not_zero(value);
        assert_non_negative(end_time);

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

    /// Start exponentially approaching the target value at the given time with
    /// a rate having the given time constant.
    ///
    /// # Panics
    ///
    /// Will panic if:
    /// - `start_time` is negative
    /// - `time_constant` is negative
    pub fn set_target_at_time(&self, value: f32, start_time: f64, time_constant: f64) -> &Self {
        assert_non_negative(start_time);
        assert_non_negative(time_constant);

        // [spec] If timeConstant is zero, the output value jumps immediately to the final value.
        let event = if time_constant == 0. {
            AudioParamEvent {
                event_type: AudioParamEventType::SetValueAtTime,
                value,
                time: start_time,
                time_constant: None,
                cancel_time: None,
                duration: None,
                values: None,
            }
        } else {
            AudioParamEvent {
                event_type: AudioParamEventType::SetTargetAtTime,
                value,
                time: start_time,
                time_constant: Some(time_constant),
                cancel_time: None,
                duration: None,
                values: None,
            }
        };

        self.send_event(event);

        self
    }

    /// Cancels all scheduled parameter changes with times greater than or equal
    /// to `cancel_time`.
    ///
    /// # Panics
    ///
    /// Will panic if `cancel_time` is negative
    pub fn cancel_scheduled_values(&self, cancel_time: f64) -> &Self {
        assert_non_negative(cancel_time);

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

    /// Cancels all scheduled parameter changes with times greater than or equal
    /// to `cancel_time` and the automation value that would have happened at
    /// that time is then propagated for all future time.
    ///
    /// # Panics
    ///
    /// Will panic if `cancel_time` is negative
    pub fn cancel_and_hold_at_time(&self, cancel_time: f64) -> &Self {
        assert_non_negative(cancel_time);

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

    /// Sets an array of arbitrary parameter values starting at the given time
    /// for the given duration.
    ///
    /// # Panics
    ///
    /// Will panic if:
    /// - `value` length is less than 2
    /// - `start_time` is negative
    /// - `duration` is negative or equal to zero
    pub fn set_value_curve_at_time(&self, values: &[f32], start_time: f64, duration: f64) -> &Self {
        assert_sequence_length(values);
        assert_non_negative(start_time);
        assert_strictly_positive(duration);

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
            is_a_rate: self.is_a_rate,
            automation_rate_constrained: self.automation_rate_constrained,
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
            is_a_rate: parts.is_a_rate,
            automation_rate_constrained: parts.automation_rate_constrained,
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
    intrinsic_value: f32,
    current_value: Arc<AtomicF32>,
    receiver: Receiver<AudioParamEvent>,
    is_a_rate: Arc<AtomicBool>,
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
        _params: AudioParamValues<'_>,
        scope: &RenderScope,
    ) -> bool {
        let period = 1. / scope.sample_rate as f64;

        let input = &inputs[0]; // single input mode
        let output = &mut outputs[0];

        self.compute_intrinsic_values(scope.current_time, period, RENDER_QUANTUM_SIZE);
        self.mix_to_output(input, output);

        true // has intrinsic value
    }
}

impl AudioParamProcessor {
    // warning: tick in called directly in the unit tests so everything important
    // for the tests should be done here
    fn compute_intrinsic_values(&mut self, block_time: f64, dt: f64, count: usize) -> &[f32] {
        if !self.receiver.is_empty() {
            self.handle_incoming_events();
        }

        self.compute_buffer(block_time, dt, count);

        self.buffer.as_slice()
    }

    fn mix_to_output(&mut self, input: &AudioRenderQuantum, output: &mut AudioRenderQuantum) {
        #[cfg(test)]
        assert!(self.buffer.len() == 1 || self.buffer.len() == RENDER_QUANTUM_SIZE);

        if self.buffer.len() == 1 && input.is_silent() {
            let mut value = self.buffer[0];

            if value.is_nan() {
                value = self.default_value;
            }

            output.set_single_valued(true);

            let output_channel = output.channel_data_mut(0);
            output_channel[0] = value.clamp(self.min_value, self.max_value);
        } else {
            // @note: we could add two other optimizations here:
            // - when buffer.len() == 1 and buffer[0] == 0., then we don't need to
            //   zip and add, but we still need to clamp
            // - when input.is_silent(), then we can copy_from_slice the buffer into
            //   output and then just clamp
            *output = input.clone();
            output.set_single_valued(false);

            output
                .channel_data_mut(0)
                .iter_mut()
                .zip(self.buffer.iter().cycle())
                .for_each(|(o, p)| {
                    *o += p;

                    if o.is_nan() {
                        *o = self.default_value;
                    }

                    *o = o.clamp(self.min_value, self.max_value)
                });
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
        diff.mul_add(phase as f32, start_value)
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
        start_value * ratio.powf(phase as f32)
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
        diff.mul_add(exponent.exp() as f32, end_value)
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
            values[values.len() - 1]
        } else {
            let position = (values.len() - 1) as f64 * (time - start_time) / duration;
            let k = position as usize;
            let phase = (position - position.floor()) as f32;
            (values[k + 1] - values[k]).mul_add(phase, values[k])
        }
    }

    fn handle_incoming_events(&mut self) {
        // cf. https://www.w3.org/TR/webaudio/#computation-of-value
        // 1. paramIntrinsicValue will be calculated at each time, which is either the
        // value set directly to the value attribute, or, if there are any automation
        // events with times before or at this time, the value as calculated from
        // these events. If automation events are removed from a given time range,
        // then the paramIntrinsicValue value will remain unchanged and stay at its
        // previous value until either the value attribute is directly set, or
        // automation events are added for the time range.
        for event in self.receiver.try_iter() {
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
                                // on this: Firefox actually restore intrinsic_value
                                // from the value at the beginning of the vent, while
                                // Chrome just keeps the current intrinsic_value
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
                                    self.intrinsic_value = last_event.value;
                                }
                            }
                            _ => (),
                        }
                    }
                }

                // remove all events in queue where event.time >= cancel_time
                // i.e. keep all events where event.time < cancel_time
                self.event_timeline
                    .retain(|queued| queued.time < event.time);
                continue; // cancel_values events are not inserted in queue
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

            // [spec] Similarly a NotSupportedError exception MUST be thrown if any
            // automation method is called at a time which is contained in [𝑇,𝑇+𝐷), 𝑇
            // being the time of the curve and 𝐷 its duration.
            // @note - Cancel methods are not automation methods
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

            // handle SetValue - param intrinsic value must be updated from event value
            if event.event_type == AudioParamEventType::SetValue {
                self.intrinsic_value = event.value;
            }

            // If no event in the timeline and event_type is `LinearRampToValueAtTime`
            // or `ExponentialRampToValue` at time, we must insert a `SetValueAtTime`
            // with intrinsic value and calling time.
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
            if self.event_timeline.is_empty()
                && self.last_event.is_none()
                && (event.event_type == AudioParamEventType::LinearRampToValueAtTime
                    || event.event_type == AudioParamEventType::ExponentialRampToValueAtTime)
            {
                let set_value_event = AudioParamEvent {
                    event_type: AudioParamEventType::SetValue,
                    value: self.intrinsic_value,
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

            // for SetTarget, this behavior is not per se specified, but it allows
            // to make sure we have a stable start_value available without having
            // to store it elsewhere.
            if self.event_timeline.is_empty()
                && event.event_type == AudioParamEventType::SetTargetAtTime
            {
                let set_value_event = AudioParamEvent {
                    event_type: AudioParamEventType::SetValue,
                    value: self.intrinsic_value,
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

        self.event_timeline.sort();
    }

    fn compute_buffer(&mut self, block_time: f64, dt: f64, count: usize) {
        // Set [[current value]] to the value of paramIntrinsicValue at the
        // beginning of this render quantum.
        let clamped = self.intrinsic_value.clamp(self.min_value, self.max_value);
        self.current_value.store(clamped, Ordering::SeqCst);

        // clear the buffer for this block
        self.buffer.clear();

        let is_a_rate = self.is_a_rate.load(Ordering::SeqCst);
        let is_k_rate = !is_a_rate;

        let next_block_time = dt.mul_add(count as f64, block_time);

        // Check if we can safely return a buffer of length 1 even for a-rate params.
        // Several cases allow us to do so:
        // - The timeline is empty
        // - The timeline is not empty: in such case if `event.time >= next_block_time`
        //   AND `event_type` is not `LinearRampToValueAtTime` or `ExponentialRampToValueAtTime`
        //   this is safe, i.e.:
        //   + For linear and exponential ramps `event.time` is the end time while their
        //   start time is `last_event.time`, therefore if `peek()` is of these
        //   types, we are in the middle of the ramp.
        //   + For all other event, `event.time` is their start time.
        //   (@note - `SetTargetAtTime` events also uses `last_event` but only for
        //   its value, not for its timing information, so no problem there)
        let is_constant_block = match self.event_timeline.peek() {
            None => true,
            Some(event) => {
                if event.event_type != AudioParamEventType::LinearRampToValueAtTime
                    && event.event_type != AudioParamEventType::ExponentialRampToValueAtTime
                {
                    event.time >= next_block_time
                } else {
                    false
                }
            }
        };

        if is_k_rate || is_constant_block {
            self.buffer.push(self.intrinsic_value);
            // nothing to compute in timeline, for both k-rate and a-rate
            if is_constant_block {
                return;
            }
        }

        loop {
            let some_event = self.event_timeline.peek();

            match some_event {
                None => {
                    if is_a_rate {
                        self.buffer.resize(count, self.intrinsic_value);
                    }
                    break;
                }
                Some(event) => {
                    match event.event_type {
                        AudioParamEventType::SetValue | AudioParamEventType::SetValueAtTime => {
                            let value = event.value;
                            let mut time = event.time;

                            // `set_value` calls and implicitly inserted events
                            // are inserted with a `time = 0.` to make sure
                            // they are processed first, replacing w/ block_time
                            // allows to conform to the spec:
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-value
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-linearramptovalueattime
                            // cf. https://www.w3.org/TR/webaudio/#dom-audioparam-exponentialramptovalueattime
                            if time == 0. {
                                time = block_time;
                            }

                            // fill buffer with current intrinsic value until `event.time`
                            if is_a_rate {
                                let end_index = ((time - block_time).max(0.) / dt) as usize;
                                let end_index_clipped = end_index.min(count);

                                for _ in self.buffer.len()..end_index_clipped {
                                    self.buffer.push(self.intrinsic_value);
                                }
                            }

                            if time > next_block_time {
                                break;
                            } else {
                                self.intrinsic_value = value;

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
                                if end_index_clipped > start_index {
                                    let mut time = (start_index as f64).mul_add(dt, block_time);

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
                                        self.intrinsic_value = value;
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
                                self.intrinsic_value = value;
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

                                self.intrinsic_value = value;

                                let mut last_event = self.event_timeline.pop().unwrap();
                                last_event.time = end_time;
                                last_event.value = value;
                                self.last_event = Some(last_event);

                            // Event ended during this block
                            } else {
                                self.intrinsic_value = end_value;
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

                                    if end_index_clipped > start_index {
                                        let mut time = (start_index as f64).mul_add(dt, block_time);

                                        for _ in start_index..end_index_clipped {
                                            let value = self.compute_exponential_ramp_sample(
                                                start_time,
                                                duration,
                                                start_value,
                                                ratio,
                                                time,
                                            );

                                            self.buffer.push(value);
                                            self.intrinsic_value = value;

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
                                    self.intrinsic_value = value;
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

                                    self.intrinsic_value = value;

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value;
                                    self.last_event = Some(last_event);

                                // Event ended during this block
                                } else {
                                    self.intrinsic_value = end_value;
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
                            // if SetTarget is the first event registered, we implicitly
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

                                if end_index_clipped > start_index {
                                    let mut time = (start_index as f64).mul_add(dt, block_time);

                                    for _ in start_index..end_index_clipped {
                                        // check if we have reached start_time
                                        let value = if time - start_time < 0. {
                                            self.intrinsic_value
                                        } else {
                                            self.compute_set_target_sample(
                                                start_time,
                                                time_constant,
                                                end_value,
                                                diff,
                                                time,
                                            )
                                        };

                                        self.buffer.push(value);
                                        self.intrinsic_value = value;
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

                                let diff = (end_value - value).abs();

                                // abort event if diff is below SNAP_TO_TARGET
                                if diff < SNAP_TO_TARGET {
                                    self.intrinsic_value = end_value;

                                    // if end_value is zero, the buffer might contain
                                    // subnormals, we need to check that and flush to zero
                                    if end_value == 0. {
                                        for v in self.buffer.iter_mut() {
                                            if v.is_subnormal() {
                                                *v = 0.;
                                            }
                                        }
                                    }

                                    let event = AudioParamEvent {
                                        event_type: AudioParamEventType::SetValueAtTime,
                                        time: next_block_time,
                                        value: end_value,
                                        time_constant: None,
                                        cancel_time: None,
                                        duration: None,
                                        values: None,
                                    };

                                    self.event_timeline.replace_peek(event);
                                } else {
                                    self.intrinsic_value = value;
                                }
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

                                self.intrinsic_value = value;
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

                                if end_index_clipped > start_index {
                                    let mut time = (start_index as f64).mul_add(dt, block_time);

                                    for _ in start_index..end_index_clipped {
                                        // check if we have reached start_time
                                        let value = if time - start_time < 0. {
                                            self.intrinsic_value
                                        } else {
                                            self.compute_set_value_curve_sample(
                                                start_time, duration, values, time,
                                            )
                                        };

                                        self.buffer.push(value);
                                        self.intrinsic_value = value;

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
                                self.intrinsic_value = value;
                                break;

                            // handle end of event during this block
                            } else {
                                // event has been cancelled
                                if event.cancel_time.is_some() {
                                    let value = self.compute_set_value_curve_sample(
                                        start_time, duration, values, end_time,
                                    );

                                    self.intrinsic_value = value;

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value;
                                    self.last_event = Some(last_event);
                                // event has ended
                                } else {
                                    let value = values[values.len() - 1];

                                    let mut last_event = self.event_timeline.pop().unwrap();
                                    last_event.time = end_time;
                                    last_event.value = value;

                                    self.intrinsic_value = value;
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
    }
}

pub(crate) fn audio_param_pair(
    opts: AudioParamDescriptor,
    registration: AudioContextRegistration,
) -> (AudioParam, AudioParamProcessor) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let current_value = Arc::new(AtomicF32::new(opts.default_value));
    let is_a_rate = Arc::new(AtomicBool::new(opts.automation_rate == AutomationRate::A));

    let param = AudioParam {
        registration,
        is_a_rate: is_a_rate.clone(),
        automation_rate_constrained: false,
        default_value: opts.default_value,
        min_value: opts.min_value,
        max_value: opts.max_value,
        current_value: current_value.clone(),
        sender,
    };

    let render = AudioParamProcessor {
        intrinsic_value: opts.default_value,
        current_value,
        receiver,
        is_a_rate,
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
    use crate::render::Alloc;

    use super::*;

    #[test]
    #[should_panic]
    fn test_assert_non_negative_fail() {
        assert_non_negative(-1.);
    }

    #[test]
    fn test_assert_non_negative() {
        assert_non_negative(0.);
    }

    #[test]
    #[should_panic]
    fn test_assert_strictly_positive_fail() {
        assert_strictly_positive(0.);
    }

    #[test]
    fn test_assert_strictly_positive() {
        assert_strictly_positive(0.1);
    }

    #[test]
    #[should_panic]
    fn test_assert_not_zero_fail() {
        assert_not_zero(0.);
    }

    #[test]
    fn test_assert_not_zero() {
        assert_not_zero(-0.1);
        assert_not_zero(0.1);
    }

    #[test]
    #[should_panic]
    fn test_assert_sequence_length_fail() {
        assert_sequence_length(&[0.; 1]);
    }

    #[test]
    fn test_assert_sequence_length() {
        assert_sequence_length(&[0.; 2]);
    }

    #[test]
    fn test_default_and_accessors() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: -10.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value(2.);
            assert_float_eq!(param.value(), 2., abs_all <= 0.);

            let vs = render.compute_intrinsic_values(0., 1., 10);

            assert_float_eq!(param.value(), 2., abs_all <= 0.);
            assert_float_eq!(vs, &[2.; 10][..], abs_all <= 0.);
        }

        // make sure param.value() is properly clamped
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 1.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value(2.);
            assert_float_eq!(param.value(), 1., abs_all <= 0.);

            let vs = render.compute_intrinsic_values(0., 1., 10);

            // value should clamped while intrinsic value should not
            assert_float_eq!(param.value(), 1., abs_all <= 0.);
            assert_float_eq!(vs, &[2.; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_steps_a_rate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0., 5., 5., 5., 5., 5., 5., 12., 12.][..],
                abs_all <= 0.
            );

            // no event left in timeline, i.e. length is 1
            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[8.; 1][..], abs_all <= 0.);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0., 5., 5., 5., 5., 5., 5., 5., 5.][..],
                abs_all <= 0.
            );

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(
                vs,
                &[5., 5., 8., 8., 8., 8., 8., 8., 8., 8.][..],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_steps_k_rate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[8.; 1][..], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &[3.; 1][..], abs_all <= 0.);
    }

    #[test]
    fn test_linear_ramp_arate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0., 5., 6., 7., 8., 7., 6., 5., 4.][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_linear_ramp_arate_end_of_block() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            abs_all <= 0.
        );
    }

    #[test]
    // @note - with real params, a set_value event is always used to provide
    // init value to the param. Therefore this test is purely theoretical and
    // in real world the param would not behave like that, which is wrong
    // @todo - open an issue to review how init value is passed (or how last_event is set)
    fn test_linear_ramp_arate_implicit_set_value() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // mimic a ramp inserted after start
        // i.e. setTimeout(() => param.linearRampToValueAtTime(10, now + 10)), 10 * 1000);

        // no event in timeline here, i.e. length is 1
        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);

        // implicitly insert a SetValue event at time 10
        param.linear_ramp_to_value_at_time(10.0, 20.0);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            abs_all <= 0.
        );

        // ramp finishes on first value of this block, i.e. length is 10
        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &[10.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_linear_ramp_arate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
            abs_all <= 0.
        );
        assert_float_eq!(param.value(), 0., abs <= 0.);

        // next quantum t = 10..20
        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(
            vs,
            &[10., 11., 12., 13., 14., 15., 16., 17., 18., 19.][..],
            abs_all <= 0.
        );
        assert_float_eq!(param.value(), 10., abs <= 0.);

        // ramp finished t = 20..30
        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &[20.0; 10][..], abs_all <= 0.);
        assert_float_eq!(param.value(), 20., abs <= 0.);
    }

    #[test]
    fn test_linear_ramp_krate_multiple_blocks() {
        // regression test for issue #9
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 0., abs <= 0.);
            // next quantum t = 10..20
            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[10.; 1][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 10., abs <= 0.);
            // ramp finished t = 20..30
            let vs = render.compute_intrinsic_values(20., 1., 10);
            assert_float_eq!(vs, &[20.0; 1][..], abs_all <= 0.);
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
            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 0., abs <= 0.);
            // next quantum t = 10..20
            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[10.; 1][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 10., abs <= 0.);
            // ramp finished t = 20..30
            let vs = render.compute_intrinsic_values(20., 1., 10);
            assert_float_eq!(vs, &[15.0; 1][..], abs_all <= 0.);
            assert_float_eq!(param.value(), 15., abs <= 0.);
        }
    }

    #[test]
    fn test_linear_ramp_start_time() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(1., 0.);
        param.linear_ramp_to_value_at_time(-1., 10.);
        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[1., 0.8, 0.6, 0.4, 0.2, 0., -0.2, -0.4, -0.6, -0.8][..],
            abs_all <= 1e-7
        );

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[-1.; 10][..], abs_all <= 0.);

        // start time should be end time of last event, i.e. 10.
        param.linear_ramp_to_value_at_time(1., 30.);

        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_exponential_ramp_a_rate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &res[..], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[1.0; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_exponential_ramp_a_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &res[0..10], abs_all <= 0.);
        assert_float_eq!(param.value(), res[0], abs <= 0.);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        assert_float_eq!(param.value(), res[10], abs <= 0.);
    }

    #[test]
    fn test_exponential_ramp_a_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &[res[0]; 1][..], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[res[10]; 1][..], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &[1.; 1][..], abs_all <= 0.);
    }

    #[test]
    fn test_exponential_ramp_k_rate_zero_and_opposite_target() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[1.; 1][..], abs_all <= 0.);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &[-1.; 1][..], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[1.; 1][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_exponential_ramp_start_time() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(0., 0.);
        param.linear_ramp_to_value_at_time(1., 10.);

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][..],
            abs_all <= 1e-7
        );

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[1.; 10][..], abs_all <= 0.);

        // start time should be end time of last event, i.e. 10.
        param.exponential_ramp_to_value_at_time(0.0001, 30.);
        let vs = render.compute_intrinsic_values(20., 1., 10);
        // compute expected on 20 samples, the 10 last ones should be in vs
        let start: f32 = 1.;
        let end: f32 = 0.0001;
        let mut res = [0.; 20];
        for (t, v) in res.iter_mut().enumerate() {
            *v = start * (end / start).powf(t as f32 / 20.);
        }

        assert_float_eq!(vs, &res[10..], abs_all <= 1e-7);
    }

    #[test]
    fn test_set_target_at_time_a_rate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
            let vs = render.compute_intrinsic_values(0., 1., 10);

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
            let vs = render.compute_intrinsic_values(0., 1., 10);

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
            // start_time is 1.
            res[0] = 0.;

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }

        {
            // handle time_constant == 0.
            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 100.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());
            param.set_target_at_time(1., 1., 0.);

            let mut res = [1.; 10];
            res[0] = 0.; // start_time is 1.

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_a_rate_followed_by_set_value() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_ends_at_threshold() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 2.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(1., 0.);
        param.set_target_at_time(0., 1., 0.2);

        let vs = render.compute_intrinsic_values(0., 1., 128);
        for v in vs.iter() {
            assert!(!v.is_subnormal());
        }

        // check peek() has been replaced with set_value event
        let peek = render.event_timeline.peek();
        assert_eq!(
            peek.unwrap().event_type,
            AudioParamEventType::SetValueAtTime
        );

        // this buffer should be filled with target values
        let vs = render.compute_intrinsic_values(10., 1., 128);
        assert_float_eq!(vs[..], [0.; 128], abs_all <= 0.);
    }

    #[test]
    fn test_set_target_at_time_waits_for_start_time() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 2.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(1., 0.);
        param.set_target_at_time(0., 5., 1.);

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs[0], 1., abs <= 0.);
        assert_float_eq!(vs[1], 1., abs <= 0.);
        assert_float_eq!(vs[2], 1., abs <= 0.);
        assert_float_eq!(vs[3], 1., abs <= 0.);
        assert_float_eq!(vs[4], 1., abs <= 0.);
        assert_float_eq!(vs[5], 1., abs <= 0.);
    }

    #[test]
    fn test_set_target_at_time_a_rate_followed_by_ramp() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 1.0e-6);
            // ramp ended
            let vs = render.compute_intrinsic_values(20., 1., 10);
            assert_float_eq!(vs, &[v1; 10][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_set_target_at_time_k_rate_multiple_blocks() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &[res[0]; 1][..], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[res[10]; 1][..], abs_all <= 0.);
        }
    }

    #[test]
    // regression test for bcebfe6
    fn test_set_target_at_time_snap_to_value() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 1.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());
        let v0: f32 = 1.;
        let v1: f32 = 0.;
        let t0: f64 = 0.;
        let time_constant: f64 = 1.;

        param.set_value_at_time(v0, t0);
        param.set_target_at_time(v1, t0, time_constant);

        let mut res = [0.; 30];
        // 𝑣(𝑡) = 𝑉1 + (𝑉0 − 𝑉1) * 𝑒^−((𝑡−𝑇0) / 𝜏)
        res.iter_mut().enumerate().for_each(|(t, r)| {
            *r = v1 + (v0 - v1) * (-1. * ((t as f64 - t0) / time_constant)).exp() as f32;
        });

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &res[..10], abs_all <= 0.);

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &res[10..20], abs_all <= 0.);

        // the distance between the target value and the value just after this block
        // is smaller than SNAP_TO_TARGET (i.e. 1e-10)
        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &res[20..30], abs_all <= 0.);

        // then this block should be [0.; 10]
        let vs = render.compute_intrinsic_values(30., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_cancel_scheduled_values() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 1., 2., 3., 4., 4., 4., 4., 4., 4.][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_cancel_scheduled_values_ramp() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
            // cancels the ramp, the set value event is kept in timeline
            param.cancel_scheduled_values(10.);

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
                abs_all <= 0.
            );

            // the SetValue event has been consumed in first tick and the ramp
            // is removed from timeline, no event left in timeline (length is 1)
            param.cancel_scheduled_values(10.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);
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

            // the SetValue from param inserted by the Ramp is left in timeline
            // i.e. length is 10
            param.linear_ramp_to_value_at_time(10., 10.);
            param.cancel_scheduled_values(10.); // cancels the ramp

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.][..],
                abs_all <= 0.
            );

            // ramp is removed from timeline, no event left
            param.cancel_scheduled_values(10.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold() {
        let context = OfflineAudioContext::new(1, 0, 48000.);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 2., 2., 2., 2., 2., 2., 2.][0..10],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_cancel_and_hold_during_set_target() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[0..10], abs_all <= 0.);

            let vs = render.compute_intrinsic_values(10., 1., 10);
            assert_float_eq!(vs, &res[10..20], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold_during_linear_ramp() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 1., 2., 3., 4., 4.5, 4.5, 4.5, 4.5, 4.5][0..10],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_cancel_and_hold_during_exponential_ramp() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(vs, &res[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_cancel_and_hold_during_set_value_curve() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

            let vs = render.compute_intrinsic_values(0., 1., 10);
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

            let vs = render.compute_intrinsic_values(0., 1., 10);
            assert_float_eq!(
                vs,
                &[0., 0.2, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9][..],
                abs_all <= 1e-7
            );
        }
    }

    #[test]
    fn test_set_value_curve_at_time_a_rate() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2][..],
            abs_all <= 1e-7
        );

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_set_value_curve_at_time_a_rate_multiple_frames() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][..],
            abs_all <= 1e-7
        );

        let vs = render.compute_intrinsic_values(10., 1., 10);
        assert_float_eq!(
            vs,
            &[1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][..],
            abs_all <= 1e-7
        );

        let vs = render.compute_intrinsic_values(20., 1., 10);
        assert_float_eq!(vs, &[0.; 10][..], abs_all <= 0.);
    }

    #[test]
    #[should_panic]
    fn test_set_value_curve_at_time_insert_while_another_event() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        let _vs = render.compute_intrinsic_values(0., 1., 10);
    }

    #[test]
    #[should_panic]
    fn test_set_value_curve_at_time_insert_another_event_inside() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

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
        let _vs = render.compute_intrinsic_values(0., 1., 10);
    }

    #[test]
    fn test_set_value_curve_waits_for_start_time() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: 0.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        // set to 0.0001 at t=0 (0. is a special case)
        let curve = [0., 0.5, 1., 0.5, 0.];
        param.set_value_curve_at_time(&curve[..], 5., 10.);

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(
            vs,
            &[0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_update_automation_rate_to_k() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(2., 0.000001);
        param.set_automation_rate(AutomationRate::K);

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);
    }

    #[test]
    fn test_update_automation_rate_to_a() {
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::K,
            default_value: 0.,
            min_value: -10.,
            max_value: 10.,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value_at_time(2., 0.000001);
        param.set_automation_rate(AutomationRate::A);

        let vs = render.compute_intrinsic_values(0., 1., 10);
        assert_float_eq!(vs, &[2.; 10][..], abs_all <= 0.);
    }

    #[test]
    fn test_varying_param_size() {
        // event registered online during rendering
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(0., 0.);
            param.linear_ramp_to_value_at_time(9., 9.);

            // first block should be length 10 (128 in real world)
            let vs = render.compute_intrinsic_values(0., 1., 10);
            let expected = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // second block should have length 1
            let vs = render.compute_intrinsic_values(10., 1., 10);
            let expected = [9.; 1];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // insert event in third block, should have length 10
            param.set_value_at_time(1., 25.);

            let vs = render.compute_intrinsic_values(20., 1., 10);
            let expected = [9., 9., 9., 9., 9., 1., 1., 1., 1., 1.];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // fourth block should have length 1
            let vs = render.compute_intrinsic_values(30., 1., 10);
            let expected = [1.; 1];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);
        }

        // event registered before rendering
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (param, mut render) = audio_param_pair(opts, context.mock_registration());

            param.set_value_at_time(0., 0.);
            param.linear_ramp_to_value_at_time(9., 9.);
            param.set_value_at_time(1., 25.);

            // first block should be length 10 (128 in real world)
            let vs = render.compute_intrinsic_values(0., 1., 10);
            let expected = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // second block should have length 1
            let vs = render.compute_intrinsic_values(10., 1., 10);
            let expected = [9.; 1];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // set value event in third block, length should be 10
            let vs = render.compute_intrinsic_values(20., 1., 10);
            let expected = [9., 9., 9., 9., 9., 1., 1., 1., 1., 1.];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);

            // fourth block should have length 1
            let vs = render.compute_intrinsic_values(30., 1., 10);
            let expected = [1.; 1];
            assert_float_eq!(vs, &expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_varying_param_size_modulated() {
        let alloc = Alloc::with_capacity(1);

        // buffer length is 1 and input is silence (no modulation)
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (_param, mut render) = audio_param_pair(opts, context.mock_registration());

            // no event in timeline, buffer length is 1
            let vs = render.compute_intrinsic_values(0., 1., 128);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);

            // mix to output step, input is silence
            let signal = alloc.silence();
            let input = AudioRenderQuantum::from(signal);

            let signal = alloc.silence();
            let mut output = AudioRenderQuantum::from(signal);

            render.mix_to_output(&input, &mut output);

            assert!(output.single_valued());
            assert_float_eq!(output.channel_data(0)[0], 0., abs <= 0.);
        }

        // buffer length is 1 and input is non silent
        {
            let context = OfflineAudioContext::new(1, 0, 48000.);

            let opts = AudioParamDescriptor {
                automation_rate: AutomationRate::A,
                default_value: 0.,
                min_value: 0.,
                max_value: 10.,
            };
            let (_param, mut render) = audio_param_pair(opts, context.mock_registration());

            // no event in timeline, buffer length is 1
            let vs = render.compute_intrinsic_values(0., 1., 128);
            assert_float_eq!(vs, &[0.; 1][..], abs_all <= 0.);

            // mix to output step, input is not silence
            let signal = alloc.silence();
            let mut input = AudioRenderQuantum::from(signal);
            input.channel_data_mut(0)[0] = 1.;

            let signal = alloc.silence();
            let mut output = AudioRenderQuantum::from(signal);

            render.mix_to_output(&input, &mut output);

            let mut expected = [0.; 128];
            expected[0] = 1.;

            assert!(!output.single_valued());
            assert_float_eq!(output.channel_data(0)[..], &expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_full_render_chain() {
        let alloc = Alloc::with_capacity(1);
        // prevent regression between the different processing stage
        let context = OfflineAudioContext::new(1, 0, 48000.);

        let min = 2.;
        let max = 42.;
        let default = 2.;

        let opts = AudioParamDescriptor {
            automation_rate: AutomationRate::A,
            default_value: default,
            min_value: min,
            max_value: max,
        };
        let (param, mut render) = audio_param_pair(opts, context.mock_registration());

        param.set_value(128.);
        param.linear_ramp_to_value_at_time(0., 128.);

        let intrinsic_values = render.compute_intrinsic_values(0., 1., 128);
        let mut expected = [0.; 128];
        for (i, v) in expected.iter_mut().enumerate() {
            *v = 128. - i as f32;
        }
        assert_float_eq!(intrinsic_values, &expected[..], abs_all <= 0.);

        let signal = alloc.silence();
        let mut input = AudioRenderQuantum::from(signal);
        input.channel_data_mut(0)[0] = f32::NAN;
        let signal = alloc.silence();
        let mut output = AudioRenderQuantum::from(signal);

        render.mix_to_output(&input, &mut output);

        // clamp expected
        expected.iter_mut().for_each(|v| *v = v.clamp(min, max));
        // fix NAN at position 0
        expected[0] = 2.;

        assert_float_eq!(output.channel_data(0)[..], &expected[..], abs_all <= 0.);
    }
}
