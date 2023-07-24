use std::any::Any;
use std::sync::{Mutex, OnceLock};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor, AutomationRate};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, AudioScheduledSourceNode, ChannelConfig};

/// Options for constructing an [`AudioBufferSourceNode`]
// dictionary AudioBufferSourceOptions {
//   AudioBuffer? buffer;
//   float detune = 0;
//   boolean loop = false;
//   double loopEnd = 0;
//   double loopStart = 0;
//   float playbackRate = 1;
// };
//
// @note - Does extend AudioNodeOptions but they are useless for source nodes as
// they instruct how to upmix the inputs.
// This is a common source of confusion, see e.g. https://github.com/mdn/content/pull/18472, and
// an issue in the spec, see discussion in https://github.com/WebAudio/web-audio-api/issues/2496
#[derive(Clone, Debug)]
pub struct AudioBufferSourceOptions {
    pub buffer: Option<AudioBuffer>,
    pub detune: f32,
    pub loop_: bool,
    pub loop_start: f64,
    pub loop_end: f64,
    pub playback_rate: f32,
}

impl Default for AudioBufferSourceOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            detune: 0.,
            loop_: false,
            loop_start: 0.,
            loop_end: 0.,
            playback_rate: 1.,
        }
    }
}

#[derive(Copy, Clone)]
struct PlaybackInfo {
    prev_frame_index: usize,
    k: f32,
}

#[derive(Debug, Clone)]
struct LoopState {
    pub is_looping: bool,
    pub start: f64,
    pub end: f64,
}

/// Instructions to start or stop processing
#[derive(Debug, Clone)]
enum ControlMessage {
    StartWithOffsetAndDuration(f64, f64, f64),
    Stop(f64),
    Loop(bool),
    LoopStart(f64),
    LoopEnd(f64),
}

/// `AudioBufferSourceNode` represents an audio source that consists of an
/// in-memory audio source (i.e. an audio file completely loaded in memory),
/// stored in an [`AudioBuffer`].
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#AudioBufferSourceNode>
/// - see also: [`BaseAudioContext::create_buffer_source`](crate::context::BaseAudioContext::create_buffer_source)
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // create an `AudioContext`
/// let context = AudioContext::default();
/// // load and decode a soundfile
/// let file = File::open("samples/sample.wav").unwrap();
/// let audio_buffer = context.decode_audio_data_sync(file).unwrap();
/// // play the sound file
/// let src = context.create_buffer_source();
/// src.set_buffer(audio_buffer);
/// src.connect(&context.destination());
/// src.start();
/// ```
///
/// # Examples
///
/// - `cargo run --release --example trigger_soundfile`
/// - `cargo run --release --example granular`
///
pub struct AudioBufferSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    detune: AudioParam,        // has constraints, no a-rate
    playback_rate: AudioParam, // has constraints, no a-rate
    buffer: OnceLock<AudioBuffer>,
    inner_state: Mutex<InnerState>,
}

#[derive(Debug, Clone)]
struct InnerState {
    loop_state: LoopState,
    source_started: bool,
}

impl AudioNode for AudioBufferSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        0
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl AudioScheduledSourceNode for AudioBufferSourceNode {
    fn start(&self) {
        let start = self.registration.context().current_time();
        self.start_at_with_offset_and_duration(start, 0., f64::MAX);
    }

    fn start_at(&self, when: f64) {
        self.start_at_with_offset_and_duration(when, 0., f64::MAX);
    }

    fn stop(&self) {
        let stop = self.registration.context().current_time();
        self.stop_at(stop);
    }

    fn stop_at(&self, when: f64) {
        assert!(
            self.inner_state.lock().unwrap().source_started,
            "InvalidStateError cannot stop before start"
        );

        self.registration.post_message(ControlMessage::Stop(when));
    }
}

impl AudioBufferSourceNode {
    /// Create a new [`AudioBufferSourceNode`] instance
    pub fn new<C: BaseAudioContext>(context: &C, options: AudioBufferSourceOptions) -> Self {
        context.register(move |registration| {
            let AudioBufferSourceOptions {
                buffer,
                detune,
                loop_,
                loop_start,
                loop_end,
                playback_rate,
            } = options;

            // these parameters can't be changed to a-rate
            // @see - <https://webaudio.github.io/web-audio-api/#audioparam-automation-rate-constraints>
            let detune_param_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 0.,
                automation_rate: AutomationRate::K,
            };
            let (mut d_param, d_proc) =
                context.create_audio_param(detune_param_options, &registration);
            d_param.set_automation_rate_constrained(true);
            d_param.set_value(detune);

            let playback_rate_param_options = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: AutomationRate::K,
            };
            let (mut pr_param, pr_proc) =
                context.create_audio_param(playback_rate_param_options, &registration);
            pr_param.set_automation_rate_constrained(true);
            pr_param.set_value(playback_rate);

            let loop_state = LoopState {
                is_looping: loop_,
                start: loop_start,
                end: loop_end,
            };

            let renderer = AudioBufferSourceRenderer {
                start_time: f64::MAX,
                stop_time: f64::MAX,
                duration: f64::MAX,
                offset: 0.,
                buffer: None,
                detune: d_proc,
                playback_rate: pr_proc,
                loop_state: loop_state.clone(),
                render_state: AudioBufferRendererState::default(),
                ended_triggered: false,
            };

            let inner_state = InnerState {
                loop_state,
                source_started: false,
            };
            let node = Self {
                registration,
                channel_config: ChannelConfig::default(),
                detune: d_param,
                playback_rate: pr_param,
                buffer: OnceLock::new(),
                inner_state: Mutex::new(inner_state),
            };

            if let Some(buf) = buffer {
                node.set_buffer(buf);
            }

            (node, Box::new(renderer))
        })
    }

    /// Start the playback at the given time and with a given offset
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    pub fn start_at_with_offset(&self, start: f64, offset: f64) {
        self.start_at_with_offset_and_duration(start, offset, f64::MAX);
    }

    /// Start the playback at the given time, with a given offset, for a given duration
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    pub fn start_at_with_offset_and_duration(&self, start: f64, offset: f64, duration: f64) {
        let inner_state = &mut self.inner_state.lock().unwrap();
        assert!(
            !inner_state.source_started,
            "InvalidStateError: Cannot call `start` twice"
        );
        inner_state.source_started = true;

        let control = ControlMessage::StartWithOffsetAndDuration(start, offset, duration);
        self.registration.post_message(control);
    }

    /// Current buffer value (nullable)
    pub fn buffer(&self) -> Option<&AudioBuffer> {
        self.buffer.get()
    }

    /// Provide an [`AudioBuffer`] as the source of data to be played bask
    ///
    /// # Panics
    ///
    /// Panics if a buffer has already been given to the source (though `new` or through
    /// `set_buffer`)
    pub fn set_buffer(&self, audio_buffer: AudioBuffer) {
        let clone = audio_buffer.clone();

        if self.buffer.set(audio_buffer).is_err() {
            panic!("InvalidStateError - cannot assign buffer twice");
        }

        self.registration.post_message(clone);
    }

    /// K-rate [`AudioParam`] that defines the speed at which the [`AudioBuffer`]
    /// will be played, e.g.:
    /// - `0.5` will play the file at half speed
    /// - `-1` will play the file in reverse
    ///
    /// Note that playback rate will also alter the pitch of the [`AudioBuffer`]
    pub fn playback_rate(&self) -> &AudioParam {
        &self.playback_rate
    }

    /// K-rate [`AudioParam`] that defines a pitch transposition of the file,
    /// expressed in cents
    ///
    /// see <https://en.wikipedia.org/wiki/Cent_(music)>
    pub fn detune(&self) -> &AudioParam {
        &self.detune
    }

    /// Defines if the playback the [`AudioBuffer`] should be looped
    #[allow(clippy::missing_panics_doc)]
    pub fn loop_(&self) -> bool {
        self.inner_state.lock().unwrap().loop_state.is_looping
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_loop(&self, value: bool) {
        self.inner_state.lock().unwrap().loop_state.is_looping = value;
        self.registration.post_message(ControlMessage::Loop(value));
    }

    /// Defines the loop start point, in the time reference of the [`AudioBuffer`]
    #[allow(clippy::missing_panics_doc)]
    pub fn loop_start(&self) -> f64 {
        self.inner_state.lock().unwrap().loop_state.start
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_loop_start(&self, value: f64) {
        self.inner_state.lock().unwrap().loop_state.start = value;
        self.registration
            .post_message(ControlMessage::LoopStart(value));
    }

    /// Defines the loop end point, in the time reference of the [`AudioBuffer`]
    #[allow(clippy::missing_panics_doc)]
    pub fn loop_end(&self) -> f64 {
        self.inner_state.lock().unwrap().loop_state.end
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn set_loop_end(&self, value: f64) {
        self.inner_state.lock().unwrap().loop_state.end = value;
        self.registration
            .post_message(ControlMessage::LoopEnd(value));
    }
}

struct AudioBufferRendererState {
    buffer_time: f64,
    started: bool,
    entered_loop: bool,
    buffer_time_elapsed: f64,
    is_aligned: bool,
}

impl Default for AudioBufferRendererState {
    fn default() -> Self {
        Self {
            buffer_time: 0.,
            started: false,
            entered_loop: false,
            buffer_time_elapsed: 0.,
            is_aligned: false,
        }
    }
}

struct AudioBufferSourceRenderer {
    start_time: f64,
    stop_time: f64,
    offset: f64,
    duration: f64,
    buffer: Option<AudioBuffer>,
    detune: AudioParamId,
    playback_rate: AudioParamId,
    loop_state: LoopState,
    render_state: AudioBufferRendererState,
    ended_triggered: bool,
}

impl AudioBufferSourceRenderer {
    fn handle_control_message(&mut self, control: &ControlMessage) {
        match control {
            ControlMessage::StartWithOffsetAndDuration(when, offset, duration) => {
                self.start_time = *when;
                self.offset = *offset;
                self.duration = *duration;
            }
            ControlMessage::Stop(when) => self.stop_time = *when,
            ControlMessage::Loop(is_looping) => self.loop_state.is_looping = *is_looping,
            ControlMessage::LoopStart(loop_start) => self.loop_state.start = *loop_start,
            ControlMessage::LoopEnd(loop_end) => self.loop_state.end = *loop_end,
        }
    }
}

impl AudioProcessor for AudioBufferSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum], // no input...
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        scope: &RenderScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        let sample_rate = scope.sample_rate as f64;
        let dt = 1. / sample_rate;
        let block_duration = dt * RENDER_QUANTUM_SIZE as f64;
        let next_block_time = scope.current_time + block_duration;

        let LoopState {
            is_looping,
            start: loop_start,
            end: loop_end,
        } = self.loop_state.clone();

        // these will only be used if `loop_` is true, so no need for `Option`
        let mut actual_loop_start = 0.;
        let mut actual_loop_end = 0.;

        // return early if start_time is beyond this block
        if self.start_time >= next_block_time {
            output.make_silent();
            return true;
        }

        // If the buffer has not been set wait for it.
        let buffer = match &self.buffer {
            None => {
                output.make_silent();
                return true;
            }
            Some(b) => b,
        };

        // compute compound parameter at k-rate, these parameters have constraints
        // https://webaudio.github.io/web-audio-api/#audioparam-automation-rate-constraints
        let detune = params.get(&self.detune)[0];
        let playback_rate = params.get(&self.playback_rate)[0];
        let computed_playback_rate = (playback_rate * (detune / 1200.).exp2()) as f64;

        let buffer_duration = buffer.duration();
        // multiplier to be applied on `position` to tackle possible difference
        // between the context and buffer sample rates. As this is an edge case,
        // we just linearly interpolate, thus favoring performance vs quality
        let sampling_ratio = buffer.sample_rate() as f64 / sample_rate;

        // In addition, if the buffer has more than one channel, then the
        // AudioBufferSourceNode output must change to a single channel of silence
        // at the beginning of a render quantum after the time at which any one of
        // the following conditions holds:

        // 1. the stop time has been reached.
        // 2. the duration has been reached.
        if scope.current_time >= self.stop_time
            || self.render_state.buffer_time_elapsed >= self.duration
        {
            output.make_silent(); // also converts to mono

            // @note: we need this check because this is called a until the program
            // ends, such as if the node was never removed from the graph
            if !self.ended_triggered {
                scope.send_ended_event();
                self.ended_triggered = true;
            }
            return false;
        }

        // 3. the end of the buffer has been reached.
        if !is_looping {
            if computed_playback_rate > 0. && self.render_state.buffer_time >= buffer_duration {
                output.make_silent(); // also converts to mono
                if !self.ended_triggered {
                    scope.send_ended_event();
                    self.ended_triggered = true;
                }
                return false;
            }

            if computed_playback_rate < 0. && self.render_state.buffer_time < 0. {
                output.make_silent(); // also converts to mono
                if !self.ended_triggered {
                    scope.send_ended_event();
                    self.ended_triggered = true;
                }
                return false;
            }
        }

        output.set_number_of_channels(buffer.number_of_channels());

        // go through the algorithm described in the spec
        // @see <https://webaudio.github.io/web-audio-api/#playback-AudioBufferSourceNode>
        let mut current_time = scope.current_time;

        // prevent scheduling in the past
        // If 0 is passed in for this value or if the value is less than
        // currentTime, then the sound will start playing immediately
        // cf. https://webaudio.github.io/web-audio-api/#dom-audioscheduledsourcenode-start-when-when
        if !self.render_state.started && self.start_time < current_time {
            self.start_time = current_time;
        }

        // Define if we can avoid the resampling interpolation in some common cases,
        // basically when:
        // - `src.start()` is called with `audio_context.current_time`,
        //   i.e. start time is aligned with a render quantum block
        // - the AudioBuffer was decoded w/ the right sample rate
        // - no detune or playback_rate changes are made
        // - loop boundaries have not been changed
        if self.start_time == current_time && self.offset == 0. {
            self.render_state.is_aligned = true;
        }

        // these two case imply resampling
        if sampling_ratio != 1. || computed_playback_rate != 1. {
            self.render_state.is_aligned = false;
        }

        // If loop points are not aligned on sample, they can imply resampling.
        // For now we just consider that we can go fast track if loop points are
        // bound to the buffer boundaries.
        //
        // by default loop_end is 0., see AudioBufferSourceOptions
        // but loop_start = 0 && loop_end = buffer.duration should go to fast track
        if loop_start != 0. || (loop_end != 0. && loop_end != self.duration) {
            self.render_state.is_aligned = false;
        }

        // ---------------------------------------------------------------
        // Fast track
        // ---------------------------------------------------------------
        if self.render_state.is_aligned {
            if self.start_time == current_time {
                self.render_state.started = true;
            }

            // check if buffer ends within this block
            if self.render_state.buffer_time + block_duration > buffer_duration
                || self.render_state.buffer_time + block_duration > self.duration
                || current_time + block_duration > self.stop_time
            {
                let buffer_time = self.render_state.buffer_time;
                let end_index = if current_time + block_duration > self.stop_time
                    || self.render_state.buffer_time + block_duration > self.duration
                {
                    let dt = (self.stop_time - current_time)
                        .min(self.duration - self.render_state.buffer_time);
                    let end_buffer_time = self.render_state.buffer_time + dt;
                    (end_buffer_time * sample_rate).round() as usize
                } else {
                    buffer.length()
                };
                // in case of a loop point in the middle of the block, this value
                // will be used to recompute `self.render_state.buffer_time` according
                // to the actual loop point.
                let mut loop_point_index: Option<usize> = None;

                buffer
                    .channels()
                    .iter()
                    .zip(output.channels_mut().iter_mut())
                    .for_each(|(buffer_channel, output_channel)| {
                        // we need to recompute that for each channel
                        let buffer_channel = buffer_channel.as_slice();
                        let mut start_index = (buffer_time * sample_rate).round() as usize;
                        let mut offset = 0;

                        for (index, o) in output_channel.iter_mut().enumerate() {
                            let mut buffer_index = start_index + index - offset;

                            *o = if buffer_index < end_index {
                                buffer_channel[buffer_index]
                            } else {
                                if is_looping && buffer_index == end_index {
                                    loop_point_index = Some(index);
                                    // reset values for the rest of the block
                                    start_index = 0;
                                    offset = index;
                                    buffer_index = 0;
                                }

                                if is_looping {
                                    buffer_channel[buffer_index]
                                } else {
                                    0.
                                }
                            };
                        }
                    });

                if let Some(loop_point_index) = loop_point_index {
                    self.render_state.buffer_time =
                        ((RENDER_QUANTUM_SIZE - loop_point_index) as f64 / sample_rate)
                            % buffer_duration;
                } else {
                    self.render_state.buffer_time += block_duration;
                }
            } else {
                let start_index = (self.render_state.buffer_time * sample_rate).round() as usize;
                let end_index = start_index + RENDER_QUANTUM_SIZE;
                // we can do memcopy
                buffer
                    .channels()
                    .iter()
                    .zip(output.channels_mut().iter_mut())
                    .for_each(|(buffer_channel, output_channel)| {
                        let buffer_channel = buffer_channel.as_slice();
                        output_channel.copy_from_slice(&buffer_channel[start_index..end_index]);
                    });

                self.render_state.buffer_time += block_duration;
            }

            // update render state
            self.render_state.buffer_time_elapsed += block_duration;

            return true;
        }

        // ---------------------------------------------------------------
        // Slow track
        // ---------------------------------------------------------------
        if is_looping {
            if loop_start >= 0. && loop_end > 0. && loop_start < loop_end {
                actual_loop_start = loop_start;
                actual_loop_end = loop_end.min(buffer_duration);
            } else {
                actual_loop_start = 0.;
                actual_loop_end = buffer_duration;
            }
        } else {
            self.render_state.entered_loop = false;
        }

        // internal buffer used to store playback infos to compute the samples
        // according to the source buffer. (prev_sample_index, k)
        let mut playback_infos = [None; RENDER_QUANTUM_SIZE];

        // compute position for each sample and store into `self.positions`
        for playback_info in playback_infos.iter_mut() {
            if current_time < self.start_time
                || current_time >= self.stop_time
                || self.render_state.buffer_time_elapsed >= self.duration
            {
                *playback_info = None;
                current_time += dt;

                continue; // nothing more to do for this sample
            }

            // we have now reached start time
            if !self.render_state.started {
                self.offset += current_time - self.start_time;

                if is_looping && computed_playback_rate >= 0. && self.offset >= actual_loop_end {
                    self.offset = actual_loop_end;
                }

                if is_looping && computed_playback_rate < 0. && self.offset < actual_loop_start {
                    self.offset = actual_loop_start;
                }

                self.render_state.buffer_time = self.offset;
                self.render_state.started = true;
            }

            if is_looping {
                if !self.render_state.entered_loop {
                    // playback began before or within loop, and playhead is now past loop start
                    if self.offset < actual_loop_end
                        && self.render_state.buffer_time >= actual_loop_start
                    {
                        self.render_state.entered_loop = true;
                    }

                    // playback began after loop, and playhead is now prior to the loop end
                    // @note - only possible when playback_rate < 0 (?)
                    if self.offset >= actual_loop_end
                        && self.render_state.buffer_time < actual_loop_end
                    {
                        self.render_state.entered_loop = true;
                    }
                }

                // check loop boundaries
                if self.render_state.entered_loop {
                    while self.render_state.buffer_time >= actual_loop_end {
                        self.render_state.buffer_time -= actual_loop_end - actual_loop_start;
                    }

                    while self.render_state.buffer_time < actual_loop_start {
                        self.render_state.buffer_time += actual_loop_end - actual_loop_start;
                    }
                }
            }

            if self.render_state.buffer_time >= 0.
                && self.render_state.buffer_time < buffer_duration
            {
                let position = self.render_state.buffer_time * sampling_ratio;
                let playhead = position * sample_rate;
                let playhead_floored = playhead.floor();
                let prev_frame_index = playhead_floored as usize; // can't be < 0.
                let k = (playhead - playhead_floored) as f32;

                *playback_info = Some(PlaybackInfo {
                    prev_frame_index,
                    k,
                });
            } else {
                *playback_info = None;
            }

            let time_incr = dt * computed_playback_rate;
            self.render_state.buffer_time += time_incr;
            self.render_state.buffer_time_elapsed += time_incr;
            current_time += dt;
        }

        // fill output according to computed positions
        buffer
            .channels()
            .iter()
            .zip(output.channels_mut().iter_mut())
            .for_each(|(buffer_channel, output_channel)| {
                let buffer_channel = buffer_channel.as_slice();

                playback_infos
                    .iter()
                    .zip(output_channel.iter_mut())
                    .for_each(|(playhead, o)| {
                        *o = match playhead {
                            Some(PlaybackInfo {
                                prev_frame_index,
                                k,
                            }) => {
                                // `prev_frame_index` cannot be out of bounds
                                let prev_sample = buffer_channel[*prev_frame_index];
                                let next_sample = match buffer_channel.get(prev_frame_index + 1) {
                                    Some(val) => *val,
                                    None => 0.,
                                };

                                (1. - k).mul_add(prev_sample, k * next_sample)
                            }
                            None => 0.,
                        };
                    });
            });

        true
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(control) = msg.downcast_ref::<ControlMessage>() {
            self.handle_control_message(control);
            return;
        };

        if let Some(buffer) = msg.downcast_mut::<AudioBuffer>() {
            if let Some(current_buffer) = &mut self.buffer {
                // Avoid deallocation in the render thread by swapping the buffers.
                std::mem::swap(current_buffer, buffer);
            } else {
                // Creating the tombstone buffer does not cause allocations.
                let tombstone_buffer = AudioBuffer {
                    channels: Default::default(),
                    sample_rate: Default::default(),
                };
                self.buffer = Some(std::mem::replace(buffer, tombstone_buffer));
            }
            return;
        };

        log::warn!("AudioBufferSourceRenderer: Dropping incoming message {msg:?}");
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::f32::consts::PI;

    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::RENDER_QUANTUM_SIZE;

    use super::*;

    #[test]
    fn test_playing_some_file() {
        let context = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE, 44_100.);

        let file = std::fs::File::open("samples/sample.wav").unwrap();
        let expected = context.decode_audio_data_sync(file).unwrap();

        // 44100 will go through fast track
        // 48000 will go through slow track
        [44100, 48000].iter().for_each(|sr| {
            let decoding_context = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE, *sr as f32);

            let mut filename = "samples/sample-".to_owned();
            filename.push_str(&sr.to_string());
            filename.push_str(".wav");

            let file = std::fs::File::open("samples/sample.wav").unwrap();
            let audio_buffer = decoding_context.decode_audio_data_sync(file).unwrap();

            assert_eq!(audio_buffer.sample_rate(), *sr as f32);

            let context = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE, 44_100.);

            let src = context.create_buffer_source();
            src.set_buffer(audio_buffer);
            src.connect(&context.destination());
            src.start_at(context.current_time());
            src.stop_at(context.current_time() + 128.);

            let res = context.start_rendering_sync();
            let diff_abs = if *sr == 44100 {
                0. // fast track
            } else {
                5e-3 // slow track w/ linear interpolation
            };

            // asserting length() is meaningless as this is controlled by the context
            assert_eq!(res.number_of_channels(), expected.number_of_channels());

            // check first 128 samples in left and right channels
            assert_float_eq!(
                res.channel_data(0).as_slice()[..],
                expected.get_channel_data(0)[0..128],
                abs_all <= diff_abs
            );

            assert_float_eq!(
                res.channel_data(1).as_slice()[..],
                expected.get_channel_data(1)[0..128],
                abs_all <= diff_abs
            );
        });
    }

    // slow track
    #[test]
    fn test_sub_quantum_start() {
        let sample_rate = 480000.;
        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

        let mut dirac = context.create_buffer(1, 1, sample_rate);
        dirac.copy_to_channel(&[1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(1. / sample_rate as f64);

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; RENDER_QUANTUM_SIZE];
        expected[1] = 1.;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_sub_sample_start() {
        // sub sample
        let sample_rate = 480000.;
        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

        let mut dirac = context.create_buffer(1, 1, sample_rate);
        dirac.copy_to_channel(&[1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(1.5 / sample_rate as f64);

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; RENDER_QUANTUM_SIZE];
        expected[2] = 0.5;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_sub_quantum_stop() {
        // fast track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            src.start_at(0. / sample_rate as f64);
            // stop at time of dirac, should not be played
            src.stop_at(4. / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);
            let expected = vec![0.; RENDER_QUANTUM_SIZE];

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        // slow track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            src.start_at(1. / sample_rate as f64);
            // stop at time of dirac, should not be played
            src.stop_at(5. / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);
            let expected = vec![0.; RENDER_QUANTUM_SIZE];

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_sub_sample_stop() {
        // fast track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            src.start_at(0. / sample_rate as f64);
            // stop at between two diracs, only first one should be played
            src.stop_at(4.5 / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 128];
            expected[4] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        // slow track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            src.start_at(1. / sample_rate as f64);
            // stop at between two diracs, only first one should be played
            src.stop_at(5.5 / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 128];
            expected[5] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_schedule_in_the_past() {
        let sample_rate = 48000.;
        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

        let mut dirac = context.create_buffer(1, 1, sample_rate);
        dirac.copy_to_channel(&[1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(-1.);

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; RENDER_QUANTUM_SIZE];
        expected[0] = 1.;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_audio_buffer_resampling() {
        [22500, 38000, 48000, 96000].iter().for_each(|sr| {
            let base_sr = 44100;
            let context = OfflineAudioContext::new(1, base_sr, base_sr as f32);

            // 1Hz sine at different sample rates
            let buf_sr = *sr;
            // safe cast for sample rate, see discussion at #113
            let sample_rate = buf_sr as f32;
            let mut buffer = context.create_buffer(1, buf_sr, sample_rate);
            let mut sine = vec![];

            for i in 0..buf_sr {
                let phase = i as f32 / buf_sr as f32 * 2. * PI;
                let sample = phase.sin();
                sine.push(sample);
            }

            buffer.copy_to_channel(&sine[..], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(buffer);
            src.start_at(0. / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            // 1Hz sine at audio context sample rate
            let mut expected = vec![];

            for i in 0..base_sr {
                let phase = i as f32 / base_sr as f32 * 2. * PI;
                let sample = phase.sin();
                expected.push(sample);
            }

            assert_float_eq!(channel[..], expected[..], abs_all <= 1e-6);
        });
    }

    #[test]
    fn test_playback_rate() {
        let sample_rate = 44100;
        let context = OfflineAudioContext::new(1, sample_rate, sample_rate as f32);

        let mut buffer = context.create_buffer(1, sample_rate, sample_rate as f32);
        let mut sine = vec![];

        // 1 Hz sine
        for i in 0..sample_rate {
            let phase = i as f32 / sample_rate as f32 * 2. * PI;
            let sample = phase.sin();
            sine.push(sample);
        }

        buffer.copy_to_channel(&sine[..], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(buffer);
        src.playback_rate.set_value(0.5);
        src.start();

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        // 0.5 Hz sine
        let mut expected = vec![];

        for i in 0..sample_rate {
            let phase = i as f32 / sample_rate as f32 * PI;
            let sample = phase.sin();
            expected.push(sample);
        }

        assert_float_eq!(channel[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn test_detune() {
        let sample_rate = 44100;
        let context = OfflineAudioContext::new(1, sample_rate, sample_rate as f32);

        let mut buffer = context.create_buffer(1, sample_rate, sample_rate as f32);
        let mut sine = vec![];

        // 1 Hz sine
        for i in 0..sample_rate {
            let phase = i as f32 / sample_rate as f32 * 2. * PI;
            let sample = phase.sin();
            sine.push(sample);
        }

        buffer.copy_to_channel(&sine[..], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(buffer);
        src.detune.set_value(-1200.);
        src.start();

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        // 0.5 Hz sine
        let mut expected = vec![];

        for i in 0..sample_rate {
            let phase = i as f32 / sample_rate as f32 * PI;
            let sample = phase.sin();
            expected.push(sample);
        }

        assert_float_eq!(channel[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn test_end_of_file() {
        // fast track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE * 2, sample_rate);

            let mut buffer = context.create_buffer(1, 129, sample_rate);
            let mut data = vec![0.; 129];
            data[0] = 1.;
            data[128] = 1.;
            buffer.copy_to_channel(&data, 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(buffer);
            src.start_at(0. / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[0] = 1.;
            expected[128] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        // slow track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE * 2, sample_rate);

            let mut buffer = context.create_buffer(1, 129, sample_rate);
            let mut data = vec![0.; 129];
            data[0] = 1.;
            data[128] = 1.;
            buffer.copy_to_channel(&data, 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(buffer);
            src.start_at(1. / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[1] = 1.;
            expected[129] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_with_duration() {
        // fast track
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            // duration is between two diracs, only first one should be played
            src.start_at_with_offset_and_duration(0., 0., 4.5 / sample_rate as f64);

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 128];
            expected[4] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
            dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_buffer(dirac);
            // duration is between two diracs, only first one should be played
            // as we force slow track with start == 1. / sample_rate as f64
            // the expected dirac will be at index 5 instead of 4
            src.start_at_with_offset_and_duration(
                1. / sample_rate as f64,
                0. / sample_rate as f64,
                4.5 / sample_rate as f64,
            );

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 128];
            expected[5] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_with_offset() {
        // offset always bypass slow track
        let sample_rate = 480000.;
        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, sample_rate);

        let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE, sample_rate);
        dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        // duration is between two diracs, only first one should be played
        // as we force slow track with start == 1. / sample_rate as f64
        // the expected dirac will be at index 5 instead of 4
        src.start_at_with_offset_and_duration(
            0. / sample_rate as f64,
            1. / sample_rate as f64,
            3.5 / sample_rate as f64,
        );

        let result = context.start_rendering_sync();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; 128];
        expected[3] = 1.;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    // just to make things more readable when populating expected values
    #[allow(clippy::erasing_op)]
    #[allow(clippy::identity_op)]
    fn test_fast_track_loop() {
        // buffer smaller than block
        {
            let sample_rate = 480000.;
            let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE * 2, sample_rate);

            let mut dirac = context.create_buffer(1, RENDER_QUANTUM_SIZE / 2, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_loop(true);
            src.set_buffer(dirac);
            src.start();

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[64 * 0] = 1.;
            expected[64 * 1] = 1.;
            expected[64 * 2] = 1.;
            expected[64 * 3] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        // buffer larger than block
        {
            let sample_rate = 480000.;
            let len = RENDER_QUANTUM_SIZE * 4;
            let context = OfflineAudioContext::new(1, len, sample_rate);

            let mut dirac = context.create_buffer(1, 129, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_loop(true);
            src.set_buffer(dirac);
            src.start();

            let result = context.start_rendering_sync();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; len];
            expected[129 * 0] = 1.;
            expected[129 * 1] = 1.;
            expected[129 * 2] = 1.;
            expected[129 * 3] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        // stereo
        {
            let sample_rate = 480000.;
            let len = RENDER_QUANTUM_SIZE * 4;
            let context = OfflineAudioContext::new(2, len, sample_rate);

            let mut dirac = context.create_buffer(2, 129, sample_rate);
            dirac.copy_to_channel(&[1.], 0);
            dirac.copy_to_channel(&[0., 1.], 1);

            let src = context.create_buffer_source();
            src.connect(&context.destination());
            src.set_loop(true);
            src.set_buffer(dirac);
            src.start();

            let result = context.start_rendering_sync();

            let mut expected_left = vec![0.; len];
            expected_left[129 * 0] = 1.;
            expected_left[129 * 1] = 1.;
            expected_left[129 * 2] = 1.;
            expected_left[129 * 3] = 1.;

            let mut expected_right = vec![0.; len];
            expected_right[129 * 0 + 1] = 1.;
            expected_right[129 * 1 + 1] = 1.;
            expected_right[129 * 2 + 1] = 1.;
            expected_right[129 * 3 + 1] = 1.;

            assert_float_eq!(
                result.get_channel_data(0)[..],
                expected_left[..],
                abs_all <= 0.
            );
            assert_float_eq!(
                result.get_channel_data(1)[..],
                expected_right[..],
                abs_all <= 0.
            );
        }
    }
}
