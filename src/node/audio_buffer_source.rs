use crossbeam_channel::{Receiver, Sender};
use once_cell::sync::OnceCell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::buffer::AudioBuffer;
use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::control::Controller;
use crate::param::{AudioParam, AudioParamOptions, AutomationRate};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

/// Options for constructing an [`AudioBufferSourceNode`]
pub struct AudioBufferSourceOptions {
    pub buffer: Option<AudioBuffer>,
    pub detune: f32,
    pub loop_: bool,
    pub loop_start: f64,
    pub loop_end: f64,
    pub playback_rate: f32,
    pub channel_config: ChannelConfigOptions,
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
            channel_config: Default::default(),
        }
    }
}

struct AudioBufferMessage(AudioBuffer);

/// `AudioBufferSourceNode` represents an audio source that consists of an
/// in-memory audio source (i.e. an audio file completely loaded in memory),
/// stored in an [`AudioBuffer`].
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#AudioBufferSourceNode>
/// - see also: [`AsBaseAudioContext::create_buffer_source`](crate::context::AsBaseAudioContext::create_buffer_source)
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
/// use web_audio_api::context::{AsBaseAudioContext, AudioContext};
/// use web_audio_api::node::AudioNode;
///
/// // create an `AudioContext`
/// let context = AudioContext::new(None);
/// // load and decode a soundfile
/// let file = File::open("samples/sample.wav").unwrap();
/// let audio_buffer = context.decode_audio_data(file).unwrap();
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
    controller: Controller,
    channel_config: ChannelConfig,
    sender: Sender<AudioBufferMessage>,
    detune: AudioParam,        // has constraints, no a-rate
    playback_rate: AudioParam, // has constraints, no a-rate
    buffer: OnceCell<AudioBuffer>,
    source_started: AtomicBool,
}

impl AudioNode for AudioBufferSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl AudioBufferSourceNode {
    /// Create a new [`AudioBufferSourceNode`] instance
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AudioBufferSourceOptions) -> Self {
        context.base().register(move |registration| {
            let AudioBufferSourceOptions {
                buffer,
                detune,
                loop_,
                loop_start,
                loop_end,
                playback_rate,
                channel_config,
            } = options;

            // @todo - these parameters can't be changed to a-rate
            // @see - <https://webaudio.github.io/web-audio-api/#audioparam-automation-rate-constraints>
            // @see - https://github.com/orottier/web-audio-api-rs/issues/29
            let detune_param_options = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 0.,
                automation_rate: AutomationRate::K,
            };
            let (d_param, d_proc) = context
                .base()
                .create_audio_param(detune_param_options, registration.id());

            d_param.set_value(detune);

            let playback_rate_param_options = AudioParamOptions {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: AutomationRate::K,
            };
            let (pr_param, pr_proc) = context
                .base()
                .create_audio_param(playback_rate_param_options, registration.id());

            pr_param.set_value(playback_rate);

            // Channel to send buffer channels references to the renderer.
            // A capacity of 1 suffices since it is not allowed to set the value multiple times
            let (sender, receiver) = crossbeam_channel::bounded(1);

            let controller = Controller::new();

            let renderer = AudioBufferSourceRenderer {
                controller: controller.clone(),
                receiver,
                buffer: None,
                detune: d_proc,
                playback_rate: pr_proc,
                render_state: AudioBufferRendererState::default(),
                // `internal_buffer` is used to compute the samples per channel at each frame. Note
                // that the `vec` will always be resized to actual buffer number_of_channels when
                // received on the render thread.
                internal_buffer: Vec::<f32>::with_capacity(crate::MAX_CHANNELS),
            };

            let node = Self {
                registration,
                controller,
                channel_config: channel_config.into(),
                sender,
                detune: d_param,
                playback_rate: pr_param,
                buffer: OnceCell::new(),
                source_started: AtomicBool::new(false),
            };

            node.controller.set_loop(loop_);
            node.controller.set_loop_start(loop_start);
            node.controller.set_loop_end(loop_end);

            if let Some(buf) = buffer {
                node.set_buffer(buf);
            }

            (node, Box::new(renderer))
        })
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

        self.sender
            .send(AudioBufferMessage(clone))
            .expect("Sending AudioBufferMessage failed");
    }

    /// Start the playback on next block
    pub fn start(&self) {
        let start = self.registration.context().current_time();
        self.start_at_with_offset_and_duration(start, 0., f64::MAX);
    }

    /// Start the playback at the given time
    pub fn start_at(&self, start: f64) {
        self.start_at_with_offset_and_duration(start, 0., f64::MAX);
    }

    /// Start the playback at the given time and with a given offset
    pub fn start_at_with_offset(&self, start: f64, offset: f64) {
        self.start_at_with_offset_and_duration(start, offset, f64::MAX);
    }

    /// Start the playback at the given time, with a given offset, for a given duration
    pub fn start_at_with_offset_and_duration(&self, start: f64, offset: f64, duration: f64) {
        if self.source_started.swap(true, Ordering::SeqCst) {
            panic!("InvalidStateError: Cannot call `start` twice");
        }

        self.controller.set_offset(offset);
        self.controller.set_duration(duration);
        self.controller.scheduler().start_at(start);
    }

    /// Stop the playback on next block
    pub fn stop(&self) {
        let stop = self.registration.context().current_time();
        self.stop_at(stop);
    }

    /// Stop the playback at given time
    pub fn stop_at(&self, stop: f64) {
        if !self.source_started.load(Ordering::SeqCst) {
            panic!("InvalidStateError cannot stop before start");
        }

        self.controller.scheduler().stop_at(stop)
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
    pub fn loop_(&self) -> bool {
        self.controller.loop_()
    }

    pub fn set_loop(&self, value: bool) {
        self.controller.set_loop(value);
    }

    /// Defines the loop start point, in the time reference of the [`AudioBuffer`]
    pub fn loop_start(&self) -> f64 {
        self.controller.loop_start()
    }

    pub fn set_loop_start(&self, value: f64) {
        self.controller.set_loop_start(value);
    }

    /// Defines the loop end point, in the time reference of the [`AudioBuffer`]
    pub fn loop_end(&self) -> f64 {
        self.controller.loop_end()
    }

    pub fn set_loop_end(&self, value: f64) {
        self.controller.set_loop_end(value);
    }
}

struct AudioBufferRendererState {
    buffer_time: f64,
    started: bool,
    entered_loop: bool,
    buffer_time_elapsed: f64,
}

impl Default for AudioBufferRendererState {
    fn default() -> Self {
        Self {
            buffer_time: 0.,
            started: false,
            entered_loop: false,
            buffer_time_elapsed: 0.,
        }
    }
}

struct AudioBufferSourceRenderer {
    controller: Controller,
    receiver: Receiver<AudioBufferMessage>,
    buffer: Option<AudioBuffer>,
    detune: AudioParamId,
    playback_rate: AudioParamId,
    render_state: AudioBufferRendererState,
    internal_buffer: Vec<f32>,
}

impl AudioProcessor for AudioBufferSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum], // no input...
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        let dt = 1. / sample_rate.0 as f64;
        let num_frames = RENDER_QUANTUM_SIZE;
        let next_block_time = timestamp + dt * num_frames as f64;

        if let Ok(msg) = self.receiver.try_recv() {
            let buffer = msg.0;

            self.internal_buffer.resize(buffer.number_of_channels(), 0.);
            self.buffer = Some(buffer);
        }

        // compute compound parameter at k-rate
        let detune_values = params.get(&self.detune);
        let playback_rate_values = params.get(&self.playback_rate);
        let detune = detune_values[0];
        let playback_rate = playback_rate_values[0];
        let computed_playback_rate = (playback_rate * (detune / 1200.).exp2()) as f64;

        // grab all timing informations
        let start_time = self.controller.scheduler().get_start_at();
        let stop_time = self.controller.scheduler().get_stop_at();
        let mut offset = self.controller.offset();
        let duration = self.controller.duration();
        let loop_ = self.controller.loop_();
        let loop_start = self.controller.loop_start();
        let loop_end = self.controller.loop_end();

        // these will only be used if `loop_` is true, so no need for `Option`
        let mut actual_loop_start = 0.;
        let mut actual_loop_end = 0.;

        // return early if start_time is beyond this block
        if start_time >= next_block_time {
            output.make_silent();
            return true;
        }

        // If the buffer has not been set wait for it.
        // @see - `set_buffer` tries to acquire the buffer if the source already started
        let buffer = match &self.buffer {
            None => {
                output.make_silent();
                return true;
            }
            Some(b) => b,
        };

        // from this point we know that we have a buffer
        let buffer_duration = buffer.duration();

        // In addition, if the buffer has more than one channel, then the
        // AudioBufferSourceNode output must change to a single channel of silence
        // at the beginning of a render quantum after the time at which any one of
        // the following conditions holds:

        // 1. the stop time has been reached.
        // 2. the duration has been reached.
        if timestamp >= stop_time || self.render_state.buffer_time_elapsed >= duration {
            output.make_silent(); // also converts to mono
            return false;
        }

        // 3. the end of the buffer has been reached.
        if !loop_ {
            if computed_playback_rate > 0. && self.render_state.buffer_time >= buffer_duration {
                output.make_silent(); // also converts to mono
                return false;
            }

            if computed_playback_rate < 0. && self.render_state.buffer_time < 0. {
                output.make_silent(); // also converts to mono
                return false;
            }
        }

        output.set_number_of_channels(buffer.number_of_channels());

        // go through the algorithm described in the spec
        // @see <https://webaudio.github.io/web-audio-api/#playback-AudioBufferSourceNode>
        let mut current_time = timestamp;

        if loop_ {
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

        for index in 0..num_frames {
            if current_time < start_time
                || current_time >= stop_time
                || self.render_state.buffer_time_elapsed >= duration
            {
                self.internal_buffer.fill(0.);
                output.set_channels_values_at(index, &self.internal_buffer);
                current_time += dt;

                continue; // nothing more to do for this sample
            }

            // we have now reached start time
            if !self.render_state.started {
                offset += current_time - start_time;

                if loop_ && computed_playback_rate >= 0. && offset >= actual_loop_end {
                    offset = actual_loop_end;
                }

                if loop_ && computed_playback_rate < 0. && offset < actual_loop_start {
                    offset = actual_loop_start;
                }

                self.render_state.buffer_time = offset;
                self.render_state.started = true;
            }

            if loop_ {
                if !self.render_state.entered_loop {
                    // playback began before or within loop, and playhead is now past loop start
                    if offset < actual_loop_end
                        && self.render_state.buffer_time >= actual_loop_start
                    {
                        self.render_state.entered_loop = true;
                    }

                    // playback began after loop, and playhead is now prior to the loop end
                    // @note - only possible when playback_rate < 0 (?)
                    if offset >= actual_loop_end && self.render_state.buffer_time < actual_loop_end
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
                self.compute_playback_at_position(
                    self.render_state.buffer_time,
                    sample_rate.0 as f64,
                );
            } else {
                self.internal_buffer.fill(0.);
            }

            output.set_channels_values_at(index, &self.internal_buffer);

            self.render_state.buffer_time += dt * computed_playback_rate;
            self.render_state.buffer_time_elapsed += dt * computed_playback_rate;
            current_time += dt;
        }

        true
    }
}

impl AudioBufferSourceRenderer {
    // Pick the closest index to the given position
    //
    // @note - this is not used but we keep that around as it could be usefull
    // for testing and/or to for perf improvement if the playback is aligned on
    // `sample_rate` and `playback_rate.abs() = 1` and as not been modified
    #[allow(dead_code)]
    fn compute_playback_at_position_direct(&mut self, position: f64, sample_rate: f64) {
        let sample_index = (position * sample_rate).round() as usize;

        let iterator = self.buffer.as_ref().unwrap().channels().iter().enumerate();
        for (channel_index, channel) in iterator {
            self.internal_buffer[channel_index] = channel.as_slice()[sample_index];
        }
    }

    // Linear interpolate betwen tow frames according to a given position
    fn compute_playback_at_position(&mut self, position: f64, sample_rate: f64) {
        let playhead = position * sample_rate;
        let playhead_floored = playhead.floor();
        let prev_index = playhead_floored as usize; // can't be < 0.
        let next_index = playhead.ceil() as usize; // can be >= length

        let k = (playhead - playhead_floored) as f32;
        let k_inv = 1. - k;

        let iterator = self.buffer.as_ref().unwrap().channels().iter().enumerate();
        for (channel_index, channel) in iterator {
            // @todo - [spec] If |position| is greater than or equal to |loopEnd|
            // and there is no subsequent sample frame in buffer, then interpolation
            // should be based on the sequence of subsequent frames beginning at |loopStart|.
            //
            // for now we just interpolate with zero
            let prev_sample = channel.as_slice()[prev_index];
            let next_sample = if next_index >= channel.len() {
                0.
            } else {
                channel.as_slice()[next_index]
            };
            let value = k_inv * prev_sample + k * next_sample;
            self.internal_buffer[channel_index] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{AsBaseAudioContext, OfflineAudioContext};
    use crate::node::AudioNode;
    use crate::{SampleRate, RENDER_QUANTUM_SIZE};

    use float_eq::assert_float_eq;

    #[test]
    fn test_playing_some_file() {
        let mut context = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE, SampleRate(44_100));

        let file = std::fs::File::open("samples/sample.wav").unwrap();
        let audio_buffer = context.decode_audio_data(file).unwrap();

        let src = context.create_buffer_source();
        src.set_buffer(audio_buffer.clone());
        src.connect(&context.destination());
        src.start_at(context.current_time());
        src.stop_at(context.current_time() + 128.);

        let res = context.start_rendering();

        // check first 128 samples in left and right channels
        assert_float_eq!(
            res.channel_data(0).as_slice()[..],
            audio_buffer.get_channel_data(0)[0..128],
            abs_all <= 0.
        );

        assert_float_eq!(
            res.channel_data(1).as_slice()[..],
            audio_buffer.get_channel_data(1)[0..128],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_sub_quantum_start() {
        let sample_rate = 128;
        let sr = SampleRate(sample_rate as u32);
        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, sr);

        let mut dirac = context.create_buffer(1, 1, sr);
        dirac.copy_to_channel(&[1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(1. / sample_rate as f64);

        let result = context.start_rendering();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; 1 * sample_rate];
        expected[1] = 1.;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_sub_sample_start() {
        // sub sample
        let sample_rate = 128;
        let sr = SampleRate(sample_rate as u32);
        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, sr);

        let mut dirac = context.create_buffer(1, sample_rate, sr);
        dirac.copy_to_channel(&[1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(1.5 / sample_rate as f64);

        let result = context.start_rendering();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; 1 * sample_rate];
        expected[2] = 0.5;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_sub_quantum_stop() {
        let sample_rate = 128;
        let sr = SampleRate(sample_rate as u32);
        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, sr);

        let mut dirac = context.create_buffer(1, sample_rate, sr);
        dirac.copy_to_channel(&[0., 0., 0., 0., 1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(0. / sample_rate as f64);
        // stop at time of dirac, shoud not be played
        src.stop_at(4. / sample_rate as f64);

        let result = context.start_rendering();
        let channel = result.get_channel_data(0);
        let expected = vec![0.; 1 * sample_rate];

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn test_sub_sample_stop() {
        let sample_rate = 128;
        let sr = SampleRate(sample_rate as u32);
        let mut context = OfflineAudioContext::new(1, 1 * sample_rate, sr);

        let mut dirac = context.create_buffer(1, sample_rate, sr);
        dirac.copy_to_channel(&[0., 0., 0., 0., 1., 1.], 0);

        let src = context.create_buffer_source();
        src.connect(&context.destination());
        src.set_buffer(dirac);
        src.start_at(0. / sample_rate as f64);
        // stop at between two diracs, only first one should be played
        src.stop_at(4.5 / sample_rate as f64);

        let result = context.start_rendering();
        let channel = result.get_channel_data(0);

        let mut expected = vec![0.; 1 * sample_rate];
        expected[4] = 1.;

        assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
    }
}
