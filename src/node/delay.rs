use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};

use std::cell::{Cell, RefCell, RefMut};
use std::rc::Rc;

/// Options for constructing a DelayNode
pub struct DelayOptions {
    pub max_delay_time: f64,
    pub delay_time: f64,
    pub channel_config: ChannelConfigOptions,
}

impl Default for DelayOptions {
    fn default() -> Self {
        Self {
            max_delay_time: 1.,
            delay_time: 0.,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Node that delays the incoming audio signal by a certain amount
///
/// The current implementation does not allow for zero delay. The minimum delay is one render
/// quantum (e.g. ~2.9ms at 44.1kHz).
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/DelayNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#DelayNode>
/// - see also: [`BaseAudioContext::create_delay`](crate::context::BaseAudioContext::create_delay)
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::AudioNode;
///
/// // create an `AudioContext` and load a sound file
/// let context = AudioContext::new(None);
/// let file = File::open("samples/sample.wav").unwrap();
/// let audio_buffer = context.decode_audio_data(file).unwrap();
///
/// // create a delay of 0.5s
/// let delay = context.create_delay(1.);
/// delay.delay_time().set_value(0.5);
/// delay.connect(&context.destination());
///
/// let src = context.create_buffer_source();
/// src.set_buffer(audio_buffer);
/// // connect to both delay and destination
/// src.connect(&delay);
/// src.connect(&context.destination());
/// src.start();
/// ```
///
/// # Examples
///
/// - `cargo run --release --example simple_delay`
/// - `cargo run --release --example feedback_delay`
///
/*
 * For simplicity in the audio graph rendering, we have made the conscious decision to deviate from
 * the spec and split the delay node up front in a reader and writer node (instead of during the
 * render loop - see https://webaudio.github.io/web-audio-api/#rendering-loop )
 *
 * This has a drawback: a delay of 0 is no longer possible. This would only be possible if the
 * writer end is rendered before the reader end in the graph, but we cannot enforce that here.
 * (The only way would be to connect the writer to the reader, but that would kill the
 * cycle-breaker feature of the delay node.)
 *
 * @note: one possible strategy here would be to create a connection between Reader
 * and Writer in `DelayNode::new` just to guarantee the order of the processing if
 * the delay is not in a loop. In the graph process if the node is found in a cycle,
 * this connection could be removed and the Reader marked as "in_cycle" so that
 * it would clamp the min delay to quantum duration.
 * > no need to make this cancellable, once in a cycle the node behaves like that
 * even if the cycle is broken later (user have to know what they are doing)
 */
pub struct DelayNode {
    reader_registration: AudioContextRegistration,
    writer_registration: AudioContextRegistration,
    delay_time: AudioParam,
    channel_config: ChannelConfig,
}

impl AudioNode for DelayNode {
    /*
     * We set the writer node as 'main' registration.  This means other nodes can say
     * `node.connect(delaynode)` and they will connect to the writer.
     * Below, we override the (dis)connect methods as they should operate on the reader node.
     */
    fn registration(&self) -> &AudioContextRegistration {
        &self.writer_registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }

    /// Connect a specific output of this AudioNode to a specific input of another node.
    fn connect_at<'a>(
        &self,
        dest: &'a dyn AudioNode,
        output: u32,
        input: u32,
    ) -> &'a dyn AudioNode {
        if self.context() != dest.context() {
            panic!("InvalidAccessError: Attempting to connect nodes from different contexts");
        }
        if self.number_of_outputs() <= output {
            panic!("IndexSizeError: output port {} is out of bounds", output);
        }
        if dest.number_of_inputs() <= input {
            panic!("IndexSizeError: input port {} is out of bounds", input);
        }

        self.context()
            .connect(self.reader_registration.id(), dest.id(), output, input);

        dest
    }

    /// Disconnects all outputs of the AudioNode that go to a specific destination AudioNode.
    fn disconnect<'a>(&self, dest: &'a dyn AudioNode) -> &'a dyn AudioNode {
        if self.context() != dest.context() {
            panic!("attempting to disconnect nodes from different contexts");
        }

        self.context()
            .disconnect(self.reader_registration.id(), dest.id());

        dest
    }

    /// Disconnects all outgoing connections from the AudioNode.
    fn disconnect_all(&self) {
        self.context().disconnect_all(self.reader_registration.id());
    }
}

impl DelayNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: DelayOptions) -> Self {
        let sample_rate = context.sample_rate_raw().0 as f64;

        // Specifies the maximum delay time in seconds allowed for the delay line.
        // If specified, this value MUST be greater than zero and less than three
        // minutes or a NotSupportedError exception MUST be thrown. If not specified,
        // then 1 will be used.
        if options.max_delay_time <= 0. || options.max_delay_time >= 180. {
            panic!("NotSupportedError: MUST be greater than zero and less than three minutes");
        }

        // we internally clamp max delay to quantum duration because the current
        // implementation doesn't allow sub-quantum delays. Later, this will
        // ensure that even if the declared max_delay_time and max_delay are smaller
        // than quantum duration, the node, if found in a loop, will gracefully
        // fallback to the clamped behavior. (e.g. we ensure that ring buffer size
        // is always >= 2)
        let quantum_duration = 1. / sample_rate * RENDER_QUANTUM_SIZE as f64;
        let max_delay_time = options.max_delay_time.max(quantum_duration);

        // allocate large enough buffer to store all delayed samples
        //
        // we add 1 here so that in edge cases where num_samples is a multiple of
        // RENDER_QUANTUM_SIZE and delay_time == max_delay_time we are sure to
        // enough room for history. (see. test_max_delay_multiple_of_quantum_size)
        let num_samples = max_delay_time * sample_rate + 1.;
        let num_quanta =
            (num_samples.ceil() as usize + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE;
        let ring_buffer = Vec::with_capacity(num_quanta);

        let shared_ring_buffer = Rc::new(RefCell::new(ring_buffer));
        let shared_ring_buffer_clone = shared_ring_buffer.clone();

        // shared value set by the writer when it is dropped
        let last_written_index = Rc::new(Cell::<Option<usize>>::new(None));
        let last_written_index_clone = last_written_index.clone();

        context.base().register(move |writer_registration| {
            let node = context.base().register(move |reader_registration| {
                let param_opts = AudioParamOptions {
                    min_value: 0.,
                    max_value: max_delay_time as f32,
                    default_value: 0.,
                    automation_rate: crate::param::AutomationRate::A,
                };
                let (param, proc) = context
                    .base()
                    .create_audio_param(param_opts, reader_registration.id());

                param.set_value_at_time(options.delay_time as f32, 0.);

                let reader_render = DelayReader {
                    delay_time: proc,
                    ring_buffer: shared_ring_buffer_clone,
                    index: 0,
                    last_written_index: last_written_index_clone,
                    last_written_index_checked: None,
                    // `internal_buffer` is used to compute the samples per channel at each frame.
                    // Note that the `vec` will always be resized to actual buffer
                    // number_of_channels when received on the render thread.
                    internal_buffer: Vec::<f32>::with_capacity(crate::MAX_CHANNELS),
                };

                let node = DelayNode {
                    reader_registration,
                    writer_registration,
                    channel_config: options.channel_config.into(),
                    delay_time: param,
                };

                (node, Box::new(reader_render))
            });

            let writer_render = DelayWriter {
                ring_buffer: shared_ring_buffer,
                index: 0,
                last_written_index,
            };

            (node, Box::new(writer_render))
        })
    }

    /// A-rate [`AudioParam`] representing the amount of delay (in seconds) to apply.
    pub fn delay_time(&self) -> &AudioParam {
        &self.delay_time
    }
}

struct DelayWriter {
    ring_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
    last_written_index: Rc<Cell<Option<usize>>>,
}

struct DelayReader {
    delay_time: AudioParamId,
    ring_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
    last_written_index: Rc<Cell<Option<usize>>>,
    // local copy of shared `last_written_index` so as to avoid render ordering issues
    last_written_index_checked: Option<usize>,
    // internal buffer used to compute output per channel at each frame
    internal_buffer: Vec<f32>,
}

// SAFETY:
// AudioRenderQuantums are not Send but we promise the `ring_buffer` Vec is
// empty before we ship it to the render thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for DelayWriter {}
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for DelayReader {}

trait RingBufferChecker {
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>>;

    // This step guarantees the ring buffer is filled with silence buffers,
    // This allow to simplify the code in both Writer and Reader as we know
    // `len() == capacity()` and all inner buffers are initialized with zeros.
    #[inline(always)]
    fn check_ring_buffer_size(&self, render_quantum: &AudioRenderQuantum) {
        let mut ring_buffer = self.ring_buffer_mut();

        if ring_buffer.len() < ring_buffer.capacity() {
            let len = ring_buffer.capacity();
            let mut silence = render_quantum.clone();
            silence.make_silent();

            ring_buffer.resize(len, silence);
        }
    }
}

impl Drop for DelayWriter {
    fn drop(&mut self) {
        let last_written_index = if self.index == 0 {
            self.ring_buffer.borrow().capacity() - 1
        } else {
            self.index - 1
        };

        self.last_written_index.set(Some(last_written_index));
    }
}

impl RingBufferChecker for DelayWriter {
    #[inline(always)]
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>> {
        self.ring_buffer.borrow_mut()
    }
}

impl AudioProcessor for DelayWriter {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let input = inputs[0].clone();
        let output = &mut outputs[0];

        // We must perform this check on both Writer and Reader as the order of
        // the rendering between them is not guaranteed.
        self.check_ring_buffer_size(&input);
        // `check_ring_buffer_up_down_mix` can only be done on the Writer
        // side as Reader do not access the "real" input
        self.check_ring_buffer_up_down_mix(&input);

        // populate ring buffer
        let mut buffer = self.ring_buffer.borrow_mut();
        buffer[self.index] = input;

        // increment cursor
        self.index = (self.index + 1) % buffer.capacity();
        // The writer end does not produce output,
        // clear the buffer so that it can be re-used
        output.make_silent();

        // let the node be decommisionned if it has no input left
        false
    }
}

impl DelayWriter {
    #[inline(always)]
    fn check_ring_buffer_up_down_mix(&self, input: &AudioRenderQuantum) {
        // [spec]
        // When the number of channels in a DelayNode's input changes (thus changing
        // the output channel count also), there may be delayed audio samples which
        // have not yet been output by the node and are part of its internal state.
        // If these samples were received earlier with a different channel count,
        // they MUST be upmixed or downmixed before being combined with newly received
        // input so that all internal delay-line mixing takes place using the single
        // prevailing channel layout.
        let mut ring_buffer = self.ring_buffer_mut();
        let buffer_number_of_channels = ring_buffer[0].number_of_channels();
        let input_number_of_channels = input.number_of_channels();

        if buffer_number_of_channels != input_number_of_channels {
            for render_quantum in ring_buffer.iter_mut() {
                render_quantum.mix(input_number_of_channels, ChannelInterpretation::Speakers);
            }
        }
    }
}

impl RingBufferChecker for DelayReader {
    #[inline(always)]
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>> {
        self.ring_buffer.borrow_mut()
    }
}

impl AudioProcessor for DelayReader {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum], // cannot be used
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let output = &mut outputs[0];
        // We must perform the checks (buffer size and up/down mix) on both Writer
        // and Reader as the order of processing between them is not guaranteed.
        self.check_ring_buffer_size(output);

        let ring_buffer = self.ring_buffer.borrow();

        // we need to rely on ring buffer to know the actual number of output channels
        let number_of_channels = ring_buffer[0].number_of_channels();
        // resize internal buffer if needed
        if self.internal_buffer.len() != number_of_channels {
            self.internal_buffer.resize(number_of_channels, 0.);
        }

        output.set_number_of_channels(number_of_channels);

        // shadow and cast sample_rate, we don't need the wrapper type here
        let sample_rate = sample_rate.0 as f64;
        let dt = 1. / sample_rate;
        let quantum_duration = RENDER_QUANTUM_SIZE as f64 * dt;

        let delay_param = params.get(&self.delay_time);

        for (index, delay) in delay_param.iter().enumerate() {
            // param is already clamped to max_delay_time internally, so it is
            // safe to only check lower boundary
            let clamped_delay = (*delay as f64).max(quantum_duration);
            let num_samples = clamped_delay * sample_rate;
            // negative position of the playhead relative to this block start
            let position = index as f64 - num_samples;

            // find address of the frame in the ring buffer just before `position`
            let prev_position = position.floor();
            let (prev_block_index, prev_frame_index) =
                self.find_frame_adress_at_position(prev_position);

            // find address of the frame in the ring buffer just after `position`
            let next_position = position.ceil();
            let (next_block_index, next_frame_index) =
                self.find_frame_adress_at_position(next_position);

            // as position is negative k will be what we expect
            let k = (position - position.floor()) as f32;
            let k_inv = 1. - k;

            // compute linear interpolation between prev and next for each channel
            for channel_number in 0..number_of_channels {
                let prev_sample =
                    ring_buffer[prev_block_index].channel_data(channel_number)[prev_frame_index];
                let next_sample =
                    ring_buffer[next_block_index].channel_data(channel_number)[next_frame_index];

                let value = k_inv * prev_sample + k * next_sample;

                self.internal_buffer[channel_number] = value;
            }

            // populate output at index w/ internal_buffer
            output.set_channels_values_at(index, &self.internal_buffer);
        }

        if matches!(self.last_written_index_checked, Some(index) if index == self.index) {
            return false;
        }

        // check if the writer has been decommisionned
        // we need this local copy because if the writer has been processed
        // before the reader, the direct check against `self.last_written_index`
        // would be true earlier than we want
        let last_written_index = self.last_written_index.get();
        if last_written_index.is_some() && self.last_written_index_checked.is_none() {
            self.last_written_index_checked = last_written_index;
        }
        // increment ring buffer cursor
        self.index = (self.index + 1) % ring_buffer.capacity();

        true
    }
}

impl DelayReader {
    #[inline(always)]
    // note that `position` is negative as we look into the past
    fn find_frame_adress_at_position(&self, position: f64) -> (usize, usize) {
        let num_frames = RENDER_QUANTUM_SIZE as i32;
        let buffer_len = self.ring_buffer.borrow().len() as i32;
        let current_index = self.index as i32;

        // offset of the block in which the target sample is recorded
        // we need to be `float` here so that `floor()` behaves as expected
        let block_offset = (position / num_frames as f64).floor();
        // offset of the block in which the target sample is recorded
        let mut block_index = current_index + block_offset as i32;
        // unroll ring buffer is needed
        if block_index < 0 {
            block_index += buffer_len;
        }

        // find frame index in the target block
        let mut frame_offset = position as i32 % num_frames;
        // handle special 0 case
        if frame_offset == 0 {
            frame_offset = -num_frames;
        }
        let frame_index = num_frames + frame_offset;

        (block_index as usize, frame_index as usize)
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::OfflineAudioContext;
    use crate::SampleRate;

    use super::*;

    #[test]
    fn test_sample_accurate() {
        for delay_in_samples in [128., 131., 197.].iter() {
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 256, sample_rate);

            let delay = context.create_delay(2.);
            delay.delay_time.set_value(delay_in_samples / 128.);
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[*delay_in_samples as usize] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_sub_sample_accurate() {
        {
            let delay_in_samples = 128.5;
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 256, sample_rate);

            let delay = context.create_delay(2.);
            delay.delay_time.set_value(delay_in_samples / 128.);
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[128] = 0.5;
            expected[129] = 0.5;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        {
            let delay_in_samples = 128.8;
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 256, sample_rate);

            let delay = context.create_delay(2.);
            delay.delay_time.set_value(delay_in_samples / 128.);
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[128] = 0.2;
            expected[129] = 0.8;

            assert_float_eq!(channel[..], expected[..], abs_all <= 1e-5);
        }
    }

    #[test]
    fn test_multichannel() {
        let delay_in_samples = 128.;
        let sample_rate = SampleRate(128);
        let mut context = OfflineAudioContext::new(2, 2 * 128, sample_rate);

        let delay = context.create_delay(2.);
        delay.delay_time.set_value(delay_in_samples / 128.);
        delay.connect(&context.destination());

        let mut two_chan_dirac = context.create_buffer(2, 256, sample_rate);
        // different channels
        two_chan_dirac.copy_to_channel(&[1.], 0);
        two_chan_dirac.copy_to_channel(&[0., 1.], 1);

        let src = context.create_buffer_source();
        src.connect(&delay);
        src.set_buffer(two_chan_dirac);
        src.start_at(0.);

        let result = context.start_rendering();

        let channel_left = result.get_channel_data(0);
        let mut expected_left = vec![0.; 256];
        expected_left[128] = 1.;
        assert_float_eq!(channel_left[..], expected_left[..], abs_all <= 0.);

        let channel_right = result.get_channel_data(1);
        let mut expected_right = vec![0.; 256];
        expected_right[128 + 1] = 1.;
        assert_float_eq!(channel_right[..], expected_right[..], abs_all <= 0.);
    }

    #[test]
    fn test_input_number_of_channels_change() {
        let delay_in_samples = 128.;
        let sample_rate = SampleRate(128);
        let mut context = OfflineAudioContext::new(2, 3 * 128, sample_rate);

        let delay = context.create_delay(2.);
        delay.delay_time.set_value(delay_in_samples / 128.);
        delay.connect(&context.destination());

        let mut one_chan_dirac = context.create_buffer(1, 128, sample_rate);
        one_chan_dirac.copy_to_channel(&[1.], 0);

        let src1 = context.create_buffer_source();
        src1.connect(&delay);
        src1.set_buffer(one_chan_dirac);
        src1.start_at(0.);

        let mut two_chan_dirac = context.create_buffer(2, 256, sample_rate);
        // the two channels are different
        two_chan_dirac.copy_to_channel(&[1.], 0);
        two_chan_dirac.copy_to_channel(&[0., 1.], 1);
        // start second buffer at next block
        let src2 = context.create_buffer_source();
        src2.connect(&delay);
        src2.set_buffer(two_chan_dirac);
        src2.start_at(1.);

        let result = context.start_rendering();

        let channel_left = result.get_channel_data(0);
        let mut expected_left = vec![0.; 3 * 128];
        expected_left[128] = 1.;
        expected_left[256] = 1.;
        assert_float_eq!(channel_left[..], expected_left[..], abs_all <= 0.);

        let channel_right = result.get_channel_data(1);
        let mut expected_right = vec![0.; 3 * 128];
        expected_right[128] = 1.;
        expected_right[256 + 1] = 1.;
        assert_float_eq!(channel_right[..], expected_right[..], abs_all <= 0.);
    }

    #[test]
    fn test_node_stays_alive_long_enough() {
        // make sure there are no hidden order problem
        for _ in 0..10 {
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 5 * 128, sample_rate);

            // Set up a source that starts only after 5 render quanta.
            // The delay writer and reader should stay alive in this period of silence.
            // We set up the nodes in a separate block {} so they are dropped in the control thread,
            // otherwise the lifecycle rules do not kick in
            {
                let delay = context.create_delay(1.);
                delay.delay_time.set_value(1.);
                delay.connect(&context.destination());

                let mut dirac = context.create_buffer(1, 1, sample_rate);
                dirac.copy_to_channel(&[1.], 0);

                let src = context.create_buffer_source();
                src.connect(&delay);
                src.set_buffer(dirac);
                // 3rd block - play buffer
                // 4th block - play silence and dropped in render thread
                src.start_at(3.);
            } // src and delay nodes are dropped

            let result = context.start_rendering();
            let mut expected = vec![0.; 5 * 128];
            // source starts after 2 * 128 samples, then is delayed another 128
            expected[4 * 128] = 1.;

            assert_float_eq!(result.get_channel_data(0), &expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_max_delay_multiple_of_quantum_size() {
        // regression test that delay node has always enough internal buffer size when
        // max_delay is a multiple of quantum size and delay == max_delay. We need
        // to test multiple times since (currently) the topological sort of the
        // graph depends on randomized hash values. This bug only occurs when the
        // Writer is called earlier than the Reader. 10 times should do:
        for _ in 0..10 {
            // set delay and max delay time exactly 1 render quantum
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 256, sample_rate);

            let delay = context.create_delay(1.);
            delay.delay_time.set_value(1.);
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[128] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }

        for _ in 0..10 {
            // set delay and max delay time exactly 2 render quantum
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 3 * 128, sample_rate);

            let delay = context.create_delay(2.);
            delay.delay_time.set_value(2.);
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 3 * 128];
            expected[256] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn test_max_delay_smaller_than_quantum_size() {
        // regression test that even if the declared max_delay_time is smaller than
        // a quantum duration (which is not allowed for now), the node internally
        // clamp it to quantum duration so that everything works even if order
        // of processing is not garanteed (which is the default behavior for now).
        // When allowing sub quantum delay, this will also guarantees that the node
        // gracefully fallback to min
        for _ in 0..10 {
            let sample_rate = SampleRate(128);
            let mut context = OfflineAudioContext::new(1, 256, sample_rate);

            let delay = context.create_delay(0.5); // this will be internally clamped to 1.
            delay.delay_time.set_value(0.5); // this will be clamped to 1. by the Reader
            delay.connect(&context.destination());

            let mut dirac = context.create_buffer(1, 1, sample_rate);
            dirac.copy_to_channel(&[1.], 0);

            let src = context.create_buffer_source();
            src.connect(&delay);
            src.set_buffer(dirac);
            src.start_at(0.);

            let result = context.start_rendering();
            let channel = result.get_channel_data(0);

            let mut expected = vec![0.; 256];
            expected[128] = 1.;

            assert_float_eq!(channel[..], expected[..], abs_all <= 0.);
        }
    }
}
