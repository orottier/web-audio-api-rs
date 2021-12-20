use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};

use std::cell::{RefCell, RefMut};
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
/*
 * For simplicity in the audio graph rendering, we have made the conscious decision to deviate from
 * the spec and split the delay node up front in a reader and writer node (instead of during the
 * render loop - see https://webaudio.github.io/web-audio-api/#rendering-loop )
 *
 * This has a drawback: a delay of 0 is no longer possible. This would only be possible if the
 * writer end is rendered before the reader end in the graph, but we cannot enforce that here.
 * (The only way would be to connect the writer to the reader, but that would kill the
 * cycle-breaker feature of the delay node.)
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
    ) -> Result<&'a dyn AudioNode, crate::IndexSizeError> {
        if self.context() != dest.context() {
            panic!("attempting to connect nodes from different contexts");
        }

        if self.number_of_outputs() <= output || dest.number_of_inputs() <= input {
            return Err(crate::IndexSizeError {});
        }

        self.context()
            .connect(self.reader_registration.id(), dest.id(), output, input);

        Ok(dest)
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
    pub fn new<C: AsBaseAudioContext>(context: &C, options: DelayOptions) -> Self {
        let max_delay_time = options.max_delay_time;
        // allocate large enough buffer to store all delayed samples
        let max_samples = max_delay_time * context.base().sample_rate().0 as f64;
        let max_quanta =
            (max_samples.ceil() as usize + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE;
        let delay_buffer = Vec::with_capacity(max_quanta);

        let shared_buffer = Rc::new(RefCell::new(delay_buffer));
        let shared_buffer_clone = shared_buffer.clone();

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
                    delay_buffer: shared_buffer_clone,
                    index: 0,
                    // internal buffer used to compute output per channel at each frame
                    internal_buffer: Vec::<f32>::with_capacity(2),
                    max_delay_time,
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
                delay_buffer: shared_buffer,
                index: 0,
            };

            (node, Box::new(writer_render))
        })
    }

    pub fn delay_time(&self) -> &AudioParam {
        &self.delay_time
    }
}

struct DelayWriter {
    delay_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
}

struct DelayReader {
    delay_time: AudioParamId,
    delay_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
    internal_buffer: Vec<f32>,
    max_delay_time: f64,
}

// SAFETY:
// AudioRenderQuantums are not Send but we promise the `delay_buffer` Vec is
// empty before we ship it to the render thread.
unsafe impl Send for DelayWriter {}
unsafe impl Send for DelayReader {}

trait RingBufferChecker {
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>>;

    // this step, while not necessary per se, guarantees the ring buffer is filled
    // with silence buffers, simplifying the code in both Writer and Reader as we
    // know `len() == capacity()` and all the buffers is initialized with zeros.
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

    #[inline(always)]
    fn check_ring_buffer_up_down_mix(&self, input: &AudioRenderQuantum) {
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
            // println!("mix ring_buffer channels to {:?}", input_number_of_channels);
            for render_quantum in ring_buffer.iter_mut() {
                // @todo - use node current channel interpretation
                render_quantum.mix(input_number_of_channels, ChannelInterpretation::Speakers);
            }
        }
    }
}

impl RingBufferChecker for DelayWriter {
    #[inline(always)]
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>> {
        self.delay_buffer.borrow_mut()
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
        // We must perform the checks (buffer size and up/down mix) on both Writer
        // and Reader as the order of processing between them is not guaranteed.
        self.check_ring_buffer_size(&input);
        self.check_ring_buffer_up_down_mix(&input);

        let output = &mut outputs[0];

        let mut buffer = self.delay_buffer.borrow_mut();
        // add to buffer
        buffer[self.index] = input;
        // increment cursor
        self.index = (self.index + 1) % buffer.capacity();
        // The writer end does not produce output.
        // Clear output buffer, it may have been re-used
        output.make_silent();

        // todo: return false when all inputs disconnected and buffer exhausted
        // if input.is_silence() store block timestamp
        // else store f64::Max
        //
        // if silence_timestamp + maxDelayTime >= timestamp
        //  return false
        // else
        //  return true
        // @note f64::MAX + something == f64::MAX but doesn't crashes

        true
    }
}

impl RingBufferChecker for DelayReader {
    #[inline(always)]
    fn ring_buffer_mut(&self) -> RefMut<Vec<AudioRenderQuantum>> {
        self.delay_buffer.borrow_mut()
    }
}

impl AudioProcessor for DelayReader {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // @note - are we sure we receive the same input as the writer?
        //
        // Input is only used here to make the checks (buffer size and up/down mix)
        // We must perform the checks (buffer size and up/down mix) on both Writer
        // and Reader as the order of processing between them is not guaranteed.
        let input = inputs[0].clone(); // @note - this clone may not be necessary
        self.check_ring_buffer_size(&input);
        self.check_ring_buffer_up_down_mix(&input);

        let number_of_channels = input.number_of_channels();
        // resize internal buffer if needed
        if self.internal_buffer.len() != number_of_channels {
            self.internal_buffer.resize(number_of_channels, 0.);
        }

        // single input/output node
        let output = &mut outputs[0];

        // handle delay_buffer up/down max, this must be tested both on the reader
        // and on the writer as we don't know which one will processed first

        // get playhead position (delay) at each sample
        // - clamp between playhead.min(RENDER_QUANTUM_SIZE * dt).max(max_delay_time);
        // - pick samples in delay_buffer (difficulty is need to abstract the wrapped buffers)
        let sample_rate = sample_rate.0 as f64; // shadow sample_rate, we don't need the type
        let dt = 1. / sample_rate;
        let num_frames = RENDER_QUANTUM_SIZE;
        let quantum_duration = num_frames as f64 * dt;

        let delay = params.get(&self.delay_time);
        let ring_buffer = self.delay_buffer.borrow(); // no need for `mut` here

        let mut current_time = 0.;

        // println!("{:?}", current_time + delay[0] as f64);

        for index in 0..num_frames {
            // Compute clamped `delayed_position` at a-rate relatively
            // to ring_buffer current `index`.
            let position = current_time + delay[index] as f64;
            // @todo - allow 0. delay if not in loop (mostly impacts graph processing)
            let delayed_position = position.clamp(quantum_duration, self.max_delay_time);

            // As previous and next samples might not be in the same render_quantum
            // stored in the ring buffer, we need to compute both block index in
            // ring buffer and sample index in given block for both previous and
            // next samples.
            let playhead = delayed_position * sample_rate;
            let playhead_floored = playhead.floor();

            let prev_raw = playhead_floored as usize;
            let prev_block_index_raw = prev_raw / RENDER_QUANTUM_SIZE; // block in ring_buffer
            let prev_sample_index = prev_raw - prev_block_index_raw * RENDER_QUANTUM_SIZE; // sample in block
            let prev_block_index = (self.index + prev_block_index_raw) % ring_buffer.len(); // offset block

            let next_raw = playhead.ceil() as usize;
            let next_block_index_raw = next_raw / RENDER_QUANTUM_SIZE;
            let next_sample_index = next_raw - next_block_index_raw * RENDER_QUANTUM_SIZE;
            let next_block_index = (self.index + next_block_index_raw) % ring_buffer.len();

            let k = (playhead - playhead_floored) as f32;
            let k_inv = 1. - k;

            for channel_number in 0..number_of_channels {
                let prev_sample =
                    ring_buffer[prev_block_index].channel_data(channel_number)[prev_sample_index];
                let next_sample =
                    ring_buffer[next_block_index].channel_data(channel_number)[next_sample_index];

                let value = k_inv * prev_sample + k * next_sample;

                self.internal_buffer[channel_number] = value;
            }

            // internal buffer is populated for this sample, push in ouput
            output.set_channels_values_at(index, &self.internal_buffer);

            current_time += dt;
        }

        // increment cursor
        self.index = (self.index + 1) % ring_buffer.len();

        // todo: return false when all inputs disconnected and buffer exhausted
        true
    }
}

#[cfg(test)]
mod tests {
    // use float_eq::assert_float_eq;

    use crate::SampleRate;
    use crate::context::OfflineAudioContext;

    use super::*;

    #[test]
    fn test_create_delay_node() {
        let mut context = OfflineAudioContext::new(1, 256, SampleRate(44_100));

        let options = DelayOptions::default();
        let delay = DelayNode::new(&context, options);

        context.start_rendering();
    }
}
