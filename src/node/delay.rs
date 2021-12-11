use crate::context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId};
use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions};

use std::cell::RefCell;
use std::rc::Rc;

/// Options for constructing a DelayNode
pub struct DelayOptions {
    pub max_delay_time: f32,
    pub delay_time: f32,
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
        // allocate large enough buffer to store all delayed samples
        let max_samples = options.max_delay_time * context.base().sample_rate().0 as f32;
        let max_quanta =
            (max_samples.ceil() as usize + RENDER_QUANTUM_SIZE - 1) / RENDER_QUANTUM_SIZE;
        let delay_buffer = Vec::with_capacity(max_quanta);

        let shared_buffer = Rc::new(RefCell::new(delay_buffer));
        let shared_buffer_clone = shared_buffer.clone();

        context.base().register(move |writer_registration| {
            let node = context.base().register(move |reader_registration| {
                let param_opts = AudioParamOptions {
                    min_value: 0.,
                    max_value: options.max_delay_time,
                    default_value: 0.,
                    automation_rate: crate::param::AutomationRate::A,
                };
                let (param, proc) = context
                    .base()
                    .create_audio_param(param_opts, reader_registration.id());

                param.set_value_at_time(options.delay_time, 0.);

                let reader_render = DelayReader {
                    delay_time: proc,
                    delay_buffer: shared_buffer_clone,
                    index: 0,
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

struct DelayReader {
    delay_time: AudioParamId,
    delay_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
}

struct DelayWriter {
    delay_buffer: Rc<RefCell<Vec<AudioRenderQuantum>>>,
    index: usize,
}

// SAFETY:
// AudioRenderQuantums are not Send but we promise the `delay_buffer` Vec is empty before we ship it to
// the render thread.
unsafe impl Send for DelayReader {}
unsafe impl Send for DelayWriter {}

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

        let mut buffer = self.delay_buffer.borrow_mut();

        // add to buffer
        if buffer.len() < buffer.capacity() {
            buffer.push(input);
        } else {
            buffer[self.index] = input;
        }

        // increment cursor
        self.index = (self.index + 1) % buffer.capacity();

        // The writer end does not produce output.
        // Clear output buffer, it may have been re-used
        output.make_silent();

        // todo: return false when all inputs disconnected and buffer exhausted
        true
    }
}

impl AudioProcessor for DelayReader {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let output = &mut outputs[0];

        // todo: a-rate processing
        let delay = params.get(&self.delay_time)[0];

        // calculate the delay in chunks of RENDER_QUANTUM_SIZE (todo: sub quantum delays)
        let quanta = (delay * sample_rate.0 as f32) as usize / RENDER_QUANTUM_SIZE;

        // a delay of zero quanta is not allowed (in cycles, we don't know wether the reader or
        // writer renders first and the ordering may change on every graph update - causing clicks)
        let quanta = quanta.max(1);

        let buffer = self.delay_buffer.borrow_mut();

        let delayed_index = (self.index + buffer.capacity() - quanta) % buffer.capacity();
        if delayed_index >= buffer.len() {
            // still waiting to fill buffer for given delay
            output.make_silent();
        } else {
            *output = buffer[delayed_index].clone();
        }

        // increment cursor
        self.index = (self.index + 1) % buffer.capacity();

        // todo: return false when all inputs disconnected and buffer exhausted
        true
    }
}
