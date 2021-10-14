use crate::{
    buffer::{ChannelConfig, ChannelConfigOptions},
    context::{AsBaseAudioContext, AudioContextRegistration, AudioParamId},
    param::{AudioParam, AudioParamOptions},
    process::{AudioParamValues, AudioProcessor},
    SampleRate, BUFFER_SIZE,
};

use super::AudioNode;

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
pub struct DelayNode {
    registration: AudioContextRegistration,
    delay_time: AudioParam,
    channel_config: ChannelConfig,
}

impl AudioNode for DelayNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
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
}

impl DelayNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: DelayOptions) -> Self {
        context.base().register(move |registration| {
            let param_opts = AudioParamOptions {
                min_value: 0.,
                max_value: options.max_delay_time,
                default_value: 0.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());

            param.set_value_at_time(options.delay_time, 0.);

            // allocate large enough buffer to store all delayed samples
            let max_samples = options.max_delay_time * context.base().sample_rate().0 as f32;
            let max_quanta = (max_samples.ceil() as u32 + BUFFER_SIZE - 1) / BUFFER_SIZE;
            let delay_buffer = Vec::with_capacity(max_quanta as usize);

            let render = DelayRenderer {
                delay_time: proc,
                delay_buffer,
                index: 0,
            };

            let node = DelayNode {
                registration,
                channel_config: options.channel_config.into(),
                delay_time: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn delay_time(&self) -> &AudioParam {
        &self.delay_time
    }
}

struct DelayRenderer {
    delay_time: AudioParamId,
    delay_buffer: Vec<crate::alloc::AudioBuffer>,
    index: usize,
}

// SAFETY:
// AudioBuffers are not Send but we promise the `delay_buffer` Vec is emtpy before we ship it to
// the render thread.
unsafe impl Send for DelayRenderer {}

impl AudioProcessor for DelayRenderer {
    fn process(
        &mut self,
        inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        sample_rate: SampleRate,
    ) {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // todo: a-rate processing
        let delay = params.get(&self.delay_time)[0];

        // calculate the delay in chunks of BUFFER_SIZE (todo: sub quantum delays)
        let quanta = (delay * sample_rate.0 as f32) as usize / BUFFER_SIZE as usize;

        if quanta == 0 {
            // when no delay is set, simply copy input to output
            *output = input.clone();
        } else if self.delay_buffer.len() < quanta {
            // still filling buffer
            self.delay_buffer.push(input.clone());
            // clear output, it may have been re-used
            output.make_silent();
        } else {
            *output = std::mem::replace(&mut self.delay_buffer[self.index], input.clone());
            // progress index
            self.index = (self.index + 1) % quanta;
        }
    }

    fn tail_time(&self) -> bool {
        // todo: return false when all inputs disconnected and buffer exhausted
        true
    }
}
