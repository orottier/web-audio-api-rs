// use std::sync::atomic::{Ordering};
use std::sync::Arc;
use crossbeam_channel::{Receiver, Sender};

use crate::audio_buffer::{AudioBuffer};
use crate::context::{AsBaseAudioContext, AudioContextRegistration};
use crate::control::{Controller, Scheduler};
// use crate::param::{AudioParam, AudioParamOptions};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::{SampleRate, RENDER_QUANTUM_SIZE};

use super::{
    AudioNode, ChannelConfig, ChannelConfigOptions,
};

struct BufferChannelsMessage(Vec<Arc<Vec<f32>>>);

pub struct AudioBufferSourceOptions {
    pub buffer: Option<AudioBuffer>,
    // pub detune: f32,
    // pub r#loop: bool,
    // pub loop_start: f64,
    // pub loop_end: f64,
    // pub playback_rate: f32,
    pub channel_config: ChannelConfigOptions,
}

impl Default for AudioBufferSourceOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            // detune: 0.,
            // r#loop: false,
            // loop_start: 0.,
            // loop_end: 0.,
            // playback_rate: 1.,
            channel_config: Default::default(),
        }
    }
}

pub struct AudioBufferSourceNode {
    registration: AudioContextRegistration,
    scheduler: Scheduler,
    _controller: Controller,
    channel_config: ChannelConfig,
    sender: Sender<BufferChannelsMessage>,

    // detune: AudioParam,         // k-rate (can't be changed to a-rate, see wait it means...)
    // playback_rate: AudioParam,  // k-rate (can't be changed to a-rate, see wait it means...)

    buffer: Option<AudioBuffer>,
    start_called: bool,
}

// impl AudioScheduledSourceNode for AudioBufferSourceNode {
//     fn scheduler(&self) -> &Scheduler {
//         self.controller.scheduler()
//     }
// }

// impl AudioControllableSourceNode for AudioBufferSourceNode {
//     fn controller(&self) -> &Controller {
//         &self.controller
//     }
// }

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
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AudioBufferSourceOptions) -> Self {
        // create render and register all that in the context
        context.base().register(move |registration| {
            // let AudioBufferSourceOptions {
            //   buffer,
            //   channel_config,
            // } = options;
            // handle loop options, cf. controller

            // create parameters

            // let buffer_set = Arc::new(AtomicBool::new(false));

            // we don't want to block the control thread waiting for next `render.tick`
            let (sender, receiver) = crossbeam_channel::bounded(1);
            let controller = Controller::new();
            let scheduler = Scheduler::new();
            // create renderer
            let renderer = AudioBufferSourceRenderer {
                scheduler: scheduler.clone(),
                _controller: controller.clone(),
                receiver,
                buffer_channels: None,
            };

            // create node
            let mut node = Self {
                registration,
                scheduler,
                _controller: controller,
                channel_config: options.channel_config.into(),
                sender,
                buffer: None, // we don't want to pass the buffer here
                start_called: false,
            };

            if options.buffer.is_some() {
                node.set_buffer(&options.buffer.unwrap());
            }

            (node, Box::new(renderer))
        })
    }

    pub fn set_buffer(&mut self, audio_buffer: &AudioBuffer) {
        // - Let new buffer be the AudioBuffer or null value to be assigned to buffer.
        // - If new buffer is not null and [[buffer set]] is true, throw an InvalidStateError and abort these steps.
        if self.buffer.is_some() {
            // buffer has already been set, panic!
            panic!("InvalidStateError - cannot assign buffer twice");
        }
        // - If new buffer is not null, set [[buffer set]] to true.
        // - Assign new buffer to the buffer attribute.
        self.buffer = Some(audio_buffer.clone());
        // - If start() has previously been called on this node, perform the operation acquire the content on buffer.
        if self.start_called {
            self.acquire_buffer_channels();
        }
    }

    pub fn start(&mut self) {
        self.start_at(0.);
    }

    pub fn start_at(&mut self, start: f64) {
        self.start_called = true;
        self.acquire_buffer_channels();

        self.scheduler.start_at(start);
    }

    // pub fn start_at_with_offset(&self) {
    //     self.start_at(0.);
    // }

    // pub fn start_at_with_offset_and_duration(&self) {
    //     self.start_at(0.);
    // }

    // here we just create a new Vec or Arc references of the AudioBuffer.internal_data
    // cf. https://webaudio.github.io/web-audio-api/#acquire-the-content
    fn acquire_buffer_channels(&self) {
        let buffer = self.buffer.as_ref().unwrap();
        let number_of_channels = buffer.number_of_channels();
        let mut channels = Vec::<Arc<Vec<f32>>>::with_capacity(number_of_channels);

         for channel_number in 0..number_of_channels {
            let channel = buffer.get_channel_clone(channel_number);
            channels.push(channel);
        }

        self.sender
            .send(BufferChannelsMessage(channels))
            .expect("Sending BufferChannelsMessage failed");
    }
}

struct AudioBufferSourceRenderer {
    scheduler: Scheduler,
    _controller: Controller,
    receiver: Receiver<BufferChannelsMessage>,
    buffer_channels: Option<Vec<Arc<Vec<f32>>>>,
    // detune: AudioParamId,
    // playback_rate: AudioParamId,
    // shared reference to the
}

impl AudioProcessor for AudioBufferSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum], // no input
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        timestamp: f64,
        sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output_quantum = &mut outputs[0];

        // see if we received a buffer_channels
        if let Ok(msg) = self.receiver.try_recv() {
            self.buffer_channels = Some(msg.0);

            // println!("- ts: {:?} - start_time {:?} - stop_time {:?}", timestamp, start_time, stop_time);
            // println!("- buffer is set: {:?}", self.buffer_channels.is_some());
            // println!("- # output channels: {:?}", outputs.len());
        }

        if self.buffer_channels.is_none() {
            output_quantum.make_silent();
            return true
        }

        // println!("- ts: {:?} - start_time {:?} - stop_time {:?}", timestamp, start_time, stop_time);
        // println!("- buffer is set: {:?}", self.buffer_channels.is_some());
        // println!("- # output channels: {:?}", outputs.len());

        let dt = 1. / sample_rate.0 as f64;
        let next_block_time = timestamp + dt * RENDER_QUANTUM_SIZE as f64;

        let start_time = self.scheduler.start_time();
        let stop_time = self.scheduler.stop_time();

        // let's go dirty
        let buffer_channels = self.buffer_channels.as_ref().unwrap();
        let num_channels = buffer_channels.len();
        let length = buffer_channels[0].len();

        output_quantum.set_number_of_channels(buffer_channels.len());

        // idiot processing
        // we must go the other side so we can compute interpolation from
        // position for all channels at once
        for channel_number in 0..num_channels {
            let mut position = timestamp - start_time;
            let mut output_channel = output_quantum.channel_data_mut(channel_number);
            let buffer_channel = &buffer_channels[channel_number];

            for i in 0..RENDER_QUANTUM_SIZE {
                // this would the fast track start_time is aligned on a sample
                // e.g. start() or start(audioContext.currentTime);
                let index = (position * sample_rate.0 as f64).round() as usize;

                // we should have some `get_value_at_position` for interpolation

                if position < 0. {
                    output_channel[i] = 0.;
                } else {
                    if index < length {
                        output_channel[i] = buffer_channel[index];
                    } else {
                        output_channel[i] = 0.;
                    }
                }

                position += dt;
            }
        }

        // println!(">> process");
        true
    }
}

#[cfg(test)]
mod tests {

    use crate::context::{OfflineAudioContext, AsBaseAudioContext};
    use crate::audio_buffer::decode_audio_data;
    use crate::{SampleRate, RENDER_QUANTUM_SIZE};
    use crate::node::{AudioNode};

    use float_eq::assert_float_eq;

    #[test]
    fn type_playing_some_file() {
        let mut context = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE * 1, SampleRate(44_100));
        // load and decode buffer

        let file = std::fs::File::open("sample.wav").unwrap();
        // the Rc should be abstracted somehow, but can't find any solution for now
        let audio_buffer = decode_audio_data(file);

        let mut src = context.create_buffer_source();
        src.set_buffer(&audio_buffer);
        src.connect(&context.destination());
        src.start_at(context.current_time());

        let res = context.start_rendering();

        println!("buffer sample rate: {:?}", audio_buffer.sample_rate());
        println!("context sample rate: {:?}", context.sample_rate());

        // right
        assert_float_eq!(
            res.channel_data(0).as_slice(),
            audio_buffer.get_slice(0, 0, 128),
            abs_all <= 0.
        );

        // left
        assert_float_eq!(
            res.channel_data(1).as_slice(),
            audio_buffer.get_slice(1, 0, 128),
            abs_all <= 0.
        );
    }
}
















