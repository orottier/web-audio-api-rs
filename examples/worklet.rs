use rand::Rng;

use web_audio_api::context::{
    AudioContext, AudioContextRegistration, AudioParamId, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, ChannelConfig, ChannelConfigOptions};
use web_audio_api::param::{AudioParam, AudioParamDescriptor, AutomationRate};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use web_audio_api::SampleRate;

/// Audio source node emitting white noise (random samples)
struct WhiteNoiseNode {
    /// handle to the audio context, required for all audio nodes
    registration: AudioContextRegistration,
    /// channel configuration (for up/down-mixing of inputs), required for all audio nodes
    channel_config: ChannelConfig,
    /// audio param controlling the volume (for educational purpose, use a GainNode otherwise)
    amplitude: AudioParam,
}

// implement required methods for AudioNode trait
impl AudioNode for WhiteNoiseNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    // source nodes take no input
    fn number_of_inputs(&self) -> u32 {
        0
    }

    // emit a single output
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl WhiteNoiseNode {
    /// Construct a new WhiteNoiseNode
    fn new<C: BaseAudioContext>(context: &C) -> Self {
        context.base().register(move |registration| {
            // setup the amplitude audio param
            let param_opts = AudioParamDescriptor {
                min_value: 0.,
                max_value: 1.,
                default_value: 1.,
                automation_rate: AutomationRate::A,
            };
            let (param, proc) = context
                .base()
                .create_audio_param(param_opts, registration.id());

            // setup the processor, this will run in the render thread
            let render = WhiteNoiseProcessor { amplitude: proc };

            // setup the audio node, this will live in the control thread (user facing)
            let node = WhiteNoiseNode {
                registration,
                channel_config: ChannelConfigOptions::default().into(),
                amplitude: param,
            };

            (node, Box::new(render))
        })
    }

    /// The Amplitude AudioParam
    fn amplitude(&self) -> &AudioParam {
        &self.amplitude
    }
}

struct WhiteNoiseProcessor {
    amplitude: AudioParamId,
}

impl AudioProcessor for WhiteNoiseProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        // get the audio param values
        let amplitude_values = params.get(&self.amplitude);

        // edit the output buffer in place
        output.modify_channels(|buf| {
            let mut rng = rand::thread_rng();
            amplitude_values
                .iter()
                .zip(buf.iter_mut())
                .for_each(|(i, o)| {
                    let rand: f32 = rng.gen_range(-1.0..1.0);
                    *o = *i * rand
                })
        });

        true // source node will always be active
    }
}

fn main() {
    env_logger::init();
    let context = AudioContext::new(None);

    // construct new node in this context
    let noise = WhiteNoiseNode::new(&context);

    // control amplitude
    noise.amplitude().set_value(0.3); // start at low volume
    noise.amplitude().set_value_at_time(1., 2.); // high volume after 2 secs

    // connect to speakers
    noise.connect(&context.destination());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs(4));
}
