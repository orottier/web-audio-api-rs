use rand::Rng;

use web_audio_api::buffer::{ChannelConfig, ChannelConfigOptions};
use web_audio_api::context::{
    AsBaseAudioContext, AudioContext, AudioContextRegistration, AudioParamId,
};
use web_audio_api::node::AudioNode;
use web_audio_api::param::{AudioParam, AudioParamOptions, AutomationRate};
use web_audio_api::process::{AudioParamValues, AudioProcessor};
use web_audio_api::SampleRate;

/// Audio source node emitting white noise (random samples)
struct WhiteNoiseNode<'a> {
    /// handle to the audio context, required for all audio nodes
    registration: AudioContextRegistration<'a>,
    /// channel configuration (for up/down-mixing of inputs), required for all audio nodes
    channel_config: ChannelConfig,
    /// audio param controlling the volume (for educational purpose, use a GainNode otherwise)
    amplitude: AudioParam<'a>,
}

// implement required methods for AudioNode trait
impl<'a> AudioNode for WhiteNoiseNode<'a> {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
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

impl<'a> WhiteNoiseNode<'a> {
    /// Construct a new WhiteNoiseNode
    fn new<C: AsBaseAudioContext>(context: &'a C) -> Self {
        context.base().register(move |registration| {
            // setup the amplitude audio param
            let param_opts = AudioParamOptions {
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
        _inputs: &[web_audio_api::alloc::AudioBuffer],
        outputs: &mut [web_audio_api::alloc::AudioBuffer],
        params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) {
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
        })
    }

    fn tail_time(&self) -> bool {
        true // source node will always be active
    }
}

fn main() {
    let context = AudioContext::new();

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
