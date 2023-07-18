use std::any::Any;

use rand::Rng;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, AudioContextRegistration,
    AudioParamId, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use web_audio_api::{AudioParam, AudioParamDescriptor, AutomationRate};

// Shocase how to create your own audio node
//
// `cargo run --release --example worklet`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example worklet`

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NoiseColor {
    White, // zero mean, constant variance, uncorrelated in time
    Red,   // zero mean, constant variance, serially correlated in time
}

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
    fn number_of_inputs(&self) -> usize {
        0
    }

    // emit a single output
    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl WhiteNoiseNode {
    /// Construct a new WhiteNoiseNode
    fn new<C: BaseAudioContext>(context: &C) -> Self {
        context.register(move |registration| {
            // setup the amplitude audio param
            let param_opts = AudioParamDescriptor {
                min_value: 0.,
                max_value: 1.,
                default_value: 1.,
                automation_rate: AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_opts, &registration);

            // setup the processor, this will run in the render thread
            let render = WhiteNoiseProcessor {
                amplitude: proc,
                color: NoiseColor::White,
            };

            // setup the audio node, this will live in the control thread (user facing)
            let node = WhiteNoiseNode {
                registration,
                channel_config: ChannelConfig::default(),
                amplitude: param,
            };

            (node, Box::new(render))
        })
    }

    /// The Amplitude AudioParam
    fn amplitude(&self) -> &AudioParam {
        &self.amplitude
    }

    fn set_noise_color(&self, color: NoiseColor) {
        self.registration.post_message(color);
    }
}

struct WhiteNoiseProcessor {
    amplitude: AudioParamId,
    color: NoiseColor,
}

impl AudioProcessor for WhiteNoiseProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        _scope: &RenderScope,
    ) -> bool {
        // single output node, with a stereo config
        let output = &mut outputs[0];
        output.set_number_of_channels(2);

        // get the audio param values
        let amplitude_values = params.get(&self.amplitude);

        // edit the output buffer in place
        output.channels_mut().iter_mut().for_each(|buf| {
            let mut rng = rand::thread_rng();

            // audio param buffer length is either 1 (k-rate, or when all a-rate samples are equal) or
            // 128 (a-rate), so use `cycle` to be able to zip it with the output buffer
            let amplitude_values_cycled = amplitude_values.iter().cycle();

            let mut prev_sample = 0.; // TODO, inherit from previous render quantum

            buf.iter_mut()
                .zip(amplitude_values_cycled)
                .for_each(|(output_sample, amplitude)| {
                    let mut value: f32 = rng.gen_range(-1.0..1.0);
                    if self.color == NoiseColor::Red {
                        // red noise samples correlate with their previous value
                        value = value * 0.2 + prev_sample * 0.8;
                        prev_sample = value;
                    }
                    *output_sample = *amplitude * value
                })
        });

        true // source node will always be active
    }

    fn onmessage(&mut self, msg: &mut Box<dyn Any + Send + 'static>) {
        // handle incoming signals requesting for change of color
        if let Some(color) = msg.downcast_ref::<NoiseColor>() {
            self.color = *color;
            return;
        }

        // ...add more message handlers here...

        log::warn!("WhiteNoiseProcessor: Dropping incoming message {msg:?}");
    }
}

fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // construct new node in this context
    let noise = WhiteNoiseNode::new(&context);

    // control amplitude
    noise.amplitude().set_value(0.3); // start at low volume
    noise.amplitude().set_value_at_time(1., 2.); // high volume after 2 secs

    // connect to speakers
    noise.connect(&context.destination());

    // enjoy listening
    println!("Low volume");
    std::thread::sleep(std::time::Duration::from_secs(2));
    println!("High volume");
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Switch to red noise");
    noise.set_noise_color(NoiseColor::Red);
    std::thread::sleep(std::time::Duration::from_secs(4));
}
