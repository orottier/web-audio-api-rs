use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::worklet::{
    AudioParamValues, AudioWorkletNode, AudioWorkletNodeOptions, AudioWorkletProcessor,
};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, GainNode, GainOptions, OscillatorNode, OscillatorOptions,
};
use web_audio_api::render::RenderScope;
use web_audio_api::{AudioParamDescriptor, AutomationRate};

// https://webaudio.github.io/web-audio-api/#the-bitcrusher-node

struct Bitcrusher {
    phase: f32,
    last_sample_value: f32,
}

impl AudioWorkletProcessor for Bitcrusher {
    type ProcessorOptions = ();

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        vec![
            AudioParamDescriptor {
                name: String::from("bit_depth"),
                min_value: 1.,
                max_value: 16.,
                default_value: 12.,
                automation_rate: AutomationRate::A,
            },
            AudioParamDescriptor {
                name: String::from("frequency_reduction"),
                min_value: 0.,
                max_value: 1.,
                default_value: 0.5,
                automation_rate: AutomationRate::A,
            },
        ]
    }

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self {
            phase: 0.,
            last_sample_value: 0.,
        }
    }

    fn process<'a, 'b>(
        &mut self,
        _scope: &'b RenderScope,
        inputs: &'b [&'a [f32]],
        outputs: &'b mut [&'a mut [f32]],
        params: AudioParamValues<'b>,
    ) -> bool {
        let bit_depth = params.get("bit_depth");
        let frequency_reduction = params.get("frequency_reduction");

        // if we are in ramp
        if bit_depth.len() > 1 {
            inputs
                .iter()
                .zip(outputs)
                .for_each(|(input_channel, output_channel)| {
                    input_channel
                        .iter()
                        .zip(output_channel.iter_mut())
                        .zip(bit_depth.iter())
                        .zip(frequency_reduction.iter().cycle())
                        .for_each(|(((i, o), d), f)| {
                            let step = (0.5_f32).powf(*d);

                            self.phase += f;

                            if self.phase > 1. {
                                self.phase -= 1.;
                                self.last_sample_value = step * (i / step + 0.5).floor();
                            }

                            *o = self.last_sample_value;
                        });
                });
        } else {
            // Because we know bitDepth is constant for this call,
            // we can lift the computation of step outside the loop,
            // saving many operations.
            let step = (0.5_f32).powf(bit_depth[0]);

            inputs
                .iter()
                .zip(outputs)
                .for_each(|(input_channel, output_channel)| {
                    input_channel
                        .iter()
                        .zip(output_channel.iter_mut())
                        .zip(frequency_reduction.iter().cycle())
                        .for_each(|((i, o), f)| {
                            self.phase += f;

                            if self.phase > 1. {
                                self.phase -= 1.;
                                self.last_sample_value = step * (i / step + 0.5).floor();
                            }

                            *o = self.last_sample_value;
                        });
                });
        }

        false
    }
}

// AudioWorkletNode example - Bitcrusher
//
// `cargo run --release --example worklet_bitcrusher`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example worklet`
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

    let mut osc = OscillatorNode::new(&context, OscillatorOptions::default());
    let gain = GainNode::new(&context, GainOptions::default());

    let mut options = AudioWorkletNodeOptions::default();
    options
        .parameter_data
        .insert(String::from("bit_depth"), 4.0);

    let bitcrusher = AudioWorkletNode::new::<Bitcrusher>(&context, options);

    osc.connect(&bitcrusher)
        .connect(&gain)
        .connect(&context.destination());
    osc.start();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
