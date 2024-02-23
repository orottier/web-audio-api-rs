use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, OscillatorNode, OscillatorOptions, OscillatorType,
};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};
use web_audio_api::{AudioParamDescriptor, AutomationRate};

// AudioWorkletNode example - BitCrusher
//
// `cargo run --release --example worklet_bitcrusher`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example worklet_bitcrusher`
//
// implementation from:
// https://webaudio.github.io/web-audio-api/#the-bitCrusher-node
//
// example usage from:
// https://github.com/GoogleChromeLabs/web-audio-samples/blob/main/src/audio-worklet/basic/bit-crusher/main.js
//
// online demo:
// https://googlechromelabs.github.io/web-audio-samples/audio-worklet/basic/bit-crusher/

struct BitCrusher {
    phase: f32,
    last_sample_value: f32,
}

impl AudioWorkletProcessor for BitCrusher {
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
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let bit_depth = params.get("bit_depth");
        let frequency_reduction = params.get("frequency_reduction");

        // if we are in ramp
        if bit_depth.len() > 1 {
            inputs[0].iter().zip(outputs[0].iter_mut()).for_each(
                |(input_channel, output_channel)| {
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
                },
            );
        } else {
            // Because we know bitDepth is constant for this call,
            // we can lift the computation of step outside the loop,
            // saving many operations.
            let step = (0.5_f32).powf(bit_depth[0]);

            inputs[0].iter().zip(outputs[0].iter_mut()).for_each(
                |(input_channel, output_channel)| {
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
                },
            );
        }

        false
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

    let mut osc = OscillatorNode::new(&context, OscillatorOptions::default());

    let options = AudioWorkletNodeOptions::default();
    let bit_crusher = AudioWorkletNode::new::<BitCrusher>(&context, options);

    let param_bit_depth = bit_crusher.parameters().get("bit_depth").unwrap();
    let param_reduction = bit_crusher.parameters().get("frequency_reduction").unwrap();

    osc.set_type(OscillatorType::Sawtooth);
    osc.frequency().set_value(5000.);
    param_bit_depth.set_value_at_time(1., 0.);

    osc.connect(&bit_crusher).connect(&context.destination());

    param_reduction.set_value_at_time(0.01, 0.);
    param_reduction.linear_ramp_to_value_at_time(0.1, 4.);
    param_reduction.exponential_ramp_to_value_at_time(0.01, 8.);

    osc.start();
    osc.stop_at(8.);

    std::thread::sleep(std::time::Duration::from_millis(8000));
}
