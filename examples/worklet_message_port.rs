use rand::Rng;

use std::any::Any;
use std::collections::HashMap;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioNodeOptions};

use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};

// Showcase how to create your own audio node with message passing
//
// `cargo run --release --example worklet_message_port`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example worklet_message_port`

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NoiseColor {
    White, // zero mean, constant variance, uncorrelated in time
    Red,   // zero mean, constant variance, serially correlated in time
}

/// Audio source node emitting white noise (random samples)
struct WhiteNoiseNode {
    node: AudioWorkletNode,
}

impl WhiteNoiseNode {
    fn new(context: &AudioContext) -> Self {
        let options = AudioWorkletNodeOptions {
            number_of_inputs: 0,
            number_of_outputs: 1,
            output_channel_count: vec![1],
            parameter_data: HashMap::new(),
            processor_options: (),
            audio_node_options: AudioNodeOptions::default(),
        };

        let node = AudioWorkletNode::new::<WhiteNoiseProcessor>(context.base(), options);
        Self { node }
    }

    fn node(&self) -> &AudioWorkletNode {
        &self.node
    }

    fn set_noise_color(&self, color: NoiseColor) {
        self.node.port().post_message(color);
    }
}

struct WhiteNoiseProcessor {
    color: NoiseColor,
}

impl AudioWorkletProcessor for WhiteNoiseProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self {
            color: NoiseColor::White,
        }
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'_>,
        scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // edit the output buffer in place
        outputs[0].iter_mut().for_each(|buf| {
            let mut rng = rand::thread_rng();
            let mut prev_sample = 0.; // TODO, inherit from previous render quantum

            buf.iter_mut().for_each(|output_sample| {
                let mut value: f32 = rng.gen_range(-1.0..1.0);
                if self.color == NoiseColor::Red {
                    // red noise samples correlate with their previous value
                    value = value * 0.2 + prev_sample * 0.8;
                    prev_sample = value;
                }
                *output_sample = value
            })
        });

        if scope.current_frame % 12800 == 0 {
            scope.post_message(Box::new(scope.current_frame));
        }

        true // tail time, source node will always be active
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(&color) = msg.downcast_ref::<NoiseColor>() {
            self.color = color;
            return;
        }

        log::warn!("WhiteNoiseProcessor: Ignoring incoming message");
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

    // connect to speakers
    noise.node().connect(&context.destination());

    // add event handling for the heartbeat events from the render thread
    noise.node().port().set_onmessage(|m| {
        let frame = m.downcast::<u64>().unwrap();
        println!("rendered frame {frame}");
    });

    // enjoy listening
    println!("White noise");
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Switch to red noise");
    noise.set_noise_color(NoiseColor::Red);
    std::thread::sleep(std::time::Duration::from_secs(4));
}
