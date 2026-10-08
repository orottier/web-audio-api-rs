//! Construct a raw node/processor pair and render it without an audio device.

use std::any::Any;
use web_audio_api::context::{
    AudioContextRegistration, AudioParamId, BaseAudioContext, OfflineAudioContext,
};
use web_audio_api::node::{AudioNode, AudioNodeOptions, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::unstable_plumbing::{
    self, AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use web_audio_api::{AudioParam, AudioParamDescriptor, AutomationRate};

#[derive(Debug)]
struct RawGainNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    gain: AudioParam,
}

impl AudioNode for RawGainNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl RawGainNode {
    fn new(context: &impl BaseAudioContext) -> Self {
        unstable_plumbing::register(context, |registration| {
            let (gain, gain_id) = unstable_plumbing::create_audio_param(
                context,
                AudioParamDescriptor {
                    name: "gain".into(),
                    min_value: f32::MIN,
                    max_value: f32::MAX,
                    default_value: 1.,
                    automation_rate: AutomationRate::A,
                },
                &registration,
            );
            let node = Self {
                registration,
                channel_config: AudioNodeOptions::default().into(),
                gain,
            };
            let processor = RawGainProcessor {
                gain: gain_id,
                invert: false,
            };
            (node, Box::new(processor))
        })
    }

    fn set_invert(&self, invert: bool) {
        unstable_plumbing::post_message(&self.registration, invert);
    }
}

struct RawGainProcessor {
    gain: AudioParamId,
    invert: bool,
}

impl AudioProcessor for RawGainProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // Cloning shares the buffer until samples are mutated (copy-on-write).
        outputs[0] = inputs[0].clone();
        let gain = params.get(&self.gain);
        let sign = if self.invert { -1. } else { 1. };
        for channel in outputs[0].channels_mut() {
            // A constant parameter has one value; otherwise there is one per frame.
            for (sample, gain) in channel.iter_mut().zip(gain.iter().cycle()) {
                *sample *= gain * sign;
            }
        }
        // This transform has no tail. The existing has_side_effects default is false.
        false
    }

    fn onmessage(&mut self, message: &mut dyn Any) {
        if let Some(invert) = message.downcast_ref::<bool>() {
            self.invert = *invert;
        }
    }
}

fn main() {
    let mut context = OfflineAudioContext::new(1, 256, 48_000.);
    let mut source = context.create_constant_source();
    let gain = RawGainNode::new(&context);
    gain.gain.set_value(0.5);
    gain.set_invert(true);
    source.connect(&gain);
    gain.connect(&context.destination());
    source.start();

    let rendered = context.start_rendering_sync();
    assert!(rendered.get_channel_data(0).iter().all(|&v| v == -0.5));
    println!(
        "Rendered {} frames through a raw gain processor.",
        rendered.length()
    );
}
