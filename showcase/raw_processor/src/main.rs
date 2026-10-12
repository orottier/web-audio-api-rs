//! Construct a raw node/processor pair and render it without an audio device.

use web_audio_api::context::{
    AudioContextRegistration, AudioParamId, BaseAudioContext, OfflineAudioContext,
};
use web_audio_api::node::{AudioNode, AudioNodeOptions, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::unstable_plumbing::{
    self, AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use web_audio_api::{AudioParam, AudioParamDescriptor, AutomationRate, MessagePort};

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
                sum_of_squares: 0.,
                sample_count: 0,
            };
            (node, Box::new(processor))
        })
    }

    fn port(&self) -> MessagePort<'_> {
        unstable_plumbing::message_port(self)
    }
}

struct RawGainProcessor {
    gain: AudioParamId,
    sum_of_squares: f64,
    sample_count: usize,
}

impl AudioProcessor for RawGainProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // Cloning shares the buffer until samples are mutated (copy-on-write).
        outputs[0] = inputs[0].clone();
        let gain = params.get(&self.gain);
        for channel in outputs[0].channels_mut() {
            // A constant parameter has one value; otherwise there is one per frame.
            for (sample, gain) in channel.iter_mut().zip(gain.iter().cycle()) {
                *sample *= gain;
                self.sum_of_squares += f64::from(*sample).powi(2);
                self.sample_count += 1;
                // Report output RMS over each 40,000 samples, including all channels.
                if self.sample_count == 40_000 {
                    let rms = (self.sum_of_squares / self.sample_count as f64).sqrt();
                    scope.post_message(Box::new(rms));
                    self.sum_of_squares = 0.;
                    self.sample_count = 0;
                }
            }
        }
        // This transform has no audio tail.
        false
    }

    fn has_side_effects(&self) -> bool {
        // This processor sends messages in addition to producing audio.
        true
    }
}

fn main() {
    let mut context = OfflineAudioContext::new(1, 120_000, 48_000.);
    let mut source = context.create_constant_source();
    let gain = RawGainNode::new(&context);
    gain.gain.set_value(0.5);
    gain.port().set_onmessage(|message| {
        let rms = message.downcast::<f64>().expect("expected RMS value");
        println!("Output RMS: {rms:.3}");
    });
    source.connect(&gain);
    gain.connect(&context.destination());
    source.start();

    let rendered = context.start_rendering_sync();
    assert!(rendered.get_channel_data(0).iter().all(|&v| v == 0.5));
    println!(
        "Rendered {} frames through a raw gain processor.",
        rendered.length()
    );
}
