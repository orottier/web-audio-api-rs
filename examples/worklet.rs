use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletNode, AudioWorkletNodeOptions, AudioWorkletProcessor, RenderScope,
};
use web_audio_api::{AudioParamDescriptor, AutomationRate, MessagePort};

struct MyProcessor;

impl AudioWorkletProcessor for MyProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions, port: MessagePort<'_>) -> Self {
        port.set_onmessage(|_| println!("rec from processor"));
        port.post_message(1234);
        Self {}
    }

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        vec![AudioParamDescriptor {
            name: String::from("gain"),
            min_value: f32::MIN,
            max_value: f32::MAX,
            default_value: 1.,
            automation_rate: AutomationRate::A,
        }]
    }

    fn process<'a, 'b>(
        &mut self,
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        scope: &'b RenderScope,
    ) -> bool {
        // passthrough with gain
        inputs[0]
            .iter()
            .zip(outputs[0].iter_mut())
            .for_each(|(ic, oc)| {
                let gain = params.get("gain");
                for ((is, os), g) in ic.iter().zip(oc.iter_mut()).zip(gain.iter().cycle()) {
                    *os = is * g;
                }
            });

        if scope.current_frame % 12800 == 0 {
            scope.port().post_message(Box::new(1234));
        }

        false
    }
}

// AudioWorkletNode example
//
// `cargo run --release --example worklet`
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

    let mut options = AudioWorkletNodeOptions::default();
    options.parameter_data.insert(String::from("gain"), 1.0);

    let worklet = AudioWorkletNode::new::<MyProcessor>(&context, options);
    worklet.connect(&context.destination());
    worklet
        .parameters()
        .get("gain")
        .unwrap()
        .linear_ramp_to_value_at_time(0., 5.);

    let mut osc = context.create_oscillator();
    osc.frequency().set_value(300.);
    osc.connect(&worklet);
    osc.start();

    worklet.port().set_onmessage(|_| println!("rec from main"));

    // TODO this message is lost because the message handler has not been set up yet for this
    // AudioWorkletProcessor (constructor will run on first `process` call)
    worklet.port().post_message(123);

    std::thread::sleep(std::time::Duration::from_millis(1000));

    worklet.port().post_message(123);
    std::thread::sleep(std::time::Duration::from_millis(5000));
}
