use std::env;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, AudioContextRegistration,
    BaseAudioContext,
};
use web_audio_api::media_devices::{
    enumerate_devices_sync, get_user_media_sync, MediaDeviceInfo, MediaDeviceInfoKind,
    MediaStreamConstraints, MediaTrackConstraints,
};
use web_audio_api::node::{AudioNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// Estimate the audio round-trip latency. Just plug your audio output into the
// audio input with some cable.
//
// `cargo run --release --example roundtrip_latency_test`
//
// For testing purposes, a feedback delay line of 17ms is created to emulate audio
// feedback, run with:
//
// `cargo run --release  --example roundtrip_latency_test -- test`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example roundtrip_latency_test`

fn ask_source_id() -> Option<String> {
    println!("Enter the input 'device_id' and press <Enter>");
    println!("- Use 0 for the default audio input device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "0" => None,
        i => Some(i.to_string()),
    }
}

fn ask_sink_id() -> String {
    println!("Enter the output 'sink' and press <Enter>");
    println!("- Use 0 for the default audio output device");

    let input = std::io::stdin().lines().next().unwrap().unwrap();
    match input.trim() {
        "0" => "none".to_string(),
        i => i.to_string(),
    }
}

struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    pub fn new(v: f64) -> Self {
        Self {
            inner: AtomicU64::new(u64::from_ne_bytes(v.to_ne_bytes())),
        }
    }

    pub fn load(&self) -> f64 {
        f64::from_ne_bytes(self.inner.load(Ordering::SeqCst).to_ne_bytes())
    }

    pub fn store(&self, v: f64) {
        self.inner
            .store(u64::from_ne_bytes(v.to_ne_bytes()), Ordering::SeqCst)
    }
}

struct LatencyTesterNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    estimated_latency: Arc<AtomicF64>,
}

// implement required methods for AudioNode trait
impl AudioNode for LatencyTesterNode {
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

impl LatencyTesterNode {
    /// Construct a new LatencyTesterNode
    pub fn new<C: BaseAudioContext>(context: &C) -> Self {
        context.register(move |registration| {
            let estimated_latency = Arc::new(AtomicF64::new(0.));

            let render = LatencyTesterProcessor {
                estimated_latency: Arc::clone(&estimated_latency),
                send_time: 0.,
            };

            let node = LatencyTesterNode {
                registration,
                channel_config: ChannelConfig::default(),
                estimated_latency,
            };

            (node, Box::new(render))
        })
    }

    pub fn estimated_latency(&self) -> f64 {
        self.estimated_latency.load()
    }
}

struct LatencyTesterProcessor {
    estimated_latency: Arc<AtomicF64>,
    send_time: f64,
}

impl AudioProcessor for LatencyTesterProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        scope: &RenderScope,
    ) -> bool {
        // send a dirac every second
        // 48000 / 128 = 375
        let output = &mut outputs[0];

        if (scope.current_frame / 128) % 375 == 0 {
            output
                .channels_mut()
                .iter_mut()
                .for_each(|channel| channel[0] = 1.);

            self.send_time = scope.current_time;
        } else {
            output.make_silent();
        }

        // check input for dirac
        let input = &inputs[0];
        let sample_rate = scope.sample_rate;
        let sample_duration = 1. / sample_rate as f64;
        let dirac_found = false;

        input.channel_data(0).iter().enumerate().for_each(|(i, s)| {
            if !dirac_found && *s > 0.5 {
                let now = scope.current_time + (i as f64 * sample_duration);
                let diff = now - self.send_time;
                self.estimated_latency.store(diff);
            }
        });

        true
    }
}

fn main() {
    let mut test = false;
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && args[1] == "test" {
        test = true;
    }

    let (_context, latency_tester) = if test {
        // emulate loopback
        let latency: f32 = 0.017;
        println!(
            "> Testing: estimated latency should be around {:?}",
            latency
        );

        let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
            Ok("playback") => AudioContextLatencyCategory::Playback,
            _ => AudioContextLatencyCategory::default(),
        };

        let context = AudioContext::new(AudioContextOptions {
            latency_hint,
            sample_rate: Some(48000.),
            ..AudioContextOptions::default()
        });

        let latency_tester = LatencyTesterNode::new(&context);
        latency_tester.connect(&context.destination());

        let delay = context.create_delay(latency.ceil() as f64);
        delay.delay_time().set_value(latency);

        latency_tester.connect(&delay);
        delay.connect(&latency_tester);

        (context, latency_tester)
    } else {
        // full round trip
        let devices = enumerate_devices_sync();

        let input_devices: Vec<MediaDeviceInfo> = devices
            .into_iter()
            .filter(|d| d.kind() == MediaDeviceInfoKind::AudioInput)
            .collect();

        dbg!(input_devices);
        let source_id = ask_source_id();

        let devices = enumerate_devices_sync();

        let output_devices: Vec<MediaDeviceInfo> = devices
            .into_iter()
            .filter(|d| d.kind() == MediaDeviceInfoKind::AudioOutput)
            .collect();

        dbg!(output_devices);
        let sink_id = ask_sink_id();

        let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
            Ok("playback") => AudioContextLatencyCategory::Playback,
            _ => AudioContextLatencyCategory::default(),
        };

        let context = AudioContext::new(AudioContextOptions {
            latency_hint,
            sample_rate: Some(48000.),
            sink_id,
            ..AudioContextOptions::default()
        });

        let latency_tester = LatencyTesterNode::new(&context);
        latency_tester.connect(&context.destination());

        // open mic stream and pipe into latency_tester
        let mut constraints = MediaTrackConstraints::default();
        constraints.device_id = source_id;
        let stream_constraints = MediaStreamConstraints::AudioWithConstraints(constraints);
        let mic = get_user_media_sync(stream_constraints);

        // create media stream source node with mic stream
        let stream_source = context.create_media_stream_source(&mic);
        stream_source.connect(&latency_tester);

        (context, latency_tester)
    };

    loop {
        let latency = latency_tester.estimated_latency();

        if latency == 0. {
            println!("- Unable to perform measurement");
        } else {
            println!(
                "- Estimated latency {:?}",
                latency_tester.estimated_latency()
            );
        }

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
