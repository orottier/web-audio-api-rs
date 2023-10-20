use rand::rngs::ThreadRng;
use rand::Rng;
use std::fs::File;

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
use web_audio_api::AudioBuffer;

// Granular synthesis example
//
// `cargo run --release --example granular`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example granular`

// note: naive lookhead scheduler implementation, a proper implementation should:
// - generalize to handle several engines and type of engines
// - run the loop in a dedicated thread
// - eventually use a proper priority queue
// see https://web.dev/audio-scheduling/ for some explanations
struct Scheduler {
    period: f64,
    lookahead: f64,
    queue: Vec<f64>,
    audio_context: AudioContext,
    engine: Option<ScrubEngine>,
}

impl Scheduler {
    fn new(audio_context: AudioContext) -> Self {
        Self {
            period: 0.05,
            lookahead: 0.1,
            queue: Vec::new(),
            audio_context,
            engine: None,
        }
    }

    fn add(&mut self, engine: Option<ScrubEngine>, start_time: f64) {
        self.engine = engine;
        self.queue.push(start_time);

        loop {
            self.tick();
            std::thread::sleep(std::time::Duration::from_millis(
                (self.period * 1000.) as u64,
            ));
        }
    }

    fn tick(&mut self) {
        let now = self.audio_context.current_time();
        // sort queue to have smaller values at the end of the vector
        self.queue.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut head = self.queue.last().cloned();

        while head.is_some() && head.unwrap() < now + self.lookahead {
            self.queue.pop();

            let trigger_time = head.unwrap();
            let next_time = self
                .engine
                .as_mut()
                .unwrap()
                .trigger_grain(&self.audio_context, trigger_time);

            if let Some(time) = next_time {
                self.queue.push(time);
                self.queue.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }

            head = self.queue.last().cloned();
        }
    }
}

struct ScrubEngine {
    audio_buffer: AudioBuffer,
    grain_period: f64,
    grain_duration: f64,
    position: f64,
    incr_position: f64,
    rng: ThreadRng,
}

impl ScrubEngine {
    fn new(audio_buffer: AudioBuffer) -> Self {
        let grain_period = 0.01;
        let grain_duration = 0.2;
        let speed = 1. / 2.;

        Self {
            audio_buffer,
            grain_period,
            grain_duration,
            position: 0.,
            incr_position: grain_period * speed, // half grain period to scrub half speed
            rng: rand::thread_rng(),
        }
    }

    fn trigger_grain(&mut self, audio_context: &AudioContext, trigger_time: f64) -> Option<f64> {
        // add some jitter to avoid some weird phase stuff
        let start_time = trigger_time + self.rng.gen::<f64>() * 0.003;

        let env = audio_context.create_gain();
        env.gain().set_value(0.);
        env.connect(&audio_context.destination());

        let mut src = audio_context.create_buffer_source();
        src.set_buffer(self.audio_buffer.clone());
        src.connect(&env);

        env.gain()
            .set_value_at_time(0., start_time)
            .linear_ramp_to_value_at_time(1., start_time + self.grain_duration / 2.)
            .linear_ramp_to_value_at_time(0., start_time + self.grain_duration);

        src.start_at_with_offset(start_time, self.position);
        src.stop_at(start_time + self.grain_duration);

        // check if we should reverse playback
        if self.position + self.incr_position
            > self.audio_buffer.duration() - (self.grain_duration + 0.2)
            || self.position + self.incr_position < 0.
        {
            self.incr_position *= -1.;
        }
        // define position for next call
        self.position += self.incr_position;

        // return the next time at which we want to trigger a grain
        Some(trigger_time + self.grain_period)
    }
}

fn main() {
    env_logger::init();
    println!("++ scrub into file forward and backward at 0.5 speed");

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let audio_context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let file = File::open("samples/sample.wav").unwrap();
    let audio_buffer = audio_context.decode_audio_data_sync(file).unwrap();

    let scrub_engine = ScrubEngine::new(audio_buffer);
    let start_time = audio_context.current_time();
    let mut scheduler = Scheduler::new(audio_context);
    scheduler.add(Some(scrub_engine), start_time);
}
