use rand::rngs::ThreadRng;
use rand::Rng;
use std::{thread, time};
use web_audio_api::context::{AsBaseAudioContext, AudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, DelayNode, GainNode};

// run in release mode
// `cargo run --release --example cyclic_graph`

struct FeedbackDelay<'a> {
    audio_context: &'a AudioContext,
    input: GainNode,
    delay: DelayNode,
    pre_gain: GainNode,
    feedback: GainNode,
    output: GainNode,
}

impl<'a> FeedbackDelay<'a> {
    fn new(audio_context: &'a AudioContext) -> Self {
        let output = audio_context.create_gain();

        let delay = audio_context.create_delay(1.);
        delay.connect(&output);

        let feedback = audio_context.create_gain();
        feedback.connect(&delay);
        delay.connect(&feedback);

        let pre_gain = audio_context.create_gain();
        pre_gain.connect(&feedback); // delay line
        pre_gain.connect(&output); // direct sound

        let input = audio_context.create_gain();
        input.connect(&pre_gain);

        Self {
            audio_context,
            input,
            delay,
            pre_gain,
            feedback,
            output,
        }
    }

    fn connect(&self, dest: &dyn AudioNode) {
        self.output.connect(dest);
    }

    fn set_pre_gain(&self, value: f32) {
        let now = self.audio_context.current_time();
        self.pre_gain.gain().set_target_at_time(value, now, 0.005);
    }

    fn set_feedback(&self, value: f32) {
        let now = self.audio_context.current_time();
        self.feedback.gain().set_target_at_time(value, now, 0.005);
    }

    fn set_delay(&self, value: f32) {
        let now = self.audio_context.current_time();
        self.delay.delay_time().set_target_at_time(value, now, 0.005);
    }
}

fn trigger_chord(audio_context: &AudioContext, delays: &Vec<FeedbackDelay>, rng: &mut ThreadRng) {
    let num_notes = rng.gen_range(2..8);
    let base_freq = 100.;
    let now = audio_context.current_time();

    for _ in 0..num_notes {
        let num_partial = rng.gen_range(1..15);
        let jitter = rng.gen_range(1..20);
        let start_time = now + (jitter as f64 / 1000.);
        let end_time = start_time + 0.2 + 1. / num_partial as f64;
        let delay_index = rng.gen_range(0..delays.len());
        let gain = rng.gen_range(1..10) as f32 / 10.;

        let feedback_delay = &delays[delay_index];
        feedback_delay.set_pre_gain(rng.gen_range(5..10) as f32 / 10.);
        feedback_delay.set_feedback(rng.gen_range(6..9) as f32 / 10.);

        let panner = audio_context.create_stereo_panner();
        panner.connect(&feedback_delay.input);
        panner.pan().set_value(rng.gen_range(0..10) as f32 / 5. - 1.);

        let env = audio_context.create_gain();
        env.connect(&panner);
        env.gain().set_value_at_time(0., start_time);
        env.gain().linear_ramp_to_value_at_time(gain, start_time + 0.02);
        env.gain().exponential_ramp_to_value_at_time(0.0001, end_time);

        let osc = audio_context.create_oscillator();
        osc.connect(&env);
        osc.frequency().set_value(base_freq * num_partial as f32);
        osc.start_at(start_time);
        osc.stop_at(end_time);
    }
}

fn main() {
    let audio_context = AudioContext::new(None);

    let mut rng = rand::thread_rng();
    let mut delays: Vec<FeedbackDelay> = vec![];

    for i in 0..10 {
        let feedback_delay = FeedbackDelay::new(&audio_context);
        feedback_delay.set_delay(1. / (i as f32 + 1.));
        feedback_delay.connect(&audio_context.destination());

        delays.push(feedback_delay);
    }

    loop {
        trigger_chord(&audio_context, &delays, &mut rng);
        thread::sleep(time::Duration::from_millis(2000));
    }
}
