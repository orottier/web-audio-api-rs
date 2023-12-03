use rand::Rng;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{stdin, stdout, BufRead, Write};
use std::time::{Duration, Instant};

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
    OfflineAudioContext,
};
use web_audio_api::node::{
    AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode, OscillatorType,
};
use web_audio_api::AudioBuffer;

// Benchmarks adapted from https://github.com/padenot/webaudio-benchmark
//
// `cargo run --release --example benchmarks`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example benchmarks`
struct BenchResult {
    name: &'static str,
    duration: Duration,
    buffer: AudioBuffer,
}

fn load_buffer(sources: &mut Vec<AudioBuffer>, pathname: &str, sample_rate: f32) {
    let context = OfflineAudioContext::new(1, 1, sample_rate);

    let file = File::open(pathname).unwrap();
    let audio_buffer = context.decode_audio_data_sync(file).unwrap();

    sources.push(audio_buffer);
}

fn get_buffer(sources: &[AudioBuffer], sample_rate: f32, number_of_channels: usize) -> AudioBuffer {
    let buffer = sources.iter().find(|&buffer| {
        buffer.sample_rate() == sample_rate && buffer.number_of_channels() == number_of_channels
    });

    buffer.unwrap().clone()
}

fn benchmark(name: &'static str, mut context: OfflineAudioContext, results: &mut Vec<BenchResult>) {
    print!("> Running benchmark: {:<70}\r", name);
    stdout().flush().unwrap();

    let start = Instant::now();
    let buffer = context.start_rendering_sync();
    let duration = start.elapsed();

    let result = BenchResult {
        name,
        duration,
        buffer,
    };

    results.push(result);
}

fn main() {
    env_logger::init();

    let mut sources = Vec::<AudioBuffer>::new();
    let mut results = Vec::<BenchResult>::new();

    const DURATION: usize = 120;
    let sample_rate = 48000.;

    load_buffer(&mut sources, "samples/think-mono-38000.wav", 38000.);
    load_buffer(&mut sources, "samples/think-mono-44100.wav", 44100.);
    load_buffer(&mut sources, "samples/think-mono-48000.wav", 48000.);
    load_buffer(&mut sources, "samples/think-stereo-38000.wav", 38000.);
    load_buffer(&mut sources, "samples/think-stereo-44100.wav", 44100.);
    load_buffer(&mut sources, "samples/think-stereo-48000.wav", 48000.);

    // -------------------------------------------------------
    // benchmarks
    // -------------------------------------------------------
    println!();

    {
        let name = "Baseline (silence)";

        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Mono)";

        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Stereo)";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Stereo and positional)";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_panner();
        panner.connect(&context.destination());
        panner.position_x().set_value(1.);
        panner.position_y().set_value(2.);
        panner.position_z().set_value(3.);
        panner.orientation_x().set_value(1.);
        panner.orientation_y().set_value(2.);
        panner.orientation_z().set_value(3.);

        let mut source = context.create_buffer_source();
        source.connect(&panner);

        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Mono)";

        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000., 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Stereo)";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000., 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Stereo and positional)";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_panner();
        panner.connect(&context.destination());
        panner.position_x().set_value(1.);
        panner.position_y().set_value(2.);
        panner.position_z().set_value(3.);
        panner.orientation_x().set_value(1.);
        panner.orientation_y().set_value(2.);
        panner.orientation_z().set_value(3.);

        let mut source = context.create_buffer_source();
        source.connect(&panner);

        let buf = get_buffer(&sources, 38000., 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Upmix without resampling (Mono -> Stereo)";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Downmix without resampling (Stereo -> Mono)";

        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple mixing (100x same buffer) - be careful w/ volume here!";

        let adjusted_duration = DURATION / 4;
        let context =
            OfflineAudioContext::new(2, adjusted_duration * sample_rate as usize, sample_rate);

        for _ in 0..100 {
            let mut source = context.create_buffer_source();
            let buf = get_buffer(&sources, 38000., 1);
            source.set_buffer(buf);
            source.set_loop(true);
            source.connect(&context.destination());
            source.start();
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple mixing (100 different buffers) - be careful w/ volume here!";

        let adjusted_duration = DURATION / 4;
        let context =
            OfflineAudioContext::new(2, adjusted_duration * sample_rate as usize, sample_rate);
        let reference = get_buffer(&sources, 38000., 1);
        let channel_data = reference.get_channel_data(0);

        for _ in 0..100 {
            let mut buffer = context.create_buffer(1, reference.length(), 38000.);
            buffer.copy_to_channel(channel_data, 0);

            let mut source = context.create_buffer_source();
            source.set_buffer(buffer);
            source.set_loop(true);
            source.connect(&context.destination());
            source.start();
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Simple mixing with gains";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let gain = context.create_gain();
        gain.connect(&context.destination());
        gain.gain().set_value(-1.);

        let mut gains_i = vec![];

        for _ in 0..4 {
            let gain_i = context.create_gain();
            gain_i.connect(&gain);
            gain_i.gain().set_value(0.25);
            gains_i.push(gain_i);
        }

        for _ in 0..2 {
            let buf = get_buffer(&sources, 38000., 1);

            let mut source = context.create_buffer_source();
            source.set_buffer(buf);
            source.set_loop(true);
            source.start();

            for gain_i in gains_i.iter() {
                let gain_ij = context.create_gain();
                gain_ij.gain().set_value(0.5);
                gain_ij.connect(gain_i);
                source.connect(&gain_ij);
            }
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Convolution reverb";

        let adjusted_duration = DURATION as f64 / 8.;
        let length = (adjusted_duration * sample_rate as f64) as usize;
        let context = OfflineAudioContext::new(1, length, sample_rate);
        let buf = get_buffer(&sources, sample_rate, 1);

        let mut rng = rand::thread_rng();

        let decay = 10.;
        let duration = 4.;
        let len = duration * sample_rate;
        let mut buffer = context.create_buffer(2, len as usize, sample_rate);

        buffer
            .get_channel_data_mut(0)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| {
                *b = (rng.gen_range(0.0..2.) - 1.) * (1. - i as f32 / len).powf(decay)
            });

        buffer
            .get_channel_data_mut(1)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| {
                *b = (rng.gen_range(0.0..2.) - 1.) * (1. - i as f32 / len).powf(decay)
            });

        let mut convolver = context.create_convolver();
        convolver.set_buffer(buffer);
        convolver.connect(&context.destination());

        let mut source = context.create_buffer_source();
        source.set_buffer(buf);
        source.set_loop(true);
        source.start();
        source.connect(&convolver);

        benchmark(name, context, &mut results);
    }

    {
        let name = "Granular synthesis";

        let adjusted_duration = DURATION as f64 / 16.;
        let length = (adjusted_duration * sample_rate as f64) as usize;
        let context = OfflineAudioContext::new(1, length, sample_rate);
        let buffer = get_buffer(&sources, sample_rate, 1);
        let mut offset = 0.;
        let mut rng = rand::thread_rng();

        // this ~1500 sources...
        while offset < adjusted_duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let mut src = context.create_buffer_source();
            src.connect(&env);
            src.set_buffer(buffer.clone());

            let rand_start = rng.gen_range(0..1000) as f64 / 1000. * 0.5;
            let rand_duration = rng.gen_range(0..1000) as f64 / 1000. * 0.999;
            let start = offset * rand_start;
            let end = start + 0.005 * rand_duration;
            let start_release = (offset + end - start).max(0.);

            env.gain().set_value_at_time(0., offset);
            env.gain().linear_ramp_to_value_at_time(0.5, offset + 0.005);
            env.gain().set_value_at_time(0.5, start_release);
            env.gain()
                .linear_ramp_to_value_at_time(0., start_release + 0.05);

            src.start_at_with_offset_and_duration(offset, start, end);

            offset += 0.005;
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Synth (Sawtooth with Envelope)";

        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let mut osc = context.create_oscillator();
            osc.connect(&env);
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);

            env.gain().set_value_at_time(0., 0.);
            env.gain().set_value_at_time(0.5, offset);
            env.gain().set_target_at_time(0., offset + 0.01, 0.1);
            osc.start_at(offset);
            osc.stop_at(offset + 1.);

            offset += 140. / 60. / 4.;
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Synth (Sawtooth with gain - no automation)";

        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let mut osc = context.create_oscillator();
            osc.connect(&env);
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);
            osc.start_at(offset);
            osc.stop_at(offset + 1.);

            offset += 140. / 60. / 4.;
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Synth (Sawtooth without gain)";

        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let mut osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);
            osc.start_at(offset);
            osc.stop_at(offset + 1.);

            offset += 140. / 60. / 4.;
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Substractive Synth";

        let sample_rate = 44100.;
        let context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let filter = context.create_biquad_filter();
        filter.connect(&context.destination());
        filter.frequency().set_value_at_time(0., 0.);
        filter.q().set_value_at_time(20., 0.);

        let env = context.create_gain();
        env.connect(&filter);
        env.gain().set_value_at_time(0., 0.);

        let mut osc = context.create_oscillator();
        osc.connect(&env);
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(110.);
        osc.start();

        let duration = DURATION as f64;

        while offset < duration {
            env.gain().set_value_at_time(1., offset);
            env.gain().set_target_at_time(0., offset, 0.1);

            filter.frequency().set_value_at_time(0., offset);
            filter.frequency().set_target_at_time(3500., offset, 0.03);

            offset += 140. / 60. / 16.;
        }

        benchmark(name, context, &mut results);
    }

    {
        let name = "Stereo panning";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_stereo_panner();
        panner.connect(&context.destination());
        panner.pan().set_value(0.1);

        let mut src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&panner);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Stereo panning with automation";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_stereo_panner();
        panner.connect(&context.destination());
        panner.pan().set_value_at_time(-1., 0.);
        panner.pan().set_value_at_time(0.2, 0.5);

        let mut src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&panner);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Sawtooth with automation";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let mut osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(2000.);
        osc.frequency().linear_ramp_to_value_at_time(20., 10.);
        osc.start_at(0.);

        benchmark(name, context, &mut results);
    }

    {
        // derived from "Simple source test without resampling (Stereo)""
        let name = "Stereo source with delay";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let delay = context.create_delay(1.);
        delay.delay_time().set_value(1.);
        delay.connect(&context.destination());

        let mut source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&delay);
        source.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "IIR filter";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        // these values correspond to a lowpass filter at 200Hz (calculated from biquad)
        let feedforward = vec![
            0.0002029799640409502,
            0.0004059599280819004,
            0.0002029799640409502,
        ];

        let feedback = vec![1.0126964557853775, -1.9991880801438362, 0.9873035442146225];

        // Create an IIR filter node
        let iir = context.create_iir_filter(feedforward, feedback);
        iir.connect(&context.destination());

        let mut src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&iir);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(name, context, &mut results);
    }

    {
        let name = "Biquad filter";

        let context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        // Create a biquad filter node (defaults to low pass)
        let biquad = context.create_biquad_filter();
        biquad.connect(&context.destination());
        biquad.frequency().set_value(200.);

        let mut src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&biquad);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(name, context, &mut results);
    }

    println!("> All done! {:<67}\n", "");

    // -------------------------------------------------------
    // display results
    // -------------------------------------------------------
    println!(
        "{0: <3} | {1: <67} | {2: <13} | {3: <20} | {4: <20}",
        "id", "name", "duration (ms)", "Speedup vs. realtime", "buffer.duration (s)",
    );

    for (index, result) in results.iter().enumerate() {
        println!(
            "{0: <3} | {1: <67} | {2: >13} | {3: >20.2} | {4: >20}",
            index + 1,
            result.name,
            result.duration.as_micros() as f64 / 1000.,
            result.buffer.duration() / (result.duration.as_micros() as f64 / 1_000_000.),
            result.buffer.duration(),
        );
    }

    println!();
    println!("+ Press <Ctrl-C> to quit");
    println!("+ Type the id of the result you want to listen and press <Enter>");
    print!("> ");
    stdout().flush().unwrap();

    // -------------------------------------------------------
    // handle input and preview
    // -------------------------------------------------------
    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    let mut current_source: Option<AudioBufferSourceNode> = None;

    let lines = stdin().lock().lines();

    for line in lines {
        let line = line.unwrap();
        let id = line.parse::<i64>().unwrap();
        let id = usize::try_from(id - 1).unwrap();

        let result = &results[id];
        let name = result.name;

        if let Some(mut cur) = current_source.take() {
            cur.stop();
        }

        let buffer = result.buffer.clone();
        let mut source = context.create_buffer_source();
        source.set_buffer(buffer);
        source.connect(&context.destination());
        source.start();
        source.set_onended(move |_| {
            println!("done playing {}", name);
            print!("> ");
            stdout().flush().unwrap();
        });

        current_source = Some(source);

        println!("+ playing output from {}", result.name);
        print!("> ");
        stdout().flush().unwrap();
    }
}
