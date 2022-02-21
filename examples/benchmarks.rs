use std::fs::File;
use std::io::{Read, Write, stdout, stdin};
use std::time::{Duration, Instant};

use termion::cursor;
use termion::color;
use termion::raw::IntoRawMode;

use web_audio_api::SampleRate;
use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AudioContext, OfflineAudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioBufferSourceNode, AudioScheduledSourceNode, OscillatorType};

// benchmark adapted from https://github.com/padenot/webaudio-benchmark
//
// run in release mode
// `cargo run --release --example benchmarks`

struct BenchResult<'a> {
    name: &'a str,
    duration: Duration,
    buffer: AudioBuffer,
}

fn load_buffer(sources: &mut Vec<AudioBuffer>, pathname: &str, sample_rate: u32) {
    let context = OfflineAudioContext::new(1, 1, SampleRate(sample_rate));

    let file = File::open(pathname).unwrap();
    let audio_buffer = context.decode_audio_data_sync(file).unwrap();

    sources.push(audio_buffer);
}

fn get_buffer(sources: &Vec<AudioBuffer>, sample_rate: u32, number_of_channels: usize) -> AudioBuffer {
    let buffer = sources.into_iter().find(|&buffer| {
        buffer.sample_rate_raw().0 == sample_rate && buffer.number_of_channels() == number_of_channels
    });

    buffer.unwrap().clone()
}

fn main() {
    let mut sources = Vec::<AudioBuffer>::new();
    let mut results = Vec::<BenchResult>::new();

    const DURATION: usize = 120;
    let sample_rate = SampleRate(48000);

    load_buffer(&mut sources, "samples/think-mono-38000.wav", 38000);
    load_buffer(&mut sources, "samples/think-mono-44100.wav", 44100);
    load_buffer(&mut sources, "samples/think-mono-48000.wav", 48000);
    load_buffer(&mut sources, "samples/think-stereo-38000.wav", 38000);
    load_buffer(&mut sources, "samples/think-stereo-44100.wav", 44100);
    load_buffer(&mut sources, "samples/think-stereo-48000.wav", 48000);

    {
        let name = "Baseline";
        let start = Instant::now();

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let buffer = context.start_rendering_sync();

        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }

    {
        let name = "Simple source test without resampling (Mono)";
        let start = Instant::now();

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        let buffer = context.start_rendering_sync();
        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }

    {
        let name = "Simple source test without resampling (Stereo)";
        let start = Instant::now();

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        let buffer = context.start_rendering_sync();
        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }

    {
        let name = "Simple source test with resampling (Mono)";
        let start = Instant::now();

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        let buffer = context.start_rendering_sync();
        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }

    {
        let name = "Simple source test with resampling (Stereo)";
        let start = Instant::now();

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        let buffer = context.start_rendering_sync();
        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }

    {
        let name = "Synth";
        let start = Instant::now();

        let sample_rate = SampleRate(44100);
        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let osc = context.create_oscillator();
            osc.connect(&env);
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);

            env.gain().set_value_at_time(0., 0.);
            env.gain().set_value_at_time(0.5, offset);
            env.gain().set_target_at_time(0., offset + 0.01, 0.1);
            osc.start_at(offset);
            osc.stop_at(offset + 1.); // why not 0.1 ?

            offset += 140. / 60. / 4.; // 140 bpm (?)
        }

        let buffer = context.start_rendering_sync();
        let duration = start.elapsed();
        let result = BenchResult { name, duration, buffer };
        results.push(result);
    }


    let stdout = stdout();
    let mut stdout = stdout.lock().into_raw_mode().unwrap();
    let stdin = stdin();
    let stdin = stdin.lock();

    write!(stdout, "\r\n").unwrap();

    write!(stdout,
        "{}results:{}\r\n",
        termion::style::Bold,
        termion::style::Reset,
    ).unwrap();

    // would be nice to draw a table
    write!(stdout, "\r\n").unwrap();

    write!(stdout,
        "> index | name | duration (ms) | Speedup vs. realtime \r\n",
    ).unwrap();

    for (index, result) in results.iter().enumerate() {
        write!(stdout,
            "- {} | {} \t| {} | {} \r\n",
            index + 1,
            result.name,
            result.duration.as_millis(),
            (DURATION as u128 * 1000) / result.duration.as_millis(),
        ).unwrap();
    }

    write!(stdout, "\r\n").unwrap();
    write!(stdout, "- Press 1, 2, 3, 4, to play output buffer for each test\r\n").unwrap();
    write!(stdout, "- Press \"q\" to quit\r\n").unwrap();
    write!(stdout, "- Press \"s\" to stop playback\r\n").unwrap();
    write!(stdout, "\r\n").unwrap();

    stdout.flush().unwrap();

    // preview results buffers
    let context = AudioContext::new(None);
    let mut current_source: Option<AudioBufferSourceNode> = None;

    let mut bytes = stdin.bytes();
    loop {
        let b = bytes.next().unwrap().unwrap();

        match b {
            // quit
            b'q' => return,
            // stop source
            b's' => {
                if let Some(source) = current_source {
                    current_source = None;
                    source.stop();
                }
            },
            // that's really dirty
            a => {
                let num = a - 49;
                if num >= 0 && num < 10 {
                    if let Some(source) = current_source {
                        current_source = None;
                        source.stop();
                    }

                    let result = &results[num as usize];

                    write!(stdout,
                        "> play outout from {:?}\r\n", result.name,
                    ).unwrap();
                    let buffer = result.buffer.clone();
                    let source = context.create_buffer_source();
                    source.set_buffer(buffer);
                    source.connect(&context.destination());
                    source.start();

                    current_source = Some(source);
                }
            },
        }

        stdout.flush().unwrap();
    }
}
