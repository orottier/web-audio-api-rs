use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::time::{Duration, Instant};

use termion::event::Key;
use termion::input::TermRead;
use termion::cursor;
use termion::clear;
use termion::raw::IntoRawMode;

use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AudioContext, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{
    AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode, OscillatorType,
};
use web_audio_api::SampleRate;

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

fn get_buffer(sources: &[AudioBuffer], sample_rate: u32, number_of_channels: usize) -> AudioBuffer {
    let buffer = sources.iter().find(|&buffer| {
        buffer.sample_rate_raw().0 == sample_rate
            && buffer.number_of_channels() == number_of_channels
    });

    buffer.unwrap().clone()
}

fn benchmark<'a>(
    name: &'a str,
    context: &mut OfflineAudioContext,
    results: &mut Vec<BenchResult<'a>>,
) {
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

    let stdout = stdout();
    let mut stdout = stdout.lock().into_raw_mode().unwrap();

    // -------------------------------------------------------
    // benchamarks
    // -------------------------------------------------------
    write!(stdout, "\r\n> Running benchmarks ").unwrap();

    {
        let name = "Baseline (silence)";

        let mut context =
            OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Simple source test without resampling (Mono)";

        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Simple source test without resampling (Stereo)";

        let mut context =
            OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Simple source test with resampling (Mono)";

        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Simple source test with resampling (Stereo)";

        let mut context =
            OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Upmix without resampling (Mono -> Stereo)";

        let mut context =
            OfflineAudioContext::new(2, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Downmix without resampling (Stereo -> Mono)";

        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate.0, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Synth (Sawtooth with Envelope)";

        let sample_rate = SampleRate(44100);
        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
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

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Synth (Sawtooth without Envelope)";

        let sample_rate = SampleRate(44100);
        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let osc = context.create_oscillator();
            osc.connect(&context.destination());
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);
            osc.start_at(offset);
            osc.stop_at(offset + 1.); // why not 0.1 ?

            offset += 140. / 60. / 4.; // 140 bpm (?)
        }

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    {
        let name = "Sawtooth with automation";

        let mut context =
            OfflineAudioContext::new(1, DURATION * sample_rate.0 as usize, sample_rate);

        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(2000.);
        osc.frequency().linear_ramp_to_value_at_time(20., 10.);
        osc.start_at(0.);

        benchmark(name, &mut context, &mut results);

        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
    }

    // -------------------------------------------------------
    // display results
    // -------------------------------------------------------
    let stdin = stdin();
    let stdin = stdin.lock();

    write!(stdout, "\r\n\r\n").unwrap();

    write!(
        stdout,
        "{}+ id {}{}| name {}{}| duration (ms) {}{}| Speedup vs. realtime {}\r\n",
        termion::style::Bold,
        cursor::Left(200),
        cursor::Right(10),
        cursor::Left(200),
        cursor::Right(65),
        cursor::Left(200),
        cursor::Right(65 + 16),
        termion::style::Reset,
    )
    .unwrap();

    for (index, result) in results.iter().enumerate() {
        write!(
            stdout,
            "- {} {}{}| {} {}{}| {} {}{}| {} \r\n",
            index + 1,
            cursor::Left(200),
            cursor::Right(10),
            result.name,
            cursor::Left(200),
            cursor::Right(65),
            result.duration.as_millis(),
            cursor::Left(200),
            cursor::Right(65 + 16),
            (DURATION as u128 * 1000) / result.duration.as_millis(),
        )
        .unwrap();
    }

    write!(stdout, "\r\n").unwrap();
    // @todo - this needs to be reviwed can only play 9 first buffers...
    write!(stdout, "+ Press \"q\" or \"ctrl + c\" to quit\r\n").unwrap();
    write!(stdout, "\r\n").unwrap();
    write!(
        stdout,
        "+ Type the id of the result you want to listen and press \"backspace\"\r\n"
    )
    .unwrap();
    write!(stdout, "+ Press \"s\" to stop playback\r\n").unwrap();
    write!(stdout, "\r\n").unwrap();

    stdout.flush().unwrap();

    // -------------------------------------------------------
    // handle input and preview
    // -------------------------------------------------------
    let context = AudioContext::new(None);
    let mut current_source: Option<AudioBufferSourceNode> = None;
    let mut inputs = vec![];

    for c in stdin.keys() {
        match c.unwrap() {
            Key::Char('q') |
            Key::Ctrl('c') => {
                write!(stdout, "\n\r").unwrap();
                stdout.flush().unwrap();
                return
            },
            Key::Char('s') => {
                if let Some(source) = current_source {
                    source.stop();
                    current_source = None;

                    write!(
                        stdout,
                        "{}{}{}",
                        cursor::Down(1),
                        clear::CurrentLine,
                        cursor::Up(1),
                    ).unwrap();
                }
            },
            Key::Char(c) => {
                if c.is_digit(10) {
                    inputs.push(c);
                    write!(stdout, "{}", c).unwrap();
                }
            },
            Key::Backspace => {
                if let Some(source) = current_source {
                    source.stop();
                    current_source = None;
                }

                write!(
                    stdout,
                    "{}{}",
                    clear::CurrentLine,
                    cursor::Left(200),
                ).unwrap();

                let id_str: String = inputs.clone().into_iter().collect();
                let id = id_str.parse::<usize>().unwrap();
                let index = id - 1;

                inputs.clear();

                if id - 1 < results.len() {
                    let result = &results[index];

                    let buffer = result.buffer.clone();
                    let source = context.create_buffer_source();
                    source.set_buffer(buffer);
                    source.connect(&context.destination());
                    source.start();

                    current_source = Some(source);

                    write!(
                        stdout,
                        "{}{}> playing outout from {}{}{}",
                        cursor::Down(1),
                        clear::CurrentLine,
                        result.name,
                        cursor::Left(200),
                        cursor::Up(1),
                    ).unwrap();
                } else {
                    write!(
                        stdout,
                        "{}{}> undefined id \"{}\"{}{}",
                        cursor::Down(1),
                        clear::CurrentLine,
                        id,
                        cursor::Left(200),
                        cursor::Up(1),
                    ).unwrap();
                }
            },
            _ => {}
        }
        stdout.flush().unwrap();
    }
}
