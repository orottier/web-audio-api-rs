use rand::Rng;
use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::time::{Duration, Instant};

use termion::clear;
use termion::cursor;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

use web_audio_api::context::{AudioContext, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{
    AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode, OscillatorType,
};
use web_audio_api::AudioBuffer;

// benchmark adapted from https://github.com/padenot/webaudio-benchmark
// missing the "Convolution Reverb" as we don't have the node implemented yet
//
// run in release mode
// `cargo run --release --example benchmarks`

struct BenchResult<'a> {
    name: &'a str,
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

fn benchmark<'a>(
    stdout: &mut termion::raw::RawTerminal<std::io::Stdout>,
    name: &'a str,
    context: &mut OfflineAudioContext,
    results: &mut Vec<BenchResult<'a>>,
) {
    write!(
        stdout,
        "{}{}> Running benchmark: {}",
        clear::CurrentLine,
        cursor::Left(200),
        name
    )
    .unwrap();
    stdout.flush().unwrap();

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
    let sample_rate = 48000.;

    load_buffer(&mut sources, "samples/think-mono-38000.wav", 38000.);
    load_buffer(&mut sources, "samples/think-mono-44100.wav", 44100.);
    load_buffer(&mut sources, "samples/think-mono-48000.wav", 48000.);
    load_buffer(&mut sources, "samples/think-stereo-38000.wav", 38000.);
    load_buffer(&mut sources, "samples/think-stereo-44100.wav", 44100.);
    load_buffer(&mut sources, "samples/think-stereo-48000.wav", 48000.);

    let mut stdout = stdout().into_raw_mode().unwrap();

    // -------------------------------------------------------
    // benchamarks
    // -------------------------------------------------------
    write!(stdout, "\r\n").unwrap();

    {
        let name = "Baseline (silence)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Mono)";

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Stereo)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test without resampling (Stereo and positionnal)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_panner();
        panner.connect(&context.destination());
        panner.position_x().set_value(1.);
        panner.position_y().set_value(2.);
        panner.position_z().set_value(3.);
        panner.orientation_x().set_value(1.);
        panner.orientation_y().set_value(2.);
        panner.orientation_z().set_value(3.);

        let source = context.create_buffer_source();
        source.connect(&panner);

        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Mono)";

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000., 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Stereo)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, 38000., 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple source test with resampling (Stereo and positionnal)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_panner();
        panner.connect(&context.destination());
        panner.position_x().set_value(1.);
        panner.position_y().set_value(2.);
        panner.position_z().set_value(3.);
        panner.orientation_x().set_value(1.);
        panner.orientation_y().set_value(2.);
        panner.orientation_z().set_value(3.);

        let source = context.create_buffer_source();
        source.connect(&panner);

        let buf = get_buffer(&sources, 38000., 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Upmix without resampling (Mono -> Stereo)";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 1);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Downmix without resampling (Stereo -> Mono)";

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let source = context.create_buffer_source();
        let buf = get_buffer(&sources, sample_rate, 2);
        source.set_buffer(buf);
        source.set_loop(true);
        source.connect(&context.destination());
        source.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple mixing (100x same buffer) - be careful w/ volume here!";

        let adjusted_duration = DURATION / 4;
        let mut context =
            OfflineAudioContext::new(1, adjusted_duration * sample_rate as usize, sample_rate);

        for _ in 0..100 {
            let source = context.create_buffer_source();
            let buf = get_buffer(&sources, 38000., 1);
            source.set_buffer(buf);
            source.set_loop(true);
            source.connect(&context.destination());
            source.start();
        }

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple mixing (100 different buffers) - be careful w/ volume here!";

        let adjusted_duration = DURATION / 4;
        let mut context =
            OfflineAudioContext::new(1, adjusted_duration * sample_rate as usize, sample_rate);
        let reference = get_buffer(&sources, 38000., 1);
        let channel_data = reference.get_channel_data(0);

        for _ in 0..100 {
            let mut buffer = context.create_buffer(1, reference.length(), 38000.);
            buffer.copy_to_channel(channel_data, 0);

            let source = context.create_buffer_source();
            source.set_buffer(buffer);
            source.set_loop(true);
            source.connect(&context.destination());
            source.start();
        }

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Simple mixing with gains";

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);

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

            let source = context.create_buffer_source();
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

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Granular synthesis";

        let adjusted_duration = DURATION as f64 / 16.;
        let length = (adjusted_duration * sample_rate as f64) as usize;
        let mut context = OfflineAudioContext::new(1, length, sample_rate);
        let buffer = get_buffer(&sources, sample_rate, 1);
        let mut offset = 0.;
        let mut rng = rand::thread_rng();

        // @todo - make a PR
        // - problem w/ env.gain().set_value_at_time(0., offset);
        // - variables are badly named, but just follow the source here

        // this 1500 sources...
        while offset < adjusted_duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let src = context.create_buffer_source();
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

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Synth (Sawtooth with Envelope)";

        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
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

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Synth (Sawtooth with gain - no automation)";

        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let duration = DURATION as f64;

        while offset < duration {
            let env = context.create_gain();
            env.connect(&context.destination());

            let osc = context.create_oscillator();
            osc.connect(&env);
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(110.);
            osc.start_at(offset);
            osc.stop_at(offset + 1.); // why not 0.1 ?

            offset += 140. / 60. / 4.; // 140 bpm (?)
        }

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Synth (Sawtooth without gain)";

        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
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

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Substractive Synth";

        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);
        let mut offset = 0.;

        let filter = context.create_biquad_filter();
        filter.connect(&context.destination());
        filter.frequency().set_value_at_time(0., 0.);
        filter.q().set_value_at_time(20., 0.);

        let env = context.create_gain();
        env.connect(&filter);
        env.gain().set_value_at_time(0., 0.);

        let osc = context.create_oscillator();
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

            offset += 140. / 60. / 16.; // 140 bpm (?)
        }

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Stereo panning";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_stereo_panner();
        panner.connect(&context.destination());
        panner.pan().set_value(0.1);

        let src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&panner);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Stereo panning with automation";

        let mut context = OfflineAudioContext::new(2, DURATION * sample_rate as usize, sample_rate);

        let panner = context.create_stereo_panner();
        panner.connect(&context.destination());
        panner.pan().set_value_at_time(-1., 0.);
        panner.pan().set_value_at_time(0.2, 0.5);

        let src = context.create_buffer_source();
        let buffer = get_buffer(&sources, sample_rate, 2);
        src.connect(&panner);
        src.set_buffer(buffer);
        src.set_loop(true);
        src.start();

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    {
        let name = "Sawtooth with automation";

        let mut context = OfflineAudioContext::new(1, DURATION * sample_rate as usize, sample_rate);

        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(2000.);
        osc.frequency().linear_ramp_to_value_at_time(20., 10.);
        osc.start_at(0.);

        benchmark(&mut stdout, name, &mut context, &mut results);
    }

    write!(
        stdout,
        "{}{}> All done!\r\n\r\n",
        clear::CurrentLine,
        cursor::Left(200),
    )
    .unwrap();
    stdout.flush().unwrap();

    // -------------------------------------------------------
    // display results
    // -------------------------------------------------------
    let stdin = stdin();
    let stdin = stdin.lock();

    write!(
        stdout,
        "{}+ id {}{}| name {}{}| duration (ms) {}{}| Speedup vs. realtime {}{}| buffer.duration (s) {}\r\n",
        termion::style::Bold,
        cursor::Left(200),
        cursor::Right(10),
        cursor::Left(200),
        cursor::Right(85),
        cursor::Left(200),
        cursor::Right(85 + 16),
        cursor::Left(200),
        cursor::Right(85 + 40),
        termion::style::Reset,
    )
    .unwrap();

    for (index, result) in results.iter().enumerate() {
        write!(
            stdout,
            "- {} {}{}| {} {}{}| {} {}{}| {:.1}x {}{}| {}\r\n",
            index + 1,
            cursor::Left(200),
            cursor::Right(10),
            result.name,
            cursor::Left(200),
            cursor::Right(85),
            result.duration.as_millis(),
            cursor::Left(200),
            cursor::Right(85 + 16),
            (result.buffer.duration() * 1000.) / result.duration.as_millis() as f64,
            cursor::Left(200),
            cursor::Right(85 + 40),
            result.buffer.duration(),
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
    let context = AudioContext::default();
    let mut current_source: Option<AudioBufferSourceNode> = None;
    let mut inputs = vec![];

    for c in stdin.keys() {
        match c.unwrap() {
            Key::Char('q') | Key::Ctrl('c') => {
                write!(stdout, "\n\r\n\r").unwrap();
                stdout.flush().unwrap();
                return;
            }
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
                    )
                    .unwrap();
                }
            }
            Key::Char(c) => {
                if c.is_ascii_digit() {
                    inputs.push(c);
                    write!(stdout, "{}", c).unwrap();
                }
            }
            Key::Backspace => {
                if !inputs.is_empty() {
                    if let Some(source) = current_source {
                        source.stop();
                        current_source = None;
                    }

                    write!(stdout, "{}{}", clear::CurrentLine, cursor::Left(200)).unwrap();

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
                        )
                        .unwrap();
                    } else {
                        write!(
                            stdout,
                            "{}{}> undefined id \"{}\"{}{}",
                            cursor::Down(1),
                            clear::CurrentLine,
                            id,
                            cursor::Left(200),
                            cursor::Up(1),
                        )
                        .unwrap();
                    }
                }
            }
            _ => {}
        }
        stdout.flush().unwrap();
    }
}
