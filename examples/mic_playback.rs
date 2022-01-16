// run in release mode
// `cargo run --release --example mic_playback`

use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AsBaseAudioContext, AudioContext, AudioContextRegistration};
use web_audio_api::media::Microphone;
use web_audio_api::node::BiquadFilterType;
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, ChannelConfig, ChannelConfigOptions,
};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use web_audio_api::SampleRate;

use std::io::{stdin, stdout, Write};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;

use crossbeam_channel::{self, Receiver, Sender};

use simplelog::{Config, LevelFilter, WriteLogger};

use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

struct MediaRecorder {
    /// handle to the audio context, required for all audio nodes
    registration: AudioContextRegistration,
    /// channel configuration (for up/down-mixing of inputs), required for all audio nodes
    channel_config: ChannelConfig,
    /// receiving end for the samples recorded in the render thread
    receiver: Receiver<Vec<Vec<f32>>>,
}

// implement required methods for AudioNode trait
impl AudioNode for MediaRecorder {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }
    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn number_of_inputs(&self) -> u32 {
        1
    }
    fn number_of_outputs(&self) -> u32 {
        0
    }
}

impl MediaRecorder {
    /// Construct a new MediaRecorder
    fn new<C: AsBaseAudioContext>(context: &C) -> Self {
        context.base().register(move |registration| {
            let (sender, receiver) = crossbeam_channel::unbounded();

            // setup the processor, this will run in the render thread
            let render = MediaRecorderProcessor { sender };

            // setup the audio node, this will live in the control thread (user facing)
            let node = MediaRecorder {
                registration,
                channel_config: ChannelConfigOptions::default().into(),
                receiver,
            };

            (node, Box::new(render))
        })
    }

    fn get_data(self, sample_rate: SampleRate) -> AudioBuffer {
        let data = self
            .receiver
            .try_iter()
            .reduce(|mut accum, item| {
                accum.iter_mut().zip(item).for_each(|(a, i)| a.extend(i));
                accum
            })
            .unwrap();

        AudioBuffer::from(data, sample_rate)
    }
}

struct MediaRecorderProcessor {
    sender: Sender<Vec<Vec<f32>>>,
}

impl AudioProcessor for MediaRecorderProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input node
        let input = &inputs[0];
        let data = input.channels().iter().map(|c| c.to_vec()).collect();

        let _ = self.sender.send(data);

        false // no tail time
    }
}

// This struct is used as an adaptor, it implements std::io::Write and forwards the buffer to a mpsc::Sender
struct WriteAdapter {
    buffer: Vec<u8>,
    sender: Sender<String>,
}

impl std::io::Write for WriteAdapter {
    // On write we forward each u8 of the buffer to the sender and return the length of the buffer
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut other = self.buffer.split_off(pos + 1);
            std::mem::swap(&mut self.buffer, &mut other);
            let string = String::from_utf8(other).unwrap();
            self.sender.send(string).unwrap();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn print_header(
    stdout: &mut termion::raw::RawTerminal<std::io::Stdout>,
    height: u16,
    width: u16,
    s: &str,
) {
    let char_count = s.chars().count() as u16; // assume ascii
    let pad_count = width / 2 - char_count / 2 - 2;
    let pad = "=".repeat(pad_count as usize);
    write!(
        stdout,
        "{}{}  {}  {}",
        termion::cursor::Goto(1, height),
        &pad,
        s,
        &pad
    )
    .unwrap();
}

fn print_static_ui(
    stdout: &mut termion::raw::RawTerminal<std::io::Stdout>,
    width: u16,
    height: u16,
) {
    // clear screen and hide cursor
    write!(stdout, "{}{}", termion::clear::All, termion::cursor::Hide).unwrap();

    print_header(stdout, 1, width, "Controls");
    write!(stdout, "{}Press q to exit", termion::cursor::Goto(1, 2)).unwrap();
    write!(
        stdout,
        "{}Use + and - for recorded output volume",
        termion::cursor::Goto(1, 3)
    )
    .unwrap();
    write!(
        stdout,
        "{}Use n and m to adjust playback speed",
        termion::cursor::Goto(1, 4)
    )
    .unwrap();
    write!(
        stdout,
        "{}Use x to cycle through output effects",
        termion::cursor::Goto(1, 5)
    )
    .unwrap();

    print_header(stdout, 10, width, "Frequency response");
    print_header(stdout, height - 11, width, "Logs");
}

fn print_logs(
    stdout: &mut termion::raw::RawTerminal<std::io::Stdout>,
    logs: &[String],
    offset: u16,
) {
    let offset = offset + 10 - logs.len() as u16;
    // Collect all messages send to the channel and parse the result as a string
    logs.iter().enumerate().for_each(|(i, msg)| {
        write!(
            stdout,
            "{}{}{}",
            termion::cursor::Goto(1, offset + i as u16),
            termion::clear::CurrentLine,
            msg
        )
        .unwrap();
    })
}

fn draw_plot(stdout: &mut termion::raw::RawTerminal<std::io::Stdout>, plot: String, offset: u16) {
    plot.split('\n').enumerate().for_each(|(i, l)| {
        write!(
            stdout,
            "{}{}",
            termion::cursor::Goto(1, offset + i as u16),
            l
        )
        .unwrap()
    });
}

fn audio_thread(
    gain_factor: Arc<AtomicI32>,
    playback_rate_factor: Arc<AtomicI32>,
    biquad_filter: Arc<AtomicU32>,
    width: u16,
    height: u16,
    plot_send: Sender<String>,
) {
    let context = AudioContext::new(None);

    let stream = Microphone::new();
    let mic_in = context.create_media_stream_source(stream);

    let analyser = Arc::new(context.create_analyser());
    let analyser_clone = analyser.clone();
    let mut freq_buffer = Some(vec![0.; analyser.frequency_bin_count()]);
    // spawn thread to poll analyser updates
    std::thread::spawn(move || {
        use textplots::{Chart, Plot, Shape};
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let tmp_buf = freq_buffer.take().unwrap();
            freq_buffer = Some(analyser_clone.get_float_frequency_data(tmp_buf));
            let points: Vec<_> = freq_buffer
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, &f)| (i as f32, f))
                .collect();
            let plot = Chart::new_with_y_range(
                width as u32 * 2,
                (height - 25) as u32 * 4,
                0.0,
                analyser_clone.frequency_bin_count() as f32,
                -80.,
                20.,
            )
            .lineplot(&Shape::Bars(&points[..]))
            .to_string();
            plot_send.send(plot).unwrap();
        }
    });

    let playback_step = 1.05_f32; // playback increases by 5% per setting
    let gain_step = 1.1_f32; // gain increases by 10% per setting

    loop {
        log::info!("Start recording - 4 seconds");
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        osc.disconnect_all();

        let recorder = MediaRecorder::new(&context);
        mic_in.connect(&recorder);
        mic_in.connect(&*analyser);
        std::thread::sleep(std::time::Duration::from_millis(4000));
        mic_in.disconnect_all();

        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.start();
        let buf = recorder.get_data(context.sample_rate_raw());

        std::thread::sleep(std::time::Duration::from_millis(200));
        osc.disconnect_all();

        log::info!("Playback recording");

        let src = context.create_buffer_source();
        let playback_rate = playback_step.powi(playback_rate_factor.load(Ordering::Relaxed));
        src.playback_rate().set_value(playback_rate);
        src.set_buffer(buf);

        let biquad = context.create_biquad_filter();
        biquad.set_type((biquad_filter.load(Ordering::Relaxed) % 8).into());

        let gain = context.create_gain();
        let gain_value = gain_step.powi(gain_factor.load(Ordering::Relaxed));
        gain.gain().set_value(gain_value);

        src.connect(&biquad);
        biquad.connect(&gain);
        gain.connect(&context.destination());
        gain.connect(&*analyser);

        src.start();
        let duration = (4. / playback_rate * 1000.) as u64; // millis
        std::thread::sleep(std::time::Duration::from_millis(duration));
    }
}

fn main() {
    let (width, height) = termion::terminal_size().unwrap();

    /*
     * First, setup logging facility. Use a WriteLogger to catch the log entries and ship them to
     * the UI. We will render the log lines at the bottom of the interface.
     */
    let (sender, receiver) = crossbeam_channel::unbounded();
    let target = WriteAdapter {
        buffer: vec![],
        sender,
    };
    WriteLogger::init(LevelFilter::Debug, Config::default(), target).unwrap();
    log::info!("Running the microphone playback example");

    /*
     * Create thread safe user controllable variables
     */
    let gain_factor = Arc::new(AtomicI32::new(0));
    let playback_rate_factor = Arc::new(AtomicI32::new(0));
    let biquad_filter = Arc::new(AtomicU32::new(7)); // allpass

    /*
     * Then, setup the audio loop in a separate thread. It will record audio for a few seconds,
     * then play it back to the speakers with some transformations applied. Repeat.
     */
    let gain_factor_clone = gain_factor.clone();
    let playback_rate_factor_clone = playback_rate_factor.clone();
    let biquad_filter_clone = biquad_filter.clone();
    let (plot_send, plot_recv) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        audio_thread(
            gain_factor_clone,
            playback_rate_factor_clone,
            biquad_filter_clone,
            width,
            height,
            plot_send,
        )
    });

    /*
     * Spawn another thread to handle user input. It reads key presses and uses message passing to
     * send them back to the main UI thread.
     */
    let (stdin_send, stdin_recv) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        stdin().keys().for_each(|c| {
            let _ = stdin_send.send(c);
        })
    });

    /*
     * Now, the `main` thread will serve as the UI thread. It will respond to key presses from the
     * 'user input thread', and log messages from the log WriteAdapter.
     */
    let mut stdout = stdout().into_raw_mode().unwrap();
    print_static_ui(&mut stdout, width, height);

    let log_offset = height - 10;

    let mut logs = vec![];
    'outer: loop {
        // collect logs
        let prev_log_len = logs.len();
        logs.extend(receiver.try_iter());
        if logs.len() > prev_log_len {
            while logs.len() > 10 {
                logs.remove(0);
            }
            print_logs(&mut stdout, &logs, log_offset);
            stdout.flush().unwrap();
        }

        if let Ok(plot) = plot_recv.try_recv() {
            draw_plot(&mut stdout, plot, 11);
        }

        for c in stdin_recv.try_iter() {
            match c.unwrap() {
                Key::Char('q') => break 'outer,
                Key::Char('+') => {
                    let prev = gain_factor.fetch_add(1, Ordering::Relaxed);
                    log::info!("Volume: {:+}", prev + 1);
                }
                Key::Char('-') => {
                    let prev = gain_factor.fetch_add(-1, Ordering::Relaxed);
                    log::info!("Volume: {:+}", prev - 1);
                }
                Key::Char('m') => {
                    let prev = playback_rate_factor.fetch_add(1, Ordering::Relaxed);
                    log::info!("Playback speed: {:+}", prev + 1);
                }
                Key::Char('n') => {
                    let prev = playback_rate_factor.fetch_add(-1, Ordering::Relaxed);
                    log::info!("Playback speed: {:+}", prev - 1);
                }
                Key::Char('x') => {
                    let prev = biquad_filter.fetch_add(1, Ordering::Relaxed);
                    let biquad_filter_type = BiquadFilterType::from((prev + 1) % 8);
                    log::info!("Filter type now: {:?}", biquad_filter_type);
                }
                Key::Char(c) => log::debug!("Unknown input - {}", c),
                _ => {}
            }
        }
    }

    // restore cursor
    write!(stdout, "{}", termion::cursor::Show).unwrap();
}
