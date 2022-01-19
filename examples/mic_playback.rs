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

/// Instruction box height
const INFO_PANEL_HEIGHT: u16 = 9;
/// The number of log lines to show at the bottom of the screen
const LOG_LEN: usize = 10;

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
    sender: Sender<UiEvent>,
}

impl std::io::Write for WriteAdapter {
    // On write we forward each u8 of the buffer to the sender and return the length of the buffer
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut other = self.buffer.split_off(pos + 1);
            std::mem::swap(&mut self.buffer, &mut other);
            let string = String::from_utf8(other).unwrap();
            let event = UiEvent::LogMessage(string);
            let _ = self.sender.send(event); // allowed to fail if the main thread is shutting down
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

    print_header(stdout, INFO_PANEL_HEIGHT + 1, width, "Frequency response");
    print_header(stdout, height - 1 - LOG_LEN as u16, width, "Logs");
}

fn print_logs(
    stdout: &mut termion::raw::RawTerminal<std::io::Stdout>,
    logs: &[String],
    offset: u16,
) {
    let offset = offset + LOG_LEN as u16 - logs.len() as u16;
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
    plot_send: Sender<UiEvent>,
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
            let event = UiEvent::GraphUpdate(plot);
            let _ = plot_send.send(event); // allowed to fail if the main thread is shutting down
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

enum UiEvent {
    UserInput(termion::event::Key),
    LogMessage(String),
    GraphUpdate(String),
}

fn main() {
    let (width, height) = termion::terminal_size().unwrap();

    /*
     * The UI is drawn on the main thread. We will spawn other threads to
     * - collect user input
     * - handle the audio graph
     * - submit logs.
     *
     * All communication is sent over a single channel, which has multiple senders and one
     * receiver.
     */
    let (ui_event_sender, ui_event_receiver) = crossbeam_channel::unbounded();

    /*
     * First, setup logging facility. Use a WriteLogger to catch the log entries and ship them to
     * the UI. We will render the log lines at the bottom of the interface.
     */
    let target = WriteAdapter {
        buffer: vec![],
        sender: ui_event_sender.clone(),
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
    {
        // create clones in a new { }-block and shadow original bindings
        let gain_factor = gain_factor.clone();
        let playback_rate_factor = playback_rate_factor.clone();
        let biquad_filter = biquad_filter.clone();
        let ui_event_sender = ui_event_sender.clone();

        std::thread::spawn(move || {
            audio_thread(
                gain_factor,
                playback_rate_factor,
                biquad_filter,
                width,
                height,
                ui_event_sender,
            )
        });
    }

    /*
     * Spawn another thread to handle user input. It reads key presses and uses message passing to
     * send them back to the main UI thread.
     */
    let key_sender = ui_event_sender.clone();
    std::thread::spawn(move || {
        stdin().keys().for_each(|key| {
            let event = UiEvent::UserInput(key.unwrap());
            let _ = key_sender.send(event); // allowed to fail if the main thread is shutting down
        })
    });

    /*
     * Now, the `main` thread will serve as the UI thread. It will respond to all the UI events
     * that are sent from the various threads we have set up above
     */
    let mut stdout = stdout().into_raw_mode().unwrap();
    print_static_ui(&mut stdout, width, height); // print headers, usage instructions

    // the log section is put at the bottom 10 lines of the terminal
    let log_offset = height - LOG_LEN as u16;
    let mut logs = Vec::with_capacity(LOG_LEN);

    for ui_event in ui_event_receiver.iter() {
        match ui_event {
            UiEvent::LogMessage(msg) => {
                logs.push(msg);
                if logs.len() > LOG_LEN {
                    logs.remove(0);
                }
                print_logs(&mut stdout, &logs, log_offset);
            }
            UiEvent::GraphUpdate(plot) => {
                draw_plot(&mut stdout, plot, INFO_PANEL_HEIGHT + 2);
            }
            UiEvent::UserInput(c) => {
                match c {
                    Key::Char('q') => break, // halt UI loop
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
                    _ => {} // ignore backspace, arrows, etc
                }
            }
        }

        // force screen update
        stdout.flush().unwrap();
    }

    // shutting down, restore cursor
    write!(stdout, "{}", termion::cursor::Show).unwrap();
}
