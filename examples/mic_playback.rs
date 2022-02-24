// run in release mode
// `cargo run --release --example mic_playback`

use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AudioContext, AudioContextRegistration, BaseAudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::BiquadFilterType;
use web_audio_api::node::{
    AnalyserNode, AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode, BiquadFilterNode,
    ChannelConfig, ChannelConfigOptions, GainNode, MediaStreamAudioSourceNode,
};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use web_audio_api::SampleRate;

use std::io::{stdin, stdout, Write};
use std::sync::Arc;

use crossbeam_channel::{self, Receiver, Sender};

use simplelog::{Config, LevelFilter, WriteLogger};

use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

use textplots::{Chart, Plot, Shape};

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
    fn new<C: BaseAudioContext>(context: &C) -> Self {
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

    fn get_data(self, sample_rate: SampleRate) -> Option<AudioBuffer> {
        self.receiver
            .try_iter()
            .reduce(|mut accum, item| {
                accum.iter_mut().zip(item).for_each(|(a, i)| a.extend(i));
                accum
            })
            .map(|data| AudioBuffer::from(data, sample_rate))
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
        "{}Tap the spacebar to start and stop recording",
        termion::cursor::Goto(1, 3)
    )
    .unwrap();
    write!(
        stdout,
        "{}Use + and - for recorded output volume",
        termion::cursor::Goto(1, 4)
    )
    .unwrap();
    write!(
        stdout,
        "{}Use n and m to adjust playback speed",
        termion::cursor::Goto(1, 5)
    )
    .unwrap();
    write!(
        stdout,
        "{}Use x to cycle through output effects",
        termion::cursor::Goto(1, 6)
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

fn poll_frequency_graph(
    analyser: Arc<AnalyserNode>,
    plot_send: Sender<UiEvent>,
    width: u16,
    height: u16,
) -> ! {
    let bin_count = analyser.frequency_bin_count() as usize;
    let mut freq_buffer = Some(vec![0.; bin_count]);

    loop {
        // 5 frames per second
        std::thread::sleep(std::time::Duration::from_millis(200));

        // todo, check BaseAudioContext.state if it is still running

        let tmp_buf = freq_buffer.take().unwrap();
        let tmp_buf = analyser.get_float_frequency_data(tmp_buf);

        let points: Vec<_> = tmp_buf
            .iter()
            .enumerate()
            .map(|(i, &f)| (i as f32, f))
            .collect();

        let plot = Chart::new_with_y_range(
            width as u32 * 2,
            (height - 25) as u32 * 4,
            0.0,
            bin_count as f32,
            -80.,
            20.,
        )
        .lineplot(&Shape::Bars(&points[..]))
        .to_string();

        let event = UiEvent::GraphUpdate(plot);
        let _ = plot_send.send(event); // allowed to fail if the main thread is shutting down

        // restore Vec
        freq_buffer = Some(tmp_buf);
    }
}

struct AudioThread {
    context: AudioContext,
    mic_in: MediaStreamAudioSourceNode,
    analyser: Arc<AnalyserNode>,
    recorder: Option<MediaRecorder>,
    buffer_source: AudioBufferSourceNode,
    gain_node: Option<GainNode>,
    biquad_node: Option<BiquadFilterNode>,
    gain_factor: i32,
    playback_rate_factor: i32,
    biquad_filter_type: u32,
}

impl Drop for AudioThread {
    fn drop(&mut self) {
        self.context.close_sync();
    }
}

impl AudioThread {
    const PLAYBACK_STEP: f32 = 1.05; // playback increases by 5% per setting
    const GAIN_STEP: f32 = 1.1_f32; // gain increases by 10% per setting

    fn new() -> Self {
        let context = AudioContext::new(None);

        let stream = Microphone::new();
        let mic_in = context.create_media_stream_source(stream);

        let analyser = Arc::new(context.create_analyser());
        let buffer_source = context.create_buffer_source();

        let mut instance = Self {
            context,
            mic_in,
            analyser,
            recorder: None,
            buffer_source,
            gain_node: None,
            biquad_node: None,
            gain_factor: 0,
            playback_rate_factor: 0,
            biquad_filter_type: 7,
        };

        instance.playback(false); // start playing silence

        instance
    }

    fn analyser(&self) -> Arc<AnalyserNode> {
        self.analyser.clone()
    }

    fn add_volume(&mut self, diff: i32) {
        self.gain_factor += diff;
        log::info!("Microphone gain: {:+}", self.gain_factor);

        if let Some(gain) = &mut self.gain_node {
            let gain_value = Self::GAIN_STEP.powi(self.gain_factor);
            gain.gain().set_value(gain_value);
        }
    }

    fn add_playback_rate(&mut self, diff: i32) {
        self.playback_rate_factor += diff;
        let playback_rate = Self::PLAYBACK_STEP.powi(self.playback_rate_factor);
        self.buffer_source.playback_rate().set_value(playback_rate);
        log::info!("Playback rate: {:+}", self.playback_rate_factor);
    }

    fn update_biquad_filter(&mut self) {
        self.biquad_filter_type = (self.biquad_filter_type + 1) % 8;
        let name = BiquadFilterType::from(self.biquad_filter_type);
        log::info!("Biquad filter type: {:?}", name);

        if let Some(biquad) = &mut self.biquad_node {
            biquad.set_type((self.biquad_filter_type % 8).into());
        }
    }

    fn playback(&mut self, beep: bool) {
        // stop mic input
        self.mic_in.disconnect_all();

        if beep {
            let osc = self.context.create_oscillator();
            osc.connect(&self.context.destination());
            osc.start();
            osc.stop_at(self.context.current_time() + 0.2);

            log::info!("Playback audio - press space to stop");
        } else {
            log::info!("Press space to start recording!");
        }

        let buf = self
            .recorder
            .take()
            .and_then(|r| r.get_data(self.context.sample_rate_raw()));

        let buffer_source = self.context.create_buffer_source();
        let playback_rate = Self::PLAYBACK_STEP.powi(self.playback_rate_factor);
        buffer_source.playback_rate().set_value(playback_rate);
        buffer_source.set_loop(true);
        if let Some(buf) = buf {
            buffer_source.set_buffer(buf);
        }

        let biquad = self.context.create_biquad_filter();
        biquad.set_type((self.biquad_filter_type % 8).into());

        let gain = self.context.create_gain();
        let gain_value = Self::GAIN_STEP.powi(self.gain_factor);
        gain.gain().set_value(gain_value);

        buffer_source.connect(&biquad);
        biquad.connect(&gain);
        gain.connect(&self.context.destination());
        gain.connect(&*self.analyser);

        buffer_source.start();

        self.buffer_source = buffer_source;
        self.gain_node = Some(gain);
        self.biquad_node = Some(biquad);
    }

    fn record(&mut self) {
        self.buffer_source.stop();
        log::info!("Start recording - press space to stop");

        let osc = self.context.create_oscillator();
        osc.connect(&self.context.destination());
        osc.start();
        osc.stop_at(self.context.current_time() + 0.2);

        let recorder = MediaRecorder::new(&self.context);
        self.mic_in.connect(&recorder);
        self.mic_in.connect(&*self.analyser);

        self.recorder = Some(recorder);
    }

    fn switch_state(&mut self) {
        if self.recorder.is_some() {
            self.playback(true)
        } else {
            self.record()
        }
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
     * Then, setup the audio graph. It has two modes: record audio and playback audio.
     */
    let mut audio_thread = AudioThread::new();

    /*
     * Spawn thread to periodically poll analyser updates from the audio graph
     */
    {
        let plot_send = ui_event_sender.clone();
        let analyser = audio_thread.analyser();
        std::thread::spawn(move || poll_frequency_graph(analyser, plot_send, width, height));
    }

    /*
     * Spawn another thread to handle user input. It reads key presses and uses message passing to
     * send them back to the main UI thread.
     */
    std::thread::spawn(move || {
        stdin().keys().for_each(|key| {
            let event = UiEvent::UserInput(key.unwrap());
            let _ = ui_event_sender.send(event); // allowed to fail if the main thread is shutting down
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
                        audio_thread.add_volume(1);
                    }
                    Key::Char('-') => {
                        audio_thread.add_volume(-1);
                    }
                    Key::Char('m') => {
                        audio_thread.add_playback_rate(1);
                    }
                    Key::Char('n') => {
                        audio_thread.add_playback_rate(-1);
                    }
                    Key::Char('x') => {
                        audio_thread.update_biquad_filter();
                    }
                    Key::Char(' ') => {
                        audio_thread.switch_state();
                    }
                    Key::Char(c) => log::debug!("Unknown input - {}", c),
                    _ => {} // control chars etc
                }
            }
        }

        // force screen update
        stdout.flush().unwrap();
    }

    // shutting down, restore cursor
    write!(
        stdout,
        "{}{}",
        termion::cursor::Goto(1, height),
        termion::cursor::Show,
    )
    .unwrap();
}
