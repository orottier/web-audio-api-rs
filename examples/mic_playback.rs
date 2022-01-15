use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{AsBaseAudioContext, AudioContext, AudioContextRegistration};
use web_audio_api::media::Microphone;
use web_audio_api::node::BiquadFilterType;
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, ChannelConfig, ChannelConfigOptions,
};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use web_audio_api::SampleRate;

use crossbeam_channel::{self, Receiver, Sender};

use simplelog::{Config, LevelFilter, WriteLogger};
use std::io::{stdin, stdout, Write};
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
    sender: Sender<u8>,
}

impl std::io::Write for WriteAdapter {
    // On write we forward each u8 of the buffer to the sender and return the length of the buffer
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for chr in buf {
            self.sender.send(*chr).unwrap();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn print_logs(stdout: &mut termion::raw::RawTerminal<std::io::Stdout>, receiver: &Receiver<u8>) {
    // Collect all messages send to the channel and parse the result as a string
    let msg = String::from_utf8(receiver.try_iter().collect::<Vec<u8>>()).unwrap();
    if !msg.is_empty() {
        write!(
            stdout,
            "{}{}{}",
            termion::cursor::Goto(1, 10),
            termion::clear::CurrentLine,
            msg
        )
        .unwrap();
    }
}

fn main() {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let target = WriteAdapter { sender };
    WriteLogger::init(LevelFilter::Debug, Config::default(), target).unwrap();

    log::info!("Running the microphone playback example");

    // setup UI
    let stdin = stdin();
    let mut stdout = stdout().into_raw_mode().unwrap();

    write!(
        stdout,
        "{}{}q to exit. Type stuff, use alt, and so on.{}",
        termion::clear::All,
        termion::cursor::Goto(1, 1),
        termion::cursor::Hide
    )
    .unwrap();

    print_logs(&mut stdout, &receiver);

    stdout.flush().unwrap();

    for c in stdin.keys() {
        log::info!("got key {:?}", c);
        write!(
            stdout,
            "{}{}",
            termion::cursor::Goto(1, 1),
            termion::clear::CurrentLine
        )
        .unwrap();

        match c.unwrap() {
            Key::Char('q') => break,
            Key::Char(c) => println!("{}", c),
            Key::Alt(c) => println!("^{}", c),
            Key::Ctrl(c) => println!("*{}", c),
            Key::Esc => println!("ESC"),
            Key::Left => println!("←"),
            Key::Right => println!("→"),
            Key::Up => println!("↑"),
            Key::Down => println!("↓"),
            Key::Backspace => println!("×"),
            _ => {}
        }

        print_logs(&mut stdout, &receiver);

        stdout.flush().unwrap();
    }

    write!(stdout, "{}", termion::cursor::Show).unwrap();
    panic!("Done");

    let context = AudioContext::new(None);

    let stream = Microphone::new();
    // register as media element in the audio context
    let mic_in = context.create_media_stream_source(stream);

    loop {
        println!("beep - now recording");
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        osc.disconnect_all();

        let recorder = MediaRecorder::new(&context);
        mic_in.connect(&recorder);
        std::thread::sleep(std::time::Duration::from_millis(4000));
        mic_in.disconnect_all();

        println!("beep - end recording");
        let osc = context.create_oscillator();
        osc.connect(&context.destination());
        osc.start();
        let buf = recorder.get_data(context.sample_rate_raw());

        std::thread::sleep(std::time::Duration::from_millis(200));
        osc.disconnect_all();

        println!("playback buf - duration {:.2}", buf.duration());
        println!("applying LowPass and Gain");
        let src = context.create_buffer_source();
        src.set_buffer(buf);
        let biquad = context.create_biquad_filter();
        biquad.set_type(BiquadFilterType::Lowpass);
        let gain = context.create_gain();
        gain.gain().set_value(2.);
        src.connect(&biquad);
        biquad.connect(&gain);
        gain.connect(&context.destination());
        src.start();

        std::thread::sleep(std::time::Duration::from_millis(4000));
        println!("end playback");
    }
}
