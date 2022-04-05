const DATA_SIZE: usize = 512;

use rand::Rng;
use std::error::Error;
use std::net::UdpSocket;
use std::time::Duration;

use web_audio_api::buffer::AudioBuffer;
use web_audio_api::buffer::AudioBufferOptions;
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media::Microphone;
use web_audio_api::node::AudioNode;
use web_audio_api::SampleRate;
use web_audio_api::RENDER_QUANTUM_SIZE;

const SERVER_ADDR: &str = "0.0.0.0:1234";
const CLIENT_ADDR: &str = "0.0.0.0:5555";

fn main() -> std::io::Result<()> {
    let mut args = std::env::args();
    args.next(); // program name
    match args
        .next()
        .expect("Role argument is required: server or client")
        .as_ref()
    {
        "server" => run_server(),
        "client" => run_client(),
        _ => panic!("Argument must be server or client"),
    }
}

/// The server is just an echo client, with some chaos introduced
fn run_server() -> std::io::Result<()> {
    let socket = UdpSocket::bind(SERVER_ADDR)?;
    println!("Server listening at {}", socket.local_addr()?);

    let mut buf = [0; DATA_SIZE];
    loop {
        let (amt, src) = socket.recv_from(&mut buf)?;
        let buf = &mut buf[..amt];

        let mut rng = rand::thread_rng();

        // introduce chaos
        match rng.gen_range(0..100) {
            0 => continue,                                      // 1% packet loss
            1 => std::thread::sleep(Duration::from_millis(50)), // 1% of packets has 50ms delay
            _ => (),
        };

        socket.send_to(buf, &src)?;
    }
}

fn serialize(audio_buf: &AudioBuffer, byte_buf: &mut [u8]) -> usize {
    todo!();
}

fn deserialize(byte_buf: &[u8]) -> AudioBuffer {
    todo!();
}

struct SocketStream {
    socket: &'static UdpSocket,
    byte_buf: Vec<u8>,
}

impl Iterator for SocketStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = match self.socket.recv(&mut self.byte_buf) {
            Ok(received) => &self.byte_buf[..received],
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // construct empty buffer
                let options = AudioBufferOptions {
                    number_of_channels: 1,
                    length: RENDER_QUANTUM_SIZE,
                    sample_rate: SampleRate(48000),
                };
                return Some(Ok(AudioBuffer::new(options)));
            }
            Err(e) => panic!("client recv IO error: {}", e),
        };
        Some(Ok(deserialize(bytes)))
    }
}

/// The client records their audio, ships it to the server and plays the received samples
fn run_client() -> std::io::Result<()> {
    let socket = UdpSocket::bind(CLIENT_ADDR)?;
    socket.set_nonblocking(true).unwrap();
    println!("Client listening at {}", socket.local_addr()?);
    socket.connect(SERVER_ADDR)?;
    println!("Client 'connected' to {}", SERVER_ADDR);

    // hack, make socket static to avoid `Arc` or similar
    let socket: &'static UdpSocket = Box::leak(Box::new(socket));

    // setup the audio context
    let context = AudioContext::default();

    // leg 1: receive server packets and play them
    let stream = SocketStream {
        socket,
        byte_buf: vec![0; 512],
    };
    let stream_in = context.create_media_stream_source(stream);
    stream_in.connect(&context.destination());

    // leg 2: record mic input and ship to server
    let mic = Microphone::new();
    let stream_in = context.create_media_stream_source(mic);
    let stream_out = context.create_media_stream_destination();
    stream_out.set_channel_count(1); // force mono
    stream_in.connect(&stream_out);

    let mut byte_buf = vec![0; 512];
    for item in stream_out.stream() {
        let buf = item.unwrap();
        let size = serialize(&buf, &mut byte_buf[..]);
        loop {
            match socket.send(&byte_buf[..size]) {
                Ok(_) => break,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => panic!("client send IO error: {}", e),
            }
        }
    }

    unreachable!()
}
