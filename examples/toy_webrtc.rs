const DATA_SIZE: usize = 512;

use std::convert::TryInto;
use std::error::Error;
use std::iter::IntoIterator;
use std::net::UdpSocket;

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::media_devices;
use web_audio_api::media_devices::MediaStreamConstraints;
use web_audio_api::media_streams::MediaStreamTrack;
use web_audio_api::node::AudioNode;
use web_audio_api::{AudioBuffer, AudioBufferOptions};

// This example features a toy VOIP app
//
// Run `cargo run --release --example toy_webrtc -- server` to run the echo server.
// The echo server simply receives the audio packets and ships them back.
//
// Run `cargo run --release --example toy_webrtc -- client` to run the VOIP client.
// The client starts recording audio, ships the data to the server and plays back what it receives.
//
// The client and server us an UDP connection for low latency audio playback over the network.
// Make sure you either run the server and client on the same machine, or within your local
// network, because any firewall may block the packets (no NAT traversal / UPnP implemented)
//
// Make sure to use headphones to prevent catastrophic feedback cycles, but protect your ears!
// Start with extremely low volume.
//
// Audio data is not encrypted over the wire, anyone in your network could eavesdrop

const MAX_UDP_SIZE: usize = 508;
const SERVER_ADDR: &str = "0.0.0.0:1234";
const CLIENT_ADDR: &str = "0.0.0.0:5555";

fn main() -> std::io::Result<()> {
    env_logger::init();

    let mut args = std::env::args();
    args.next(); // program name

    /*
    let samples: Vec<_> = (0..128).map(|i| i as f32 / 128.).collect();
    let buf = AudioBuffer::from(vec![samples], SampleRate(48000));
    let mut bytes = vec![0; 512];
    let ser = serialize(&buf, &mut bytes);
    let deser = deserialize(&bytes[..256], SampleRate(48000));
    dbg!(deser);
    todo!();
    */

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

        // introduce chaos
        /*
        use rand::Rng;
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..100) {
            0 => continue, // 1% packet loss
            1 => std::thread::sleep(std::time::Duration::from_millis(1)), // 1% of packets has 1ms delay
            _ => (),
        };
        */

        socket.send_to(buf, src)?;
    }
}

// cram 128 floats with a bit of overhead into 508 bytes
fn serialize(audio_buf: &AudioBuffer, byte_buf: &mut [u8]) -> usize {
    // convert f32 to i16, get big endian bytes, write to byte_buf[0..256]
    let n = audio_buf
        .get_channel_data(0)
        .iter()
        .map(|f| (f * i16::MAX as f32) as i16)
        .map(|i| i.to_be_bytes())
        .flat_map(IntoIterator::into_iter)
        .zip(byte_buf.iter_mut())
        .map(|(i, o)| *o = i)
        .count();

    assert!(n < MAX_UDP_SIZE);

    n
}

fn deserialize(byte_buf: &[u8], sample_rate: f32) -> AudioBuffer {
    let samples: Vec<f32> = byte_buf
        .chunks_exact(2)
        .take(128)
        .map(|bs| i16::from_be_bytes(bs.try_into().unwrap()))
        .map(|i| i as f32 / i16::MAX as f32)
        .collect();
    AudioBuffer::from(vec![samples], sample_rate)
}

struct SocketStream {
    socket: &'static UdpSocket,
    sample_rate: f32,
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
                    length: 128,
                    sample_rate: self.sample_rate,
                };
                return Some(Ok(AudioBuffer::new(options)));
            }
            Err(e) => panic!("client recv IO error: {}", e),
        };
        Some(Ok(deserialize(bytes, self.sample_rate)))
    }
}

/// The client
///     1. records the audio and ships it to the server
///     2. plays back the samples received from the server
fn run_client() -> std::io::Result<()> {
    let socket = UdpSocket::bind(CLIENT_ADDR)?;
    socket.set_nonblocking(true).unwrap();
    println!("Client listening at {}", socket.local_addr()?);
    socket.connect(SERVER_ADDR)?;
    println!("Client 'connected' to {SERVER_ADDR}");

    // hack, make socket static to avoid `Arc` or similar
    let socket: &'static UdpSocket = Box::leak(Box::new(socket));

    let context = AudioContext::default();

    // leg 1: receive server packets and play them
    let stream = SocketStream {
        socket,
        sample_rate: context.sample_rate(),
        byte_buf: vec![0; 512],
    };
    let track = MediaStreamTrack::from_iter(stream);
    let stream_in = context.create_media_stream_track_source(&track);
    stream_in.connect(&context.destination());

    // leg 2: record mic input and ship to server
    let mic = media_devices::get_user_media_sync(MediaStreamConstraints::Audio);
    let stream_in = context.create_media_stream_source(&mic);
    let stream_out = context.create_media_stream_destination();
    stream_out.set_channel_count(1); // force mono
    stream_in.connect(&stream_out);

    let mut byte_buf = vec![0; 512];
    for item in stream_out.stream().get_tracks()[0].iter() {
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
