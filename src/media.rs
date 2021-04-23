//! Microphone input and OGG, WAV and MP3 decoding

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc::{Receiver, TryRecvError};

use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::buffer::{AudioBuffer, ChannelData};
use crate::control::{Controller, Scheduler};
use crate::{SampleRate, BUFFER_SIZE};

#[cfg(not(test))]
use std::sync::mpsc::{self, SyncSender};

#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Sample, Stream};

/// Interface for media streaming.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator, for example the [`OggVorbisDecoder`]
///
/// Below is an example showing how to play the stream directly.
///
/// If you want to control the media playback (play/pause, offsets, loops), wrap the `MediaStream`
/// in a [`MediaElement`].
///
/// # Example
///
/// ```rust
/// use web_audio_api::SampleRate;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::buffer::AudioBuffer;
///
/// // create a new buffer: 512 samples of silence
/// let silence = AudioBuffer::new(1, 512, SampleRate(44_100));
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(5);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream` and can be used in the audio graph
/// let context = AudioContext::new();
/// let node = context.create_media_stream_source(media);
/// ```
pub trait MediaStream:
    Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send>>> + Send + 'static
{
}
impl<M: Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send>>> + Send + 'static> MediaStream
    for M
{
}

/// Wrapper for [`MediaStream`]s, to control playback.
///
/// # Example
///
/// ```rust
/// use web_audio_api::SampleRate;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::buffer::{AudioBuffer, ChannelData};
/// use web_audio_api::media::MediaElement;
/// use web_audio_api::node::AudioControllableSourceNode;
///
/// // create a new buffer with a few samples of silence
/// let silence = AudioBuffer::from_channels(vec![ChannelData::from(vec![0.; 20])], SampleRate(44_100));
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(3);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream`, we can wrap it in a `MediaElement`
/// let mut element = MediaElement::new(media);
/// element.set_loop(true);
///
/// // the media element provides an infinite iterator now
/// for buf in element.take(5) {
///     assert_eq!(
///         buf.unwrap().channel_data(0).unwrap(),
///         &ChannelData::from(vec![0.; 20])
///     )
/// }
///
/// ```
pub struct MediaElement<S> {
    input: S,
    buffer: Vec<AudioBuffer>,
    buffer_complete: bool,
    buffer_index: usize,
    controller: Controller,
    timestamp: f64,
}

use crate::node::AudioControllableSourceNode;
impl<S> AudioControllableSourceNode for MediaElement<S> {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

use crate::node::AudioScheduledSourceNode;
impl<S> AudioScheduledSourceNode for MediaElement<S> {
    fn scheduler(&self) -> &Scheduler {
        &self.controller.scheduler()
    }
}

impl<S: MediaStream> MediaElement<S> {
    /// Create a new MediaElement by buffering a MediaStream
    pub fn new(input: S) -> Self {
        Self {
            input,
            buffer: vec![],
            buffer_complete: false,
            buffer_index: 0,
            controller: Controller::new(),
            timestamp: 0.,
        }
    }

    fn load_next(&mut self) -> Option<Result<AudioBuffer, Box<dyn Error + Send>>> {
        if !self.buffer_complete {
            match self.input.next() {
                Some(Err(e)) => {
                    // no further streaming
                    self.buffer_complete = true;

                    return Some(Err(e));
                }
                Some(Ok(data)) => {
                    self.buffer.push(data.clone());
                    self.timestamp += data.duration();
                    return Some(Ok(data));
                }
                None => {
                    self.buffer_complete = true;
                    return None;
                }
            }
        }

        None
    }

    /// Seek to a timestamp offset in the media buffer
    pub fn seek(&mut self, ts: f64) {
        if ts == 0. {
            self.timestamp = 0.;
            self.buffer_index = 0;
            return;
        }

        self.timestamp = 0.;
        for (i, buf) in self.buffer.iter().enumerate() {
            self.buffer_index = i;
            self.timestamp += buf.duration();
            if self.timestamp >= ts {
                return;
            }
        }

        while let Some(Ok(buf)) = self.load_next() {
            self.timestamp += buf.duration();
            if self.timestamp >= ts {
                return;
            }
        }
    }
}

impl<S: MediaStream> Iterator for MediaElement<S> {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.loop_() && self.timestamp > self.loop_end() {
            self.seek(self.loop_start());
        }

        if let Some(data) = self.buffer.get(self.buffer_index) {
            self.buffer_index += 1;
            self.timestamp += data.duration();
            return Some(Ok(data.clone()));
        }

        if let Some(data) = self.load_next() {
            self.buffer_index += 1;
            return Some(data);
        }

        if !self.loop_() || self.buffer.is_empty() {
            return None;
        }

        self.seek(self.loop_start());

        self.next()
    }
}

/// Microphone input stream
///
/// It implements the [`MediaStream`] trait so can be used inside a [`crate::node::MediaStreamAudioSourceNode`]
///
/// Check the `microphone.rs` example for usage.
pub struct Microphone {
    receiver: Receiver<AudioBuffer>,
    channels: usize,
    sample_rate: SampleRate,

    #[cfg(not(test))]
    stream: Stream,
}

// Todo, the Microphone struct is shipped to the render thread
// but it contains a Stream which is not Send.
unsafe impl Send for Microphone {}

impl Microphone {
    /// Setup the default microphone input stream
    #[cfg(not(test))]
    pub fn new() -> Self {
        let buffer = 1; // todo, use buffering to smooth frame drops
        let (sender, receiver) = mpsc::sync_channel(buffer);

        let io_builder = io::InputBuilder::new();
        let config = io_builder.config();
        log::debug!("Input {:?}", config);

        let sample_rate = SampleRate(config.sample_rate.0);
        let channels = config.channels as usize;
        let render = MicrophoneRender {
            channels,
            sample_rate,
            sender,
        };

        let stream = io_builder.build(render);

        Self {
            receiver,
            channels,
            sample_rate,
            stream,
        }
    }

    /// Suspends the input stream, temporarily halting audio hardware access and reducing
    /// CPU/battery usage in the process.
    pub fn suspend(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.pause().unwrap()
    }

    /// Resumes the input stream that has previously been suspended/paused.
    pub fn resume(&self) {
        #[cfg(not(test))] // in tests, do not set up a cpal Stream
        self.stream.play().unwrap()
    }
}

#[cfg(not(test))]
impl Default for Microphone {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for Microphone {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.receiver.try_recv() {
            Ok(buffer) => {
                // new frame was ready
                buffer
            }
            Err(TryRecvError::Empty) => {
                // frame not received in time, emit silence
                log::debug!("input frame delayed");
                AudioBuffer::new(self.channels, BUFFER_SIZE as usize, self.sample_rate)
            }
            Err(TryRecvError::Disconnected) => {
                // MicrophoneRender has stopped, close stream
                return None;
            }
        };

        Some(Ok(next))
    }
}

#[cfg(not(test))]
pub(crate) struct MicrophoneRender {
    channels: usize,
    sample_rate: SampleRate,
    sender: SyncSender<AudioBuffer>,
}

#[cfg(not(test))]
impl MicrophoneRender {
    pub fn render<S: Sample>(&self, data: &[S]) {
        let mut channels = Vec::with_capacity(self.channels);

        // copy rendered audio into output slice
        for i in 0..self.channels {
            channels.push(ChannelData::from(
                data.iter()
                    .skip(i)
                    .step_by(self.channels)
                    .map(|v| v.to_f32())
                    .collect(),
            ));
        }

        let buffer = AudioBuffer::from_channels(channels, self.sample_rate);
        let result = self.sender.try_send(buffer); // can fail (frame dropped)
        if result.is_err() {
            log::debug!("input frame dropped");
        }
    }
}

/// Ogg Vorbis (.ogg) file decoder
///
/// It implements the [`MediaStream`] trait so can be used inside a `MediaElementAudioSourceNode`
///
/// # Usage
///
/// ``` rust
/// use web_audio_api::media::OggVorbisDecoder;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::node::AudioNode;
///
/// // construct the decoder
/// let file = std::fs::File::open("sample.ogg").unwrap();
/// let media = OggVorbisDecoder::try_new(file).unwrap();
///
/// // register the media node
/// let context = AudioContext::new();
/// let node = context.create_media_stream_source(media);
///
/// // play media
/// node.connect(&context.destination());
/// ```
///
pub struct OggVorbisDecoder {
    stream: OggStreamReader<BufReader<File>>,
}

impl OggVorbisDecoder {
    /// Try to construct a new instance from a [`File`]
    pub fn try_new(file: File) -> Result<Self, VorbisError> {
        OggStreamReader::new(BufReader::new(file)).map(|stream| Self { stream })
    }
}

impl Iterator for OggVorbisDecoder {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        let packet: Vec<Vec<f32>> = match self.stream.read_dec_packet_generic() {
            Err(e) => return Some(Err(Box::new(e))),
            Ok(None) => return None,
            Ok(Some(packet)) => packet,
        };

        let channel_data: Vec<_> = packet.into_iter().map(ChannelData::from).collect();
        let sample_rate = SampleRate(self.stream.ident_hdr.audio_sample_rate);
        let result = AudioBuffer::from_channels(channel_data, sample_rate);

        Some(Ok(result))
    }
}

/// WAV file decoder
///
/// It implements the [`MediaStream`] trait so can be used inside a `MediaElementAudioSourceNode`
///
/// # Usage
///
/// ``` rust
/// use web_audio_api::media::WavDecoder;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::node::AudioNode;
///
/// // construct the decoder
/// let file = std::fs::File::open("sample.wav").unwrap();
/// let media = WavDecoder::try_new(file).unwrap();
///
/// // register the media node
/// let context = AudioContext::new();
/// let node = context.create_media_stream_source(media);
///
/// // play media
/// node.connect(&context.destination());
/// ```
///
pub struct WavDecoder {
    //stream: hound::WavIntoSamples<BufReader<File>, f32>,
    stream: Box<dyn Iterator<Item = Result<f32, hound::Error>> + Send>,
    channels: u32,
    sample_rate: SampleRate,
}

impl WavDecoder {
    /// Try to construct a new instance from a [`File`]
    pub fn try_new(file: File) -> Result<Self, hound::Error> {
        hound::WavReader::new(BufReader::new(file)).map(|wav| {
            let channels = wav.spec().channels as u32;
            let sample_rate = SampleRate(wav.spec().sample_rate);

            // convert samples to f32, always
            let stream: Box<dyn Iterator<Item = Result<_, _>> + Send> =
                match wav.spec().sample_format {
                    hound::SampleFormat::Float => Box::new(wav.into_samples::<f32>()),
                    hound::SampleFormat::Int => {
                        let bits = wav.spec().bits_per_sample as f32;
                        Box::new(
                            wav.into_samples::<i32>()
                                .map(move |r| r.map(|i| i as f32 / bits)),
                        )
                    }
                };

            Self {
                stream,
                channels,
                sample_rate,
            }
        })
    }
}

impl Iterator for WavDecoder {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        // read data in chunks of channels * BUFFER_SIZE
        let mut data = vec![vec![]; self.channels as usize];
        for (i, res) in self
            .stream
            .by_ref()
            .take((self.channels * crate::BUFFER_SIZE) as usize)
            .enumerate()
        {
            match res {
                Err(e) => return Some(Err(Box::new(e))),
                Ok(v) => data[i % self.channels as usize].push(v),
            }
        }

        // exhausted?
        if data[0].is_empty() {
            return None;
        }

        // convert data to AudioBuffer
        let channels = data.into_iter().map(ChannelData::from).collect();
        let result = AudioBuffer::from_channels(channels, self.sample_rate);

        Some(Ok(result))
    }
}
