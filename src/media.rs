//! Microphone input and OGG, WAV and MP3 decoding

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::buffer::{AudioBuffer, AudioBufferOptions, ChannelData};
use crate::control::Controller;
use crate::{BufferDepletedError, SampleRate, RENDER_QUANTUM_SIZE};

#[cfg(not(test))]
use crossbeam_channel::Sender;
use crossbeam_channel::{self, Receiver, TryRecvError};

#[cfg(not(test))]
use crate::io;

#[cfg(not(test))]
use cpal::{traits::StreamTrait, Sample, Stream};

/// Interface for media streaming.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator, for example the [`OggVorbisDecoder`] or
/// [`Microphone`].
///
/// Below is an example showing how to play the stream directly in the audio context. However, this
/// is typically not what you should do. The media stream will be polled on the render thread which
/// will have catastrophic effects if the iterator blocks or for another reason takes too much time
/// to yield a new sample frame.
///
/// The solution is to wrap the `MediaStream` inside a [`MediaElement`]. This will take care of
/// buffering and timely delivery of audio to the render thread. It also allows for media playback
/// controls (play/pause, offsets, loops, etc.)
///
/// # Example
///
/// ```no_run
/// use web_audio_api::SampleRate;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::buffer::{AudioBuffer, AudioBufferOptions};
///
/// // create a new buffer: 512 samples of silence
/// let options = AudioBufferOptions {
///     number_of_channels: 0,
///     length: 512,
///     sample_rate: SampleRate(44_100),
/// };
/// let silence = AudioBuffer::new(options);
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(5);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream` and can be used in the audio graph
/// let context = AudioContext::new(None);
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

/// Wrapper for [`MediaStream`]s, for buffering and playback controls.
///
/// Currently, the media element will start a new thread to buffer all available media. (todo
/// async executor)
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
/// let silence = AudioBuffer::from_channels(
///     vec![ChannelData::from(vec![0.; 20])],
///     SampleRate(44_100)
/// );
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(3);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream`, we can wrap it in a `MediaElement`
/// let mut element = MediaElement::new(media);
/// element.controller().set_loop(true);
///
/// // the media element provides an infinite iterator now
/// for buf in element.take(5) {
///   match buf {
///       Ok(b) => {
///           assert_eq!(
///               b.get_channel_data(0)[..],
///               vec![0.; 20][..]
///           )
///       },
///       Err(e) => (),
///   }
/// }
/// ```
pub struct MediaElement {
    /// input media stream
    input: Receiver<Option<Result<AudioBuffer, Box<dyn Error + Send>>>>,
    /// media buffer
    buffer: Vec<AudioBuffer>,
    /// true when input stream is finished
    buffer_complete: bool,
    /// current position in buffer when filling/looping
    buffer_index: usize,
    /// user facing controller
    controller: Controller,
    /// current playback timestamp of this stream
    timestamp: f64,
    /// indicates if we are currently seeking but the data is not available
    seeking: Option<f64>,
}

impl MediaElement {
    /// Create a new MediaElement by buffering a MediaStream
    pub fn new<S: MediaStream>(input: S) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();

        let fill_buffer = move || {
            let _ = sender.send(None); // signal thread started
            input.map(Some).for_each(|i| {
                let _ = sender.send(i);
            });
            let _ = sender.send(None); // signal depleted
        };

        std::thread::spawn(fill_buffer);
        // wait for thread startup before handing out the MediaElement
        let ping = receiver.recv().expect("buffer channel disconnected");
        assert!(ping.is_none());

        Self {
            input: receiver,
            buffer: vec![],
            buffer_complete: false,
            buffer_index: 0,
            controller: Controller::new(),
            timestamp: 0.,
            seeking: None,
        }
    }

    pub fn controller(&self) -> &Controller {
        &self.controller
    }

    fn load_next(&mut self) -> Option<Result<AudioBuffer, Box<dyn Error + Send>>> {
        if !self.buffer_complete {
            let next = match self.input.try_recv() {
                Err(_) => return Some(Err(Box::new(BufferDepletedError {}))),
                Ok(v) => v,
            };

            match next {
                Some(Err(e)) => {
                    // no further streaming
                    self.buffer_complete = true;

                    return Some(Err(e));
                }
                Some(Ok(data)) => {
                    self.buffer.push(data.clone());
                    self.buffer_index += 1;
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

        // seek within currently buffered data
        for (i, buf) in self.buffer.iter().enumerate() {
            self.buffer_index = i;
            self.timestamp += buf.duration();
            if self.timestamp > ts {
                return; // seeking complete
            }
        }

        // seek by consuming the leftover input stream
        loop {
            match self.load_next() {
                Some(Ok(buf)) => {
                    self.timestamp += buf.duration();
                    if self.timestamp > ts {
                        return; // seeking complete
                    }
                }
                Some(Err(e)) if e.is::<BufferDepletedError>() => {
                    // mark incomplete seeking
                    self.seeking = Some(ts);
                    return;
                }
                // stop seeking if stream finished or errors occur
                _ => {
                    // prevent playback of last available frame
                    self.buffer_index += 1;
                    return;
                }
            }
        }
    }
}

impl Iterator for MediaElement {
    type Item = Result<AudioBuffer, Box<dyn Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        // handle seeking
        if let Some(seek) = self.controller().should_seek() {
            self.seek(seek);
        } else if let Some(seek) = self.seeking.take() {
            self.seek(seek);
        }
        if self.seeking.is_some() {
            return Some(Err(Box::new(BufferDepletedError {})));
        }

        // handle looping
        if self.controller.loop_() && self.timestamp > self.controller.loop_end() {
            self.seek(self.controller.loop_start());
        }

        // read from cache if available
        if let Some(data) = self.buffer.get(self.buffer_index) {
            self.buffer_index += 1;
            self.timestamp += data.duration();
            return Some(Ok(data.clone()));
        }

        // read from backing media stream
        match self.load_next() {
            Some(Ok(data)) => {
                return Some(Ok(data));
            }
            Some(Err(e)) if e.is::<BufferDepletedError>() => {
                // hickup when buffering was too slow
                return Some(Err(e));
            }
            _ => (), // stream finished or errored out
        };

        // signal depleted if we're not looping
        if !self.controller.loop_() || self.buffer.is_empty() {
            return None;
        }

        // loop and get next
        self.seek(self.controller.loop_start());
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
        let (stream, config, receiver) = io::build_input();
        log::debug!("Input {:?}", config);

        let sample_rate = SampleRate(config.sample_rate.0);
        let channels = config.channels as usize;

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

                let options = AudioBufferOptions {
                    number_of_channels: self.channels,
                    length: RENDER_QUANTUM_SIZE,
                    sample_rate: self.sample_rate,
                };

                AudioBuffer::new(options)
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
    sender: Sender<AudioBuffer>,
}

#[cfg(not(test))]
impl MicrophoneRender {
    pub fn new(channels: usize, sample_rate: SampleRate, sender: Sender<AudioBuffer>) -> Self {
        Self {
            channels,
            sample_rate,
            sender,
        }
    }

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
/// ```no_run
/// use web_audio_api::media::{MediaElement, OggVorbisDecoder};
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // construct the decoder
/// let file = std::fs::File::open("sample.ogg").unwrap();
/// let media = OggVorbisDecoder::try_new(file).unwrap();
///
/// // Wrap in a `MediaElement` so buffering/decoding does not take place on the render thread
/// let element = MediaElement::new(media);
///
/// // register the media element node
/// let context = AudioContext::new(None);
/// let node = context.create_media_element_source(element);
///
/// // play media
/// node.connect(&context.destination());
/// node.start();
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
/// ```no_run
/// use web_audio_api::media::{MediaElement, WavDecoder};
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // construct the decoder
/// let file = std::fs::File::open("sample.wav").unwrap();
/// let media = WavDecoder::try_new(file).unwrap();
///
/// // Wrap in a `MediaElement` so buffering/decoding does not take place on the render thread
/// let element = MediaElement::new(media);
///
/// // register the media element node
/// let context = AudioContext::new(None);
/// let node = context.create_media_element_source(element);
///
/// // play media
/// node.connect(&context.destination());
/// node.start();
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
        // read data in chunks of channels * RENDER_QUANTUM_SIZE
        let mut data = vec![vec![]; self.channels as usize];
        for (i, res) in self
            .stream
            .by_ref()
            .take(self.channels as usize * RENDER_QUANTUM_SIZE)
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
