//! Microphone input and media decoding (OGG, WAV, FLAC, ..)

use std::error::Error;
use std::io::{Read, Seek, SeekFrom};

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

use symphonia::core::audio::Signal;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::conv::FromSample;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Interface for media streaming.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator, for example the [`MediaDecoder`] or
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
/// use web_audio_api::node::AudioNode;
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
/// node.connect(&context.destination());
/// ```
pub trait MediaStream:
    Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>> + Send + 'static
{
}
impl<M: Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>> + Send + 'static>
    MediaStream for M
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
/// use web_audio_api::buffer::AudioBuffer;
/// use web_audio_api::media::MediaElement;
/// use web_audio_api::node::AudioControllableSourceNode;
///
/// // create a new buffer with a few samples of silence
/// let samples = vec![vec![0.; 20]];
/// let silence = AudioBuffer::from(samples, SampleRate(44_100));
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
    input: Receiver<Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>>>,
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

    fn load_next(&mut self) -> Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> {
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
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

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
/// It implements the [`MediaStream`] trait so can be used inside a
/// [`MediaStreamAudioSourceNode`](crate::node::MediaStreamAudioSourceNode)
///
/// # Example
///
/// ```no_run
/// use web_audio_api::context::{AsBaseAudioContext, AudioContext};
/// use web_audio_api::media::Microphone;
/// use web_audio_api::node::AudioNode;
///
/// let context = AudioContext::new(None);
///
/// let stream = Microphone::new();
/// // register as media element in the audio context
/// let background = context.create_media_stream_source(stream);
/// // connect the node directly to the destination node (speakers)
/// background.connect(&context.destination());
///
/// // enjoy listening
/// std::thread::sleep(std::time::Duration::from_secs(4));
/// ```
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
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

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

/// Wrapper for `Read` implementors to be used in Symphonia decoding
///
/// Symphonia requires its input to impl `Seek` - but allows non-seekable sources. Hence we
/// implement Seek but return false for `is_seekable()`.
struct MediaInput<R> {
    input: R,
}

impl<R: Read> MediaInput<R> {
    pub fn new(input: R) -> Self {
        Self { input }
    }
}

impl<R: Read> Read for MediaInput<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.input.read(buf)
    }
}

impl<R> Seek for MediaInput<R> {
    fn seek(&mut self, _pos: SeekFrom) -> std::io::Result<u64> {
        panic!("MediaInput does not support seeking")
    }
}

impl<R: Read + Send> symphonia::core::io::MediaSource for MediaInput<R> {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

/// Media stream decoder (OGG, WAV, FLAC, ..)
///
/// Using the `MediaDecoder` is the preferred way to play large audio files and streams. For small
/// soundbites, consider using `decode_audio_data` on the audio context which will create a single
/// AudioBuffer which can be played/looped with high precision in an `AudioBufferSourceNode`.
///
/// The MediaDecoder implements the [`MediaStream`] trait so can be used inside a
/// `MediaElementAudioSourceNode`
///
/// The current implementation can decode FLAC, Opus, PCM, Vorbis, and Wav.
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::media::{MediaElement, MediaDecoder};
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // construct the decoder
/// let file = std::fs::File::open("sample.ogg").unwrap();
/// let media = MediaDecoder::try_new(file).unwrap();
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
pub struct MediaDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
}

impl MediaDecoder {
    /// Try to construct a new instance from a `Read` implementor
    ///
    /// # Errors
    ///
    /// This method returns an Error in various cases (IO, mime sniffing, decoding).
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Cursor;
    /// use web_audio_api::media::MediaDecoder;
    ///
    /// let input = Cursor::new(vec![0; 32]); // or a File, TcpStream, ...
    /// let media = MediaDecoder::try_new(input);
    ///
    /// assert!(media.is_err()); // the input was not a valid MIME type
    pub fn try_new<R: std::io::Read + Send + 'static>(
        input: R,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Symfonia lib needs a Box<dyn MediaSource> - use our own MediaInput
        let input = Box::new(MediaInput::new(input));

        // Create the media source stream using the boxed media source from above.
        let mss = symphonia::core::io::MediaSourceStream::new(input, Default::default());

        // Create a hint to help the format registry guess what format reader is appropriate. In this
        // function we'll leave it empty.
        let hint = Hint::new();

        // Use the default options when reading and decoding.
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let decoder_opts: DecoderOptions = Default::default();

        // Probe the media source stream for a format.
        let probed =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        // Get the format reader yielded by the probe operation.
        let format = probed.format;

        // Get the default track.
        let track = format.default_track().unwrap();

        // Create a (stateful) decoder for the track.
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .unwrap();

        Ok(Self { format, decoder })
    }
}

impl Iterator for MediaDecoder {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let format = &mut self.format;
        let decoder = &mut self.decoder;

        // Get the default track.
        let track = format.default_track().unwrap();
        let number_of_channels = track.codec_params.channels.unwrap().count();
        let input_sample_rate = SampleRate(track.codec_params.sample_rate.unwrap() as _);

        // Store the track identifier, we'll use it to filter packets.
        let track_id = track.id;

        let data: Vec<Vec<f32>> = loop {
            // Get the next packet from the format reader.
            let packet = match format.next_packet() {
                Err(e) => {
                    log::error!("next packet err {:?}", e);
                    return None;
                }
                Ok(p) => p,
            };

            // If the packet does not belong to the selected track, skip it.
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet into audio samples, ignoring any decode errors.
            break match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // This looks a bit awkward but this may be the only way to get the f32 samples
                    // out without making double copies.
                    let chans = 0..number_of_channels;
                    use symphonia::core::audio::AudioBufferRef::*;
                    match audio_buf {
                        U8(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        U16(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        U24(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        U32(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        S8(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        S16(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        S24(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        S32(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        F32(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                        F64(buf) => chans
                            .map(|i| buf.chan(i).iter().copied().map(f32::from_sample).collect())
                            .collect(),
                    }
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    // continue processing but log the error
                    log::error!("decode err {:?}", e);
                    continue;
                }
                Err(e) => {
                    // do not continue processing, return error result
                    return Some(Err(Box::new(e)));
                }
            };
        };

        let channels = data.into_iter().map(ChannelData::from).collect();
        let buffer = AudioBuffer::from_channels(channels, input_sample_rate);

        Some(Ok(buffer))
    }
}
