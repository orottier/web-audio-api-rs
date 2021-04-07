//! OGG, WAV and MP3 encoding/decoding

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::buffer::{AudioBuffer, ChannelData};
use crate::control::{Controller, Scheduler};
use crate::SampleRate;

/// Interface for media decoding.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator that can run in a separate thread, for
/// example the [`OggVorbisDecoder`]
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
/// use crate::web_audio_api::node::AudioControllableSourceNode;
///
/// // create a new buffer: 20 samples of silence
/// let silence = AudioBuffer::from_channels(vec![ChannelData::from(vec![0.; 20])], SampleRate(44_100));
///
/// // create a sequence of this buffer
/// let sequence = std::iter::repeat(silence).take(5);
///
/// // the sequence should actually yield `Result<AudioBuffer, _>`s
/// let media = sequence.map(|b| Ok(b));
///
/// // media is now a proper `MediaStream`, we can wrap it in a `MediaElement`
/// let mut element = MediaElement::new(media);
/// element.set_loop(true);
///
/// for buf in element.take(3) {
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

/// Ogg Vorbis (.ogg) file decoder
///
/// It implements the [`MediaStream`] trait so can be used inside a `MediaElementAudioSourceNode`
///
/// # Usage
///
/// ``` rust
/// use web_audio_api::media::OggVorbisDecoder;
/// use web_audio_api::context::{AudioContext, AsBaseAudioContext};
/// use crate::web_audio_api::node::AudioNode;
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
/// use crate::web_audio_api::node::AudioNode;
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
