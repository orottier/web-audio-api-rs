//! OGG, WAV and MP3 encoding/decoding

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::buffer::{AudioBuffer, ChannelData};
use crate::SampleRate;

/// Interface for external media decoding.
///
/// This is a trait alias for an [`AudioBuffer`] Iterator that can run in a separate thread, for example the [`OggVorbisDecoder`]
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
/// // media is now a proper `MediaElement` and can be used in the audio graph
/// let context = AudioContext::new();
/// let node = context.create_media_element_source(media);
/// ```
pub trait MediaStream:
    Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send>>> + Send + 'static
{
}
impl<M: Iterator<Item = Result<AudioBuffer, Box<dyn Error + Send>>> + Send + 'static> MediaStream
    for M
{
}

/// Ogg Vorbis (.ogg) file decoder
///
/// It implements the [`MediaElement`] trait so can be used inside a `MediaElementAudioSourceNode`
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
/// let node = context.create_media_element_source(media);
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
    type Item = Result<AudioBuffer, Box<dyn Error + Send + 'static>>;

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
