//! OGG, WAV and MP3 encoding/decoding

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::buffer::{AudioBuffer, ChannelData};

/// Interface for external media decoding
pub trait MediaElement {
    /// Decode a single audio chunk, return `None` at end of stream
    fn stream_chunk(&mut self) -> Result<Option<AudioBuffer>, Box<dyn Error>>;
}

/// Ogg Vorbis (.ogg) file decoder
pub struct OggVorbisDecoder {
    stream: OggStreamReader<BufReader<File>>,
}

impl OggVorbisDecoder {
    pub fn try_new(file: File) -> Result<Self, VorbisError> {
        OggStreamReader::new(BufReader::new(file)).map(|stream| Self { stream })
    }
}

impl MediaElement for OggVorbisDecoder {
    fn stream_chunk(&mut self) -> Result<Option<AudioBuffer>, Box<dyn Error>> {
        let maybe_packet: Option<Vec<Vec<f32>>> = self.stream.read_dec_packet_generic()?;

        let result = maybe_packet.map(|packet| {
            let channel_data: Vec<_> = packet.into_iter().map(ChannelData::from).collect();
            AudioBuffer::from_channels(channel_data)
        });

        Ok(result)
    }
}
