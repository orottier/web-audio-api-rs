//! Media file decoding

use std::io::{Read, Seek, SeekFrom};

use crate::buffer::AudioBuffer;
use crate::media::MediaDecoder;
use crate::SampleRate;

pub fn decode_audio_data<R: std::io::Read + Send + 'static>(
    input: R,
    sample_rate: SampleRate,
) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
    // Set up a media decoder, consume the stream in full and construct a single buffer out of it
    let mut buffer = MediaDecoder::try_new(input)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .reduce(|mut accum, item| {
            accum.extend(&item);
            accum
        })
        .unwrap();

    // resample to desired rate (no-op if already matching)
    buffer.resample(sample_rate);

    Ok(buffer)
}

/// Wrapper for `Read` implementors to be used in Symphonia decoding
///
/// Symphonia requires its input to impl `Seek` - but allows non-seekable sources. Hence we
/// implement Seek but return false for `is_seekable()`.
pub struct MediaInput<R> {
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
