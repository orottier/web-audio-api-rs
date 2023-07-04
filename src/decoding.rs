use std::error::Error;
use std::io::{Read, Seek, SeekFrom};

use crate::buffer::{AudioBuffer, ChannelData};

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::{Decoder, DecoderOptions, FinalizeResult};
use symphonia::core::conv::FromSample;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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

impl<R: Read + Send + Sync> symphonia::core::io::MediaSource for MediaInput<R> {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

/// Media stream decoder (OGG, WAV, FLAC, ..)
///
/// The current implementation can decode FLAC, Opus, PCM, Vorbis, and Wav.
pub(crate) struct MediaDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    packet_count: usize,
}

impl MediaDecoder {
    /// Try to construct a new instance from a `Read` implementor
    ///
    /// # Errors
    ///
    /// This method returns an Error in various cases (IO, mime sniffing, decoding).
    pub fn try_new<R: std::io::Read + Send + Sync + 'static>(
        input: R,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Symfonia lib needs a Box<dyn MediaSource> - use our own MediaInput
        let input = Box::new(MediaInput::new(input));

        // Create the media source stream using the boxed media source from above.
        let mss = symphonia::core::io::MediaSourceStream::new(input, Default::default());

        // Create a hint to help the format registry guess what format reader is appropriate. In this
        // function we'll leave it empty.
        let hint = Hint::new();

        // TODO: Allow to customize some options.
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let decoder_opts = DecoderOptions {
            // Opt-in to verify the decoded data against the checksums in the container.
            verify: true,
        };

        // Probe the media source stream for a format.
        let probed =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        // Get the format reader yielded by the probe operation.
        let format = probed.format;

        // Get the default track.
        let track = format.default_track().ok_or(SymphoniaError::Unsupported(
            "no default media track available",
        ))?;
        let track_id = track.id;

        // Create a (stateful) decoder for the track.
        let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

        Ok(Self {
            format,
            decoder,
            track_id,
            packet_count: 0,
        })
    }
}

impl Iterator for MediaDecoder {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            format,
            decoder,
            track_id,
            packet_count,
        } = self;

        // Get the default track.
        let track = format.default_track()?;
        let number_of_channels = track.codec_params.channels?.count();
        let input_sample_rate = track.codec_params.sample_rate? as f32;

        loop {
            // Get the next packet from the format reader.
            let packet = match format.next_packet() {
                Err(err) => {
                    if let SymphoniaError::IoError(err) = &err {
                        if err.kind() == std::io::ErrorKind::UnexpectedEof {
                            // End of stream
                            log::debug!("Decoding finished after {packet_count} packet(s)");
                            let FinalizeResult { verify_ok } = decoder.finalize();
                            if verify_ok == Some(false) {
                                log::warn!("Verification of decoded data failed");
                            }
                            return None;
                        }
                    }
                    log::warn!(
                        "Failed to fetch next packet following packet #{packet_count}: {err}"
                    );
                    return Some(Err(Box::new(err)));
                }
                Ok(packet) => {
                    *packet_count += 1;
                    packet
                }
            };

            // If the packet does not belong to the selected track, skip it.
            let packet_track_id = packet.track_id();
            if packet_track_id != *track_id {
                log::debug!(
                    "Skipping packet from other track {packet_track_id} while decoding track {track_id}"
                );
                continue;
            }

            // Decode the packet into audio samples.
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    let output = convert_buf(audio_buf, number_of_channels, input_sample_rate);
                    return Some(Ok(output));
                }
                Err(SymphoniaError::DecodeError(err)) => {
                    // Recoverable error, continue with the next packet.
                    log::warn!("Failed to decode packet #{packet_count}: {err}");
                }
                Err(SymphoniaError::IoError(err)) => {
                    // Recoverable error, continue with the next packet.
                    log::warn!("I/O error while decoding packet #{packet_count}: {err}");
                }
                Err(err) => {
                    // All other errors are considered fatal and decoding must be aborted.
                    return Some(Err(Box::new(err)));
                }
            };
        }
    }
}

/// Convert a Symphonia AudioBufferRef to our own AudioBuffer
fn convert_buf(
    input: AudioBufferRef<'_>,
    number_of_channels: usize,
    input_sample_rate: f32,
) -> AudioBuffer {
    let chans = 0..number_of_channels;

    // This looks a bit awkward but this may be the only way to get the f32 samples
    // out without making double copies.
    use symphonia::core::audio::AudioBufferRef::*;

    let data: Vec<Vec<f32>> = match input {
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
    };

    let channels = data.into_iter().map(ChannelData::from).collect();
    AudioBuffer::from_channels(channels, input_sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_media_decoder() {
        let input = Cursor::new(vec![0; 32]);
        let media = MediaDecoder::try_new(input);

        assert!(media.is_err()); // the input was not a valid MIME type
    }
}
