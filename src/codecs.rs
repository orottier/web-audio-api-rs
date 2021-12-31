//! Media file decoding

use std::io::{Read, Seek, SeekFrom};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::buffer::{AudioBuffer, ChannelData};
use crate::SampleRate;

pub fn decode_audio_data<R: std::io::Read + Send + 'static>(
    input: R,
    sample_rate: SampleRate,
) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
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
    let mut format = probed.format;

    // Get the default track.
    let track = format.default_track().unwrap();
    let number_of_channels = track.codec_params.channels.unwrap().count();
    let input_sample_rate = SampleRate(track.codec_params.sample_rate.unwrap() as _);

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    // Store the track identifier, we'll use it to filter packets.
    let track_id = track.id;

    let mut sample_buf = None;

    let mut data = vec![vec![]; number_of_channels as usize];

    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Err(e) => {
                log::error!("next packet err {:?}", e);
                break;
            }
            Ok(p) => p,
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // The decoded audio samples may now be accessed via the audio buffer if per-channel
                // slices of samples in their native decoded format is desired. Use-cases where
                // the samples need to be accessed in an interleaved order or converted into
                // another sample format, or a byte buffer is required, are covered by copying the
                // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                // example below, we will copy the audio buffer into a sample buffer in an
                // interleaved order while also converting to a f32 sample format.

                // If this is the *first* decoded packet, create a sample buffer matching the
                // decoded audio buffer format.
                if sample_buf.is_none() {
                    // Get the audio buffer specification.
                    let spec = *audio_buf.spec();

                    // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                    let duration = audio_buf.capacity() as u64;

                    // Create the f32 sample buffer.
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);

                    for (i, v) in buf.samples().iter().enumerate() {
                        data[i % number_of_channels].push(*v);
                    }
                }
            }
            Err(SymphoniaError::DecodeError(e)) => {
                // continue processing but log the error
                log::error!("decode err {:?}", e);
            }
            Err(e) => {
                // do not continue processing, return error result
                return Err(Box::new(e));
            }
        }
    }

    let channels = data.into_iter().map(ChannelData::from).collect();
    let mut buffer = AudioBuffer::from_channels(channels, input_sample_rate);

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
