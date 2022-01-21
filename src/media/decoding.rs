use std::error::Error;
use std::io::{Read, Seek, SeekFrom};

use crate::buffer::{AudioBuffer, ChannelData};
use crate::SampleRate;

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::{Decoder, DecoderOptions};
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
/// soundbites, consider using
/// [`decode_audio_data`](crate::context::Context::decode_audio_data) on the audio
/// context which will create a single AudioBuffer which can be played/looped with high precision
/// in an `AudioBufferSourceNode`.
///
/// The MediaDecoder implements the [`MediaStream`](crate::media::MediaStream) trait so can be used
/// inside a `MediaElementAudioSourceNode`
///
/// The current implementation can decode FLAC, Opus, PCM, Vorbis, and Wav.
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::media::{MediaElement, MediaDecoder};
/// use web_audio_api::context::{AudioContext, Context};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// // construct the decoder
/// let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
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

        loop {
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
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    let output = convert_buf(audio_buf, number_of_channels, input_sample_rate);
                    return Some(Ok(output));
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    // Todo: treat decoding errors as fatal or move to next packet? Context:
                    // https://github.com/RustAudio/rodio/issues/401#issuecomment-974747404
                    log::error!("Symphonia DecodeError {:?} - abort stream", e);
                    return Some(Err(Box::new(SymphoniaError::DecodeError(e))));
                }
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // this happens for Wav-files, running into EOF is expected
                }
                Err(e) => {
                    // do not continue processing, return error result
                    return Some(Err(Box::new(e)));
                }
            };
        }
    }
}

/// Convert a Symphonia AudioBufferRef to our own AudioBuffer
fn convert_buf(
    input: AudioBufferRef<'_>,
    number_of_channels: usize,
    input_sample_rate: SampleRate,
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
