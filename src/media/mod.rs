//! Microphone input and media decoding (OGG, WAV, FLAC, ..)

mod decoding;
pub use decoding::MediaDecoder;

mod mic;
pub use mic::Microphone;

#[cfg(not(test))]
pub(crate) use mic::MicrophoneRender;

use std::error::Error;

use crate::buffer::AudioBuffer;
use crate::control::Controller;
use crate::BufferDepletedError;

use crossbeam_channel::{self, Receiver};

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
/// use web_audio_api::context::{AudioContext, Context};
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
/// use web_audio_api::context::{AudioContext, Context};
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
