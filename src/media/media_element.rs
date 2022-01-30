use crossbeam_channel::{self, Receiver};
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::buffer::AudioBuffer;
use crate::control::Controller;
use crate::{BufferDepletedError, AtomicF64};

use super::MediaStream;

/// Wrapper for [`MediaStream`]s, for buffering and playback controls.
/// Mimic API of HTMLMediaElement: <https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/>
///
/// Warning: this abstraction is not part of the Web Audio API and does not aim at
/// implementing the full HTMLMediaElement API. It is only provided for convenience
/// reasons.
///
/// Currently, the media element will start a new thread to buffer all available media. (todo
/// async executor)
///
///
/// # Example
///
/// ```rust
/// use web_audio_api::SampleRate;
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
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
/// element.set_loop(true);
///
/// // the media element provides an infinite iterator now
/// for buf in element.take(5) {
///   match buf {
///       Ok(b) => {
///           assert_eq!(
///               b.get_channel_data(0)[..],
///               vec![0.; 20][..]
///           )
///       }
///       Err(e) => (),
///   }
/// }
/// ```
struct MediaElementInner {
    /// input media stream
    input: Receiver<Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>>>,
    /// media buffer
    buffer: Vec<AudioBuffer>,
    /// true when input stream is finished
    buffer_complete: AtomicBool,
    /// current position in buffer when filling/looping
    buffer_index: AtomicUsize,
    /// user facing controller
    controller: Controller,
    /// current time of this stream
    current_time: AtomicF64,
    /// state of the element
    paused: AtomicBool,
    /// indicates if we are currently seeking but the data is not available
    seeking: Option<AtomicF64>,
}

#[derive(Clone)]
pub struct MediaElement {
    inner: Arc<MediaElementInner>,
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

        let media_element_inner = MediaElementInner {
            input: receiver,
            buffer: vec![],
            buffer_complete: AtomicBool::new(false),
            buffer_index: AtomicUsize::new(0),
            controller: Controller::new(),
            current_time: AtomicF64::new(0.),
            paused: AtomicBool::new(true),
            seeking: None,
        };

        Self {
            inner: Arc::new(media_element_inner),
        }
    }

    pub fn current_time(&self) -> f64 {
        self.inner.current_time
    }

    pub fn paused(&self) -> bool {
        self.inner.paused.load(Ordering::SeqCst)
    }

    pub fn start(&self) {
        self.inner.paused.store(false, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        self.inner.paused.store(true, Ordering::SeqCst);
    }

    pub fn loop_(&self) -> bool {
        self.inner.controller.loop_()
    }

    pub fn set_loop(&self, loop_: bool) {
        self.inner.controller.set_loop(loop_);
    }

    pub fn loop_start(&self) -> f64 {
        self.inner.controller.loop_start()
    }

    pub fn set_loop_start(&self, start_time: f64) {
        self.inner.controller.set_loop_start(start_time);
    }

    pub fn loop_end(&self) -> f64 {
        self.inner.controller.loop_end()
    }

    pub fn set_loop_end(&self, end_time: f64) {
        self.inner.controller.set_loop_end(end_time);
    }

    pub fn seek(&self, position: f64) {
        self.inner.controller.seek(position);
    }

    /// Seek to a current_time offset in the media buffer
    fn do_seek(&mut self, ts: f64) {
        if ts == 0. {
            self.inner.current_time = 0.;
            self.inner.buffer_index = 0;
            return;
        }

        self.inner.current_time = 0.;

        // seek within currently buffered data
        for (i, buf) in self.inner.buffer.iter().enumerate() {
            self.inner.buffer_index = i;
            self.inner.current_time += buf.duration();
            if self.inner.current_time > ts {
                return; // seeking complete
            }
        }

        // seek by consuming the leftover input stream
        loop {
            match self.load_next() {
                Some(Ok(buf)) => {
                    self.inner.current_time += buf.duration();
                    if self.inner.current_time > ts {
                        return; // seeking complete
                    }
                }
                Some(Err(e)) if e.is::<BufferDepletedError>() => {
                    // mark incomplete seeking
                    self.inner.seeking = Some(ts);
                    return;
                }
                // stop seeking if stream finished or errors occur
                _ => {
                    // prevent playback of last available frame
                    self.inner.buffer_index += 1;
                    return;
                }
            }
        }
    }

    fn load_next(&mut self) -> Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> {
        if !self.inner.buffer_complete {
            let next = match self.inner.input.try_recv() {
                Err(_) => return Some(Err(Box::new(BufferDepletedError {}))),
                Ok(v) => v,
            };

            match next {
                Some(Err(e)) => {
                    // no further streaming
                    self.inner.buffer_complete = true;

                    return Some(Err(e));
                }
                Some(Ok(data)) => {
                    self.inner.buffer.push(data.clone());
                    self.inner.buffer_index += 1;
                    self.inner.current_time += data.duration();
                    return Some(Ok(data));
                }
                None => {
                    self.inner.buffer_complete = true;
                    return None;
                }
            }
        }

        None
    }
}

impl Iterator for MediaElement {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.paused() {
            return None;
        }

        // handle seeking
        if let Some(seek) = self.inner.controller.should_seek() {
            self.do_seek(seek);
        } else if let Some(seek) = self.inner.seeking.take() {
            self.do_seek(seek);
        }
        if self.inner.seeking.is_some() {
            return Some(Err(Box::new(BufferDepletedError {})));
        }

        // handle looping
        if self.inner.controller.loop_() && self.inner.current_time > self.inner.controller.loop_end() {
            self.do_seek(self.inner.controller.loop_start());
        }

        // read from cache if available
        if let Some(data) = self.inner.buffer.get(self.inner.buffer_index) {
            self.inner.buffer_index += 1;
            self.inner.current_time += data.duration();
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
        if !self.inner.controller.loop_() || self.inner.buffer.is_empty() {
            return None;
        }

        // loop and get next
        self.do_seek(self.inner.controller.loop_start());
        self.next()
    }
}
