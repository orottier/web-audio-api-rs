use crossbeam_channel::{self, Receiver};
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::buffer::AudioBuffer;
use crate::control::Controller;
use crate::{AtomicF64, BufferDepletedError};

use super::MediaStream;

struct MediaElementInner {
    /// input media stream
    input: Receiver<Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>>>,
    /// media buffer
    buffer: Mutex<Vec<AudioBuffer>>,
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
    // treat NaN as niche: no seeking
    seeking: AtomicF64,
}

/// Simple Audio Player
///
/// Wrapper for [`MediaStream`]s, for buffering and playback controls.
/// Mimic API of HTMLMediaElement: <https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/>
///
/// Currently, the media element will start a new thread to buffer all available media. (todo
/// async executor)
///
/// # Warning
///
/// This abstraction is not part of the Web Audio API and does not aim at implementing
/// the full HTMLMediaElement API. It is only provided for convenience reasons.
///
/// # Example
///
/// ```no_run
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::media::{MediaDecoder, MediaElement};
/// use web_audio_api::node::{AudioNode};
///
/// // build a decoded audio stream the decoder
/// let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
/// let stream = MediaDecoder::try_new(file).unwrap();
/// // wrap in a `MediaElement`
/// let media_element = MediaElement::new(stream);
/// // pipe the media element into the web audio graph
/// let context = AudioContext::new(None);
/// let node = context.create_media_element_source(&media_element);
/// node.connect(&context.destination());
/// // start media playback
/// media_element.start();
/// ```
///
pub struct MediaElement {
    inner: Arc<MediaElementInner>,
}

// In current implementation if the media element is passed twice to a
// `MediaElementAudioSourceNode`, the stream is played at double speed.
// therefore we panic if MediaElement is cloned more than once.
impl Clone for MediaElement {
    fn clone(&self) -> Self {
        if Arc::strong_count(&self.inner) >= 2 {
            panic!("Cannot use MediaElement in MediaElementAudioSourceNode more than once");
        }

        Self {
            inner: self.inner.clone(),
        }
    }
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
            buffer: Mutex::new(vec![]),
            buffer_complete: AtomicBool::new(false),
            buffer_index: AtomicUsize::new(0),
            controller: Controller::new(),
            current_time: AtomicF64::new(0.),
            paused: AtomicBool::new(true),
            seeking: AtomicF64::new(f64::NAN),
        };

        Self {
            inner: Arc::new(media_element_inner),
        }
    }

    pub fn current_time(&self) -> f64 {
        self.inner.current_time.load()
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

    // @note - do not expose, not part of MediaElement spec
    // will simplify later refactoring / improvements
    // pub fn loop_start(&self) -> f64 {
    //     self.inner.controller.loop_start()
    // }

    // pub fn set_loop_start(&self, start_time: f64) {
    //     self.inner.controller.set_loop_start(start_time);
    // }

    // pub fn loop_end(&self) -> f64 {
    //     self.inner.controller.loop_end()
    // }

    // pub fn set_loop_end(&self, end_time: f64) {
    //     self.inner.controller.set_loop_end(end_time);
    // }

    pub fn seek(&self, position: f64) {
        self.inner.controller.seek(position);
    }

    /// Seek to a current_time offset in the media buffer
    fn do_seek(&mut self, ts: f64) {
        if ts == 0. {
            self.inner.current_time.store(0.);
            self.inner.buffer_index.store(0, Ordering::SeqCst);
            return;
        }

        self.inner.current_time.store(0.);

        // seek within currently buffered data
        for (i, buf) in self.inner.buffer.lock().unwrap().iter().enumerate() {
            self.inner.buffer_index.store(i, Ordering::SeqCst);
            self.inner.current_time.fetch_add(buf.duration());

            if self.inner.current_time.load() > ts {
                return; // seeking complete
            }
        }

        // seek by consuming the leftover input stream
        loop {
            match self.load_next() {
                Some(Ok(buf)) => {
                    self.inner.current_time.fetch_add(buf.duration());

                    if self.inner.current_time.load() > ts {
                        return; // seeking complete
                    }
                }
                Some(Err(e)) if e.is::<BufferDepletedError>() => {
                    // mark incomplete seeking
                    self.inner.seeking.store(ts);
                    return;
                }
                // stop seeking if stream finished or errors occur
                _ => {
                    // prevent playback of last available frame
                    self.inner.buffer_index.fetch_add(1, Ordering::SeqCst);
                    return;
                }
            }
        }
    }

    fn load_next(&mut self) -> Option<Result<AudioBuffer, Box<dyn Error + Send + Sync>>> {
        if !self.inner.buffer_complete.load(Ordering::SeqCst) {
            let next = match self.inner.input.try_recv() {
                Err(_) => return Some(Err(Box::new(BufferDepletedError {}))),
                Ok(v) => v,
            };

            match next {
                Some(Err(e)) => {
                    // no further streaming
                    self.inner.buffer_complete.store(true, Ordering::SeqCst);

                    return Some(Err(e));
                }
                Some(Ok(data)) => {
                    self.inner.buffer.lock().unwrap().push(data.clone());
                    self.inner.buffer_index.fetch_add(1, Ordering::SeqCst);
                    self.inner.current_time.fetch_add(data.duration());
                    return Some(Ok(data));
                }
                None => {
                    self.inner.buffer_complete.store(true, Ordering::SeqCst);
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
        } else if !self.inner.seeking.load().is_nan() {
            let seek = self.inner.seeking.swap(f64::NAN);
            self.do_seek(seek);
        }

        // didn't manage to load enough data
        if !self.inner.seeking.load().is_nan() {
            return Some(Err(Box::new(BufferDepletedError {})));
        }

        // handle looping
        if self.inner.controller.loop_()
            && self.inner.current_time.load() > self.inner.controller.loop_end()
        {
            self.do_seek(self.inner.controller.loop_start());
        }

        // read from cache if available
        if let Some(data) = self
            .inner
            .buffer
            .lock()
            .unwrap()
            .get(self.inner.buffer_index.load(Ordering::SeqCst))
        {
            self.inner.buffer_index.fetch_add(1, Ordering::SeqCst);
            self.inner.current_time.fetch_add(data.duration());
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
        if !self.inner.controller.loop_() || self.inner.buffer.lock().unwrap().is_empty() {
            return None;
        }

        // loop and get next
        self.do_seek(self.inner.controller.loop_start());
        self.next()
    }
}
