//! The `OfflineAudioContext` type

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextState, BaseAudioContext, ConcreteBaseAudioContext};
use crate::events::{
    Event, EventDispatch, EventHandler, EventPayload, EventType, OfflineAudioCompletionEvent,
};
use crate::render::RenderThread;
use crate::stats::AudioStats;
use crate::{
    assert_valid_buffer_length, assert_valid_number_of_channels, assert_valid_sample_rate,
    RENDER_QUANTUM_SIZE,
};

use crate::events::EventLoop;
use futures_channel::{mpsc, oneshot};
use futures_util::SinkExt as _;

pub(crate) type OfflineAudioContextCallback =
    dyn FnOnce(&mut OfflineAudioContext) + Send + Sync + 'static;

/// The `OfflineAudioContext` doesn't render the audio to the device hardware; instead, it generates
/// it, as fast as it can, and outputs the result to an `AudioBuffer`.
// the naming comes from the web audio specification
#[allow(clippy::module_name_repetitions)]
pub struct OfflineAudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// the size of the buffer in sample-frames
    length: usize,
    /// actual renderer of the audio graph, can only be called once
    renderer: Mutex<Option<OfflineAudioContextRenderer>>,
    /// channel to notify resume actions on the rendering
    resume_sender: mpsc::Sender<()>,
    /// channel to inject `suspend` points scheduled after rendering has started
    /// (see the matching `suspend_injection_receiver` on the renderer)
    suspend_injection_sender: mpsc::UnboundedSender<(usize, oneshot::Sender<()>)>,
    /// Incremental rendering state (embedders' drive model; see
    /// [`Self::render_upto_sync`])
    incremental: Mutex<Option<IncrementalRender>>,
    /// Finished incremental render, waiting to be taken by
    /// [`Self::take_rendered_sync`]
    rendered: Mutex<Option<AudioBuffer>>,
    /// Number of frames rendered so far. This must be a lock-free atomic and
    /// not derived from `incremental`: the early-return paths of
    /// `render_upto_sync` report the frame count *while holding* the
    /// `incremental` MutexGuard, and a std Mutex is not re-entrant, so reading
    /// the count through the mutex from there would self-deadlock the calling
    /// thread (both sequences are legal: render to the end, take the result,
    /// call render_upto_sync again; or start_rendering_sync followed by
    /// render_upto_sync). With an atomic, that deadlock path does not exist by
    /// construction. It also fixes a semantic wart: after the result has been
    /// taken, both `incremental` and `rendered` are None and the count would
    /// otherwise read as 0 instead of `length`.
    rendered_frames: AtomicUsize,
}

/// Cross-call state of an incremental offline render.
struct IncrementalRender {
    renderer: crate::render::RenderThread,
    buffer: Vec<Vec<f32>>,
    /// Number of render quanta produced so far
    quantum: usize,
    event_loop: crate::events::EventLoop,
}

impl std::fmt::Debug for OfflineAudioContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineAudioContext")
            .field("length", &self.length())
            .field("base", &self.base())
            .finish_non_exhaustive()
    }
}

struct OfflineAudioContextRenderer {
    /// the rendering 'thread', fully controlled by the offline context
    renderer: RenderThread,
    /// sorted list of promises to resolve at certain render quanta (via `suspend`)
    suspend_promises: Vec<(usize, oneshot::Sender<()>)>,
    /// sorted list of callbacks to run at certain render quanta (via `suspend_sync`)
    suspend_callbacks: Vec<(usize, Box<OfflineAudioContextCallback>)>,
    /// channel to listen for `resume` calls on a suspended context
    resume_receiver: mpsc::Receiver<()>,
    /// channel for `suspend` points registered *after* rendering has started
    /// (e.g. scheduled from inside a suspend promise callback); drained by the
    /// render loop and merged into the pending suspend list
    suspend_injection_receiver: mpsc::UnboundedReceiver<(usize, oneshot::Sender<()>)>,
    /// event loop to run after each render quantum
    event_loop: EventLoop,
}

impl BaseAudioContext for OfflineAudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }
}

impl OfflineAudioContext {
    /// Creates an `OfflineAudioContext` instance
    ///
    /// # Arguments
    ///
    /// * `channels` - number of output channels to render
    /// * `length` - length of the rendering audio buffer
    /// * `sample_rate` - output sample rate
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn new(number_of_channels: usize, length: usize, sample_rate: f32) -> Self {
        assert_valid_number_of_channels(number_of_channels);
        assert_valid_buffer_length(length);
        assert_valid_sample_rate(sample_rate);

        // communication channel to the render thread,
        // unbounded is fine because it does not need to be realtime safe
        let (sender, receiver) = crossbeam_channel::unbounded();

        let (node_id_producer, node_id_consumer) = llq::Queue::new().split();
        let graph = crate::render::graph::Graph::new(node_id_producer);
        let message = crate::message::ControlMessage::Startup { graph };
        sender.send(message).unwrap();

        // track number of frames - synced from render thread to control thread
        let frames_played = Arc::new(AtomicU64::new(0));
        let frames_played_clone = Arc::clone(&frames_played);
        let state = Arc::new(AtomicU8::new(AudioContextState::Suspended as u8));
        let state_clone = Arc::clone(&state);

        // Communication channel for events from the render thread to the control thread.
        // Use an unbounded channel because we do not require real-time safety.
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);

        // setup the render 'thread', which will run inside the control thread
        let renderer = RenderThread::new(
            sample_rate,
            number_of_channels,
            receiver,
            state_clone,
            frames_played_clone,
            AudioStats::new(),
            event_send.clone(),
        );

        // first, setup the base audio context
        let base = ConcreteBaseAudioContext::new(
            sample_rate,
            number_of_channels,
            state,
            frames_played,
            sender,
            event_send,
            event_loop.clone(),
            true,
            node_id_consumer,
        );

        let (resume_sender, resume_receiver) = mpsc::channel(0);
        let (suspend_injection_sender, suspend_injection_receiver) = mpsc::unbounded();

        let renderer = OfflineAudioContextRenderer {
            renderer,
            suspend_promises: Vec::new(),
            suspend_callbacks: Vec::new(),
            resume_receiver,
            suspend_injection_receiver,
            event_loop,
        };

        Self {
            base,
            length,
            renderer: Mutex::new(Some(renderer)),
            resume_sender,
            suspend_injection_sender,
            incremental: Mutex::new(None),
            rendered: Mutex::new(None),
            rendered_frames: AtomicUsize::new(0),
        }
    }

    /// Given the current connections and scheduled changes, starts rendering audio.
    ///
    /// This function will block the current thread and returns the rendered `AudioBuffer`
    /// synchronously.
    ///
    /// This method will only adhere to scheduled suspensions via [`Self::suspend_sync`] and
    /// will ignore those provided via [`Self::suspend`].
    ///
    /// # Panics
    ///
    /// Panics if this method is called multiple times

    // ──────────────────────────────────────────────────────────────────────────
    // Incremental offline rendering (for embedders)
    //
    // start_rendering_sync renders the whole buffer in one call and consumes
    // the renderer, and suspend points must all be registered before rendering
    // starts (suspending after the renderer has been taken panics). Embedders
    // hosting a JavaScript engine drive rendering incrementally instead:
    // render up to a suspend point, hand control back to script (which mutates
    // the graph inside the suspend promise callback), resume, render the next
    // segment. The three methods below provide that capability; they are
    // mutually exclusive with suspend/suspend_sync.
    // ──────────────────────────────────────────────────────────────────────────

    /// Renders up to `upto_frame` (exclusive; rounded up to whole render
    /// quanta; >= length renders to the end). May be called repeatedly to
    /// continue. Returns the total number of rendered frames. When the end is
    /// reached the result is stored and can be taken once through
    /// [`Self::take_rendered_sync`].
    ///
    /// Calling after the render has finished (result stored or already taken)
    /// or after `start_rendering*` has claimed the renderer is a no-op that
    /// returns the current frame count. Those early returns run while the
    /// `incremental` lock is held, which is why the frame count lives in a
    /// lock-free atomic (see the `rendered_frames` field comment).
    pub fn render_upto_sync(&mut self, upto_frame: usize) -> usize {
        let length = self.length;
        let num_quanta = length.div_ceil(RENDER_QUANTUM_SIZE);

        let mut inc_guard = self.incremental.lock().unwrap();
        if inc_guard.is_none() {
            // First call: claim the renderer (mutually exclusive with
            // start_rendering_sync - both take() it).
            let Some(r) = self.renderer.lock().unwrap().take() else {
                // Already finished or claimed by start_rendering* - no-op.
                // Note: inc_guard is still held here, so only the lock-free
                // rendered_frames may be read.
                return self.rendered_frames_sync();
            };
            let mut buffer = Vec::with_capacity(r.renderer.output_channels());
            buffer.resize_with(buffer.capacity(), || Vec::with_capacity(length));
            *inc_guard = Some(IncrementalRender {
                renderer: r.renderer,
                buffer,
                quantum: 0,
                event_loop: r.event_loop,
            });
        }

        let Some(inc) = inc_guard.as_mut() else {
            // As above: `incremental` must not be locked again from here.
            return self.rendered_frames_sync();
        };

        // A segment is being rendered - Running (the previous segment may have
        // parked the state at Suspended, see the end of this function).
        self.base.set_state(AudioContextState::Running);

        let target_q = if upto_frame >= length {
            num_quanta
        } else {
            upto_frame.div_ceil(RENDER_QUANTUM_SIZE).min(num_quanta)
        };
        if target_q > inc.quantum {
            let n = target_q - inc.quantum;
            let ev = inc.event_loop.clone();
            inc.renderer.render_offline_quanta(&mut inc.buffer, n, &ev);
            inc.quantum = target_q;
        }

        let done = inc.quantum >= num_quanta;
        let frames = (inc.quantum * RENDER_QUANTUM_SIZE).min(length);
        self.rendered_frames.store(frames, Ordering::Release);

        if done {
            // Rendered to the end: finalize and store the result.
            // finish_offline_render consumes the RenderThread (unload_graph
            // takes self by value), so the whole IncrementalRender is taken out
            // and destructured.
            let owned = inc_guard.take().expect("just checked");
            drop(inc_guard);
            let IncrementalRender {
                renderer,
                mut buffer,
                event_loop,
                ..
            } = owned;
            renderer.finish_offline_render(&event_loop);
            for ch in buffer.iter_mut() {
                ch.truncate(length); // the last quantum may overshoot `length`
            }
            let result = AudioBuffer::from(buffer, self.base.sample_rate());
            *self.rendered.lock().unwrap() = Some(result.clone());
            self.base.set_state(AudioContextState::Closed);
            let _ = self.base.send_event(EventDispatch::complete(result));
            // Spin the event loop once more after finalizing: the
            // complete/statechange events are only queued after
            // finish_offline_render, past the last spin inside it. Without this
            // extra spin the crate-side oncomplete/onstatechange handlers would
            // never fire. Matches the tail of start_rendering_sync.
            event_loop.handle_pending_events();
        } else {
            // Parked at a suspend point: per the spec, an OfflineAudioContext
            // must report "suspended" while parked (leaving it at Running lets
            // script observe "running" from inside the suspend callback).
            self.base.set_state(AudioContextState::Suspended);
        }
        frames
    }

    /// Number of frames rendered so far (currentTime = frames / sampleRate).
    /// Lock-free (see the `rendered_frames` field comment).
    #[must_use]
    pub fn rendered_frames_sync(&self) -> usize {
        self.rendered_frames.load(Ordering::Acquire)
    }

    /// Takes the finished result after rendering reached the end (None if the
    /// render is unfinished or the result was already taken).
    #[must_use]
    pub fn take_rendered_sync(&mut self) -> Option<AudioBuffer> {
        self.rendered.lock().unwrap().take()
    }

    #[must_use]
    pub fn start_rendering_sync(&mut self) -> AudioBuffer {
        let renderer = self
            .renderer
            .lock()
            .unwrap()
            .take()
            .expect("InvalidStateError - Cannot call `startRendering` twice");

        let OfflineAudioContextRenderer {
            renderer,
            suspend_callbacks,
            event_loop,
            ..
        } = renderer;

        self.base.set_state(AudioContextState::Running);

        let result = renderer.render_audiobuffer_sync(self, suspend_callbacks, &event_loop);

        self.base.set_state(AudioContextState::Closed);
        let _ = self
            .base
            .send_event(EventDispatch::complete(result.clone()));

        // spin the event loop once more to handle the statechange/complete events
        event_loop.handle_pending_events();

        result
    }

    /// Given the current connections and scheduled changes, starts rendering audio.
    ///
    /// Rendering is purely CPU bound and contains no `await` points, so calling this method will
    /// block the executor until completion or until the context is suspended.
    ///
    /// This method will only adhere to scheduled suspensions via [`Self::suspend`] and will
    /// ignore those provided via [`Self::suspend_sync`].
    ///
    /// # Panics
    ///
    /// Panics if this method is called multiple times.
    pub async fn start_rendering(&self) -> AudioBuffer {
        // We are mixing async with a std Mutex, so be sure not to `await` while the lock is held
        let renderer = self
            .renderer
            .lock()
            .unwrap()
            .take()
            .expect("InvalidStateError - Cannot call `startRendering` twice");

        let OfflineAudioContextRenderer {
            renderer,
            suspend_promises,
            resume_receiver,
            suspend_injection_receiver,
            event_loop,
            ..
        } = renderer;

        self.base.set_state(AudioContextState::Running);

        let result = renderer
            .render_audiobuffer(
                self.length,
                suspend_promises,
                resume_receiver,
                suspend_injection_receiver,
                &event_loop,
            )
            .await;

        self.base.set_state(AudioContextState::Closed);
        let _ = self
            .base
            .send_event(EventDispatch::complete(result.clone()));

        // spin the event loop once more to handle the statechange/complete events
        event_loop.handle_pending_events();

        result
    }

    /// get the length of rendering audio buffer
    // false positive: OfflineAudioContext is not const
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }

    #[track_caller]
    fn calculate_suspend_frame(&self, suspend_time: f64) -> usize {
        assert!(
            suspend_time >= 0.,
            "InvalidStateError: suspendTime cannot be negative"
        );
        assert!(
            suspend_time < self.length as f64 / self.sample_rate() as f64,
            "InvalidStateError: suspendTime cannot be greater than or equal to the total render duration"
        );
        (suspend_time * self.base.sample_rate() as f64 / RENDER_QUANTUM_SIZE as f64).ceil() as usize
    }

    /// Schedules a suspension of the time progression in the audio context at the specified time
    /// and returns a promise
    ///
    /// The specified time is quantized and rounded up to the render quantum size.
    ///
    /// Suspend points may be scheduled before rendering starts *or dynamically
    /// after it has started* - e.g. from inside a previous suspend point's
    /// promise callback, which is how the standard suspend/resume contract is
    /// meant to be driven (the suspend times are generally not known up front).
    ///
    /// # Panics
    ///
    /// Panics (synchronously, when this method is called) if the quantized frame number
    ///
    /// - is negative or
    /// - is less than or equal to the current time or
    /// - is greater than or equal to the total render duration or
    /// - is scheduled by another suspend for the same time
    ///
    /// # Example usage
    ///
    /// ```rust
    /// use futures::{executor, join};
    /// use futures::FutureExt as _;
    /// use std::sync::Arc;
    ///
    /// use web_audio_api::context::BaseAudioContext;
    /// use web_audio_api::context::OfflineAudioContext;
    /// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
    ///
    /// let context = Arc::new(OfflineAudioContext::new(1, 512, 44_100.));
    /// let context_clone = Arc::clone(&context);
    ///
    /// let suspend_promise = context.suspend(128. / 44_100.).then(|_| async move {
    ///     let mut src = context_clone.create_constant_source();
    ///     src.connect(&context_clone.destination());
    ///     src.start();
    ///     context_clone.resume().await;
    /// });
    ///
    /// let render_promise = context.start_rendering();
    ///
    /// let buffer = executor::block_on(async move { join!(suspend_promise, render_promise).1 });
    /// assert_eq!(buffer.number_of_channels(), 1);
    /// assert_eq!(buffer.length(), 512);
    /// ```
    pub fn suspend(&self, suspend_time: f64) -> impl std::future::Future<Output = ()> + '_ {
        let quantum = self.calculate_suspend_frame(suspend_time);

        let (sender, receiver) = oneshot::channel();

        // Register the suspend point synchronously (before returning the future),
        // so that a point scheduled right before `resume()` - the standard driving
        // pattern - is guaranteed to be in place before the render loop advances
        // past it. Only waiting for the point to be reached is deferred to the
        // returned future.
        {
            let mut lock = self.renderer.lock().unwrap();
            match lock.as_mut() {
                // Rendering has not started yet: register the suspend point on
                // the renderer directly, as before.
                Some(renderer) => {
                    let insert_pos = renderer
                        .suspend_promises
                        .binary_search_by_key(&quantum, |&(q, _)| q)
                        .expect_err(
                            "InvalidStateError - cannot suspend multiple times at the same render quantum",
                        );

                    renderer
                        .suspend_promises
                        .insert(insert_pos, (quantum, sender));
                }
                // Rendering is already underway (the renderer has been moved into
                // `render_audiobuffer`, typically because we are being called from
                // inside a suspend promise callback). Inject the new suspend point
                // into the running render loop instead of panicking. The render
                // loop drains this channel and merges the point into its pending
                // list, so suspend points can be scheduled dynamically - which is
                // what the standard suspend/resume contract requires.
                None => {
                    self.suspend_injection_sender
                        .unbounded_send((quantum, sender))
                        .expect("InvalidStateError - cannot suspend, rendering has finished");
                }
            }
        } // lock is dropped

        async move {
            receiver.await.unwrap();
            self.base().set_state(AudioContextState::Suspended);
        }
    }

    /// Schedules a suspension of the time progression in the audio context at the specified time
    /// and runs a callback.
    ///
    /// This is a synchronous version of [`Self::suspend`] that runs the provided callback at
    /// the `suspendTime`. The rendering resumes automatically after the callback has run, so there
    /// is no `resume_sync` method.
    ///
    /// The specified time is quantized and rounded up to the render quantum size.
    ///
    /// # Panics
    ///
    /// Panics if the quantized frame number
    ///
    /// - is negative or
    /// - is less than or equal to the current time or
    /// - is greater than or equal to the total render duration or
    /// - is scheduled by another suspend for the same time
    ///
    /// # Example usage
    ///
    /// ```rust
    /// use web_audio_api::context::BaseAudioContext;
    /// use web_audio_api::context::OfflineAudioContext;
    /// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
    ///
    /// let mut context = OfflineAudioContext::new(1, 512, 44_100.);
    ///
    /// context.suspend_sync(128. / 44_100., |context| {
    ///     let mut src = context.create_constant_source();
    ///     src.connect(&context.destination());
    ///     src.start();
    /// });
    ///
    /// let buffer = context.start_rendering_sync();
    /// assert_eq!(buffer.number_of_channels(), 1);
    /// assert_eq!(buffer.length(), 512);
    /// ```
    pub fn suspend_sync<F: FnOnce(&mut Self) + Send + Sync + 'static>(
        &mut self,
        suspend_time: f64,
        callback: F,
    ) {
        let quantum = self.calculate_suspend_frame(suspend_time);

        let mut lock = self.renderer.lock().unwrap();
        let renderer = lock
            .as_mut()
            .expect("InvalidStateError - cannot suspend when rendering has already started");

        let insert_pos = renderer
            .suspend_callbacks
            .binary_search_by_key(&quantum, |(q, _c)| *q)
            .expect_err(
                "InvalidStateError - cannot suspend multiple times at the same render quantum",
            );

        let boxed_callback = Box::new(|ctx: &mut OfflineAudioContext| {
            ctx.base().set_state(AudioContextState::Suspended);
            (callback)(ctx);
            ctx.base().set_state(AudioContextState::Running);
        });

        renderer
            .suspend_callbacks
            .insert(insert_pos, (quantum, boxed_callback));
    }

    /// Resumes the progression of the OfflineAudioContext's currentTime when it has been suspended
    ///
    /// # Panics
    ///
    /// Panics when the context is closed or rendering has not started
    pub async fn resume(&self) {
        self.base().set_state(AudioContextState::Running);
        self.resume_sender.clone().send(()).await.unwrap()
    }

    /// Register callback to run when the rendering has completed
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    #[allow(clippy::missing_panics_doc)]
    pub fn set_oncomplete<F: FnOnce(OfflineAudioCompletionEvent) + Send + 'static>(
        &self,
        callback: F,
    ) {
        let callback = move |v| match v {
            EventPayload::Complete(v) => {
                let event = OfflineAudioCompletionEvent {
                    rendered_buffer: v,
                    event: Event { type_: "complete" },
                };
                callback(event)
            }
            _ => unreachable!(),
        };

        self.base()
            .set_event_handler(EventType::Complete, EventHandler::Once(Box::new(callback)));
    }

    /// Unset the callback to run when the rendering has completed
    pub fn clear_oncomplete(&self) {
        self.base().clear_event_handler(EventType::Complete);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::node::AudioNode;
    use crate::node::AudioScheduledSourceNode;

    #[test]
    fn test_sample_rate_length() {
        let context = OfflineAudioContext::new(1, 48000, 96000.);
        assert_float_eq!(context.sample_rate(), 96000., abs_all <= 0.);
        assert_eq!(context.length(), 48000);
    }

    #[test]
    fn render_empty_graph() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);
        assert_eq!(context.state(), AudioContextState::Suspended);
        let buffer = context.start_rendering_sync();

        assert_eq!(context.length(), 555);

        assert_eq!(buffer.number_of_channels(), 2);
        assert_eq!(buffer.length(), 555);
        assert_float_eq!(buffer.get_channel_data(0), &[0.; 555][..], abs_all <= 0.);
        assert_float_eq!(buffer.get_channel_data(1), &[0.; 555][..], abs_all <= 0.);

        assert_eq!(context.state(), AudioContextState::Closed);
    }

    #[test]
    #[should_panic]
    fn render_twice_panics() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);
        let _ = context.start_rendering_sync();
        let _ = context.start_rendering_sync();
    }

    #[test]
    fn test_suspend_sync() {
        use crate::node::ConstantSourceNode;
        use std::sync::OnceLock;

        let len = RENDER_QUANTUM_SIZE * 4;
        let sample_rate = 48000_f64;

        let mut context = OfflineAudioContext::new(1, len, sample_rate as f32);
        static SOURCE: OnceLock<ConstantSourceNode> = OnceLock::new();

        context.suspend_sync(RENDER_QUANTUM_SIZE as f64 / sample_rate, |context| {
            assert_eq!(context.state(), AudioContextState::Suspended);
            let mut src = context.create_constant_source();
            src.connect(&context.destination());
            src.start();
            SOURCE.set(src).unwrap();
        });

        context.suspend_sync((3 * RENDER_QUANTUM_SIZE) as f64 / sample_rate, |context| {
            assert_eq!(context.state(), AudioContextState::Suspended);
            SOURCE.get().unwrap().disconnect();
        });

        let output = context.start_rendering_sync();

        assert_float_eq!(
            output.get_channel_data(0)[..RENDER_QUANTUM_SIZE],
            &[0.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            output.get_channel_data(0)[RENDER_QUANTUM_SIZE..3 * RENDER_QUANTUM_SIZE],
            &[1.; 2 * RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            output.get_channel_data(0)[3 * RENDER_QUANTUM_SIZE..4 * RENDER_QUANTUM_SIZE],
            &[0.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn render_suspend_resume_async() {
        use futures::executor;
        use futures::join;
        use futures::FutureExt as _;

        let context = Arc::new(OfflineAudioContext::new(1, 512, 44_100.));
        let context_clone = Arc::clone(&context);

        let suspend_promise = context.suspend(128. / 44_100.).then(|_| async move {
            let mut src = context_clone.create_constant_source();
            src.connect(&context_clone.destination());
            src.start();
            context_clone.resume().await;
        });

        let render_promise = context.start_rendering();

        let buffer = executor::block_on(async move { join!(suspend_promise, render_promise).1 });

        assert_eq!(buffer.number_of_channels(), 1);
        assert_eq!(buffer.length(), 512);

        assert_float_eq!(
            buffer.get_channel_data(0)[..128],
            &[0.; 128][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            buffer.get_channel_data(0)[128..],
            &[1.; 384][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn render_dynamic_suspend_after_start_async() {
        // The standard suspend/resume driving pattern schedules each suspend
        // point from inside the previous point's promise callback - i.e. the
        // suspend times are not known up front and later ones are scheduled
        // *after* `start_rendering()` has begun. Before dynamic suspend
        // injection, `suspend()` panicked in that case ("cannot suspend when
        // rendering has already started"). This mirrors that pattern and checks
        // the dynamically scheduled points take effect at the right frames.
        use futures::executor;
        use futures::join;

        let sample_rate = 48_000.0_f32;
        let quantum = RENDER_QUANTUM_SIZE as f64; // suspend(k * quantum / sr) lands on quantum k
        let length = RENDER_QUANTUM_SIZE * 5; // 5 quanta
        let context = Arc::new(OfflineAudioContext::new(1, length, sample_rate));

        let mut src = context.create_constant_source();
        src.offset().set_value(1.0);
        src.connect(&context.destination());
        src.start();

        let ctx = Arc::clone(&context);
        let suspend_time = |k: u32| k as f64 * quantum / sample_rate as f64;
        let driver = async move {
            // suspend(1) is registered before rendering starts; the remaining
            // points are scheduled from *inside* the previous point's handler,
            // i.e. after `start_rendering` has claimed the renderer. Each next
            // point is scheduled *before* the matching resume - mirroring the
            // standard `suspend(t).then(|| { schedule_next(); resume(); })`
            // driving pattern - so it is registered while the render is parked.
            let mut current = Some(ctx.suspend(suspend_time(1)));
            for k in 1..5u32 {
                current.take().unwrap().await;
                // mutate the graph at the suspend point: the next segment carries k+1
                src.offset().set_value((k + 1) as f32);
                if k + 1 < 5 {
                    // dynamic: registered now (render is parked), before resume
                    current = Some(ctx.suspend(suspend_time(k + 1)));
                }
                ctx.resume().await;
            }
        };

        let render = context.start_rendering();
        let buffer = executor::block_on(async move { join!(driver, render).1 });

        assert_eq!(buffer.length(), length);
        let ch = buffer.get_channel_data(0);
        // segment [k, k+1) carries value k+1: q0 = 1 (initial), q1 = 2, ... q4 = 5
        for k in 0..5usize {
            assert_float_eq!(ch[k * RENDER_QUANTUM_SIZE + 10], (k + 1) as f32, abs <= 1e-6);
        }
    }

    #[test]
    #[should_panic]
    fn test_suspend_negative_panics() {
        let mut context = OfflineAudioContext::new(2, 128, 44_100.);
        context.suspend_sync(-1.0, |_| ());
    }

    #[test]
    #[should_panic]
    fn test_suspend_after_duration_panics() {
        let mut context = OfflineAudioContext::new(2, 128, 44_100.);
        context.suspend_sync(1.0, |_| ());
    }

    #[test]
    #[should_panic]
    fn test_suspend_after_render_panics() {
        let mut context = OfflineAudioContext::new(2, 128, 44_100.);
        let _ = context.start_rendering_sync();
        context.suspend_sync(0.0, |_| ());
    }

    #[test]
    #[should_panic]
    fn test_suspend_identical_frame_panics() {
        let mut context = OfflineAudioContext::new(2, 128, 44_100.);
        context.suspend_sync(0.0, |_| ());
        context.suspend_sync(0.0, |_| ());
    }

    #[test]
    fn test_onstatechange() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);

        let changed = Arc::new(AtomicBool::new(false));
        let changed_clone = Arc::clone(&changed);
        context.set_onstatechange(move |_event| {
            changed_clone.store(true, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();

        assert!(changed.load(Ordering::Relaxed));
    }

    #[test]
    fn test_onstatechange_async() {
        use futures::executor;

        let context = OfflineAudioContext::new(2, 555, 44_100.);

        let changed = Arc::new(AtomicBool::new(false));
        let changed_clone = Arc::clone(&changed);
        context.set_onstatechange(move |_event| {
            changed_clone.store(true, Ordering::Relaxed);
        });

        let _ = executor::block_on(context.start_rendering());

        assert!(changed.load(Ordering::Relaxed));
    }

    #[test]
    fn test_oncomplete() {
        let mut context = OfflineAudioContext::new(2, 555, 44_100.);

        let complete = Arc::new(AtomicBool::new(false));
        let complete_clone = Arc::clone(&complete);
        context.set_oncomplete(move |event| {
            assert_eq!(event.rendered_buffer.length(), 555);
            complete_clone.store(true, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();

        assert!(complete.load(Ordering::Relaxed));
    }

    #[test]
    fn test_oncomplete_async() {
        use futures::executor;

        let context = OfflineAudioContext::new(2, 555, 44_100.);

        let complete = Arc::new(AtomicBool::new(false));
        let complete_clone = Arc::clone(&complete);
        context.set_oncomplete(move |event| {
            assert_eq!(event.rendered_buffer.length(), 555);
            complete_clone.store(true, Ordering::Relaxed);
        });

        let _ = executor::block_on(context.start_rendering());

        assert!(complete.load(Ordering::Relaxed));
    }

    fn require_send_sync<T: Send + Sync>(_: T) {}

    #[test]
    fn test_all_futures_thread_safe() {
        let context = OfflineAudioContext::new(2, 555, 44_100.);

        require_send_sync(context.start_rendering());
        // suspend now registers synchronously, so the time must be valid on call
        require_send_sync(context.suspend(0.));
        require_send_sync(context.resume());
    }
}

#[cfg(test)]
mod incremental_render_tests {
    use float_eq::assert_float_eq;

    use super::*;
    use crate::node::{AudioNode, AudioScheduledSourceNode};

    fn build_graph(context: &mut OfflineAudioContext) {
        let mut osc = context.create_oscillator();
        osc.frequency().set_value(440.);
        osc.connect(&context.destination());
        osc.start();
    }

    #[test]
    fn test_chunked_render_equals_one_shot() {
        // Rendering in arbitrary increments must produce the exact same samples
        // as a single start_rendering_sync pass over an identical graph.
        let length = 1000; // deliberately not a multiple of the quantum size

        let mut reference = OfflineAudioContext::new(1, length, 48000.);
        build_graph(&mut reference);
        let expected = reference.start_rendering_sync();

        let mut chunked = OfflineAudioContext::new(1, length, 48000.);
        build_graph(&mut chunked);
        assert_eq!(chunked.render_upto_sync(100), 128); // rounded up to quanta
        assert_eq!(chunked.state(), AudioContextState::Suspended);
        assert_eq!(chunked.render_upto_sync(600), 640);
        let frames = chunked.render_upto_sync(usize::MAX);
        assert_eq!(frames, length);
        assert_eq!(chunked.state(), AudioContextState::Closed);

        let result = chunked.take_rendered_sync().expect("render finished");
        assert_float_eq!(
            result.get_channel_data(0),
            expected.get_channel_data(0),
            abs_all <= 0.
        );

        // The result can only be taken once; afterwards the frame count keeps
        // reporting the full length and further render calls are no-ops (this
        // sequence used to self-deadlock when the frame count was derived from
        // the incremental state under its own mutex).
        assert!(chunked.take_rendered_sync().is_none());
        assert_eq!(chunked.render_upto_sync(usize::MAX), length);
        assert_eq!(chunked.rendered_frames_sync(), length);
    }

    #[test]
    fn test_render_upto_after_start_rendering_is_noop() {
        // start_rendering_sync claims the renderer; a later incremental call
        // must not panic or deadlock, just report the current frame count.
        let mut context = OfflineAudioContext::new(1, 256, 48000.);
        build_graph(&mut context);
        let _ = context.start_rendering_sync();
        let _ = context.render_upto_sync(usize::MAX);
    }
}
