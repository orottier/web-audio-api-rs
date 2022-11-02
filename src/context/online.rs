//! The `AudioContext` type and constructor options
use crate::context::{AudioContextState, BaseAudioContext, ConcreteBaseAudioContext};
use crate::io::{self, enumerate_devices, AudioBackendManager, ControlThreadInit, RenderThreadInit};
use crate::media::{MediaElement, MediaStream};
use crate::message::ControlMessage;
use crate::node::{self, ChannelConfigOptions};
use crate::AudioRenderCapacity;

use std::error::Error;
use std::sync::Mutex;

/// Identify the type of playback, which affects tradeoffs
/// between audio output latency and power consumption
#[derive(Clone, Debug)]
pub enum AudioContextLatencyCategory {
    /// Balance audio output latency and power consumption.
    Balanced,
    /// Provide the lowest audio output latency possible without glitching. This is the default.
    Interactive,
    /// Prioritize sustained playback without interruption
    /// over audio output latency. Lowest power consumption.
    Playback,
    /// Specify the number of seconds of latency
    /// this latency is not guaranted to be applied,
    /// it depends on the audio hardware capabilities
    Custom(f64),
}

impl Default for AudioContextLatencyCategory {
    fn default() -> Self {
        Self::Interactive
    }
}

/// Specify the playback configuration for the [`AudioContext`] constructor.
///
/// All fields are optional and will default to the value best suited for interactive playback on
/// your hardware configuration.
///
/// Check the documentation of the [`AudioContext` constructor](AudioContext::new) for usage
/// instructions.
#[derive(Clone, Debug, Default)]
pub struct AudioContextOptions {
    /// Identify the type of playback, which affects
    /// tradeoffs between audio output latency and power consumption
    pub latency_hint: AudioContextLatencyCategory,

    /// Sample rate of the audio context and audio output hardware
    pub sample_rate: Option<f32>,

    /// The audio output device
    /// - use `None` for the default device
    /// - use `Some(None)` to process the audio graph without playing through an audio output device.
    /// - use `Some(Some(sinkId))` to use the specified audio sink id, obtained with `enumerate_devices`
    pub sink_id: Option<Option<String>>,
}

/// This interface represents an audio graph whose `AudioDestinationNode` is routed to a real-time
/// output device that produces a signal directed at the user.
// the naming comes from the web audio specfication
#[allow(clippy::module_name_repetitions)]
pub struct AudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// audio backend (play/pause functionality)
    backend_manager: Mutex<Box<dyn AudioBackendManager>>,
    /// Provider for rendering performance metrics
    render_capacity: AudioRenderCapacity,
    /// Initializer for the render thread (when restart is required)
    render_thread_init: RenderThreadInit,
}

impl BaseAudioContext for AudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new(AudioContextOptions::default())
    }
}

impl AudioContext {
    /// Creates and returns a new `AudioContext` object.
    ///
    /// This will play live audio on the default output device.
    ///
    /// ```no_run
    /// use web_audio_api::context::{AudioContext, AudioContextOptions};
    ///
    /// // Request a sample rate of 44.1 kHz and default latency (buffer size 128, if available)
    /// let opts = AudioContextOptions {
    ///     sample_rate: Some(44100.),
    ///     ..AudioContextOptions::default()
    /// };
    ///
    /// // Setup the audio context that will emit to your speakers
    /// let context = AudioContext::new(opts);
    ///
    /// // Alternatively, use the default constructor to get the best settings for your hardware
    /// // let context = AudioContext::default();
    /// ```
    ///
    /// # Panics
    ///
    /// The `AudioContext` constructor will panic when an invalid `sinkId` is provided in the
    /// `AudioContextOptions`. In a future version, a `try_new` constructor will be introduced that
    /// will never panic.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(options: AudioContextOptions) -> Self {
        if let Some(Some(sink_id)) = &options.sink_id {
            if !enumerate_devices()
                .into_iter()
                .any(|d| d.device_id() == sink_id)
            {
                panic!("NotFoundError: invalid sinkId");
            }
        }

        let (control_thread_init, render_thread_init) = io::thread_init();
        let backend = io::build_output(options, render_thread_init.clone());

        let ControlThreadInit {
            frames_played,
            ctrl_msg_send,
            load_value_recv,
        } = control_thread_init;

        let graph = crate::render::graph::Graph::new();
        let message = crate::message::ControlMessage::Startup { graph };
        ctrl_msg_send.send(message).unwrap();

        let base = ConcreteBaseAudioContext::new(
            backend.sample_rate(),
            backend.number_of_channels(),
            frames_played,
            ctrl_msg_send,
            false,
        );
        base.set_state(AudioContextState::Running);

        // setup AudioRenderCapacity for this context
        let base_clone = base.clone();
        let render_capacity = AudioRenderCapacity::new(base_clone, load_value_recv);

        Self {
            base,
            backend_manager: Mutex::new(backend),
            render_capacity,
            render_thread_init,
        }
    }

    /// This represents the number of seconds of processing latency incurred by
    /// the `AudioContext` passing the audio from the `AudioDestinationNode`
    /// to the audio subsystem.
    // We don't do any buffering between rendering the audio and sending
    // it to the audio subsystem, so this value is zero. (see Gecko)
    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn base_latency(&self) -> f64 {
        0.
    }

    /// The estimation in seconds of audio output latency, i.e., the interval
    /// between the time the UA requests the host system to play a buffer and
    /// the time at which the first sample in the buffer is actually processed
    /// by the audio output device.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn output_latency(&self) -> f64 {
        self.backend_manager.lock().unwrap().output_latency()
    }

    /// Identifier or the information of the current audio output device.
    ///
    /// The initial value is `Some("")`, which means the default audio output device.
    #[allow(clippy::missing_panics_doc)]
    pub fn sink_id(&self) -> Option<String> {
        self.backend_manager
            .lock()
            .unwrap()
            .sink_id()
            .map(str::to_string)
    }

    /// Update the current audio output device.
    ///
    /// The provided `sink_id` string must match a device name `enumerate_devices`.
    ///
    /// Supplying `None` for the `sink_id` will process the audio graph without playing through an
    /// audio output device.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    #[allow(clippy::needless_collect, clippy::missing_panics_doc)]
    pub fn set_sink_id_sync(&self, sink_id: Option<String>) -> Result<(), Box<dyn Error>> {
        if self.sink_id() == sink_id {
            return Ok(()); // sink is already active
        }

        if let Some(sink_id_value) = &sink_id {
            if !enumerate_devices()
                .into_iter()
                .any(|d| d.device_id() == sink_id_value)
            {
                Err("NotFoundError: invalid sinkId")?;
            }
        }

        let mut backend_manager_guard = self.backend_manager.lock().unwrap();
        let state = self.state();
        if state == AudioContextState::Closed {
            return Ok(());
        }

        // Acquire exclusive lock on ctrl msg sender
        let ctrl_msg_send = self.base.lock_control_msg_sender();

        // Flush out the ctrl msg receiver, cache
        let mut pending_msgs: Vec<_> = self.render_thread_init.ctrl_msg_recv.try_iter().collect();

        // Acquire the active audio graph from the current render thread, shutting it down
        let graph = if matches!(pending_msgs.get(0), Some(ControlMessage::Startup { .. })) {
            // Handle the edge case where the previous backend was suspended for its entire lifetime.
            // In this case, the `Startup` control message was never processed.
            let msg = pending_msgs.remove(0);
            match msg {
                ControlMessage::Startup { graph } => graph,
                _ => unreachable!(),
            }
        } else {
            // Acquire the audio graph from the current render thread, shutting it down
            let (graph_send, graph_recv) = crossbeam_channel::bounded(1);
            let message = ControlMessage::Shutdown { sender: graph_send };
            ctrl_msg_send.send(message).unwrap();
            if state == AudioContextState::Suspended {
                // We must wake up the render thread to be able to handle the shutdown.
                // No new audio will be produced because it will receive the shutdown command first.
                backend_manager_guard.resume();
            }
            graph_recv.recv().unwrap()
        };

        // hotswap the backend
        let options = AudioContextOptions {
            sample_rate: Some(self.sample_rate()),
            latency_hint: AudioContextLatencyCategory::default(), // todo reuse existing setting
            sink_id: Some(sink_id),
        };
        *backend_manager_guard = io::build_output(options, self.render_thread_init.clone());

        // if the previous backend state was suspend, suspend the new one before shipping the graph
        if state == AudioContextState::Suspended {
            backend_manager_guard.suspend();
        }

        // send the audio graph to the new render thread
        let message = ControlMessage::Startup { graph };
        ctrl_msg_send.send(message).unwrap();

        self.base().set_state(AudioContextState::Running);

        // flush the cached msgs
        pending_msgs
            .into_iter()
            .for_each(|m| self.base().send_control_msg(m).unwrap());

        // explicitly release the lock to prevent concurrent render threads
        drop(backend_manager_guard);

        Ok(())
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn suspend_sync(&self) {
        if self.backend_manager.lock().unwrap().suspend() {
            self.base().set_state(AudioContextState::Suspended);
        }
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn resume_sync(&self) {
        if self.backend_manager.lock().unwrap().resume() {
            self.base().set_state(AudioContextState::Running);
        }
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    ///
    /// # Panics
    ///
    /// Will panic when this function is called multiple times
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn close_sync(&self) {
        self.backend_manager.lock().unwrap().close();

        self.base().set_state(AudioContextState::Closed);
    }

    /// Creates a [`MediaStreamAudioSourceNode`](node::MediaStreamAudioSourceNode) from a
    /// [`MediaStream`]
    #[must_use]
    pub fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode {
        let opts = node::MediaStreamAudioSourceOptions {
            media_stream: media,
        };
        node::MediaStreamAudioSourceNode::new(self, opts)
    }

    /// Creates a [`MediaStreamAudioDestinationNode`](node::MediaStreamAudioDestinationNode)
    #[must_use]
    pub fn create_media_stream_destination(&self) -> node::MediaStreamAudioDestinationNode {
        let opts = ChannelConfigOptions::default();
        node::MediaStreamAudioDestinationNode::new(self, opts)
    }

    /// Creates a [`MediaElementAudioSourceNode`](node::MediaElementAudioSourceNode) from a
    /// [`MediaElement`]
    #[must_use]
    pub fn create_media_element_source(
        &self,
        media_element: &mut MediaElement,
    ) -> node::MediaElementAudioSourceNode {
        let opts = node::MediaElementAudioSourceOptions { media_element };
        node::MediaElementAudioSourceNode::new(self, opts)
    }

    /// Returns an [`AudioRenderCapacity`] instance associated with an AudioContext.
    #[must_use]
    pub fn render_capacity(&self) -> &AudioRenderCapacity {
        &self.render_capacity
    }
}
