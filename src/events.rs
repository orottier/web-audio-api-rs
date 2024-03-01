use crate::context::{AudioContextState, AudioNodeId};
use crate::{AudioBuffer, AudioRenderCapacityEvent};

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;

/// The Event interface
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Event {
    pub type_: &'static str,
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub(crate) enum EventType {
    Ended(AudioNodeId),
    SinkChange,
    StateChange,
    RenderCapacity,
    ProcessorError(AudioNodeId),
    Diagnostics,
    Message(AudioNodeId),
}

/// The Error Event interface
#[non_exhaustive]
#[derive(Debug)]
pub struct ErrorEvent {
    /// The error message
    pub message: String,
    /// The object with which panic was originally invoked.
    pub error: Box<dyn Any + Send>,
    /// Inherits from this base Event
    pub event: Event,
}

/// The OfflineAudioCompletionEvent Event interface
#[non_exhaustive]
#[derive(Debug)]
pub struct OfflineAudioCompletionEvent {
    /// The rendered AudioBuffer
    pub rendered_buffer: AudioBuffer,
    /// Inherits from this base Event
    pub event: Event,
}

#[derive(Debug)]
pub(crate) enum EventPayload {
    None,
    RenderCapacity(AudioRenderCapacityEvent),
    ProcessorError(ErrorEvent),
    Diagnostics(Vec<u8>),
    Message(Box<dyn Any + Send + 'static>),
    AudioContextState(AudioContextState),
}

#[derive(Debug)]
pub(crate) struct EventDispatch {
    type_: EventType,
    payload: EventPayload,
}

impl EventDispatch {
    pub fn ended(id: AudioNodeId) -> Self {
        EventDispatch {
            type_: EventType::Ended(id),
            payload: EventPayload::None,
        }
    }

    pub fn sink_change() -> Self {
        EventDispatch {
            type_: EventType::SinkChange,
            payload: EventPayload::None,
        }
    }

    pub fn state_change(state: AudioContextState) -> Self {
        EventDispatch {
            type_: EventType::StateChange,
            payload: EventPayload::AudioContextState(state),
        }
    }

    pub fn render_capacity(value: AudioRenderCapacityEvent) -> Self {
        EventDispatch {
            type_: EventType::RenderCapacity,
            payload: EventPayload::RenderCapacity(value),
        }
    }

    pub fn processor_error(id: AudioNodeId, value: ErrorEvent) -> Self {
        EventDispatch {
            type_: EventType::ProcessorError(id),
            payload: EventPayload::ProcessorError(value),
        }
    }

    pub fn diagnostics(value: Vec<u8>) -> Self {
        EventDispatch {
            type_: EventType::Diagnostics,
            payload: EventPayload::Diagnostics(value),
        }
    }

    pub fn message(id: AudioNodeId, value: Box<dyn Any + Send + 'static>) -> Self {
        EventDispatch {
            type_: EventType::Message(id),
            payload: EventPayload::Message(value),
        }
    }
}

pub(crate) enum EventHandler {
    Once(Box<dyn FnOnce(EventPayload) + Send + 'static>),
    Multiple(Box<dyn FnMut(EventPayload) + Send + 'static>),
}

#[derive(Clone, Default)]
pub(crate) struct EventLoop {
    event_handlers: Arc<Mutex<HashMap<EventType, EventHandler>>>,
}

impl EventLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&self, event_channel: Receiver<EventDispatch>) {
        log::debug!("Entering event loop");
        let self_clone = self.clone();

        std::thread::spawn(move || {
            // This thread is dedicated to event handling so we can block
            for mut event in event_channel.iter() {
                // Terminate the event loop when the audio context is closing
                let mut terminate = false;
                if matches!(
                    event.payload,
                    EventPayload::AudioContextState(AudioContextState::Closed)
                ) {
                    event.payload = EventPayload::None; // the statechange handler takes no argument
                    terminate = true;
                }

                let mut event_handler_lock = self_clone.event_handlers.lock().unwrap();
                let callback_option = event_handler_lock.remove(&event.type_);
                drop(event_handler_lock); // release Mutex while running callback

                if let Some(callback) = callback_option {
                    match callback {
                        EventHandler::Once(f) => (f)(event.payload),
                        EventHandler::Multiple(mut f) => {
                            (f)(event.payload);
                            self_clone
                                .event_handlers
                                .lock()
                                .unwrap()
                                .insert(event.type_, EventHandler::Multiple(f));
                        }
                    };
                }

                if terminate {
                    break;
                }
            }

            log::debug!("Event loop has terminated");
        });
    }

    pub fn set_handler(&self, event: EventType, callback: EventHandler) {
        self.event_handlers.lock().unwrap().insert(event, callback);
    }

    pub fn clear_handler(&self, event: EventType) {
        self.event_handlers.lock().unwrap().remove(&event);
    }
}
