use crate::context::AudioNodeId;
use crate::AudioRenderCapacityEvent;

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

#[derive(Hash, Eq, PartialEq)]
pub(crate) enum EventType {
    Ended(AudioNodeId),
    SinkChange,
    RenderCapacity,
    ProcessorError(AudioNodeId),
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

pub(crate) enum EventPayload {
    None,
    RenderCapacity(AudioRenderCapacityEvent),
    ProcessorError(ErrorEvent),
}

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
        let self_clone = self.clone();

        std::thread::spawn(move || loop {
            // this thread is dedicated to event handling so we can block
            for event in event_channel.iter() {
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
            }
        });
    }

    pub fn set_handler(&self, event: EventType, callback: EventHandler) {
        self.event_handlers.lock().unwrap().insert(event, callback);
    }

    pub fn clear_handler(&self, event: EventType) {
        self.event_handlers.lock().unwrap().remove(&event);
    }
}
