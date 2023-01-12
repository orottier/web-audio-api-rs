use crate::context::AudioNodeId;
use crate::AudioRenderCapacityEvent;

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;

#[derive(Hash, Eq, PartialEq)]
pub(crate) enum EventType {
    Ended(AudioNodeId),
    SinkChanged,
    RenderCapacity,
    //
}

pub(crate) enum EventPayload {
    None,
    RenderCapacity(AudioRenderCapacityEvent),
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

    pub fn sink_changed() -> Self {
        EventDispatch {
            type_: EventType::SinkChanged,
            payload: EventPayload::None,
        }
    }

    pub fn render_capacity(value: AudioRenderCapacityEvent) -> Self {
        EventDispatch {
            type_: EventType::RenderCapacity,
            payload: EventPayload::RenderCapacity(value),
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
                let mut handlers = self_clone.event_handlers.lock().unwrap();
                if let Some(callback) = handlers.remove(&event.type_) {
                    match callback {
                        EventHandler::Once(f) => (f)(event.payload),
                        EventHandler::Multiple(mut f) => {
                            (f)(event.payload);
                            handlers.insert(event.type_, EventHandler::Multiple(f));
                        }
                    };
                }
                // handlers Mutex guard drops here
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
