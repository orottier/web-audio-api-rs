use crate::context::AudioNodeId;
use crate::AudioRenderCapacityEvent;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;

#[derive(Debug, Clone)]
pub(crate) enum Event {
    Ended(AudioNodeId),
    SinkChanged,
    RenderCapacity(AudioRenderCapacityEvent),
}

/*
 * Hack: derive custom PartialEq and Hash implementation for Event, ignoring the payload of the
 * RenderCapacity variant.
 *
 * TODO, remove this hack and use a proper EventPayload enum alongside.
 */

impl Hash for Event {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self {
            Event::Ended(id) => {
                state.write_u8(0);
                state.write_u64(id.0);
            }
            Event::SinkChanged => state.write_u8(1),
            Event::RenderCapacity(_) => state.write_u8(2),
        }
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        use Event::*;

        match (&self, other) {
            (Ended(s), Ended(o)) => s.eq(o),
            (SinkChanged, SinkChanged) => true,
            (RenderCapacity(_), RenderCapacity(_)) => true,
            _ => false,
        }
    }
}

impl Eq for Event {}

pub(crate) enum EventHandler {
    Once(Box<dyn FnOnce(Event) + Send + 'static>),
    Multiple(Box<dyn FnMut(Event) + Send + 'static>),
}

#[derive(Clone, Default)]
pub(crate) struct EventLoop {
    event_handlers: Arc<Mutex<HashMap<Event, EventHandler>>>,
}

impl EventLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&self, event_channel: Receiver<Event>) {
        let self_clone = self.clone();

        std::thread::spawn(move || loop {
            // this thread is dedicated to event handling so we can block
            for event in event_channel.iter() {
                let mut handlers = self_clone.event_handlers.lock().unwrap();
                if let Some(callback) = handlers.remove(&event) {
                    match callback {
                        EventHandler::Once(f) => (f)(event),
                        EventHandler::Multiple(mut f) => {
                            (f)(event.clone());
                            handlers.insert(event, EventHandler::Multiple(f));
                        }
                    };
                }
                // handlers Mutex guard drops here
            }
        });
    }

    pub fn set_handler(&self, event: Event, callback: Option<EventHandler>) {
        let mut event_handlers = self.event_handlers.lock().unwrap();
        match callback {
            Some(callback) => event_handlers.insert(event, callback),
            None => event_handlers.remove(&event),
        };
    }
}
