use crate::context::AudioNodeId;
use crate::AudioRenderCapacityEvent;
use crossbeam_channel::Receiver;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub(crate) enum Event {
    Ended(AudioNodeId),
    SinkChanged,
    RenderCapacity(AudioRenderCapacityEvent),
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

pub(crate) enum Callback {
    Once(Box<dyn FnOnce(Event) + Send + 'static>),
    Multiple(Box<dyn FnMut(Event) + Send + 'static>),
}

impl Callback {
    fn run(self, event: Event) {
        match self {
            Self::Once(f) => (f)(event),
            Self::Multiple(mut f) => (f)(event),
        }
    }
}

pub(crate) struct EventHandler {
    pub event: Event,
    pub callback: Callback,
}

#[derive(Clone, Default)]
pub(crate) struct EventLoop {
    callbacks: Arc<Mutex<Vec<EventHandler>>>,
}

impl EventLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&self, event_channel: Receiver<Event>) {
        let self_clone = self.clone();

        std::thread::spawn(move || loop {
            // this thread is dedicated to event handling so we can block
            for message in event_channel.iter() {
                let mut handlers = self_clone.callbacks.lock().unwrap();
                // find EventHandlerInfos that matches messsage and execute callback

                let mut i = 0;
                while i < handlers.len() {
                    let handler = &mut handlers[i];
                    if handler.event != message {
                        i += 1;
                        continue;
                    }
                    if let Callback::Multiple(f) = &mut handler.callback {
                        (f)(message.clone());
                        i += 1;
                    } else {
                        let handler = handlers.remove(i);
                        handler.callback.run(message.clone());
                    }
                }
            }
        });
    }

    pub fn add_handler(&self, handler: EventHandler) {
        self.callbacks.lock().unwrap().push(handler)
    }
}
