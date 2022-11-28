use crate::context::AudioNodeId;
use crossbeam_channel::Receiver;
use std::sync::{Arc, Mutex};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum EventType {
    Ended,
}

pub(crate) enum Callback {
    Once(Box<dyn FnOnce() + Send + 'static>),
    Multiple(Box<dyn FnMut() + Send + 'static>),
}

impl Callback {
    fn run(self) {
        match self {
            Self::Once(f) => (f)(),
            Self::Multiple(mut f) => (f)(),
        }
    }
}

pub(crate) struct EventHandler {
    // could be optional meaning that its a context event (cf. onSinkChange, onStateChange, etc.)
    pub node_id: AudioNodeId,
    pub event_type: EventType,
    pub callback: Callback,
}

#[derive(Debug)]
pub(crate) struct TriggerEventMessage {
    // could be Option w/ None meaning that its a context event
    pub node_id: AudioNodeId,
    // could be Option w/ None meaning the node is dropped on the render thread
    // and listeners can be cleared
    pub event_type: EventType,
}

#[derive(Clone, Default)]
pub(crate) struct EventLoop {
    callbacks: Arc<Mutex<Vec<EventHandler>>>,
}

impl EventLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&self, event_channel: Receiver<TriggerEventMessage>) {
        let self_clone = self.clone();

        std::thread::spawn(move || loop {
            // this thread is dedicated to event handling so we can block
            for message in event_channel.iter() {
                let mut handlers = self_clone.callbacks.lock().unwrap();
                // find EventHandlerInfos that matches messsage and execute callback

                let mut i = 0;
                while i < handlers.len() {
                    let handler = &mut handlers[i];
                    if handler.node_id != message.node_id
                        || handler.event_type != message.event_type
                    {
                        i += 1;
                        continue;
                    }
                    if let Callback::Multiple(f) = &mut handler.callback {
                        (f)();
                    } else {
                        let handler = handlers.remove(i);
                        handler.callback.run();
                    }
                }
            }
        });
    }

    pub fn add_handler(&self, handler: EventHandler) {
        self.callbacks.lock().unwrap().push(handler)
    }
}
