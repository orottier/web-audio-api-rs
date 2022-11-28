use crate::context::AudioNodeId;
use crossbeam_channel::Receiver;
use std::sync::{Arc, Mutex};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum EventType {
    Ended,
}

pub(crate) struct EventHandler {
    // could be optional meaning that its a context event (cf. onSinkChange, onStateChange, etc.)
    pub node_id: AudioNodeId,
    pub event_type: EventType,
    pub callback: Box<dyn FnMut() + Send + 'static>,
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
                handlers
                    .iter_mut()
                    .filter(|item| {
                        item.node_id == message.node_id && item.event_type == message.event_type
                    })
                    .for_each(|handler| (handler.callback)())
            }
        });
    }

    pub fn add_handler(&self, handler: EventHandler) {
        self.callbacks.lock().unwrap().push(handler)
    }
}
