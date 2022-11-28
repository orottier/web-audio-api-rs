use crate::context::AudioNodeId;
use crossbeam_channel::Receiver;
use std::sync::{Arc, Mutex};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Event {
    Ended(AudioNodeId),
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
