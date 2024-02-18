use std::any::Any;

use crate::context::AudioContextRegistration;
use crate::node::AudioNode;

use crate::events::{EventHandler, EventPayload, EventType};

pub struct MessagePort<'a> {
    inner: MessagePortFlavour<'a>,
}

impl<'a> std::fmt::Debug for MessagePort<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessagePort").finish_non_exhaustive()
    }
}

enum MessagePortFlavour<'a> {
    AudioNode(&'a AudioContextRegistration),
    AudioProcessor,
}

impl<'a> MessagePort<'a> {
    pub(crate) fn from_node(node: &'a dyn AudioNode) -> Self {
        let inner = MessagePortFlavour::AudioNode(node.registration());
        Self { inner }
    }

    pub fn post_message<M: Any + Send + 'static>(&self, msg: M) {
        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                registration.post_message(msg);
            }
            _ => todo!(),
        }
    }

    pub fn set_onmessage<F: FnMut(Box<dyn Any + Send + 'static>) + Send + 'static>(
        &self,
        mut callback: F,
    ) {
        let callback = move |v| match v {
            EventPayload::Message(v) => callback(v),
            _ => unreachable!(),
        };

        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                registration.context().set_event_handler(
                    EventType::Message(registration.id()),
                    EventHandler::Multiple(Box::new(callback)),
                );
            }
            _ => todo!(),
        }
    }

    pub fn clear_onmessage(&self) {
        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                registration
                    .context()
                    .clear_event_handler(EventType::Message(registration.id()));
            }
            _ => todo!(),
        }
    }
}
