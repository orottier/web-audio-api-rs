use std::any::Any;

use crate::context::AudioContextRegistration;
use crate::node::AudioNode;
use crate::render::RenderScope;

use crate::events::{EventHandler, EventPayload, EventType};

pub(crate) type MessageHandler = Option<Box<dyn FnMut(&mut dyn Any)>>;

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
    AudioProcessor(&'a RenderScope),
}

impl<'a> MessagePort<'a> {
    pub(crate) fn from_node(node: &'a dyn AudioNode) -> Self {
        let inner = MessagePortFlavour::AudioNode(node.registration());
        Self { inner }
    }

    pub(crate) fn from_processor(scope: &'a RenderScope) -> Self {
        let inner = MessagePortFlavour::AudioProcessor(scope);
        Self { inner }
    }

    pub fn post_message<M: Any + Send + 'static>(&self, msg: M) {
        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                registration.post_message(msg);
            }
            MessagePortFlavour::AudioProcessor(scope) => {
                scope.post_message(Box::new(msg));
            }
        }
    }

    pub fn set_onmessage<F: FnMut(&mut dyn Any) + Send + 'static>(&self, mut callback: F) {
        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                let callback = move |v| match v {
                    EventPayload::Message(mut v) => callback(&mut v),
                    _ => unreachable!(),
                };

                registration.context().set_event_handler(
                    EventType::Message(registration.id()),
                    EventHandler::Multiple(Box::new(callback)),
                );
            }
            MessagePortFlavour::AudioProcessor(scope) => {
                scope.set_message_handler(Some(Box::new(callback)))
            }
        }
    }

    pub fn clear_onmessage(&self) {
        match self.inner {
            MessagePortFlavour::AudioNode(registration) => {
                registration
                    .context()
                    .clear_event_handler(EventType::Message(registration.id()));
            }
            MessagePortFlavour::AudioProcessor(scope) => {
                scope.clear_message_handler();
            }
        }
    }
}
