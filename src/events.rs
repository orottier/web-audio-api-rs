use crate::context::AudioNodeId;

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
