#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EventType {
    Ended,
}

pub(crate) struct EventHandlerInfos {
    // could be optional meaning that its a context event (cf. onSinkChange, onStateChange, etc.)
    pub node_id: u64,
    pub event_type: EventType,
    pub callback: Box<dyn Fn() + Send + Sync + 'static>,
}

#[derive(Debug)]
pub(crate) struct EventEmitterMessage {
    // could be Option w/ None meaning that its a context event
    pub node_id: u64, // we use raw u64 to
    // could be Option w/ None meaning the node is dropped on the render thread
    // and listeners can be cleared
    pub event_type: EventType,
}
