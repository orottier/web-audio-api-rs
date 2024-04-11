use super::AudioNode;
use crate::events::{Event, EventHandler, EventType};

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
pub trait AudioScheduledSourceNode: AudioNode {
    /// Play immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start(&mut self);

    /// Schedule playback start at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start_at(&mut self, when: f64);

    /// Stop immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was already stopped
    fn stop(&mut self);

    /// Schedule playback stop at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was already stopped
    fn stop_at(&mut self, when: f64);

    /// Register callback to run when the source node has stopped playing
    ///
    /// For all [`AudioScheduledSourceNode`]s, the ended event is dispatched when the stop time
    /// determined by stop() is reached. For an [`AudioBufferSourceNode`], the event is also
    /// dispatched because the duration has been reached or if the entire buffer has been played.
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    fn set_onended<F: FnOnce(Event) + Send + 'static>(&self, callback: F) {
        let callback = move |_| callback(Event { type_: "ended" });

        self.context().set_event_handler(
            EventType::Ended(self.registration().id()),
            EventHandler::Once(Box::new(callback)),
        );
    }

    /// Unset the callback to run when the source node has stopped playing
    fn clear_onended(&self) {
        self.context()
            .clear_event_handler(EventType::Ended(self.registration().id()));
    }
}
