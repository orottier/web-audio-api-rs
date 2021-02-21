use crate::node::AudioNode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Helper struct to start and stop audio streams
#[derive(Clone, Debug)]
pub struct Scheduler {
    start: Arc<AtomicU64>,
    stop: Arc<AtomicU64>,
}

impl Scheduler {
    /// Create a new Scheduler. Initial playback state will be: inactive.
    pub fn new() -> Self {
        Self {
            start: Arc::new(AtomicU64::new(u64::MAX)),
            stop: Arc::new(AtomicU64::new(u64::MAX)),
        }
    }

    /// Check if the stream should be active at this frame
    pub fn is_active(&self, frame: u64) -> bool {
        frame >= self.start.load(Ordering::SeqCst) && frame < self.stop.load(Ordering::SeqCst)
    }

    /// Schedule playback start at this frame
    pub fn start(&self, start: u64) {
        self.start.store(start, Ordering::SeqCst);
    }

    /// Stop playback at this frame
    pub fn stop(&self, stop: u64) {
        self.stop.store(stop, Ordering::SeqCst);
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have a [`Scheduler`]
///
/// Note: an impl of [`AudioNode`] `+` [`Schedule`] = [`AudioScheduledSourceNode`]
pub trait Schedule {
    fn scheduler(&self) -> &Scheduler;

    /// Check if the stream should be active at this frame
    fn is_active(&self, frame: u64) -> bool {
        self.scheduler().is_active(frame)
    }

    /// Schedule playback start at this frame
    fn start(&self, start: u64) {
        self.scheduler().start(start)
    }

    /// Stop playback at this frame
    fn stop(&self, stop: u64) {
        self.scheduler().stop(stop)
    }
}

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
///
/// Note: an impl of [`AudioNode`] `+` [`Schedule`] = [`AudioScheduledSourceNode`]
pub trait AudioScheduledSourceNode: AudioNode + Schedule {
    /// Schedules a sound to playback at an exact time.
    fn start_at(&self, timestamp: f64) {
        let frame = (timestamp * self.context().sample_rate().0 as f64) as u64;
        self.scheduler().start(frame);
    }
    /// Schedules a sound to stop playback at an exact time.
    fn stop_at(&self, timestamp: f64) {
        let frame = (timestamp * self.context().sample_rate().0 as f64) as u64;
        self.scheduler().stop(frame);
    }
    /// Play immediately
    fn start(&self) {
        self.start_at(0.);
    }
    /// Stop immediately
    fn stop(&self) {
        self.stop_at(0.);
    }
}

impl<M: AudioNode + Schedule> AudioScheduledSourceNode for M {}

/// Helper struct to control audio streams
#[derive(Clone, Debug)]
pub struct Controller {
    scheduler: Arc<Scheduler>,
    offset: Arc<AtomicU64>,
    duration: Arc<AtomicU64>,
    loop_start: Arc<AtomicU64>,
    loop_end: Arc<AtomicU64>,
    //playback_rate: Arc<AudioParam>,
}

impl Schedule for Controller {
    fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl Controller {
    /// Create a new Controller. It will not be active
    pub fn new() -> Self {
        Self {
            scheduler: Arc::new(Scheduler::new()),
            offset: Arc::new(AtomicU64::new(0)),
            duration: Arc::new(AtomicU64::new(u64::MAX)),
            loop_start: Arc::new(AtomicU64::new(u64::MAX)),
            loop_end: Arc::new(AtomicU64::new(u64::MAX)),
            //playback_rate: ... create audio param pair
        }
    }
}

impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have a [`Controller`]
pub trait Control {
    fn controller(&self) -> &Controller;
}
