//! Scheduler and Controller for precise timings

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::AtomicF64;

/// Helper struct to start and stop audio streams
#[derive(Clone, Debug)]
pub(crate) struct Scheduler {
    inner: Arc<SchedulerInner>,
}

#[derive(Debug)]
struct SchedulerInner {
    start: AtomicF64,
    stop: AtomicF64,
}

impl Scheduler {
    /// Create a new Scheduler. Initial playback state will be: inactive.
    pub fn new() -> Self {
        let inner = SchedulerInner {
            start: AtomicF64::new(f64::MAX),
            stop: AtomicF64::new(f64::MAX),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Retrieve playback start value
    pub fn get_start_at(&self) -> f64 {
        self.inner.start.load(Ordering::SeqCst)
    }

    /// Schedule playback start at this timestamp
    pub fn start_at(&self, start: f64) {
        // todo panic on invalid values, or when already called
        self.inner.start.store(start, Ordering::SeqCst);
    }

    /// Retrieve playback stop value
    pub fn get_stop_at(&self) -> f64 {
        self.inner.stop.load(Ordering::SeqCst)
    }

    /// Stop playback at this timestamp
    pub fn stop_at(&self, stop: f64) {
        // todo panic on invalid values, or when already called
        self.inner.stop.store(stop, Ordering::SeqCst);
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper struct to control audio streams
#[derive(Clone, Debug)]
pub(crate) struct Controller {
    inner: Arc<ControllerInner>,
}

#[derive(Debug)]
struct ControllerInner {
    scheduler: Scheduler,
    loop_: AtomicBool,
    loop_start: AtomicF64,
    loop_end: AtomicF64,
    offset: AtomicF64,
    duration: AtomicF64,
}

impl Controller {
    /// Create a new Controller. It will not be active
    pub fn new() -> Self {
        let inner = ControllerInner {
            scheduler: Scheduler::new(),
            loop_: AtomicBool::new(false),
            loop_start: AtomicF64::new(0.),
            loop_end: AtomicF64::new(f64::MAX),
            offset: AtomicF64::new(f64::MAX),
            duration: AtomicF64::new(f64::MAX),
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.inner.scheduler
    }

    pub fn loop_(&self) -> bool {
        self.inner.loop_.load(Ordering::SeqCst)
    }

    pub fn set_loop(&self, loop_: bool) {
        self.inner.loop_.store(loop_, Ordering::SeqCst);
    }

    pub fn loop_start(&self) -> f64 {
        self.inner.loop_start.load(Ordering::SeqCst)
    }

    pub fn set_loop_start(&self, loop_start: f64) {
        self.inner.loop_start.store(loop_start, Ordering::SeqCst);
    }

    pub fn loop_end(&self) -> f64 {
        self.inner.loop_end.load(Ordering::SeqCst)
    }

    pub fn set_loop_end(&self, loop_end: f64) {
        self.inner.loop_end.store(loop_end, Ordering::SeqCst);
    }

    pub fn offset(&self) -> f64 {
        self.inner.offset.load(Ordering::SeqCst)
    }

    pub fn set_offset(&self, offset: f64) {
        self.inner.offset.store(offset, Ordering::SeqCst);
    }

    pub fn duration(&self) -> f64 {
        self.inner.duration.load(Ordering::SeqCst)
    }

    pub fn set_duration(&self, duration: f64) {
        self.inner.duration.store(duration, Ordering::SeqCst)
    }
}

impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller() {
        let controller = Controller::new();

        assert!(!controller.loop_());
        assert!(controller.loop_start() == 0.);
        assert!(controller.loop_end() == f64::MAX);
    }
}
