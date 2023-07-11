use crate::AtomicF64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Helper struct to control playback settings
pub(crate) struct Controller {
    /// Raw values
    values: ControllerValues,
    /// Values to be shared with another thread
    shared: Arc<ControllerShared>,
}

impl Default for Controller {
    /// Create a new Controller. It will not be active
    fn default() -> Self {
        let values = ControllerValues::default();
        Self {
            values: values.clone(),
            shared: Arc::new(values.into()),
        }
    }
}

impl Controller {
    pub fn shared(&self) -> &Arc<ControllerShared> {
        &self.shared
    }

    pub fn loop_(&self) -> bool {
        self.values.loop_
    }

    pub fn set_loop(&mut self, loop_: bool) {
        self.values.loop_ = loop_;
        self.shared.loop_.store(loop_, Ordering::Release);
    }

    pub fn loop_start(&self) -> f64 {
        self.values.loop_start
    }

    pub fn set_loop_start(&mut self, loop_start: f64) {
        self.values.loop_start = loop_start;
        self.shared.loop_start.store(loop_start, Ordering::Release);
    }

    pub fn loop_end(&self) -> f64 {
        self.values.loop_end
    }

    pub fn set_loop_end(&mut self, loop_end: f64) {
        self.values.loop_end = loop_end;
        self.shared.loop_end.store(loop_end, Ordering::Release);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ControllerValues {
    pub loop_: bool,
    pub loop_start: f64,
    pub loop_end: f64,
}

impl Default for ControllerValues {
    fn default() -> Self {
        Self {
            loop_: false,
            loop_start: 0.,
            loop_end: f64::MAX,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ControllerShared {
    loop_: AtomicBool,
    loop_start: AtomicF64,
    loop_end: AtomicF64,
}

impl From<ControllerValues> for ControllerShared {
    fn from(values: ControllerValues) -> Self {
        Self {
            loop_: AtomicBool::new(values.loop_),
            loop_start: AtomicF64::new(values.loop_start),
            loop_end: AtomicF64::new(values.loop_end),
        }
    }
}

// Uses the canonical ordering for handover of values, i.e. `Acquire` on load and `Release` on
// store.
impl ControllerShared {
    pub fn loop_(&self) -> bool {
        self.loop_.load(Ordering::Acquire)
    }

    pub fn loop_start(&self) -> f64 {
        self.loop_start.load(Ordering::Acquire)
    }

    pub fn loop_end(&self) -> f64 {
        self.loop_end.load(Ordering::Acquire)
    }
}
