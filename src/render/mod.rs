//! Primitives related to audio graph rendering

use std::fmt::Debug;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct NodeIndex(pub u64);

// private mods
pub(crate) mod graph;

// pub(crate) mods
mod thread;
pub(crate) use thread::*;

// public mods
mod processor;
pub use processor::*;
mod quantum;
pub use quantum::*;
