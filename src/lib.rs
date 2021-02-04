/// Render quantum size (audio graph is rendered in blocks of this size)
pub const BUFFER_SIZE: u32 = 512;

pub mod context;
pub mod node;

pub(crate) mod control;
pub(crate) mod graph;
