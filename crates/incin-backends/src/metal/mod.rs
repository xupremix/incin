//! Native Metal backend for Apple Silicon and macOS devices.

pub mod backend;
pub mod executor;
pub mod storage;
pub mod tape;

pub use backend::{MetalBackendImpl, MetalVar};
pub use storage::{MetalStorage, MetalStorageMode, is_unified_memory};
pub use tape::MetalGrads;
