//! # Kindle Data
//!
//! Provides dataset and dataloader abstractions for the Kindle framework.
//! Handles batching, shuffling, and data transformation pipelines.

#[macro_use]
extern crate alloc;
/// Auto-generated documentation for dataset.
pub mod dataset;
/// Auto-generated documentation for downloader.
pub mod downloader;
/// Auto-generated documentation for hub.
pub mod hub;
/// Auto-generated documentation for loader.
pub mod loader;
/// Auto-generated documentation for vision.
pub mod vision;

pub use dataset::Dataset;
pub use downloader::Downloader;
pub use loader::{Collate, DataLoader};

/// Auto-generated documentation for prelude.
pub mod prelude {
    pub use super::hub::*;
    pub use super::loader::*;
}
