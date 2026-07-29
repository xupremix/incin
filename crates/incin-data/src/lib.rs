//! # Incin Data
//!
//! Provides dataset and dataloader abstractions for the Incin framework.
//! Handles batching, shuffling, and data transformation pipelines.
//!
//! ## Examples
//!
//! A `Dataset` supplies items by index; a `Collate` turns a batch of them into
//! whatever the training step wants; a `DataLoader` joins the two.
//!
//! ```rust
//! use incin_data::{Collate, DataLoader, Dataset};
//!
//! struct Squares;
//!
//! impl Dataset for Squares {
//!     type Item = u32;
//!
//!     fn len(&self) -> usize {
//!         100
//!     }
//!
//!     fn get(&self, index: usize) -> Option<Self::Item> {
//!         u32::try_from(index).ok().filter(|_| index < self.len()).map(|i| i * i)
//!     }
//! }
//!
//! struct IntoBatch;
//!
//! impl Collate<u32> for IntoBatch {
//!     type Output = Vec<u32>;
//!
//!     fn collate(&self, batch: Vec<u32>) -> Self::Output {
//!         batch
//!     }
//! }
//!
//! // Batches of 32, shuffled. Loading happens on worker threads; iterating
//! // borrows the loader, so the same one can be iterated each epoch.
//! let loader = DataLoader::new(Squares, IntoBatch, 32).with_shuffle(true);
//!
//! let batches: Vec<Vec<u32>> = (&loader).into_iter().collect();
//! assert_eq!(batches.len(), 4); // 100 items is three full batches and a short one
//! assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), 100);
//! ```

#[macro_use]
extern crate alloc;
/// Dataset.
pub mod dataset;
/// Downloader.
pub mod downloader;
/// Hub.
pub mod hub;
/// Loader.
pub mod loader;
/// Transforms and data processing pipelines.
pub mod transforms;
/// Vision.
pub mod vision;

pub use dataset::Dataset;
pub use downloader::Downloader;
pub use loader::{Collate, DataLoader};
pub use transforms::{CenterCrop, Compose, Normalize, RandomHorizontalFlip, Scale, Transform};

/// Prelude.
pub mod prelude {
    pub use super::hub::*;
    pub use super::loader::*;
    pub use super::transforms::*;
}
