//! # Incin Data
//!
//! Provides dataset and dataloader abstractions for the Incin framework.
//! Handles batching, shuffling, and data transformation pipelines.
//!
//! ## Examples
//!
//! A `Dataset` supplies items by index; a `Collate` can turn a batch into the
//! model's input type; a `DataLoader` joins the two and yields typed `Result`
//! values so worker failures cannot look like end-of-data. Use
//! `DataLoader::builder` for the default scalar, tuple, and tensor batching
//! policy.
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
//!     fn get(&self, index: usize) -> Result<Option<Self::Item>, incin_data::DataError> {
//!         Ok(u32::try_from(index).ok().filter(|_| index < self.len()).map(|i| i * i))
//!     }
//! }
//!
//! struct IntoBatch;
//!
//! impl Collate<u32> for IntoBatch {
//!     type Output = Vec<u32>;
//!
//!     fn collate(&self, batch: Vec<u32>) -> incin_data::BatchResult<Self::Output> {
//!         Ok(batch)
//!     }
//! }
//!
//! // Batches of 32, shuffled. Loading happens on worker threads; iterating
//! // borrows the loader, so the same one can be iterated each epoch.
//! let loader = DataLoader::new(Squares, IntoBatch, 32).unwrap().with_shuffle(true);
//!
//! let batches: Vec<Vec<u32>> = (&loader).into_iter().collect::<Result<_, _>>().unwrap();
//! assert_eq!(batches.len(), 4); // 100 items is three full batches and a short one
//! assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), 100);
//! ```

#[macro_use]
extern crate alloc;
/// Dataset.
pub mod dataset;
/// Downloader.
#[cfg(feature = "download")]
pub mod downloader;
/// Hub.
#[cfg(feature = "hub")]
pub mod hub;
/// Loader.
pub mod loader;
/// Transforms and data processing pipelines.
pub mod transforms;
/// Vision.
pub mod vision;

pub use dataset::Dataset;
#[cfg(feature = "download")]
pub use downloader::Downloader;
pub use loader::{
    BatchResult, Collate, DataError, DataLoader, DataLoaderBuilder, DefaultCollate,
    DistributedSampler, RemainderPolicy,
};
pub use transforms::{CenterCrop, Compose, Normalize, RandomHorizontalFlip, Scale, Transform};

/// Prelude.
pub mod prelude {
    #[cfg(feature = "hub")]
    pub use super::hub::{HubApi, HubRepo, download, from_pretrained};
    pub use super::loader::{
        BatchResult, Collate, DataError, DataLoader, DataLoaderBuilder, DataLoaderIter,
        DefaultCollate, DistributedSampler, RemainderPolicy,
    };
    pub use super::transforms::{
        CenterCrop, Compose, Normalize, RandomHorizontalFlip, Scale, Transform,
    };
}
