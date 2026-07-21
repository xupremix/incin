//! # Kindle Data
//!
//! Provides dataset and dataloader abstractions for the Kindle framework.
//! Handles batching, shuffling, and data transformation pipelines.
//!
//! ## Examples
//! 
//! Using a basic `DataLoader` with an in-memory `Dataset`:
//! 
//! ```rust,no_run
//! use kindle_data::prelude::*;
//! use kindle_data::{Dataset, DataLoader};
//! use kindle::prelude::*;
//! 
//! struct MyDataset {
//!     images: Tensor<s![100, 3, 224, 224], CpuBackend>,
//!     labels: Tensor<s![100], CpuBackend>,
//! }
//! 
//! impl Dataset for MyDataset {
//!     type Item = (Tensor<s![3, 224, 224], CpuBackend>, Tensor<s![], CpuBackend>);
//!     
//!     fn len(&self) -> usize {
//!         100
//!     }
//!     
//!     fn get(&self, idx: usize) -> Option<Self::Item> {
//!         if idx >= 100 { return None; }
//!         let img = self.images.slice_dyn(vec![idx..idx+1, 0..3, 0..224, 0..224]).unwrap();
//!         let label = self.labels.slice_dyn(vec![idx..idx+1]).unwrap();
//!         // Return properly shaped items...
//!         todo!()
//!     }
//! }
//! 
//! fn main() {
//!     let dataset = MyDataset {
//!         images: Tensor::zeros(()).unwrap(),
//!         labels: Tensor::zeros(()).unwrap(),
//!     };
//!     
//!     // Create a dataloader with batch size 32, shuffling enabled
//!     let loader = DataLoader::new(dataset).batch_size(32).shuffle(true);
//!     
//!     for batch in loader.iter() {
//!         let (images, labels) = batch;
//!         // train model...
//!     }
//! }
//! ```

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
