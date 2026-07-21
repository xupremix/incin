use crate::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

/// Auto-generated documentation for Collate.
pub trait Collate<T>: Send + Sync {
    /// Auto-generated documentation for Output.
    type Output: Send + 'static;
    /// Auto-generated documentation for collate.
    fn collate(&self, batch: Vec<T>) -> Self::Output;
}

/// Auto-generated documentation for DataLoader.
pub struct DataLoader<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    dataset: Arc<D>,
    collate_fn: Arc<C>,
    batch_size: usize,
    num_workers: usize,
    shuffle: bool,
}

impl<D, C> DataLoader<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    /// Auto-generated documentation for new.
    pub fn new(dataset: D, collate_fn: C, batch_size: usize) -> Self {
        Self {
            dataset: Arc::new(dataset),
            collate_fn: Arc::new(collate_fn),
            batch_size,
            num_workers: 0,
            shuffle: false,
        }
    }

    /// Auto-generated documentation for with_num_workers.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Auto-generated documentation for with_shuffle.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

/// Auto-generated documentation for DataLoaderIter.
pub struct DataLoaderIter<T> {
    receiver: Receiver<T>,
}

impl<T> Iterator for DataLoaderIter<T> {
    /// Auto-generated documentation for Item.
    type Item = T;

    /// Auto-generated documentation for next.
    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

impl<D, C> IntoIterator for &DataLoader<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    /// Auto-generated documentation for Item.
    type Item = C::Output;
    /// Auto-generated documentation for IntoIter.
    type IntoIter = DataLoaderIter<C::Output>;

    /// Auto-generated documentation for into_iter.
    fn into_iter(self) -> Self::IntoIter {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }

        let num_batches = indices.len().div_ceil(self.batch_size);
        let mut batch_indices = Vec::with_capacity(num_batches);

        for i in 0..num_batches {
            let start = i * self.batch_size;
            let end = core::cmp::min(start + self.batch_size, indices.len());
            batch_indices.push(indices[start..end].to_vec());
        }

        // Bounded channel to prevent over-fetching
        let (tx, rx) = sync_channel(self.num_workers * 2 + 2);

        let dataset = self.dataset.clone();
        let collate_fn = self.collate_fn.clone();

        if self.num_workers == 0 {
            // Single-threaded
            thread::spawn(move || {
                for batch_idx in batch_indices {
                    let mut batch = Vec::with_capacity(batch_idx.len());
                    for idx in batch_idx {
                        if let Some(item) = dataset.get(idx) {
                            batch.push(item);
                        }
                    }
                    if !batch.is_empty() {
                        let collated = collate_fn.collate(batch);
                        if tx.send(collated).is_err() {
                            break;
                        }
                    }
                }
            });
        } else {
            // Multi-threaded
            let batch_indices = Arc::new(Mutex::new(batch_indices.into_iter()));

            for _ in 0..self.num_workers {
                let dataset = dataset.clone();
                let collate_fn = collate_fn.clone();
                let tx = tx.clone();
                let batch_indices = batch_indices.clone();

                thread::spawn(move || {
                    loop {
                        let next_batch = {
                            let mut iter = batch_indices.lock().unwrap();
                            iter.next()
                        };

                        if let Some(batch_idx) = next_batch {
                            let mut batch = Vec::with_capacity(batch_idx.len());
                            for idx in batch_idx {
                                if let Some(item) = dataset.get(idx) {
                                    batch.push(item);
                                }
                            }
                            if !batch.is_empty() {
                                let collated = collate_fn.collate(batch);
                                if tx.send(collated).is_err() {
                                    break;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                });
            }
        }

        DataLoaderIter { receiver: rx }
    }
}
