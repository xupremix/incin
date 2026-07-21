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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    /// A trivial in-memory dataset of `i32`s, `0..len`.
    struct RangeDataset(usize);

    impl Dataset for RangeDataset {
        type Item = i32;
        fn len(&self) -> usize {
            self.0
        }
        fn get(&self, index: usize) -> Option<i32> {
            if index < self.0 {
                Some(index as i32)
            } else {
                None
            }
        }
    }

    /// Collates a batch by just returning it as-is (`Vec<i32>`).
    struct VecCollate;
    impl Collate<i32> for VecCollate {
        type Output = Vec<i32>;
        fn collate(&self, batch: Vec<i32>) -> Vec<i32> {
            batch
        }
    }

    // Every test below has a hard wall-clock timeout: a regression in the
    // worker-thread loop (e.g. a channel deadlock, an off-by-one in
    // `batch_indices` that spins forever) must fail loudly and quickly
    // instead of hanging the test suite indefinitely.
    fn collect_with_timeout<T>(iter: impl Iterator<Item = T> + Send + 'static) -> Vec<T>
    where
        T: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(iter.collect::<Vec<T>>());
        });
        rx.recv_timeout(Duration::from_secs(10))
            .expect("DataLoader iteration did not finish within 10s (possible deadlock/hang)")
    }

    #[test]
    fn single_threaded_yields_every_item_in_order_when_not_shuffled() {
        let loader = DataLoader::new(RangeDataset(10), VecCollate, 3);
        let batches = collect_with_timeout((&loader).into_iter());

        // 10 items / batch_size 3 -> batches of [3,3,3,1], nothing dropped,
        // nothing duplicated, original order preserved (no shuffle).
        assert_eq!(
            batches,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
        );
    }

    #[test]
    fn single_threaded_exact_division_produces_no_short_final_batch() {
        let loader = DataLoader::new(RangeDataset(9), VecCollate, 3);
        let batches = collect_with_timeout((&loader).into_iter());
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
    }

    #[test]
    fn empty_dataset_produces_zero_batches() {
        let loader = DataLoader::new(RangeDataset(0), VecCollate, 4);
        let batches = collect_with_timeout((&loader).into_iter());
        assert!(batches.is_empty());
    }

    #[test]
    fn batch_size_larger_than_dataset_produces_one_short_batch() {
        let loader = DataLoader::new(RangeDataset(3), VecCollate, 100);
        let batches = collect_with_timeout((&loader).into_iter());
        assert_eq!(batches, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn shuffle_preserves_the_full_set_of_items_just_reorders_them() {
        let loader = DataLoader::new(RangeDataset(50), VecCollate, 7).with_shuffle(true);
        let batches = collect_with_timeout((&loader).into_iter());

        let all_items: HashSet<i32> = batches.iter().flatten().copied().collect();
        assert_eq!(
            all_items,
            (0..50).collect::<HashSet<_>>(),
            "shuffling must not drop or duplicate any item"
        );
        // 50 / 7 = 7 batches of 7 plus one short batch of 1.
        assert_eq!(batches.len(), 8);
    }

    #[test]
    fn multi_worker_yields_every_item_exactly_once_no_duplicates_no_drops() {
        // Regression guard for the DataLoader's core concurrency invariant:
        // multiple worker threads pull batch-index chunks from a single
        // Mutex<Iterator> (loader.rs's `batch_indices`) and push collated
        // batches through a shared bounded mpsc channel. A bug here (e.g. two
        // workers racing on the same chunk, or one silently starving) would
        // show up as a missing or duplicated item — this is exactly the kind
        // of concurrency bug that's invisible without a real multi-thread run.
        let loader = DataLoader::new(RangeDataset(1000), VecCollate, 10).with_num_workers(8);
        let batches = collect_with_timeout((&loader).into_iter());

        let mut all_items: Vec<i32> = batches.into_iter().flatten().collect();
        assert_eq!(
            all_items.len(),
            1000,
            "no item should be dropped or duplicated"
        );
        all_items.sort_unstable();
        assert_eq!(all_items, (0..1000).collect::<Vec<i32>>());
    }

    #[test]
    fn multi_worker_with_shuffle_still_covers_every_item_exactly_once() {
        let loader = DataLoader::new(RangeDataset(500), VecCollate, 6)
            .with_num_workers(4)
            .with_shuffle(true);
        let batches = collect_with_timeout((&loader).into_iter());

        let mut all_items: Vec<i32> = batches.into_iter().flatten().collect();
        assert_eq!(all_items.len(), 500);
        all_items.sort_unstable();
        assert_eq!(all_items, (0..500).collect::<Vec<i32>>());
    }

    #[test]
    fn more_workers_than_batches_does_not_hang_or_lose_items() {
        // Edge case: 3 batches total but 8 workers requested. Workers that
        // find the shared iterator already exhausted must exit cleanly
        // instead of spinning or blocking the others.
        let loader = DataLoader::new(RangeDataset(25), VecCollate, 10).with_num_workers(8);
        let batches = collect_with_timeout((&loader).into_iter());

        let mut all_items: Vec<i32> = batches.into_iter().flatten().collect();
        assert_eq!(all_items.len(), 25);
        all_items.sort_unstable();
        assert_eq!(all_items, (0..25).collect::<Vec<i32>>());
    }

    #[test]
    fn collate_function_is_actually_invoked_per_batch() {
        // A Collate impl that transforms the batch (sums it) rather than
        // passing it through unchanged, proving collate_fn is really called
        // (not bypassed).
        struct SumCollate;
        impl Collate<i32> for SumCollate {
            type Output = i32;
            fn collate(&self, batch: Vec<i32>) -> i32 {
                batch.into_iter().sum()
            }
        }

        let loader = DataLoader::new(RangeDataset(6), SumCollate, 3);
        let mut sums = collect_with_timeout((&loader).into_iter());
        sums.sort_unstable();
        // batch [0,1,2] -> 3, batch [3,4,5] -> 12
        assert_eq!(sums, vec![3, 12]);
    }
}
