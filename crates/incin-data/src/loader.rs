use crate::dataset::Dataset;
use incin_core::error::{Error, ErrorMessage, Result as CoreResult};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::num::NonZeroUsize;
use std::time::Duration;

/// A recoverable data-pipeline failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// The dataset reported malformed or otherwise invalid data.
    Dataset(String),
    /// A requested index was outside the dataset's declared range.
    IndexOutOfBounds { index: usize, len: usize },
    /// A worker caught a panic while reading a sample.
    WorkerPanicked { worker_id: usize, stage: &'static str },
    /// A worker caught a panic while collating a batch.
    CollatePanicked { worker_id: usize },
    /// A worker stopped unexpectedly before completing the epoch.
    WorkerDisconnected,
    /// The configured worker receive timeout elapsed.
    Timeout { duration: Duration },
}

impl core::fmt::Display for DataError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Dataset(message) => write!(f, "dataset error: {message}"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "dataset index {index} is outside length {len}")
            }
            Self::WorkerPanicked { worker_id, stage } => {
                write!(f, "worker {worker_id} panicked while {stage}")
            }
            Self::CollatePanicked { worker_id } => {
                write!(f, "worker {worker_id} panicked while collating")
            }
            Self::WorkerDisconnected => f.write_str("data-loader worker disconnected"),
            Self::Timeout { duration } => write!(f, "data-loader timed out after {duration:?}"),
        }
    }
}

/// A batch result returned by [`DataLoaderIter`].
pub type BatchResult<T> = core::result::Result<T, DataError>;

/// Collate.
pub trait Collate<T>: Send + Sync {
    /// Output.
    type Output: Send + 'static;
    /// Collate.
    fn collate(&self, batch: Vec<T>) -> BatchResult<Self::Output>;
}

/// Data loader.
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
    drop_last: bool,
    prefetch: NonZeroUsize,
    seed: u64,
    epoch: u64,
    timeout: Option<Duration>,
}

/// Configures a [`DataLoader`] before construction.
pub struct DataLoaderBuilder<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    dataset: D,
    collate_fn: C,
    batch_size: NonZeroUsize,
    workers: usize,
    prefetch: NonZeroUsize,
    drop_last: bool,
    shuffle: bool,
    seed: u64,
    epoch: u64,
    timeout: Option<Duration>,
}

impl<D, C> DataLoaderBuilder<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    /// Sets the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> CoreResult<Self> {
        self.batch_size = NonZeroUsize::new(batch_size).ok_or_else(|| {
            Error::InvalidModuleState {
                operation: "data_loader_builder",
                reason: ErrorMessage::new(
                    "batch size must be non-zero",
                ),
            }
        })?;
        Ok(self)
    }

    /// Sets the number of worker threads.
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    /// Sets the bounded worker prefetch capacity.
    pub fn prefetch(mut self, prefetch: usize) -> CoreResult<Self> {
        self.prefetch = NonZeroUsize::new(prefetch).ok_or_else(|| {
            Error::InvalidModuleState {
                operation: "data_loader_builder",
                reason: ErrorMessage::new(
                    "prefetch capacity must be non-zero",
                ),
            }
        })?;
        Ok(self)
    }

    /// Drops the final incomplete batch when enabled.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Enables or disables deterministic shuffling.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the deterministic shuffle seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the epoch used in deterministic seed derivation.
    #[must_use]
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    /// Sets an explicit worker receive timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builds the configured loader.
    #[must_use]
    pub fn build(self) -> DataLoader<D, C> {
        DataLoader {
            dataset: Arc::new(self.dataset),
            collate_fn: Arc::new(self.collate_fn),
            batch_size: self.batch_size.get(),
            num_workers: self.workers,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            prefetch: self.prefetch,
            seed: self.seed,
            epoch: self.epoch,
            timeout: self.timeout,
        }
    }
}

impl<D, C> DataLoader<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    /// New.
    pub fn new(dataset: D, collate_fn: C, batch_size: usize) -> CoreResult<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidModuleState {
                operation: "data_loader_new",
                reason: ErrorMessage::new("batch size must be non-zero"),
            });
        }
        Ok(Self {
            dataset: Arc::new(dataset),
            collate_fn: Arc::new(collate_fn),
            batch_size,
            num_workers: 0,
            shuffle: false,
            drop_last: false,
            prefetch: NonZeroUsize::new(2).expect("constant is non-zero"),
            seed: 0,
            epoch: 0,
            timeout: None,
        })
    }

    /// Starts a builder with a default batch size of one.
    #[must_use]
    pub fn builder(dataset: D, collate_fn: C) -> DataLoaderBuilder<D, C> {
        DataLoaderBuilder {
            dataset,
            collate_fn,
            batch_size: NonZeroUsize::new(1).expect("constant is non-zero"),
            workers: 0,
            prefetch: NonZeroUsize::new(2).expect("constant is non-zero"),
            drop_last: false,
            shuffle: false,
            seed: 0,
            epoch: 0,
            timeout: None,
        }
    }

    /// With num workers.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// With shuffle.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Drops the final incomplete batch when enabled.
    #[must_use]
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets bounded worker prefetch capacity.
    #[must_use]
    pub fn with_prefetch(mut self, prefetch: NonZeroUsize) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// Sets deterministic shuffle seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the deterministic epoch number.
    #[must_use]
    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    /// Sets an explicit worker receive timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Data loader iter.
pub struct DataLoaderIter<T> {
    sync_next: Option<Box<dyn FnMut() -> Option<BatchResult<T>> + Send>>,
    receiver: Option<Receiver<(usize, BatchResult<T>)>>,
    pending: BTreeMap<usize, BatchResult<T>>,
    next_sequence: usize,
    total_batches: usize,
    terminated: bool,
    timeout: Option<Duration>,
    cancel: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl<T> Iterator for DataLoaderIter<T> {
    /// Item.
    type Item = BatchResult<T>;

    /// Next.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(sync_next) = self.sync_next.as_mut() {
            let batch = sync_next();
            if batch.is_none() {
                self.sync_next = None;
            }
            if batch.as_ref().is_some_and(|item| item.is_err()) {
                self.terminated = true;
                self.sync_next = None;
            }
            return batch;
        }
        if self.terminated || self.next_sequence >= self.total_batches {
            return None;
        }
        if let Some(batch) = self.pending.remove(&self.next_sequence) {
            self.next_sequence += 1;
            if batch.is_err() {
                self.terminated = true;
                self.cancel.store(true, Ordering::Release);
                self.receiver.take();
            }
            return Some(batch);
        }
        let receiver = self.receiver.as_ref()?;
        loop {
            let received = match self.timeout {
                Some(duration) => receiver
                    .recv_timeout(duration)
                    .map_err(|error| match error {
                        std::sync::mpsc::RecvTimeoutError::Timeout => {
                            DataError::Timeout { duration }
                        }
                        std::sync::mpsc::RecvTimeoutError::Disconnected => {
                            DataError::WorkerDisconnected
                        }
                    }),
                None => receiver.recv().map_err(|_| DataError::WorkerDisconnected),
            };
            match received {
                Ok((sequence, batch)) if sequence == self.next_sequence => {
                    self.next_sequence += 1;
                    if batch.is_err() {
                        self.terminated = true;
                        self.cancel.store(true, Ordering::Release);
                        self.receiver.take();
                    }
                    return Some(batch);
                }
                Ok((sequence, batch)) => {
                    self.pending.insert(sequence, batch);
                }
                Err(error) => {
                    self.receiver = None;
                    self.terminated = true;
                    return Some(Err(error));
                }
            }
            if let Some(batch) = self.pending.remove(&self.next_sequence) {
                self.next_sequence += 1;
                return Some(batch);
            }
        }
    }
}

impl<T> Drop for DataLoaderIter<T> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        self.sync_next.take();
        self.receiver.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<D, C> IntoIterator for &DataLoader<D, C>
where
    D: Dataset + 'static,
    C: Collate<D::Item> + 'static,
{
    /// Item.
    type Item = BatchResult<C::Output>;
    /// Into iter.
    type IntoIter = DataLoaderIter<C::Output>;

    /// Into iter.
    fn into_iter(self) -> Self::IntoIter {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            let shuffle_seed = self
                .seed
                .wrapping_add(self.epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            indices.shuffle(&mut StdRng::seed_from_u64(shuffle_seed));
        }

        let num_batches = if self.drop_last {
            indices.len() / self.batch_size
        } else {
            indices.len().div_ceil(self.batch_size)
        };
        let mut batch_indices = Vec::with_capacity(num_batches);

        for i in 0..num_batches {
            let start = i * self.batch_size;
            let end = core::cmp::min(start + self.batch_size, indices.len());
            batch_indices.push(indices[start..end].to_vec());
        }

        let dataset = self.dataset.clone();
        let collate_fn = self.collate_fn.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::new();

        if self.num_workers == 0 {
            // Keep the synchronous path lazy: constructing the iterator only
            // prepares indices. Dataset access and collation happen in the
            // caller's `next()` call, one batch at a time.
            let mut batches = batch_indices.into_iter();
            let sync_next = Box::new(move || {
                let batch_idx = batches.next()?;
                let mut batch = Vec::with_capacity(batch_idx.len());
                for idx in batch_idx {
                    match dataset.get(idx) {
                        Ok(Some(item)) => batch.push(item),
                        Ok(None) => {
                            return Some(Err(DataError::IndexOutOfBounds {
                                index: idx,
                                len: dataset.len(),
                            }));
                        }
                        Err(error) => {
                            return Some(Err(error));
                        }
                    }
                }
                if !batch.is_empty() {
                    return Some(catch_unwind(AssertUnwindSafe(|| collate_fn.collate(batch)))
                        .map_err(|_| DataError::CollatePanicked { worker_id: 0 })
                        .and_then(|result| result));
                }
                Some(Err(DataError::WorkerDisconnected))
            });
            return DataLoaderIter {
                sync_next: Some(sync_next),
                receiver: None,
                pending: BTreeMap::new(),
                next_sequence: 0,
                total_batches: num_batches,
                terminated: false,
                timeout: self.timeout,
                cancel,
                workers,
            };
        } else {
            // Multi-threaded
            let (tx, rx) = sync_channel(self.prefetch.get());
            let batch_indices = Arc::new(Mutex::new(batch_indices.into_iter().enumerate()));

            for worker_id in 0..self.num_workers {
                let dataset = dataset.clone();
                let collate_fn = collate_fn.clone();
                let tx = tx.clone();
                let batch_indices = batch_indices.clone();
                let cancel = cancel.clone();

                workers.push(thread::spawn(move || {
                    loop {
                        if cancel.load(Ordering::Acquire) {
                            break;
                        }
                        let next_batch = {
                            // The lock only protects `Iterator::next`; user
                            // dataset/collate code runs after it is released.
                            // Recover the iterator if a worker was aborted
                            // rather than turning mutex poison into a panic.
                            let mut iter = batch_indices
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            iter.next()
                        };

                        if let Some((sequence, batch_idx)) = next_batch {
                            let mut batch = Vec::with_capacity(batch_idx.len());
                            for idx in batch_idx {
                                let result = catch_unwind(AssertUnwindSafe(|| dataset.get(idx)))
                                    .map_err(|_| DataError::WorkerPanicked {
                                        worker_id,
                                        stage: "reading a sample",
                                    })
                                    .and_then(|result| result);
                                match result {
                                    Ok(Some(item)) => batch.push(item),
                                    Ok(None) => {
                                        let _ = tx.send((
                                            sequence,
                                            Err(DataError::IndexOutOfBounds {
                                                index: idx,
                                                len: dataset.len(),
                                            }),
                                        ));
                                        cancel.store(true, Ordering::Release);
                                        return;
                                    }
                                    Err(error) => {
                                        let _ = tx.send((sequence, Err(error)));
                                        cancel.store(true, Ordering::Release);
                                        return;
                                    }
                                }
                            }
                            if !batch.is_empty() {
                                let collated = catch_unwind(AssertUnwindSafe(|| collate_fn.collate(batch)))
                                    .map_err(|_| DataError::CollatePanicked { worker_id })
                                    .and_then(|result| result);
                                if tx.send((sequence, collated)).is_err() {
                                    break;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }));
            }
            drop(tx);
            DataLoaderIter {
                sync_next: None,
                receiver: Some(rx),
                pending: BTreeMap::new(),
                next_sequence: 0,
                total_batches: num_batches,
                terminated: false,
                timeout: self.timeout,
                cancel,
                workers,
            }
        }
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
        fn get(&self, index: usize) -> BatchResult<Option<i32>> {
            if index < self.0 {
                Ok(Some(index as i32))
            } else {
                Ok(None)
            }
        }
    }

    /// Collates a batch by just returning it as-is (`Vec<i32>`).
    struct VecCollate;
    impl Collate<i32> for VecCollate {
        type Output = Vec<i32>;
        fn collate(&self, batch: Vec<i32>) -> BatchResult<Vec<i32>> {
            Ok(batch)
        }
    }

    struct CountingDataset {
        reads: Arc<std::sync::atomic::AtomicUsize>,
        len: usize,
    }

    impl Dataset for CountingDataset {
        type Item = i32;

        fn len(&self) -> usize {
            self.len
        }

        fn get(&self, index: usize) -> BatchResult<Option<Self::Item>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok((index < self.len).then_some(index as i32))
        }
    }

    #[test]
    fn zero_worker_iterator_fetches_only_the_next_batch() {
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let loader = DataLoader::new(
            CountingDataset {
                reads: reads.clone(),
                len: 9,
            },
            VecCollate,
            3,
        )
        .unwrap();

        let mut iter = (&loader).into_iter();
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(iter.next().unwrap().unwrap(), vec![0, 1, 2]);
        assert_eq!(reads.load(Ordering::Relaxed), 3);
        assert_eq!(iter.next().unwrap().unwrap(), vec![3, 4, 5]);
        assert_eq!(reads.load(Ordering::Relaxed), 6);
    }

    // Every test below has a hard wall-clock timeout: a regression in the
    // worker-thread loop (e.g. a channel deadlock, an off-by-one in
    // `batch_indices` that spins forever) must fail loudly and quickly
    // instead of hanging the test suite indefinitely.
    fn collect_with_timeout<T>(
        iter: impl Iterator<Item = BatchResult<T>> + Send + 'static,
    ) -> Vec<T>
    where
        T: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(iter.collect::<Result<Vec<T>, DataError>>());
        });
        rx.recv_timeout(Duration::from_secs(10))
            .expect("DataLoader iteration did not finish within 10s (possible deadlock/hang)")
            .expect("data loader returned an unexpected test error")
    }

    #[test]
    fn single_threaded_yields_every_item_in_order_when_not_shuffled() {
        let loader = DataLoader::new(RangeDataset(10), VecCollate, 3).unwrap();
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
        let loader = DataLoader::new(RangeDataset(9), VecCollate, 3).unwrap();
        let batches = collect_with_timeout((&loader).into_iter());
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
    }

    #[test]
    fn empty_dataset_produces_zero_batches() {
        let loader = DataLoader::new(RangeDataset(0), VecCollate, 4).unwrap();
        let batches = collect_with_timeout((&loader).into_iter());
        assert!(batches.is_empty());
    }

    #[test]
    fn zero_batch_size_is_a_typed_construction_error() {
        let result = DataLoader::new(RangeDataset(4), VecCollate, 0);
        assert!(matches!(
            result,
            Err(Error::InvalidModuleState {
                operation: "data_loader_new",
                ..
            })
        ));
    }

    #[test]
    fn batch_size_larger_than_dataset_produces_one_short_batch() {
        let loader = DataLoader::new(RangeDataset(3), VecCollate, 100).unwrap();
        let batches = collect_with_timeout((&loader).into_iter());
        assert_eq!(batches, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn shuffle_preserves_the_full_set_of_items_just_reorders_them() {
        let loader = DataLoader::new(RangeDataset(50), VecCollate, 7)
            .unwrap()
            .with_shuffle(true);
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
        let loader = DataLoader::new(RangeDataset(1000), VecCollate, 10)
            .unwrap()
            .with_num_workers(8);
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
            .unwrap()
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
        let loader = DataLoader::new(RangeDataset(25), VecCollate, 10)
            .unwrap()
            .with_num_workers(8);
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
            fn collate(&self, batch: Vec<i32>) -> BatchResult<i32> {
                Ok(batch.into_iter().sum())
            }
        }

        let loader = DataLoader::new(RangeDataset(6), SumCollate, 3).unwrap();
        let mut sums = collect_with_timeout((&loader).into_iter());
        sums.sort_unstable();
        // batch [0,1,2] -> 3, batch [3,4,5] -> 12
        assert_eq!(sums, vec![3, 12]);
    }

    struct PanicDataset;
    impl Dataset for PanicDataset {
        type Item = i32;

        fn len(&self) -> usize {
            1
        }

        fn get(&self, _index: usize) -> BatchResult<Option<Self::Item>> {
            panic!("fixture panic")
        }
    }

    struct PanicCollate;
    impl Collate<i32> for PanicCollate {
        type Output = i32;

        fn collate(&self, _batch: Vec<i32>) -> BatchResult<Self::Output> {
            panic!("fixture panic")
        }
    }

    #[test]
    fn worker_panics_are_explicit_iterator_errors() {
        let loader = DataLoader::new(PanicDataset, VecCollate, 1)
            .unwrap()
            .with_num_workers(2);
        let error = (&loader).into_iter().next().unwrap().unwrap_err();
        assert!(matches!(
            error,
            DataError::WorkerPanicked {
                stage: "reading a sample",
                ..
            }
        ));
    }

    #[test]
    fn collate_panics_are_explicit_iterator_errors() {
        let loader = DataLoader::new(RangeDataset(1), PanicCollate, 1).unwrap();
        let error = (&loader).into_iter().next().unwrap().unwrap_err();
        assert!(matches!(error, DataError::CollatePanicked { worker_id: 0 }));
    }

    struct FailingDataset;
    impl Dataset for FailingDataset {
        type Item = i32;

        fn len(&self) -> usize {
            1
        }

        fn get(&self, _index: usize) -> BatchResult<Option<Self::Item>> {
            Err(DataError::Dataset("fixture failure".into()))
        }
    }

    #[test]
    fn dataset_errors_are_not_treated_as_end_of_data() {
        let loader = DataLoader::new(FailingDataset, VecCollate, 1).unwrap();
        assert!(matches!(
            (&loader).into_iter().next().unwrap(),
            Err(DataError::Dataset(message)) if message == "fixture failure"
        ));
    }

    struct SlowDataset;
    impl Dataset for SlowDataset {
        type Item = i32;

        fn len(&self) -> usize {
            1
        }

        fn get(&self, _index: usize) -> BatchResult<Option<Self::Item>> {
            std::thread::sleep(Duration::from_millis(25));
            Ok(Some(1))
        }
    }

    #[test]
    fn worker_timeout_is_an_explicit_iterator_error() {
        let loader = DataLoader::new(SlowDataset, VecCollate, 1)
            .unwrap()
            .with_num_workers(1)
            .with_timeout(Some(Duration::from_millis(1)));
        assert!(matches!(
            (&loader).into_iter().next().unwrap(),
            Err(DataError::Timeout { duration }) if duration == Duration::from_millis(1)
        ));
    }

    #[test]
    fn builder_controls_drop_last_and_deterministic_shuffle() {
        let make = || {
            DataLoader::builder(RangeDataset(10), VecCollate)
                .batch_size(3)
                .unwrap()
                .drop_last(true)
                .shuffle(true)
                .seed(17)
                .epoch(4)
                .prefetch(1)
                .unwrap()
                .build()
        };
        let first_loader = make();
        let second_loader = make();
        let first = collect_with_timeout((&first_loader).into_iter());
        let second = collect_with_timeout((&second_loader).into_iter());
        assert_eq!(first, second);
        assert_eq!(first.iter().map(Vec::len).sum::<usize>(), 9);
    }
}
