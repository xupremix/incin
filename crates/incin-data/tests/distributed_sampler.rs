//! Acceptance tests for rank-aware dataset sharding (`DistributedSampler`).
//!
//! The invariants under test are set-theoretic, per issue #98: disjointness
//! and coverage across ranks, equal step counts under `Pad`, cross-rank
//! permutation consistency, identity at world size 1, both worker paths, and
//! a typed construction error for an out-of-range rank.

use std::collections::BTreeSet;

use incin_data::{DataError, DataLoader, DistributedSampler, RemainderPolicy};

/// A dataset whose samples are their own indices, so batches double as the
/// index sequences the invariants are stated over.
struct Indices(usize);

impl incin_data::Dataset for Indices {
    type Item = i64;

    fn len(&self) -> usize {
        self.0
    }

    fn get(&self, index: usize) -> Result<Option<Self::Item>, DataError> {
        if index < self.0 {
            Ok(Some(index as i64))
        } else {
            Ok(None)
        }
    }
}

fn sampler(world_size: usize, rank: usize, remainder: RemainderPolicy) -> DistributedSampler {
    DistributedSampler {
        world_size,
        rank,
        remainder,
    }
}

/// Collects every sample one loader yields over a full epoch.
fn epoch_items(loader: &DataLoader<Indices>) -> Vec<i64> {
    let mut items = Vec::new();
    for batch in loader {
        for item in batch.expect("epoch should not fail") {
            items.push(item);
        }
    }
    items
}

#[test]
fn drop_policy_partitions_disjointly_and_covers_all_but_the_tail() {
    const LEN: usize = 103;
    const WORLD: usize = 4;

    let mut per_rank = Vec::new();
    for rank in 0..WORLD {
        let loader = DataLoader::builder(Indices(LEN))
            .batch_size(8)
            .shuffle(true)
            .seed(42)
            .sampler(sampler(WORLD, rank, RemainderPolicy::Drop))
            .build()
            .expect("valid rank");
        per_rank.push(epoch_items(&loader));
    }

    // Disjointness: no sample appears on two ranks.
    for i in 0..WORLD {
        for j in (i + 1)..WORLD {
            assert!(
                per_rank[i].iter().all(|s| !per_rank[j].contains(s)),
                "ranks {i} and {j} shared a sample"
            );
        }
    }

    // Coverage: the union misses at most `world_size - 1` trailing samples of
    // the shared permutation, so it is exactly LEN rounded down to a multiple
    // of the world size.
    let mut union: BTreeSet<i64> = BTreeSet::new();
    for rank_items in &per_rank {
        union.extend(rank_items.iter().copied());
    }
    assert_eq!(union.len(), LEN - LEN % WORLD);
    // Equal unique-sample counts per rank.
    for rank_items in &per_rank {
        assert_eq!(rank_items.len(), LEN / WORLD);
    }
}

#[test]
fn pad_policy_gives_every_rank_identical_step_counts() {
    const LEN: usize = 10;
    const WORLD: usize = 4;
    const BATCH: usize = 3;

    let mut step_counts = Vec::new();
    for rank in 0..WORLD {
        let loader = DataLoader::builder(Indices(LEN))
            .batch_size(BATCH)
            .shuffle(true)
            .seed(7)
            .sampler(sampler(WORLD, rank, RemainderPolicy::Pad))
            .build()
            .expect("valid rank");
        let batches: Vec<_> = (&loader).into_iter().collect();
        assert_eq!(batches.len(), 1, "rank {rank} step count diverged");
        assert_eq!(batches[0].as_ref().expect("batch ok").len(), BATCH);
        step_counts.push(batches.len());
    }
    assert!(step_counts.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn all_ranks_derive_the_same_shared_permutation() {
    const LEN: usize = 64;
    const WORLD: usize = 4;
    const SEED: u64 = 1234;

    // Stride-slicing one shared permutation means rank r sees positions
    // r, r+W, r+2W, ... . Interleaving the ranks' sequences must therefore
    // reproduce the shared permutation exactly - and that permutation is
    // precisely what an unsharded loader with the same (seed, epoch) yields.
    let per_rank: Vec<Vec<i64>> = (0..WORLD)
        .map(|rank| {
            let loader = DataLoader::builder(Indices(LEN))
                .batch_size(LEN)
                .shuffle(true)
                .seed(SEED)
                .sampler(sampler(WORLD, rank, RemainderPolicy::Drop))
                .build()
                .expect("valid rank");
            epoch_items(&loader)
        })
        .collect();

    let mut shared = Vec::with_capacity(LEN);
    for k in 0..per_rank[0].len() {
        for rank_items in &per_rank {
            shared.push(rank_items[k]);
        }
    }

    let plain = DataLoader::builder(Indices(LEN))
        .batch_size(LEN)
        .shuffle(true)
        .seed(SEED)
        .build()
        .expect("plain builds");
    assert_eq!(
        shared,
        epoch_items(&plain),
        "ranks disagreed on the permutation"
    );
}

#[test]
fn world_size_one_is_behaviorally_identical_to_no_sampler() {
    let plain = DataLoader::builder(Indices(50))
        .batch_size(6)
        .shuffle(true)
        .seed(99)
        .build()
        .expect("plain builds");
    let sharded = DataLoader::builder(Indices(50))
        .batch_size(6)
        .shuffle(true)
        .seed(99)
        .sampler(sampler(1, 0, RemainderPolicy::Pad))
        .build()
        .expect("sharded builds");

    assert_eq!(epoch_items(&plain), epoch_items(&sharded));
}

#[test]
fn set_epoch_rotates_partitions_while_staying_consistent_across_ranks() {
    const WORLD: usize = 3;

    let partition = |rank: usize, epoch: u64| -> Vec<i64> {
        let mut loader = DataLoader::builder(Indices(48))
            .batch_size(16)
            .shuffle(true)
            .seed(5)
            .sampler(sampler(WORLD, rank, RemainderPolicy::Drop))
            .build()
            .expect("valid rank");
        loader.set_epoch(epoch);
        epoch_items(&loader)
    };

    let e0_r0 = partition(0, 0);
    let e1_r0 = partition(0, 1);
    assert_ne!(e0_r0, e1_r0, "the epoch must change rank 0's view");

    // Cross-rank consistency at the new epoch: same permutation, different
    // slices, verified by disjointness within the epoch.
    let e1 = (0..WORLD).map(|r| partition(r, 1)).collect::<Vec<_>>();
    for i in 0..WORLD {
        for j in (i + 1)..WORLD {
            assert!(!e1[i].iter().any(|s| e1[j].contains(s)));
        }
    }
}

#[test]
fn worker_backed_path_shards_identically_to_the_sync_path() {
    const LEN: usize = 40;
    const WORLD: usize = 4;

    let sharded = |workers: usize, rank: usize| -> Vec<i64> {
        let loader = DataLoader::builder(Indices(LEN))
            .batch_size(5)
            .shuffle(true)
            .seed(11)
            .workers(workers)
            .sampler(sampler(WORLD, rank, RemainderPolicy::Drop))
            .build()
            .expect("valid rank");
        epoch_items(&loader)
    };

    // Same shard regardless of worker count...
    for rank in 0..WORLD {
        assert_eq!(sharded(0, rank), sharded(2, rank), "rank {rank} drifted");
    }
    // ...and still disjoint across ranks.
    let all: Vec<_> = (0..WORLD).map(|r| sharded(2, r)).collect();
    for i in 0..WORLD {
        for j in (i + 1)..WORLD {
            assert!(!all[i].iter().any(|s| all[j].contains(s)));
        }
    }
}

#[test]
fn rank_at_or_above_world_size_is_a_typed_construction_error() {
    let err = DataLoader::builder(Indices(10))
        .sampler(sampler(4, 4, RemainderPolicy::Drop))
        .build()
        .err()
        .expect("rank == world_size must be rejected");
    assert!(err.to_string().contains("out of range"), "{err}");

    let err = DataLoader::builder(Indices(10))
        .sampler(sampler(0, 0, RemainderPolicy::Drop))
        .build()
        .err()
        .expect("world_size zero must be rejected");
    assert!(err.to_string().contains("non-zero"), "{err}");
}
