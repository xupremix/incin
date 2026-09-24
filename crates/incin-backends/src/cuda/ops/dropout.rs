//! Counter-based, reproducible dropout mask generation for the CUDA backend.
//!
//! Issue #84: training dropout must be reproducible from a seed rather than
//! drawn from a thread-local host RNG. The previous `Execute<op::Dropout>`
//! path called `CudaBackendImpl::rand`, which samples `rand::rng()` with no
//! seed carried through `ExecutionContext` - two runs of the same graph on
//! the same inputs produced different masks, and there was no way to replay
//! a draw for debugging.
//!
//! This module owns a pure, host-side counter hash: `hash_uniform(seed, index)`
//! maps a (seed, flat index) pair to a `f32` in `[0, 1)`. The same function
//! is the single source of truth for both the unit tests below and the mask
//! upload in `launch_dropout_mask`, so a test that pins the hash pins the
//! kernel input. A process-wide monotonically increasing draw offset advances
//! by the element count on every call, so consecutive dropouts on the same
//! seed still see disjoint index ranges and cannot collide.
//!
//! The launcher returns the raw uniform draws; the executor still applies the
//! existing `add_scalar(-p) -> step -> mul -> mul_scalar(1/(1-p))` chain so
//! the materialized mask stays on the tape and the backward pass replays the
//! exact forward draw (see `dropout_trains_through_the_replayed_mask_on_cuda`).

use crate::cuda::storage::CudaStorage;
use incin_core::error::Result;
use incin_core::tensor::dtype::DTypeId;

/// Module-level seed for every dropout draw on this process.
///
/// Defaults to a fixed constant so tests are deterministic out of the box;
/// `set_dropout_seed` resets it (and the draw offset) when a caller wants a
/// fresh stream.
static DROPOUT_SEED: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);

/// Monotonic flat-index offset advanced by `numel` on every draw.
static DROPOUT_OFFSET: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Replace the process-wide dropout seed and rewind the draw offset.
///
/// Intended for tests and deterministic replay; ordinary training leaves the
/// default seed alone.
#[cfg(test)]
pub(crate) fn set_dropout_seed(seed: u64) {
    DROPOUT_SEED.store(seed, core::sync::atomic::Ordering::Relaxed);
    DROPOUT_OFFSET.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// SplitMix64 finalizer: the mix that turns a (seed, index) pair into a
/// well-scattered 64-bit value. Kept as a pure function so tests can call it
/// directly without touching atomics.
#[must_use]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map `(seed, index)` to a uniform `f32` in `[0, 1)`.
///
/// Deterministic, pure, and independent of call order: the same pair always
/// yields the same float. The high 24 bits of the mix are shifted into the
/// mantissa of a `1.0` bit pattern, then `1.0` is subtracted - the standard
/// "u32 to unit float" trick that keeps every bit of entropy the hash
/// produced.
#[must_use]
pub(crate) fn hash_uniform(seed: u64, index: u64) -> f32 {
    let mixed = mix64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(index));
    let bits = (mixed >> 40) as u32; // top 24 bits -> mantissa
    f32::from_bits(0x3F80_0000 | bits) - 1.0
}

/// Reserve `numel` consecutive counter indices and return
/// `(seed, start_index)`.
fn reserve_draw(numel: u64) -> (u64, u64) {
    let seed = DROPOUT_SEED.load(core::sync::atomic::Ordering::Relaxed);
    let start = DROPOUT_OFFSET.fetch_add(numel, core::sync::atomic::Ordering::Relaxed);
    (seed, start)
}

/// Generate `numel` counter-based uniform draws for an explicit
/// `(seed, start)` range.
///
/// Pure: no atomics and no device. `launch_dropout_mask` reserves a
/// disjoint range first and then calls this, so a test that pins these
/// values pins exactly what the mask upload sends.
#[must_use]
pub(crate) fn dropout_draws(seed: u64, start: u64, numel: u64) -> alloc::vec::Vec<f32> {
    (0..numel)
        .map(|i| hash_uniform(seed, start.wrapping_add(i)))
        .collect()
}

/// Generate `numel` counter-based uniform draws and upload them as `f32`
/// CUDA storage with `shape`.
///
/// Fail-closed on an empty tensor (returns a zero-length buffer without
/// advancing the counter) and on a numel that does not fit `i32` for the
/// downstream elementwise launches.
pub(crate) fn launch_dropout_mask(shape: &[usize]) -> Result<CudaStorage> {
    let numel = crate::bytes::checked_numel(shape)?;
    if numel == 0 {
        return crate::cuda::backend::cuda_from_f32(
            shape,
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cuda(0),
            alloc::vec::Vec::new(),
            "dropout_mask",
        );
    }
    let numel_u64 = u64::try_from(numel).map_err(|_| {
        incin_core::error::Error::Msg("dropout mask element count exceeds u64".into())
    })?;
    let (seed, start) = reserve_draw(numel_u64);
    let values = dropout_draws(seed, start, numel_u64);
    crate::cuda::backend::cuda_from_f32(
        shape,
        DTypeId::F32.descriptor(),
        &incin_core::tensor::device::DeviceId::cuda(0),
        values,
        "dropout_mask",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_uniform_is_in_the_unit_interval() {
        for i in 0..10_000 {
            let v = hash_uniform(42, i);
            assert!((0.0..1.0).contains(&v), "hash_uniform(42, {i}) = {v}");
        }
    }

    #[test]
    fn hash_uniform_is_deterministic_per_pair() {
        assert_eq!(hash_uniform(7, 123), hash_uniform(7, 123));
        assert_ne!(hash_uniform(7, 123), hash_uniform(7, 124));
        assert_ne!(hash_uniform(7, 123), hash_uniform(8, 123));
    }

    #[test]
    fn hash_uniform_roughly_fills_the_interval() {
        // 4 buckets over 40k draws: no bucket should be empty or dominate.
        let mut buckets = [0usize; 4];
        for i in 0..40_000u64 {
            let v = hash_uniform(0xDEAD_BEEF, i);
            let b = ((v * 4.0) as usize).min(3);
            buckets[b] += 1;
        }
        for count in buckets {
            assert!(count > 8_000 && count < 12_000, "skewed bucket {buckets:?}");
        }
    }

    #[test]
    fn mix64_scatters_adjacent_inputs() {
        let a = mix64(0);
        let b = mix64(1);
        assert_ne!(a, b);
        // Adjacent seeds should not produce adjacent outputs.
        assert!(a ^ b > 1_000_000);
    }

    #[test]
    fn set_dropout_seed_resets_the_draw_offset() {
        set_dropout_seed(1);
        let (seed_a, start_a) = reserve_draw(10);
        let (_seed_b, start_b) = reserve_draw(10);
        assert_eq!(seed_a, 1);
        assert_eq!(start_b, start_a + 10);
        set_dropout_seed(1);
        let (_, start_c) = reserve_draw(10);
        assert_eq!(start_c, 0);
    }
}
