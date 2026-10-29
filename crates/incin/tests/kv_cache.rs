//! `KvCache` and `MultiHeadAttention::forward_with_cache` (issue #104).
//!
//! The cache is only worth having if (a) it stores and returns the keys and
//! values appended to it, (b) overflow is a typed failure rather than a
//! silent growth or panic, and (c) decoding a sequence in chunks against the
//! cache matches a single full forward of the same tokens.
#![cfg(feature = "cpu")]

use incin::Error;
use incin::nn::{AttentionConfig, KvCache, MultiHeadAttention, PositionEncoding};
use incin::prelude::*;

type Cpu = incin_backends::cpu::CpuBackendImpl;

/// Deterministic ramp input so a mismatch is never hidden by symmetry.
fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + (v as f32) * 0.01)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Appended keys and values come back from `kv()` in order, at the right
/// length, and `len` tracks them.
#[test]
fn append_exposes_the_stored_prefix_and_len() -> Result<()> {
    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.capacity(), 8);
    assert_eq!(cache.dims(), vec![1, 2, 8, 4]);

    let keys1 = ramp(vec![1, 2, 3, 4])?;
    let values1 = ramp(vec![1, 2, 3, 4])?.mul_scalar(2.0)?;
    cache.append(&keys1, &values1)?;
    assert_eq!(cache.len(), 3);

    let keys2 = ramp(vec![1, 2, 2, 4])?.add_scalar(10.0)?;
    let values2 = ramp(vec![1, 2, 2, 4])?.add_scalar(20.0)?;
    cache.append(&keys2, &values2)?;
    assert_eq!(cache.len(), 5);

    let (k, v) = cache.kv()?;
    assert_eq!(k.dims().dims(), &[1, 2, 5, 4]);
    assert_eq!(v.dims().dims(), &[1, 2, 5, 4]);

    // The cache stores `[batch, kv_heads, seq, head_dim]`, so the flat order
    // of the read-back is by head, not by append chunk: compare against an
    // axis-2 concat of the inputs, which is what appending means.
    let expected_k = keys1.concat(&keys2, 2isize)?.to_vec1::<f32>()?;
    let got_k = k.to_vec1::<f32>()?;
    assert!(
        max_abs_diff(&expected_k, &got_k) < 1e-6,
        "stored keys diverge from what was appended"
    );

    let expected_v = values1.concat(&values2, 2isize)?.to_vec1::<f32>()?;
    let got_v = v.to_vec1::<f32>()?;
    assert!(
        max_abs_diff(&expected_v, &got_v) < 1e-6,
        "stored values diverge from what was appended"
    );

    cache.reset();
    assert_eq!(cache.len(), 0);
    let (k0, _) = cache.kv()?;
    assert_eq!(k0.dims().dims(), &[1, 2, 0, 4]);
    Ok(())
}

/// Capacity is part of the type: overflow names the numbers instead of
/// reallocating.
#[test]
fn append_past_capacity_is_a_typed_failure() -> Result<()> {
    let mut cache = KvCache::<s![1, 1, 4, 2], Cpu, f32>::new(())?;
    let first = Tensor::<Dyn, Cpu>::zeros(vec![1, 1, 3, 2])?;
    cache.append(&first, &first)?;
    assert_eq!(cache.len(), 3);

    let overflow = Tensor::<Dyn, Cpu>::zeros(vec![1, 1, 2, 2])?;
    let err = cache
        .append(&overflow, &overflow)
        .expect_err("3+2 exceeds capacity 4");
    assert!(
        matches!(
            err,
            Error::CacheCapacityExceeded {
                operation: "KvCache::append",
                requested: 5,
                capacity: 4,
            }
        ),
        "expected CacheCapacityExceeded, got {err:?}"
    );
    assert_eq!(cache.len(), 3, "a failed append must not advance len");
    Ok(())
}

/// Geometry mismatches are refused with the offending shapes named.
#[test]
fn append_with_a_mismatched_head_geometry_is_refused() -> Result<()> {
    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    // batch/kv/head_dim disagree with [1, 2, 8, 4].
    let bad = Tensor::<Dyn, Cpu>::zeros(vec![1, 3, 2, 4])?;
    let err = cache
        .append(&bad, &bad)
        .expect_err("kv_heads 3 does not match cache kv_heads 2");
    assert!(
        matches!(
            err,
            Error::InvalidModuleState {
                operation: "KvCache::append",
                ..
            }
        ),
        "expected InvalidModuleState, got {err:?}"
    );
    Ok(())
}

/// A `Dyn` cache shape is rejected: capacity must be fixed by the type.
#[test]
fn a_dynamic_cache_shape_is_refused() {
    let err = KvCache::<Dyn, Cpu, f32>::new(vec![1, 2, 8, 4])
        .err()
        .expect("Dyn cannot be a KvCache shape");
    assert!(
        err.to_string().contains("static rank-4"),
        "the refusal should name the static rank-4 requirement, said: {err}"
    );
}

/// Decoding a sequence in chunks against the cache matches one full forward.
///
/// Causal + rotary + grouped-query: the three features whose absolute-position
/// bookkeeping a chunked path can get wrong while still producing finite
/// outputs of the right shape.
#[test]
fn chunked_decode_matches_a_full_forward() -> Result<()> {
    let width = 16;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = width / n_heads;
    let seq = 6;
    let config = AttentionConfig::causal().with_rotary(10_000.0, 32);

    let attention = MultiHeadAttention::<16, 4, 2, Cpu>::build(config, (), ())?;
    let x = ramp(vec![1, seq, width])?;

    let full = attention.forward(x.clone())?;

    // [batch=1, kv_heads=2, capacity=8, head_dim=4]
    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    assert_eq!(cache.kv_heads(), n_kv_heads);
    assert_eq!(cache.head_dim(), head_dim);

    let chunk = 2;
    let mut pieces: Vec<f32> = Vec::new();
    for start in (0..seq).step_by(chunk) {
        let end = (start + chunk).min(seq);
        let step = x
            .clone()
            .try_narrow(1isize, start, end - start)?
            .forget_layout();
        let y = attention.forward_with_cache(step, &mut cache)?;
        assert_eq!(y.dims().dims(), &[1, end - start, width]);
        assert!(
            !y.requires_grad(),
            "forward_with_cache must not report gradients"
        );
        pieces.extend(y.to_vec1::<f32>()?);
    }
    assert_eq!(cache.len(), seq);

    let full_vals = full.to_vec1::<f32>()?;
    assert_eq!(pieces.len(), full_vals.len());
    let diff = max_abs_diff(&pieces, &full_vals);
    assert!(
        diff < 1e-4,
        "chunked decode diverged from the full forward by {diff:e}"
    );
    Ok(())
}

/// A cache whose head geometry disagrees with the module is refused.
#[test]
fn forward_with_cache_rejects_a_mismatched_cache() -> Result<()> {
    let attention = MultiHeadAttention::<16, 2, 2, Cpu>::build(AttentionConfig::causal(), (), ())?;
    // head_dim 4 but the module uses 8; kv_heads match by accident of shape.
    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    let step = Tensor::<Dyn, Cpu>::zeros(vec![1, 1, 16])?;
    let err = attention
        .forward_with_cache(step, &mut cache)
        .expect_err("head_dim 4 cannot serve a module with head_dim 8");
    assert!(
        matches!(
            err,
            Error::InvalidModuleState {
                operation: "attention forward_with_cache",
                ..
            }
        ),
        "expected InvalidModuleState, got {err:?}"
    );
    Ok(())
}

/// Rotary + cache: a chunk that would pass `max_seq_len` fails loudly.
#[test]
fn forward_with_cache_enforces_the_rotary_max_seq_len() -> Result<()> {
    let config = AttentionConfig::causal().with_rotary(10_000.0, 4);
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ())?;
    // capacity allows 6 tokens but the rotary tables stop at 4.
    let mut cache = KvCache::<s![1, 2, 6, 4], Cpu, f32>::new(())?;
    let ok = Tensor::<Dyn, Cpu>::zeros(vec![1, 4, 8])?;
    attention.forward_with_cache(ok, &mut cache)?;
    assert_eq!(cache.len(), 4);

    let overflow = Tensor::<Dyn, Cpu>::zeros(vec![1, 1, 8])?;
    let err = attention
        .forward_with_cache(overflow, &mut cache)
        .expect_err("position 5 exceeds max_seq_len 4");
    assert!(
        err.to_string().contains("max_seq_len"),
        "the refusal should name max_seq_len, said: {err}"
    );
    // A failed step must not have appended anything.
    assert_eq!(cache.len(), 4);
    Ok(())
}

/// Position encoding is part of the same surface: `None` still caches.
#[test]
fn forward_with_cache_works_without_rotary() -> Result<()> {
    let attention = MultiHeadAttention::<8, 2, 1, Cpu>::build(AttentionConfig::causal(), (), ())?;
    let mut cache = KvCache::<s![1, 1, 8, 4], Cpu, f32>::new(())?;
    let step = Tensor::<Dyn, Cpu>::zeros(vec![1, 3, 8])?;
    let y = attention.forward_with_cache(step, &mut cache)?;
    assert_eq!(y.dims().dims(), &[1, 3, 8]);
    assert_eq!(cache.len(), 3);
    assert!(matches!(attention.config.position, PositionEncoding::None));
    Ok(())
}
