//! Integration tests for Metal autotuning, fingerprinting, and storage mode selection.

#![cfg(feature = "metal")]

use incin_backends::metal::MetalStorageMode;
use incin_backends::metal::tuning::{
    default_metal_pointwise_candidate, default_metal_reduction_candidate,
    metal_environment_fingerprint, metal_matmul_candidates, metal_normalization_candidates,
    metal_pointwise_candidates, metal_reduction_candidates, preferred_metal_storage_mode,
};
use incin_core::prelude::DTypeId;

#[test]
fn test_metal_fingerprint_creation() {
    let fp = metal_environment_fingerprint("Apple M2 Ultra", true).unwrap();
    assert_eq!(fp.device().persistent_id(), "Apple M2 Ultra");
    assert_eq!(fp.device().architecture(), "apple-silicon-unified");
}

#[test]
fn test_metal_storage_mode_tuning_policy() {
    // Unified memory policy
    assert_eq!(
        preferred_metal_storage_mode(4096, true),
        MetalStorageMode::Shared
    );
    assert_eq!(
        preferred_metal_storage_mode(128 * 1024 * 1024, true),
        MetalStorageMode::Private
    );

    // Discrete memory policy
    assert_eq!(
        preferred_metal_storage_mode(512 * 1024, false),
        MetalStorageMode::Shared
    );
    assert_eq!(
        preferred_metal_storage_mode(8 * 1024 * 1024, false),
        MetalStorageMode::Private
    );
}

#[test]
fn test_metal_pointwise_candidate_generation() {
    let candidates =
        metal_pointwise_candidates(DTypeId::F32, 8192, true, true, MetalStorageMode::Shared);
    assert!(!candidates.is_empty());
    for c in &candidates {
        assert_eq!(c.storage_mode, MetalStorageMode::Shared);
    }
    let default_c = default_metal_pointwise_candidate(&candidates).unwrap();
    assert_eq!(default_c.block_size, 256);
}

#[test]
fn test_metal_reduction_candidates() {
    let candidates = metal_reduction_candidates(true, 4096, MetalStorageMode::Shared);
    assert_eq!(candidates.len(), 4);
    let default_c = default_metal_reduction_candidate(&candidates).unwrap();
    assert_eq!(default_c.block_size, 256);
}

#[test]
fn test_metal_matmul_and_norm_candidates() {
    let matmul_c = metal_matmul_candidates(512, 512, 512, MetalStorageMode::Shared);
    assert_eq!(matmul_c.len(), 3);

    let norm_c = metal_normalization_candidates(true, 1024, MetalStorageMode::Shared);
    assert_eq!(norm_c.len(), 3);
}
