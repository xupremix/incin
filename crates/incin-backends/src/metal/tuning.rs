//! Metal kernel and storage-mode autotuning with a fingerprinted cache.
//!
//! Provides hardware-aware fingerprinting for Metal devices, candidate
//! enumeration for MSL and MPS graph kernels, storage-mode tuning policy,
//! and single-flight benchmark caching using the core tuning infrastructure.

use crate::kernel::KernelAccess;
#[cfg(any(feature = "autotune", test))]
use crate::kernel::KernelKey;
use crate::metal::mps::{
    MpsMatMulCandidate, MpsNormalizationCandidate, MpsPointwiseCandidate, MpsReductionCandidate,
};
use crate::metal::storage::MetalStorageMode;
#[cfg(any(feature = "metal", feature = "autotune", test))]
use crate::tuning::identity::{
    CompilerFingerprint, DeviceFingerprint, SoftwareVersion, TuningEnvironmentFingerprint,
};
#[cfg(any(feature = "autotune", test))]
use crate::tuning::{
    CandidateMeasurement, LaunchCandidate, TunedLaunch, TuningDecision, TuningKey, WorkloadBucket,
    claim_tuning, select_fastest,
};
use alloc::vec::Vec;
use incin_core::prelude::{DTypeId, DeviceKind, Error, Result};

// ---------------------------------------------------------------------------
// Metal Launch Candidate & Storage Mode
// ---------------------------------------------------------------------------

/// Complete launch and storage configuration candidate for a Metal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetalLaunchCandidate {
    /// Threadgroup size (block size) for MSL execution.
    pub block_size: u16,
    /// Memory access and unrolling pattern.
    pub access: KernelAccess,
    /// Preferred Metal buffer storage mode.
    pub storage_mode: MetalStorageMode,
    /// Whether MPS / MPSGraph acceleration is selected over native MSL.
    pub use_mps: bool,
}

impl MetalLaunchCandidate {
    /// Converts to the core `LaunchCandidate` representation for cache storage.
    #[cfg(any(feature = "autotune", test))]
    pub fn to_core_candidate(&self) -> LaunchCandidate {
        LaunchCandidate {
            block_size: self.block_size,
            access: self.access,
        }
    }

    /// Reconstructs a `MetalLaunchCandidate` from a core `LaunchCandidate` and storage options.
    #[cfg(any(feature = "autotune", test))]
    pub fn from_core(core: LaunchCandidate, storage_mode: MetalStorageMode, use_mps: bool) -> Self {
        Self {
            block_size: core.block_size,
            access: core.access,
            storage_mode,
            use_mps,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment Fingerprinting for Metal
// ---------------------------------------------------------------------------

/// Generates a stable hardware and compiler fingerprint for a Metal device.
#[cfg(any(feature = "metal", feature = "autotune", test))]
pub fn metal_environment_fingerprint(
    device_name: &str,
    is_unified_memory: bool,
) -> Result<TuningEnvironmentFingerprint<incin_core::prelude::Dyn>> {
    let dev_id = if device_name.is_empty() {
        "Apple Metal GPU"
    } else {
        device_name
    };
    let arch = if is_unified_memory {
        "apple-silicon-unified"
    } else {
        "metal-discrete"
    };

    let dev_fp = DeviceFingerprint::new_dyn(
        DeviceKind::Metal,
        dev_id,
        arch,
        SoftwareVersion::new(3, 0, 0),
    )
    .map_err(|e| Error::Msg(format!("{e:?}")))?;

    let comp_fp = CompilerFingerprint::new_dyn(
        DeviceKind::Metal,
        "metal-msl",
        SoftwareVersion::new(3, 0, 0),
        arch,
        &["incin-msl-options-v1"],
    )
    .map_err(|e| Error::Msg(format!("{e:?}")))?;

    TuningEnvironmentFingerprint::new_dyn(dev_fp, comp_fp).map_err(|e| Error::Msg(format!("{e:?}")))
}

// ---------------------------------------------------------------------------
// Storage-Mode Tuning Policy
// ---------------------------------------------------------------------------

/// Returns the optimal `MetalStorageMode` given element count and unified memory.
pub fn preferred_metal_storage_mode(n_bytes: usize, is_unified_memory: bool) -> MetalStorageMode {
    if is_unified_memory {
        // On Apple Silicon unified memory, Shared mode avoids copies.
        // For large temporary scratch buffers (> 64MB), Private mode can reduce host address space pressure.
        if n_bytes >= 64 * 1024 * 1024 {
            MetalStorageMode::Private
        } else {
            MetalStorageMode::Shared
        }
    } else {
        // Discrete Metal GPU: Shared for small buffers, Private for large GPU compute.
        if n_bytes < 1024 * 1024 {
            MetalStorageMode::Shared
        } else {
            MetalStorageMode::Private
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate Generators
// ---------------------------------------------------------------------------

/// Generates execution candidates for pointwise Metal operations.
pub fn metal_pointwise_candidates(
    dtype: DTypeId,
    n_elements: usize,
    dense: bool,
    packed_aligned: bool,
    storage_mode: MetalStorageMode,
) -> Vec<MetalLaunchCandidate> {
    let mps_candidate = MpsPointwiseCandidate::preferred(n_elements);
    let use_mps = mps_candidate == MpsPointwiseCandidate::MpsGraph;

    let width = match dtype {
        DTypeId::F16 | DTypeId::BF16 | DTypeId::F64 => 2,
        DTypeId::F32 => 4,
        _ => 1,
    };

    let mut accesses = vec![KernelAccess::Scalar { unroll_width: 1 }];
    if dense && n_elements >= 1024 && width > 1 {
        accesses.push(KernelAccess::Scalar {
            unroll_width: width,
        });
        if packed_aligned {
            accesses.push(KernelAccess::Packed {
                vector_width: width,
            });
        }
    }

    let mut candidates = Vec::with_capacity(accesses.len() * 3);
    for access in accesses {
        for block_size in [128, 256, 512] {
            candidates.push(MetalLaunchCandidate {
                block_size,
                access,
                storage_mode,
                use_mps,
            });
        }
    }
    candidates
}

/// Generates execution candidates for Metal reduction operations.
pub fn metal_reduction_candidates(
    contiguous_last_axis: bool,
    reduction_size: usize,
    storage_mode: MetalStorageMode,
) -> Vec<MetalLaunchCandidate> {
    let mps_candidate = MpsReductionCandidate::preferred(reduction_size);
    let use_mps = mps_candidate == MpsReductionCandidate::MpsGraph;

    let access = if contiguous_last_axis {
        KernelAccess::WarpReduction
    } else {
        KernelAccess::Scalar { unroll_width: 1 }
    };

    [64, 128, 256, 512]
        .into_iter()
        .map(|block_size| MetalLaunchCandidate {
            block_size,
            access,
            storage_mode,
            use_mps,
        })
        .collect()
}

/// Generates execution candidates for Metal GEMM / MatMul operations.
pub fn metal_matmul_candidates(
    m: usize,
    k: usize,
    n: usize,
    storage_mode: MetalStorageMode,
) -> Vec<MetalLaunchCandidate> {
    let mps_candidate = MpsMatMulCandidate::preferred(m, k, n);
    let use_mps = mps_candidate == MpsMatMulCandidate::Mps;

    [128, 256, 512]
        .into_iter()
        .map(|block_size| MetalLaunchCandidate {
            block_size,
            access: KernelAccess::Scalar { unroll_width: 4 },
            storage_mode,
            use_mps,
        })
        .collect()
}

/// Generates execution candidates for Metal normalization operations.
pub fn metal_normalization_candidates(
    is_layer_norm: bool,
    norm_size: usize,
    storage_mode: MetalStorageMode,
) -> Vec<MetalLaunchCandidate> {
    let mps_candidate = MpsNormalizationCandidate::preferred(norm_size);
    let use_mps = mps_candidate == MpsNormalizationCandidate::MpsGraph;

    let access = if is_layer_norm {
        KernelAccess::Welford
    } else {
        KernelAccess::Scalar { unroll_width: 1 }
    };

    [128, 256, 512]
        .into_iter()
        .map(|block_size| MetalLaunchCandidate {
            block_size,
            access,
            storage_mode,
            use_mps,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Default Fallback Selection
// ---------------------------------------------------------------------------

/// Returns default pointwise Metal launch candidate.
pub fn default_metal_pointwise_candidate(
    candidates: &[MetalLaunchCandidate],
) -> Result<MetalLaunchCandidate> {
    candidates
        .iter()
        .copied()
        .find(|c| c.block_size == 256)
        .ok_or_else(|| {
            Error::Msg("Metal pointwise candidate set has no 256-thread fallback".into())
        })
}

/// Returns default reduction Metal launch candidate.
pub fn default_metal_reduction_candidate(
    candidates: &[MetalLaunchCandidate],
) -> Result<MetalLaunchCandidate> {
    candidates
        .iter()
        .copied()
        .find(|c| c.block_size == 256)
        .ok_or_else(|| {
            Error::Msg("Metal reduction candidate set has no 256-thread fallback".into())
        })
}

// ---------------------------------------------------------------------------
// Autotune Coordination & Benchmark Winners
// ---------------------------------------------------------------------------

/// Evaluates benchmark measurements and selects the fastest Metal launch candidate.
#[cfg(any(feature = "autotune", test))]
#[allow(dead_code)]
pub(crate) fn select_fastest_metal(
    metal_candidates: &[MetalLaunchCandidate],
    measurements: &[CandidateMeasurement],
) -> Result<(MetalLaunchCandidate, TunedLaunch)> {
    let winner = select_fastest(measurements)?;
    let metal_winner = metal_candidates
        .iter()
        .copied()
        .find(|mc| mc.to_core_candidate() == winner.candidate)
        .ok_or_else(|| {
            Error::Msg("winning launch candidate not found in Metal candidate set".into())
        })?;

    Ok((metal_winner, winner))
}

/// Claims autotuning for a Metal problem key, coordinating concurrent threads.
#[cfg(any(feature = "autotune", test))]
#[allow(dead_code)]
pub(crate) fn claim_metal_tuning(
    env: TuningEnvironmentFingerprint<incin_core::prelude::Dyn>,
    kernel_key: &KernelKey,
    workload: WorkloadBucket,
    metal_candidates: &[MetalLaunchCandidate],
) -> Result<TuningDecision> {
    let key = TuningKey::new(env, kernel_key, workload);
    let core_candidates: Vec<_> = metal_candidates
        .iter()
        .map(|mc| mc.to_core_candidate())
        .collect();
    claim_tuning(key, &core_candidates)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_fingerprint_generation_is_stable() {
        let fp1 = metal_environment_fingerprint("Apple M1 Max", true).unwrap();
        let fp2 = metal_environment_fingerprint("Apple M1 Max", true).unwrap();
        assert_eq!(fp1, fp2);

        let fp_discrete = metal_environment_fingerprint("AMD Radeon", false).unwrap();
        assert_ne!(fp1, fp_discrete);
    }

    #[test]
    fn storage_mode_tuning_policy_respects_memory_architecture() {
        assert_eq!(
            preferred_metal_storage_mode(1024, true),
            MetalStorageMode::Shared
        );
        assert_eq!(
            preferred_metal_storage_mode(100 * 1024 * 1024, true),
            MetalStorageMode::Private
        );
        assert_eq!(
            preferred_metal_storage_mode(1024, false),
            MetalStorageMode::Shared
        );
        assert_eq!(
            preferred_metal_storage_mode(10 * 1024 * 1024, false),
            MetalStorageMode::Private
        );
    }

    #[test]
    fn metal_pointwise_candidates_generate_correct_spectrum() {
        let candidates =
            metal_pointwise_candidates(DTypeId::F32, 4096, true, true, MetalStorageMode::Shared);
        assert!(!candidates.is_empty());
        assert!(
            candidates
                .iter()
                .all(|c| c.storage_mode == MetalStorageMode::Shared)
        );
        let default_c = default_metal_pointwise_candidate(&candidates).unwrap();
        assert_eq!(default_c.block_size, 256);
    }

    #[test]
    fn metal_reduction_and_matmul_candidates_generate_valid_sets() {
        let red_c = metal_reduction_candidates(true, 2048, MetalStorageMode::Shared);
        assert_eq!(red_c.len(), 4);
        let default_red = default_metal_reduction_candidate(&red_c).unwrap();
        assert_eq!(default_red.block_size, 256);

        let mm_c = metal_matmul_candidates(1024, 1024, 1024, MetalStorageMode::Shared);
        assert_eq!(mm_c.len(), 3);
    }

    #[test]
    fn metal_tuning_claim_and_select_fastest() {
        use crate::kernel::KernelFamily;
        use incin_core::exec::LayoutClass;

        let env = metal_environment_fingerprint("Apple M1 Max", true).unwrap();
        let kernel_key = KernelKey::cuda(
            incin_core::prelude::OperationKind::Pointwise,
            KernelFamily::PointwiseUnary,
            "neg",
            DTypeId::F32,
            LayoutClass::Contiguous,
            KernelAccess::Scalar { unroll_width: 1 },
        )
        .unwrap();

        let candidates =
            metal_pointwise_candidates(DTypeId::F32, 4096, true, true, MetalStorageMode::Shared);

        let decision = claim_metal_tuning(
            env,
            &kernel_key,
            WorkloadBucket::pointwise(4096, true),
            &candidates,
        )
        .unwrap();

        match decision {
            TuningDecision::Measure(_permit) => {}
            TuningDecision::Cached(_) => panic!("expected un-cached initial claim"),
        }

        let measurements = vec![CandidateMeasurement {
            candidate: candidates[0].to_core_candidate(),
            synchronized_samples_ns: vec![100, 110, 120],
        }];

        let (winning_metal_c, launch) = select_fastest_metal(&candidates, &measurements).unwrap();
        assert_eq!(winning_metal_c, candidates[0]);
        assert_eq!(launch.median_ns, 110);
    }
}
