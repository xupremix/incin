//! Candidate generation and bounded benchmark-result caching.
//!
//! This module deliberately separates candidate enumeration from measurement.
//! A caller must supply real synchronized device timings before a winner is
//! cached; static fallback order is never recorded as if it were benchmarked.

/// Atomic, bounded persistent storage for verified tuning results.
#[cfg(any(feature = "autotune", test))]
pub mod cache;
/// Stable device, compiler, and topology identities used by tuning keys.
#[cfg(any(feature = "metal", feature = "autotune", test))]
pub mod identity;
/// Policy-aware kernel, collective, and execution-plan tuning service.
#[cfg(any(feature = "autotune", test))]
pub mod service;
/// Shape and layout driven signatures and candidate pruning.
pub mod signature;
/// Tuning telemetry, provenance, and explain formatting.
#[cfg(any(feature = "autotune", test))]
pub mod telemetry;

#[cfg(feature = "autotune")]
pub use cache::{CacheLimits, CacheRecovery, PersistentTuningCache};
#[cfg(feature = "autotune")]
pub use identity::{CompilerFingerprint, DeviceFingerprint, TuningEnvironmentFingerprint};
#[cfg(feature = "autotune")]
pub use service::{AutotunePolicy, SelectionSource, TuningContext, TuningScope, TuningSelection};
pub use signature::{AlignmentClass, DTypePolicyId, KernelSignature, RankClass};
#[cfg(feature = "autotune")]
pub use telemetry::{TuningExplain, TuningProvenance};

use crate::kernel::KernelAccess;
#[cfg(any(feature = "autotune", test))]
use crate::kernel::KernelKey;
#[cfg(any(feature = "autotune", feature = "cuda", test))]
use alloc::vec::Vec;
#[cfg(any(feature = "autotune", test))]
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
};
#[cfg(any(feature = "autotune", feature = "cuda", test))]
use incin_core::error::{Error, Result};
#[cfg(any(feature = "cuda", test))]
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};
#[cfg(any(feature = "autotune", test))]
use std::sync::{Condvar, Mutex, OnceLock};

#[cfg(any(feature = "cuda", test))]
const POINTWISE_TUNING_THRESHOLD: usize = 1024;
#[cfg(any(feature = "autotune", test))]
const MAX_TUNING_ENTRIES: usize = 1024;
#[cfg(all(feature = "cuda", feature = "autotune"))]
const CUDA_TUNING_WARMUPS: usize = 2;
#[cfg(all(feature = "cuda", feature = "autotune"))]
const CUDA_TUNING_SAMPLES: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg(any(feature = "autotune", test))]
pub(crate) struct WorkloadBucket {
    pub(crate) elements_log2: u8,
    pub(crate) reduction_log2: u8,
    pub(crate) packed_alignment: bool,
}

#[cfg(any(feature = "autotune", test))]
impl WorkloadBucket {
    pub(crate) fn pointwise(elements: usize, packed_alignment: bool) -> Self {
        Self {
            elements_log2: size_log2_bucket(elements),
            reduction_log2: 0,
            packed_alignment,
        }
    }

    pub(crate) fn reduction(rows: usize, reduction_size: usize) -> Self {
        Self {
            elements_log2: size_log2_bucket(rows),
            reduction_log2: size_log2_bucket(reduction_size),
            packed_alignment: false,
        }
    }

    pub(crate) fn normalization(batch_size: usize, norm_size: usize) -> Self {
        Self {
            elements_log2: size_log2_bucket(batch_size),
            reduction_log2: size_log2_bucket(norm_size),
            packed_alignment: false,
        }
    }
}

#[cfg(any(feature = "autotune", test))]
fn size_log2_bucket(size: usize) -> u8 {
    if size <= 1 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()).min(u8::MAX.into()) as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg(any(feature = "autotune", test))]
pub(crate) struct TuningKey {
    pub(crate) environment: identity::TuningEnvironmentFingerprint,
    pub(crate) problem: String,
    pub(crate) workload: WorkloadBucket,
}

#[cfg(any(feature = "autotune", test))]
impl TuningKey {
    pub(crate) fn new(
        environment: identity::TuningEnvironmentFingerprint,
        kernel: &KernelKey,
        workload: WorkloadBucket,
    ) -> Self {
        Self {
            environment,
            problem: kernel.tuning_problem_id(),
            workload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchCandidate {
    pub block_size: u16,
    pub access: KernelAccess,
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn preferred_pointwise_width(dtype: DTypeDescriptor) -> u8 {
    match dtype.builtin_id() {
        Some(DTypeId::F16 | DTypeId::BF16 | DTypeId::F64) => 2,
        Some(DTypeId::F32) => 4,
        _ => 1,
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn pointwise_candidates(
    dtype: DTypeDescriptor,
    elements: usize,
    dense: bool,
    packed_aligned: bool,
) -> Vec<LaunchCandidate> {
    let width = preferred_pointwise_width(dtype);
    let mut accesses = vec![KernelAccess::Scalar { unroll_width: 1 }];
    if dense && elements >= POINTWISE_TUNING_THRESHOLD && width > 1 {
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
            candidates.push(LaunchCandidate { block_size, access });
        }
    }
    candidates
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn default_pointwise_candidate(
    candidates: &[LaunchCandidate],
) -> Result<LaunchCandidate> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.block_size == 256)
        .max_by_key(|candidate| match candidate.access {
            KernelAccess::Packed { .. } => 2,
            KernelAccess::Scalar { unroll_width } if unroll_width > 1 => 1,
            _ => 0,
        })
        .ok_or_else(|| Error::Msg("pointwise candidate set has no 256-thread fallback".into()))
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn reduction_candidates(contiguous_last_axis: bool) -> Vec<LaunchCandidate> {
    let access = if contiguous_last_axis {
        KernelAccess::WarpReduction
    } else {
        KernelAccess::Scalar { unroll_width: 1 }
    };
    [64, 128, 256, 512]
        .into_iter()
        .map(|block_size| LaunchCandidate { block_size, access })
        .collect()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn default_reduction_candidate(
    candidates: &[LaunchCandidate],
) -> Result<LaunchCandidate> {
    candidates
        .iter()
        .copied()
        .find(|candidate| candidate.block_size == 256)
        .ok_or_else(|| Error::Msg("reduction candidate set has no 256-thread fallback".into()))
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn normalization_candidates(is_layer_norm: bool) -> Vec<LaunchCandidate> {
    let access = if is_layer_norm {
        KernelAccess::Welford
    } else {
        KernelAccess::Scalar { unroll_width: 1 }
    };
    [128, 256, 512]
        .into_iter()
        .map(|block_size| LaunchCandidate { block_size, access })
        .collect()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn default_normalization_candidate(
    candidates: &[LaunchCandidate],
) -> Result<LaunchCandidate> {
    candidates
        .iter()
        .copied()
        .find(|candidate| candidate.block_size == 256)
        .ok_or_else(|| Error::Msg("normalization candidate set has no 256-thread fallback".into()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(feature = "autotune", test))]
pub(crate) struct CandidateMeasurement {
    pub(crate) candidate: LaunchCandidate,
    pub(crate) synchronized_samples_ns: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(feature = "autotune", test))]
pub(crate) struct TunedLaunch {
    pub(crate) candidate: LaunchCandidate,
    pub(crate) median_ns: u64,
    pub(crate) sample_count: u16,
}

#[cfg(any(feature = "autotune", test))]
pub(crate) fn select_fastest(measurements: &[CandidateMeasurement]) -> Result<TunedLaunch> {
    let mut winners = Vec::with_capacity(measurements.len());
    for measurement in measurements {
        if measurement.synchronized_samples_ns.is_empty() {
            return Err(Error::Msg(format!(
                "candidate {:?} has no synchronized timing samples",
                measurement.candidate
            )));
        }
        let mut samples = measurement.synchronized_samples_ns.clone();
        samples.sort_unstable();
        let median_ns = samples[samples.len() / 2];
        let sample_count = u16::try_from(samples.len())
            .map_err(|_| Error::Msg("autotune sample count exceeds u16".into()))?;
        winners.push(TunedLaunch {
            candidate: measurement.candidate,
            median_ns,
            sample_count,
        });
    }
    winners
        .into_iter()
        .min_by_key(|winner| (winner.median_ns, winner.candidate))
        .ok_or_else(|| Error::Msg("cannot tune an empty candidate set".into()))
}

#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "autotune", test))]
struct CacheEntry {
    launch: TunedLaunch,
    generation: u64,
}

#[derive(Default)]
#[cfg(any(feature = "autotune", test))]
struct TuningCache {
    entries: BTreeMap<TuningKey, CacheEntry>,
    in_flight: BTreeSet<TuningKey>,
    generation: u64,
}

#[cfg(any(feature = "autotune", test))]
impl TuningCache {
    fn get(&mut self, key: &TuningKey) -> Option<TunedLaunch> {
        self.generation = self.generation.wrapping_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.generation = self.generation;
        Some(entry.launch)
    }

    fn insert(&mut self, key: TuningKey, launch: TunedLaunch) {
        self.generation = self.generation.wrapping_add(1);
        if !self.entries.contains_key(&key)
            && self.entries.len() >= MAX_TUNING_ENTRIES
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.generation)
                .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(
            key,
            CacheEntry {
                launch,
                generation: self.generation,
            },
        );
    }
}

#[cfg(any(feature = "autotune", test))]
struct TuningCoordinator {
    state: Mutex<TuningCache>,
    ready: Condvar,
}

#[cfg(any(feature = "autotune", test))]
fn coordinator() -> &'static TuningCoordinator {
    static COORDINATOR: OnceLock<TuningCoordinator> = OnceLock::new();
    COORDINATOR.get_or_init(|| TuningCoordinator {
        state: Mutex::new(TuningCache::default()),
        ready: Condvar::new(),
    })
}

// Direct cache accessors, kept test-only: production dispatch always goes
// through `claim_tuning`/`TuningPermit::record` so concurrent callers for the
// same key are coordinated instead of redundantly measuring in parallel.
#[cfg(test)]
pub(crate) fn cached_launch(key: &TuningKey) -> Option<TunedLaunch> {
    coordinator()
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(key)
}

#[cfg(test)]
pub(crate) fn cache_measured_launch(key: TuningKey, launch: TunedLaunch) {
    coordinator()
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, launch);
}

#[cfg(any(feature = "autotune", test))]
pub(crate) enum TuningDecision {
    Cached(TunedLaunch),
    Measure(TuningPermit),
}

#[cfg(any(feature = "autotune", test))]
#[derive(Clone, Debug)]
pub(crate) struct TuningPermit {
    key: Option<TuningKey>,
}

#[cfg(any(feature = "autotune", test))]
impl TuningPermit {
    #[cfg(all(feature = "cuda", feature = "autotune"))]
    pub(crate) fn key(&self) -> Option<&TuningKey> {
        self.key.as_ref()
    }

    pub(crate) fn record(mut self, measurements: &[CandidateMeasurement]) -> Result<TunedLaunch> {
        let winner = select_fastest(measurements)?;
        let key = self
            .key
            .take()
            .ok_or_else(|| Error::Msg("CUDA tuning permit was already completed".into()))?;
        let coordinator = coordinator();
        let mut state = coordinator
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.insert(key.clone(), winner);
        state.in_flight.remove(&key);
        drop(state);
        coordinator.ready.notify_all();
        Ok(winner)
    }
}

#[cfg(any(feature = "autotune", test))]
impl Drop for TuningPermit {
    fn drop(&mut self) {
        let Some(key) = self.key.take() else {
            return;
        };
        let coordinator = coordinator();
        let mut state = coordinator
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.in_flight.remove(&key);
        drop(state);
        coordinator.ready.notify_all();
    }
}

#[cfg(any(feature = "autotune", test))]
pub(crate) fn claim_tuning(
    key: TuningKey,
    legal_candidates: &[LaunchCandidate],
) -> Result<TuningDecision> {
    if legal_candidates.is_empty() {
        return Err(Error::Msg("cannot tune an empty candidate set".into()));
    }
    let coordinator = coordinator();
    let mut state = coordinator
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    loop {
        if let Some(cached) = state.get(&key) {
            if legal_candidates.contains(&cached.candidate) {
                return Ok(TuningDecision::Cached(cached));
            }
            state.entries.remove(&key);
        }
        if state.in_flight.insert(key.clone()) {
            return Ok(TuningDecision::Measure(TuningPermit { key: Some(key) }));
        }
        state = coordinator
            .ready
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
}

#[cfg(test)]
pub(crate) fn record_measurements(
    key: TuningKey,
    measurements: &[CandidateMeasurement],
) -> Result<TunedLaunch> {
    let winner = select_fastest(measurements)?;
    cache_measured_launch(key, winner);
    Ok(winner)
}

#[cfg(all(feature = "cuda", feature = "autotune"))]
pub(crate) fn measure_cuda_candidate<F>(
    stream: &cudarc::driver::CudaStream,
    candidate: LaunchCandidate,
    mut launch: F,
) -> Result<CandidateMeasurement>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..CUDA_TUNING_WARMUPS {
        launch()?;
    }
    stream.synchronize().map_err(|error| {
        Error::Msg(format!(
            "CUDA candidate warmup synchronization failed: {error:?}"
        ))
    })?;

    let mut synchronized_samples_ns = Vec::with_capacity(CUDA_TUNING_SAMPLES);
    for _ in 0..CUDA_TUNING_SAMPLES {
        let start = stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| Error::Msg(format!("CUDA timing start event failed: {error:?}")))?;
        launch()?;
        let end = stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| Error::Msg(format!("CUDA timing end event failed: {error:?}")))?;
        let elapsed_ms = start.elapsed_ms(&end).map_err(|error| {
            Error::Msg(format!(
                "CUDA candidate event measurement failed: {error:?}"
            ))
        })?;
        synchronized_samples_ns.push(elapsed_milliseconds_to_nanoseconds(elapsed_ms)?);
    }
    Ok(CandidateMeasurement {
        candidate,
        synchronized_samples_ns,
    })
}

#[cfg(any(feature = "autotune", test))]
fn elapsed_milliseconds_to_nanoseconds(elapsed_ms: f32) -> Result<u64> {
    let elapsed_ns = f64::from(elapsed_ms) * 1_000_000.0;
    if !elapsed_ns.is_finite() || elapsed_ns < 0.0 || elapsed_ns > u64::MAX as f64 {
        return Err(Error::Msg(format!(
            "invalid CUDA event duration {elapsed_ms} ms"
        )));
    }
    Ok(elapsed_ns.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelFamily;
    use incin_core::exec::LayoutClass;

    fn test_environment(id: &str) -> identity::TuningEnvironmentFingerprint {
        identity::TuningEnvironmentFingerprint::<incin_core::shapes::Dyn>::new_dyn(
            identity::DeviceFingerprint::new_dyn(
                incin_core::tensor::device::DeviceKind::Cuda,
                id,
                "sm_90",
                identity::SoftwareVersion::new(12, 8, 0),
            )
            .unwrap(),
            identity::CompilerFingerprint::new_dyn(
                incin_core::tensor::device::DeviceKind::Cuda,
                "nvrtc",
                identity::SoftwareVersion::new(12, 8, 0),
                "sm_90",
                &["incin-nvrtc-options-v1"],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn test_kernel() -> KernelKey {
        KernelKey::cuda(
            incin_core::shapes::error::OperationKind::Pointwise,
            KernelFamily::PointwiseUnary,
            "neg",
            DTypeId::F32,
            LayoutClass::Contiguous,
            KernelAccess::Scalar { unroll_width: 1 },
        )
        .unwrap()
    }

    #[test]
    fn candidate_generation_separates_access_and_launch_width() {
        let aligned = pointwise_candidates(DTypeId::F32.descriptor(), 4096, true, true);
        assert_eq!(aligned.len(), 9);
        assert!(aligned.iter().any(|candidate| {
            candidate.block_size == 256
                && candidate.access == KernelAccess::Packed { vector_width: 4 }
        }));
        assert_eq!(
            default_pointwise_candidate(&aligned).unwrap().access,
            KernelAccess::Packed { vector_width: 4 }
        );

        let strided = pointwise_candidates(DTypeId::F32.descriptor(), 4096, false, false);
        assert_eq!(strided.len(), 3);
        assert!(
            strided
                .iter()
                .all(|candidate| { candidate.access == KernelAccess::Scalar { unroll_width: 1 } })
        );

        let reduction = reduction_candidates(true);
        assert_eq!(reduction.len(), 4);
        assert_eq!(
            default_reduction_candidate(&reduction).unwrap(),
            LaunchCandidate {
                block_size: 256,
                access: KernelAccess::WarpReduction,
            }
        );

        let normalization_ln = normalization_candidates(true);
        assert_eq!(normalization_ln.len(), 3);
        assert_eq!(
            default_normalization_candidate(&normalization_ln).unwrap(),
            LaunchCandidate {
                block_size: 256,
                access: KernelAccess::Welford,
            }
        );

        let normalization_bn = normalization_candidates(false);
        assert_eq!(normalization_bn.len(), 3);
        assert_eq!(
            default_normalization_candidate(&normalization_bn).unwrap(),
            LaunchCandidate {
                block_size: 256,
                access: KernelAccess::Scalar { unroll_width: 1 },
            }
        );
    }

    #[test]
    fn tuning_uses_synchronized_medians_and_shape_buckets() {
        let scalar = LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 4 },
        };
        let packed = LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Packed { vector_width: 4 },
        };
        let winner = select_fastest(&[
            CandidateMeasurement {
                candidate: scalar,
                synchronized_samples_ns: vec![90, 100, 10_000],
            },
            CandidateMeasurement {
                candidate: packed,
                synchronized_samples_ns: vec![70, 80, 90],
            },
        ])
        .unwrap();
        assert_eq!(winner.candidate, packed);
        assert_eq!(winner.median_ns, 80);
        assert_eq!(WorkloadBucket::pointwise(1025, true).elements_log2, 11);
        assert_ne!(
            WorkloadBucket::pointwise(4096, true),
            WorkloadBucket::pointwise(4096, false)
        );
        assert_eq!(WorkloadBucket::reduction(17, 257).reduction_log2, 9);
        assert_eq!(elapsed_milliseconds_to_nanoseconds(0.125).unwrap(), 125_000);
        assert!(elapsed_milliseconds_to_nanoseconds(f32::NAN).is_err());
    }

    #[test]
    fn measured_results_round_trip_through_device_scoped_cache() {
        let key = TuningKey::new(
            test_environment("GPU-00000000-0000-0000-0000-000000000007"),
            &test_kernel(),
            WorkloadBucket::pointwise(4096, true),
        );
        let candidate = LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Packed { vector_width: 4 },
        };
        let launch = record_measurements(
            key.clone(),
            &[CandidateMeasurement {
                candidate,
                synchronized_samples_ns: vec![122, 123, 124],
            }],
        )
        .unwrap();
        assert_eq!(
            launch,
            TunedLaunch {
                candidate,
                median_ns: 123,
                sample_count: 3,
            }
        );
        assert_eq!(cached_launch(&key), Some(launch));
    }

    #[test]
    fn tuning_claim_is_single_flight_and_commits_a_measured_winner() {
        let key = TuningKey::new(
            test_environment("GPU-00000000-0000-0000-0000-000000000099"),
            &test_kernel(),
            WorkloadBucket::pointwise(8192, true),
        );
        let candidate = LaunchCandidate {
            block_size: 128,
            access: KernelAccess::Packed { vector_width: 4 },
        };
        let permit = match claim_tuning(key.clone(), &[candidate]).unwrap() {
            TuningDecision::Measure(permit) => permit,
            TuningDecision::Cached(_) => panic!("fresh tuning key was unexpectedly cached"),
        };
        assert!(coordinator().state.lock().unwrap().in_flight.contains(&key));
        let winner = permit
            .record(&[CandidateMeasurement {
                candidate,
                synchronized_samples_ns: vec![11, 12, 13],
            }])
            .unwrap();
        assert_eq!(winner.candidate, candidate);
        assert!(!coordinator().state.lock().unwrap().in_flight.contains(&key));
        assert!(matches!(
            claim_tuning(key, &[candidate]).unwrap(),
            TuningDecision::Cached(cached) if cached == winner
        ));
    }
}
