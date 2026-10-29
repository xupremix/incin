//! Vendor-vs-native GEMM path selection by measurement (issue #85, item 10).
//!
//! The orchestrators in `cuda/ops/matmul.rs` serve every epilogue-free GEMM
//! request twice over: once through cuBLASLt (`cuda/ops/cublaslt.rs`, only
//! under `cuda-vendor`) and once through the NVRTC kernels in
//! `kernels/matmul.cu`. Until this module, the order was fixed - vendor
//! first, native on refusal or failure. That order is a guess: on small
//! shapes the vendor call's descriptor and heuristic overhead can exceed the
//! kernel launch it replaces, while on large shapes the tensor-core route
//! wins by an order of magnitude. This module replaces the guess with a
//! measurement.
//!
//! # The pieces
//!
//! - [`select_gemm_path`] is the whole policy, pure and host-side: given the
//!   workload's FLOP count, whether the vendor fit policy admitted it, the
//!   caller's [`GemmPreference`], and the learned [`GemmCrossover`], it names
//!   the [`GemmPath`] and the [`GemmSelectionSource`] that explains why. No
//!   device required, so unit tests pin every arm.
//! - [`GemmCrossover`] is the learned threshold: `vendor_min_flops` is the
//!   smallest FLOP count at which the vendor path measured faster, or `None`
//!   while nothing has been measured yet ([`GemmCrossover::unknown`]). An
//!   unknown crossover selects native - the path that is always available -
//!   which is what makes the first encounter of a shape safe.
//! - Under `cuda-vendor` + `autotune`, the orchestrators measure each
//!   workload bucket once ([`WorkloadBucket::gemm`](crate::tuning::WorkloadBucket::gemm))
//!   with the shared [`measure_cuda_candidate`](crate::tuning::measure_cuda_candidate)
//!   event timing, refine the cached crossover with
//!   [`refine_gemm_crossover`], and serve the winner's own output buffer, so
//!   measurement never costs a second launch of the winning path.
//!
//! # What selection does not touch
//!
//! - The static fallback. Without `cuda-vendor` there is no vendor module to
//!   call; without `autotune` there is no measurement. Both builds keep
//!   today's vendor-first order exactly, expressed as a selection with a zero
//!   threshold (see the orchestrators): everything admitted runs vendor
//!   first, everything else runs native. The policy below proves that order
//!   is the degenerate case, not a second implementation.
//! - Epilogue requests. `launch_matmul` and `try_launch_batched_matmul` take
//!   no bias or activation parameters - every call they make is
//!   epilogue-free - so selection only ever chooses between two
//!   implementations of the same product. Fused bias/activation callers
//!   reach `cublaslt::try_launch_matmul` directly (see `cuda/executor.rs`'s
//!   fused path) and never pass through selection, because a fused epilogue
//!   has no native equivalent to select.
//! - Failures. A vendor attempt that refuses or errors falls through to the
//!   native kernels, exactly as before; a measurement whose vendor side
//!   errors counts as a native win. Selection only ever changes which
//!   implementation runs, never whether an answer comes back.

#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
use crate::tuning::{CandidateMeasurement, LaunchCandidate};
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
use incin_core::error::Result;

/// A GEMM workload size in floating-point operations: `2 * m * n * k *
/// batch` for a `[batch, m, k] @ [batch, k, n]` product (batch 1 when
/// unbatched), saturating at [`u64::MAX`]. See [`gemm_flops`].
pub(crate) type GemmFlops = u64;

/// Counts one GEMM's floating-point operations: every output element
/// (`m * n * batch`) performs `k` multiply-adds, two FLOPs each. Saturates
/// instead of wrapping - a threshold comparison against a wrapped count
/// could route a huge workload to the wrong path.
pub(crate) fn gemm_flops(m: usize, n: usize, k: usize, batch: Option<usize>) -> GemmFlops {
    let total = (batch.unwrap_or(1) as u128)
        .saturating_mul(m as u128)
        .saturating_mul(n as u128)
        .saturating_mul(k as u128)
        .saturating_mul(2);
    total.min(u64::MAX as u128) as GemmFlops
}

/// Which implementation computes an epilogue-free GEMM request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmPath {
    /// The cuBLASLt route in `cuda/ops/cublaslt.rs` (`cuda-vendor` only).
    Vendor,
    /// The NVRTC kernels in `kernels/matmul.cu`.
    Native,
}

impl GemmPath {
    /// The stable telemetry spelling of this path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            GemmPath::Vendor => "vendor",
            GemmPath::Native => "native",
        }
    }
}

/// Why [`select_gemm_path`] named the path it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmSelectionSource {
    /// The caller forced the path; no policy or measurement was consulted.
    Forced,
    /// The vendor fit policy refused the request, so native serves it.
    VendorUnavailable,
    /// The workload is below the learned vendor crossover: native measured
    /// faster (or nothing has been measured yet, and native is the path that
    /// is always available).
    BelowCrossover,
    /// The workload is at or above the learned vendor crossover: vendor
    /// measured faster.
    AboveCrossover,
}

impl GemmSelectionSource {
    /// The stable telemetry spelling of this source.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            GemmSelectionSource::Forced => "forced",
            GemmSelectionSource::VendorUnavailable => "vendor-unavailable",
            GemmSelectionSource::BelowCrossover => "below-crossover",
            GemmSelectionSource::AboveCrossover => "above-crossover",
        }
    }
}

/// One path decision: what runs, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmSelection {
    /// The implementation that serves the request.
    pub(crate) path: GemmPath,
    /// Why that implementation was named.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) source: GemmSelectionSource,
}

/// A caller's override for [`select_gemm_path`]. Production dispatch always
/// passes [`GemmPreference::Auto`]; the forced variants exist so hardware
/// tests can pin each path independently and compare them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GemmPreference {
    /// Consult the vendor fit policy and the learned crossover.
    #[default]
    Auto,
    /// Attempt the vendor path, falling back to native on refusal or
    /// failure. Forcing selects the attempt order, not a hard requirement:
    /// the advisory invariant (selection never changes whether an answer
    /// comes back) holds for forced calls too.
    ///
    /// Only tests construct the forced variants today (production passes
    /// `Auto`); the orchestrators match on them in every build.
    #[cfg_attr(not(test), allow(dead_code))]
    ForceVendor,
    /// Run the native kernels without attempting the vendor path.
    ///
    /// Only tests construct the forced variants today (production passes
    /// `Auto`); the orchestrators match on them in every build.
    #[cfg_attr(not(test), allow(dead_code))]
    ForceNative,
}

/// The learned vendor-vs-native crossover for one
/// [`gemm_problem_id`](crate::tuning::gemm_problem_id): the smallest FLOP
/// count at which the vendor path measured faster than the native kernels.
/// `None` ([`GemmCrossover::unknown`]) means nothing has been measured yet
/// and selects native everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct GemmCrossover {
    /// Minimum FLOPs at which vendor measured faster, or `None` if unknown.
    pub(crate) vendor_min_flops: Option<u64>,
}

impl GemmCrossover {
    /// No measurement yet: selects native for every admitted workload.
    ///
    /// Only measurement builds and tests consult the unknown crossover
    /// today (the static fallback carries an explicit zero threshold); the
    /// orchestrators that will read it are the autotuned ones.
    #[cfg_attr(
        not(any(test, all(feature = "cuda-vendor", feature = "autotune"))),
        allow(dead_code)
    )]
    pub(crate) fn unknown() -> Self {
        Self {
            vendor_min_flops: None,
        }
    }

    /// A measured threshold: `flops` is the smallest FLOP count at which the
    /// vendor path measured faster.
    pub(crate) fn measured(vendor_min_flops: u64) -> Self {
        Self {
            vendor_min_flops: Some(vendor_min_flops),
        }
    }
}

/// The pure vendor-vs-native policy: names the path and the reason.
///
/// - A forced preference wins over everything, even a vendor refusal: the
///   launcher still falls back on refusal or failure, so forcing is an
///   attempt order, not a way to lose an answer.
/// - An `Auto` request the vendor fit policy refused runs native; the
///   refusal is the policy's own verdict, not a benchmark.
/// - Otherwise the FLOP count is compared against the crossover. An unknown
///   crossover behaves as an infinite threshold, so unmeasured workloads run
///   native - the path that needs no feature and no heuristic to serve.
pub(crate) fn select_gemm_path(
    flops: GemmFlops,
    vendor_admitted: bool,
    preference: GemmPreference,
    crossover: &GemmCrossover,
) -> GemmSelection {
    match preference {
        GemmPreference::ForceVendor => GemmSelection {
            path: GemmPath::Vendor,
            source: GemmSelectionSource::Forced,
        },
        GemmPreference::ForceNative => GemmSelection {
            path: GemmPath::Native,
            source: GemmSelectionSource::Forced,
        },
        GemmPreference::Auto if !vendor_admitted => GemmSelection {
            path: GemmPath::Native,
            source: GemmSelectionSource::VendorUnavailable,
        },
        GemmPreference::Auto => {
            let threshold = crossover.vendor_min_flops.unwrap_or(u64::MAX);
            if flops >= threshold {
                GemmSelection {
                    path: GemmPath::Vendor,
                    source: GemmSelectionSource::AboveCrossover,
                }
            } else {
                GemmSelection {
                    path: GemmPath::Native,
                    source: GemmSelectionSource::BelowCrossover,
                }
            }
        }
    }
}

/// The autotuned half of selection: per-problem crossover records plus the
/// set of workload buckets already measured, so each bucket pays the
/// two-path benchmark once per process. Keyed by `(problem id, device)` -
/// the crossover depends on the arithmetic and the call shape (see
/// [`gemm_problem_id`](crate::tuning::gemm_problem_id)), and different
/// devices measure differently.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
#[derive(Default)]
struct GemmAutotuneState {
    crossovers: std::collections::BTreeMap<(alloc::string::String, usize), GemmCrossover>,
    measured:
        std::collections::BTreeSet<(alloc::string::String, usize, crate::tuning::WorkloadBucket)>,
}

#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
fn gemm_autotune_state() -> &'static std::sync::Mutex<GemmAutotuneState> {
    static STATE: std::sync::OnceLock<std::sync::Mutex<GemmAutotuneState>> =
        std::sync::OnceLock::new();
    STATE.get_or_init(|| std::sync::Mutex::new(GemmAutotuneState::default()))
}

/// The cached crossover for one problem on one device, or
/// [`GemmCrossover::unknown`] when nothing has been measured yet.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn gemm_crossover_for(problem: &str, device_id: usize) -> GemmCrossover {
    gemm_autotune_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .crossovers
        .get(&(problem.into(), device_id))
        .copied()
        .unwrap_or_else(GemmCrossover::unknown)
}

/// Whether this workload bucket has already paid the two-path benchmark.
/// Buckets are log2 bins (see
/// [`WorkloadBucket::gemm`](crate::tuning::WorkloadBucket::gemm)), so one
/// measurement covers the shapes whose vendor/native tradeoff measures the
/// same.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn gemm_bucket_measured(
    problem: &str,
    device_id: usize,
    bucket: &crate::tuning::WorkloadBucket,
) -> bool {
    gemm_autotune_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .measured
        .contains(&(problem.into(), device_id, *bucket))
}

/// Records a finished benchmark: stores the refined crossover and marks the
/// bucket measured, so later requests for the bucket consult the cache
/// instead of re-running both paths.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn note_gemm_measurement(
    problem: &str,
    device_id: usize,
    bucket: crate::tuning::WorkloadBucket,
    crossover: GemmCrossover,
) {
    let mut state = gemm_autotune_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state
        .crossovers
        .insert((problem.into(), device_id), crossover);
    state.measured.insert((problem.into(), device_id, bucket));
}

/// Moves the crossover bracket after one benchmark at `flops`:
/// - vendor won: the threshold drops to `flops` (or stays lower), so this
///   workload and everything larger runs vendor;
/// - native won (or the vendor side errored, which counts as a native win):
///   the threshold rises past `flops` (or stays higher), so this workload
///   and everything smaller runs native.
///
/// The update is a conservative bracket, not a proof: contradictory later
/// measurements move it again, and the losing side always keeps its fallback
/// in the orchestrator, so a wrong threshold costs time, never correctness.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn refine_gemm_crossover(
    current: &GemmCrossover,
    flops: GemmFlops,
    vendor_won: bool,
) -> GemmCrossover {
    if vendor_won {
        GemmCrossover::measured(flops.min(current.vendor_min_flops.unwrap_or(u64::MAX)))
    } else {
        GemmCrossover::measured(
            current
                .vendor_min_flops
                .map_or(flops.saturating_add(1), |threshold| {
                    threshold.max(flops.saturating_add(1))
                }),
        )
    }
}

/// Probe tags that tell the two benchmark runs apart when
/// [`select_fastest`](crate::tuning::select_fastest) picks the winner.
/// These are labels, not launch configurations: the measured closures run
/// the real vendor and native launchers, and only the median comparison
/// reads these tags back.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) const VENDOR_PROBE_CANDIDATE: LaunchCandidate = LaunchCandidate {
    block_size: 0,
    access: crate::kernel::KernelAccess::Scalar { unroll_width: 1 },
};

/// See [`VENDOR_PROBE_CANDIDATE`].
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) const NATIVE_PROBE_CANDIDATE: LaunchCandidate = LaunchCandidate {
    block_size: 1,
    access: crate::kernel::KernelAccess::Scalar { unroll_width: 1 },
};

/// Times one GEMM path with the shared autotune event methodology (warmups
/// plus synchronized samples - see
/// [`measure_cuda_candidate`](crate::tuning::measure_cuda_candidate)). The
/// probe tag names the returned measurement so the caller can tell the two
/// sides apart without comparing buffers.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn time_gemm_path(
    stream: &cudarc::driver::CudaStream,
    probe: LaunchCandidate,
    launch: impl FnMut() -> Result<()>,
) -> Result<CandidateMeasurement> {
    crate::tuning::measure_cuda_candidate(stream, probe, launch)
}

/// Compares the two benchmark runs by synchronized median latency: `true`
/// when the vendor side measured strictly faster. Ties go to native - the
/// path with no descriptor or heuristic overhead to amortize.
#[cfg(all(feature = "cuda-vendor", feature = "autotune"))]
pub(crate) fn vendor_won_measurement(
    vendor: &CandidateMeasurement,
    native: &CandidateMeasurement,
) -> Result<bool> {
    let winner = crate::tuning::select_fastest(&[vendor.clone(), native.clone()])?;
    Ok(winner.candidate == VENDOR_PROBE_CANDIDATE)
}

/// Pure selection, unit-testable with no device: every arm of
/// [`select_gemm_path`], the telemetry spellings, the crossover
/// constructors, and the FLOP counter.
#[cfg(all(test, feature = "cuda"))]
mod selection_tests {
    use super::*;

    #[test]
    fn forced_preferences_win_over_admission_and_crossover() {
        let crossover = GemmCrossover::measured(1_000);
        assert_eq!(
            select_gemm_path(10, false, GemmPreference::ForceVendor, &crossover),
            GemmSelection {
                path: GemmPath::Vendor,
                source: GemmSelectionSource::Forced,
            }
        );
        assert_eq!(
            select_gemm_path(10_000_000, true, GemmPreference::ForceNative, &crossover),
            GemmSelection {
                path: GemmPath::Native,
                source: GemmSelectionSource::Forced,
            }
        );
    }

    #[test]
    fn refused_vendor_requests_run_native() {
        let crossover = GemmCrossover::measured(0);
        assert_eq!(
            select_gemm_path(u64::MAX, false, GemmPreference::Auto, &crossover),
            GemmSelection {
                path: GemmPath::Native,
                source: GemmSelectionSource::VendorUnavailable,
            }
        );
    }

    #[test]
    fn crossover_splits_below_and_above() {
        let crossover = GemmCrossover::measured(1_000);
        assert_eq!(
            select_gemm_path(999, true, GemmPreference::Auto, &crossover),
            GemmSelection {
                path: GemmPath::Native,
                source: GemmSelectionSource::BelowCrossover,
            }
        );
        assert_eq!(
            select_gemm_path(1_000, true, GemmPreference::Auto, &crossover),
            GemmSelection {
                path: GemmPath::Vendor,
                source: GemmSelectionSource::AboveCrossover,
            }
        );
        assert_eq!(
            select_gemm_path(1_001, true, GemmPreference::Auto, &crossover),
            GemmSelection {
                path: GemmPath::Vendor,
                source: GemmSelectionSource::AboveCrossover,
            }
        );
    }

    #[test]
    fn unknown_crossover_runs_native() {
        assert_eq!(
            select_gemm_path(
                u64::MAX - 1,
                true,
                GemmPreference::Auto,
                &GemmCrossover::unknown()
            ),
            GemmSelection {
                path: GemmPath::Native,
                source: GemmSelectionSource::BelowCrossover,
            }
        );
        assert_eq!(GemmCrossover::default(), GemmCrossover::unknown());
        assert_eq!(GemmPreference::default(), GemmPreference::Auto);
    }

    #[test]
    fn telemetry_spellings_are_stable() {
        assert_eq!(GemmPath::Vendor.as_str(), "vendor");
        assert_eq!(GemmPath::Native.as_str(), "native");
        assert_eq!(GemmSelectionSource::Forced.as_str(), "forced");
        assert_eq!(
            GemmSelectionSource::VendorUnavailable.as_str(),
            "vendor-unavailable"
        );
        assert_eq!(
            GemmSelectionSource::BelowCrossover.as_str(),
            "below-crossover"
        );
        assert_eq!(
            GemmSelectionSource::AboveCrossover.as_str(),
            "above-crossover"
        );
    }

    #[test]
    fn flops_count_multiply_adds_and_saturate() {
        assert_eq!(gemm_flops(2, 2, 3, None), 24);
        assert_eq!(gemm_flops(2, 2, 3, Some(4)), 96);
        assert_eq!(
            gemm_flops(usize::MAX, usize::MAX, usize::MAX, None),
            u64::MAX
        );
    }
}

/// The crossover bracket update: vendor wins lower the threshold, native
/// wins raise it past the measured point, and contradictory measurements
/// keep the more conservative bound.
#[cfg(all(test, feature = "cuda-vendor", feature = "autotune"))]
mod crossover_tests {
    use super::{GemmCrossover, refine_gemm_crossover};

    #[test]
    fn vendor_win_lowers_an_unknown_threshold_to_the_measured_point() {
        assert_eq!(
            refine_gemm_crossover(&GemmCrossover::unknown(), 500, true),
            GemmCrossover::measured(500)
        );
    }

    #[test]
    fn native_win_raises_an_unknown_threshold_past_the_measured_point() {
        assert_eq!(
            refine_gemm_crossover(&GemmCrossover::unknown(), 500, false),
            GemmCrossover::measured(501)
        );
    }

    #[test]
    fn later_measurements_keep_the_conservative_bound() {
        let crossover = GemmCrossover::measured(1_000);
        assert_eq!(
            refine_gemm_crossover(&crossover, 500, true),
            GemmCrossover::measured(500)
        );
        assert_eq!(
            refine_gemm_crossover(&crossover, 2_000, true),
            GemmCrossover::measured(1_000)
        );
        assert_eq!(
            refine_gemm_crossover(&crossover, 500, false),
            GemmCrossover::measured(1_000)
        );
        assert_eq!(
            refine_gemm_crossover(&crossover, 2_000, false),
            GemmCrossover::measured(2_001)
        );
    }
}
