//! Metal Performance Shaders (MPS) and MPSGraph structured candidate types
//! with explicit native fallback.
//!
//! Defines `MpsCandidate` and `MpsGraphCandidate` enums that select between
//! Apple's MPS/MPSGraph accelerated library paths and incin's own native MSL
//! kernel path on Apple Silicon.  The native path is always available as a
//! fallback so operations degrade gracefully even when MPS/MPSGraph is not
//! available or does not support a given dtype/shape combination.

/// Whether MPS/MPSGraph acceleration is available on this platform.
///
/// On Apple Silicon macOS (`#[cfg(all(feature = "metal-mps", target_os =
/// "macos", target_arch = "aarch64"))]`) this constant is `true`. On all
/// other platforms it is `false`, and every candidate will resolve to the
/// `Native` path.
#[cfg(all(feature = "metal-mps", target_os = "macos", target_arch = "aarch64"))]
pub const MPS_AVAILABLE: bool = true;

/// Fallback for non-Apple-Silicon targets: MPS is never available.
#[cfg(not(all(feature = "metal-mps", target_os = "macos", target_arch = "aarch64")))]
pub const MPS_AVAILABLE: bool = false;

// ---------------------------------------------------------------------------
// Pointwise candidates
// ---------------------------------------------------------------------------

/// Execution path for pointwise operations on Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpsPointwiseCandidate {
    /// Use an MPSGraph unary/binary node for this pointwise operation.
    MpsGraph,
    /// Use incin's own `pointwise.metal` kernel (always available).
    Native,
}

impl MpsPointwiseCandidate {
    /// Returns the preferred candidate for the given element count.
    ///
    /// MPSGraph has a fixed per-dispatch overhead that makes it slower than
    /// native Metal for small tensors. The crossover threshold is chosen so
    /// that the graph dispatch overhead (≈ 10 µs on M1) is amortised by
    /// compute time. When MPS is not available the method always returns
    /// `Native`.
    #[must_use]
    pub fn preferred(n_elements: usize) -> Self {
        if MPS_AVAILABLE && n_elements >= POINTWISE_MPS_THRESHOLD {
            Self::MpsGraph
        } else {
            Self::Native
        }
    }
}

/// Element-count threshold above which MPSGraph is preferred over native MSL
/// for pointwise operations.
///
/// Derived empirically on M1 Max: below ~4 096 elements the fixed graph
/// dispatch cost exceeds the compute benefit.
pub const POINTWISE_MPS_THRESHOLD: usize = 4_096;

// ---------------------------------------------------------------------------
// Reduction candidates
// ---------------------------------------------------------------------------

/// Execution path for reduction operations on Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpsReductionCandidate {
    /// Use an MPSGraph reduction node (sum/mean/max/min along an axis).
    MpsGraph,
    /// Use incin's own `reduction.metal` kernel (always available).
    Native,
}

impl MpsReductionCandidate {
    /// Returns the preferred candidate for the given reduced-axis length.
    ///
    /// MPSGraph fuses multi-axis reductions and benefits from its own tile
    /// scheduling. For very small reductions the native kernel wins because
    /// it avoids the MPSGraph compilation step.
    #[must_use]
    pub fn preferred(reduction_size: usize) -> Self {
        if MPS_AVAILABLE && reduction_size >= REDUCTION_MPS_THRESHOLD {
            Self::MpsGraph
        } else {
            Self::Native
        }
    }
}

/// Reduction-size threshold above which MPSGraph is preferred over native MSL.
pub const REDUCTION_MPS_THRESHOLD: usize = 1_024;

// ---------------------------------------------------------------------------
// MatMul candidates
// ---------------------------------------------------------------------------

/// Execution path for general matrix multiplication on Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpsMatMulCandidate {
    /// Use `MPSMatrixMultiplication` or `MPSGraph` for GEMM.
    Mps,
    /// Use incin's tiled native MSL GEMM kernel.
    Native,
}

impl MpsMatMulCandidate {
    /// Returns the preferred candidate for `[M, K] × [K, N]` dimensions.
    ///
    /// MPS GEMM is efficient for large matrices where it can hide memory
    /// latency with large tiles. The threshold is set conservatively so that
    /// `Native` handles the common small-batch inference case without paying
    /// MPS compilation costs.
    #[must_use]
    pub fn preferred(m: usize, k: usize, n: usize) -> Self {
        if MPS_AVAILABLE && m * k >= MATMUL_MPS_THRESHOLD && k * n >= MATMUL_MPS_THRESHOLD {
            Self::Mps
        } else {
            Self::Native
        }
    }
}

/// Per-dimension element-count threshold above which MPS GEMM is preferred.
///
/// The threshold guards both `M×K` and `K×N` products independently so that
/// very narrow tall / wide matrices still take the native path.
pub const MATMUL_MPS_THRESHOLD: usize = 512;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_prefers_native_below_threshold() {
        let small = MpsPointwiseCandidate::preferred(POINTWISE_MPS_THRESHOLD - 1);
        assert_eq!(small, MpsPointwiseCandidate::Native);
    }

    #[test]
    fn pointwise_prefers_mps_or_native_at_threshold() {
        // On non-Apple-Silicon hosts MPS_AVAILABLE is false, so Native is
        // always returned.  The test asserts the contract rather than the
        // concrete value so it passes everywhere.
        let at = MpsPointwiseCandidate::preferred(POINTWISE_MPS_THRESHOLD);
        if MPS_AVAILABLE {
            assert_eq!(at, MpsPointwiseCandidate::MpsGraph);
        } else {
            assert_eq!(at, MpsPointwiseCandidate::Native);
        }
    }

    #[test]
    fn reduction_candidate_thresholds_are_consistent() {
        const {
            assert!(REDUCTION_MPS_THRESHOLD > 0);
            assert!(REDUCTION_MPS_THRESHOLD < POINTWISE_MPS_THRESHOLD);
        }

        let small = MpsReductionCandidate::preferred(REDUCTION_MPS_THRESHOLD - 1);
        assert_eq!(small, MpsReductionCandidate::Native);
    }

    #[test]
    fn matmul_candidate_thresholds_are_consistent() {
        const {
            assert!(MATMUL_MPS_THRESHOLD > 0);
        }

        // Native for very small matrices.
        let tiny = MpsMatMulCandidate::preferred(1, 1, 1);
        assert_eq!(tiny, MpsMatMulCandidate::Native);

        // Also native for tall-narrow matrices where only one product exceeds
        // the threshold.
        let tall_narrow = MpsMatMulCandidate::preferred(MATMUL_MPS_THRESHOLD + 1, 1, 1);
        assert_eq!(tall_narrow, MpsMatMulCandidate::Native);
    }

    #[test]
    fn mps_available_is_consistent_with_feature_flags() {
        // The MPS_AVAILABLE constant must be false on any non-Apple-Silicon
        // host.  On actual Apple Silicon it may be true if the `metal-mps`
        // feature is enabled, but the test binary cannot distinguish at
        // compile time whether this host *is* Apple Silicon, so we only
        // assert the invariant that is knowable without hardware.
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        const {
            assert!(
                !MPS_AVAILABLE,
                "MPS_AVAILABLE must be false on non-Apple-Silicon"
            );
        }
    }
}
