//! The one tolerance table for the conformance matrix (issue #83).
//!
//! Scattering `assert!((a - b).abs() < 1e-5)` across suites is how a backend
//! silently gets a looser bar than its neighbours. Every numeric comparison
//! the matrix harness makes reads its bound from here, and every row carries
//! the reason for its bound: a tolerance without a recorded reason is how one
//! operation quietly grows a looser bar than its neighbours.
//!
//! Two comparisons, two halves of the table:
//!
//! * [`for_dtype`] bounds a backend's value against the CPU oracle's for one
//!   output element. The matrix compares backends against the oracle, so the
//!   bound is keyed by dtype, not by operation.
//! * [`gradient_options`] bounds an analytic gradient against its central
//!   difference. It returns [`GradCheckOptions::for_f32`](incin_core::exec::GradCheckOptions::for_f32),
//!   the same options every gradient test in this crate is written against,
//!   rather than a second copy of the numbers.
//!
//! A row that needs a looser bound than its dtype's gets one in
//! [`EXCEPTIONS`], with the reason recorded beside it. The list is empty
//! today: emptiness is itself the claim that no operation has needed one yet,
//! and adding the first entry is a reviewable decision rather than a constant
//! quietly edited in place.
//!
//! What this table is not: `tests/conformance_values.rs` keeps its own
//! per-operation table because it compares `f32` output against an `f64`
//! scalar-loop oracle, a different comparison with different error terms
//! (libm path differences, accumulation order over 64 elements). Folding the
//! two together would make one table say two things. Consolidating them is
//! tracked as a follow-up in that file's docs.

use incin_core::exec::GradCheckOptions;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

/// How far one value may land from the reference before it fails.
///
/// A value passes if it is within *either* bound: an absolute bound alone
/// rejects large values that are correct to every bit a float has, and a
/// relative bound alone rejects values near zero, where the reference has no
/// magnitude to be relative to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Absolute error allowed before failure, in value units.
    pub absolute: f64,
    /// Relative error allowed before failure, as a fraction of magnitude.
    pub relative: f64,
    /// Why the bound is what it is. Required for every row.
    pub reason: &'static str,
}

impl Tolerance {
    /// Whether `actual` is an acceptable answer for `expected`.
    #[must_use]
    pub fn accepts(self, expected: f64, actual: f64) -> bool {
        if expected == actual {
            return true;
        }
        if !expected.is_finite() || !actual.is_finite() {
            // Matching magnitudes of non-finite values says nothing; only
            // bit-level agreement (handled by the equality above) does.
            return false;
        }
        let difference = (expected - actual).abs();
        difference <= self.absolute || difference <= self.relative * expected.abs()
    }
}

/// The bound for one dtype's values against the CPU oracle.
///
/// Keyed by dtype because the error terms are properties of the format, not
/// of the operation: the rounding of one `f32` multiply is the same fact in
/// `mul` and in `matmul`.
#[must_use]
pub const fn for_dtype(dtype: DTypeId) -> Tolerance {
    match dtype {
        // Integer and logical values are bit-exact. There is no rounding to
        // budget for: any difference means the kernel computed a value it was
        // not asked to compute.
        DTypeId::Bool | DTypeId::U8 | DTypeId::U32 | DTypeId::I64 => Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "integer and logical values are bit-exact; any difference \
                     is a wrong value, not rounding",
        },
        // One correctly rounded `f32` operation differs from the exact result
        // by at most half an ulp (2^-24 relative). The absolute half covers
        // cancellation to near zero, where the relative bound has no
        // magnitude to work with. Accumulated kernels (reductions, matmul)
        // need an exception row in EXCEPTIONS, not a looser default here.
        DTypeId::F32 => Tolerance {
            absolute: 1e-6,
            relative: 1e-5,
            reason: "f32 EPSILON is 2^-24; one correctly rounded op differs by \
                     half an ulp, and the absolute half covers cancellation \
                     to near zero",
        },
        // `f16` EPSILON is 2^-11 (~4.9e-4). One rounding step is half an ulp
        // of the magnitude, so the bound sits just above one ulp.
        DTypeId::F16 => Tolerance {
            absolute: 1e-3,
            relative: 1e-3,
            reason: "f16 EPSILON is 2^-11; a single rounding step costs half \
                     an ulp of the magnitude",
        },
        // `bf16` EPSILON is 2^-8 (~3.9e-3): eight mantissa bits. Same shape
        // as the f16 row, one decimal order looser.
        DTypeId::BF16 => Tolerance {
            absolute: 1e-2,
            relative: 1e-2,
            reason: "bf16 EPSILON is 2^-8; a single rounding step costs half \
                     an ulp of the magnitude",
        },
        // `f8e4m3` EPSILON is 2^-3 (0.125): three mantissa bits. Same
        // shape as the f16/bf16 rows, at twice the epsilon: one rounding
        // step costs half an ulp of the magnitude.
        DTypeId::F8E4M3 => Tolerance {
            absolute: 0.25,
            relative: 0.25,
            reason: "f8e4m3 EPSILON is 2^-3; a single rounding step costs \
                     half an ulp of the magnitude",
        },
        // `f8e5m2` EPSILON is 2^-2 (0.25): two mantissa bits. Same shape
        // as the e4m3 row, one binade looser.
        DTypeId::F8E5M2 => Tolerance {
            absolute: 0.5,
            relative: 0.5,
            reason: "f8e5m2 EPSILON is 2^-2; a single rounding step costs \
                     half an ulp of the magnitude",
        },
        // The `f64` oracle path is near-exact: both sides compute in double
        // precision, so the bound only absorbs accumulation-order differences
        // across a handful of elements.
        DTypeId::F64 => Tolerance {
            absolute: 1e-12,
            relative: 1e-9,
            reason: "both sides compute in f64; only accumulation order over \
                     a few elements can differ",
        },
        // Thirty-two logical values share one `f16` scale per block, so the
        // quantization step is `scale / 127` and the worst rounding of one
        // value is half a step. Fixture magnitudes stay below 8, where the
        // scale is at most ~0.06 and half a step lands near 2.5e-4; the bound
        // is set two orders above that to also absorb the f16 scale rounding
        // itself. A fixture with larger magnitudes needs an exception row.
        DTypeId::Q8_0 => Tolerance {
            absolute: 5e-2,
            relative: 1e-2,
            reason: "Q8_0 shares one f16 scale across 32 values; one value \
                     rounds by half a step of scale/127, plus the scale's own \
                     f16 rounding, for fixture magnitudes below 8",
        },
        // NVFP4 shares one E4M3 scale across 16 E2M1 values: the widest
        // half step is 1.0 (the 4→6 binade) times the block scale
        // (≈ block_amax/6), plus the E4M3 scale's own ≤1/16 relative
        // rounding — under a quarter of the block's own amax (issue #95).
        DTypeId::NVFP4 => Tolerance {
            absolute: 0.25,
            relative: 0.25,
            reason: "NVFP4 quantizes to a 1-mantissa-bit grid under one E4M3 \
                     scale per 16 values; the worst half step is 1.0 times \
                     the block scale, under a quarter of the block amax",
        },
        // MXFP4 shares one exact power-of-two E8M0 scale across 32 E2M1
        // values: same grid, no scale-rounding term, scale at most twice
        // block_amax/6 — under half the block's own amax (issue #95).
        DTypeId::MXFP4 => Tolerance {
            absolute: 0.5,
            relative: 0.5,
            reason: "MXFP4 quantizes to a 1-mantissa-bit grid under one exact \
                     power-of-two scale per 32 values; the worst half step \
                     is 1.0 times the block scale, under half the block amax",
        },
        // Custom dtypes are fail-closed exact. An unknown format has no error
        // terms this table can budget for, so any deviation fails loudly
        // rather than passing under a guessed bound; the fix is a real row
        // with a measured reason, not a wider default.
        _ => Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "unknown (custom) dtype: no error terms are known, so any \
                     deviation fails rather than passing under a guess",
        },
    }
}

/// The bound an analytic gradient is held to against its central difference.
///
/// Not a second copy of the numbers: these are
/// [`GradCheckOptions::for_f32`](incin_core::exec::GradCheckOptions::for_f32),
/// the same options every gradient test in this crate is written against.
/// The step minimizes total error at `f32` precision (central-difference
/// rounding grows as `1/step`, truncation as `step^2`, meeting near
/// `(6 * f32::EPSILON).cbrt()`); the ceiling clears the measured noise floor
/// by 10x while catching gradient errors an order of magnitude smaller.
#[must_use]
pub const fn gradient_options() -> GradCheckOptions {
    GradCheckOptions::for_f32()
}

/// Per-operation overrides to [`for_dtype`], each with its recorded reason.
///
/// Empty today: no operation has needed a looser bound than its dtype's yet.
/// The shape of an entry is `(operation, tolerance, why)`, and a reviewer
/// reading a new one should ask for the measurement that set the numbers.
pub const EXCEPTIONS: &[(OperationKind, Tolerance, &str)] = &[];

/// The bound for one operation's values: its [`EXCEPTIONS`] row, or its
/// dtype's row when it has none.
#[must_use]
pub fn for_operation(operation: OperationKind, dtype: DTypeId) -> Tolerance {
    EXCEPTIONS
        .iter()
        .find(|(candidate, _, _)| *candidate == operation)
        .map(|(_, tolerance, _)| *tolerance)
        .unwrap_or_else(|| for_dtype(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dtype the registry can name has a row: a new dtype without one
    /// would silently fall through to whatever default a caller invented.
    #[test]
    fn every_dtype_has_a_row_with_a_reason() {
        for dtype in [
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::I64,
            DTypeId::BF16,
            DTypeId::F16,
            DTypeId::F8E4M3,
            DTypeId::F8E5M2,
            DTypeId::F32,
            DTypeId::F64,
            DTypeId::Q8_0,
            DTypeId::NVFP4,
            DTypeId::MXFP4,
        ] {
            let tolerance = for_dtype(dtype);
            assert!(
                !tolerance.reason.is_empty(),
                "{dtype:?} has a tolerance with no recorded reason"
            );
            assert!(
                tolerance.absolute >= 0.0 && tolerance.relative >= 0.0,
                "{dtype:?} has a negative tolerance bound"
            );
        }
    }

    /// Exact means exact, and non-finite values only match bit-identically.
    #[test]
    fn exact_rows_accept_only_identical_values() {
        let exact = for_dtype(DTypeId::I64);
        assert!(exact.accepts(3.0, 3.0));
        assert!(!exact.accepts(3.0, 4.0));
        assert!(!for_dtype(DTypeId::F32).accepts(1.0, f64::NAN));
        assert!(!for_dtype(DTypeId::F32).accepts(1.0, f64::INFINITY));
    }

    /// Exception rows are unique and reasoned: two rows for one operation
    /// would let the second silently shadow the first.
    #[test]
    fn exceptions_are_unique_and_reasoned() {
        let mut operations: Vec<OperationKind> = EXCEPTIONS
            .iter()
            .map(|(operation, _, _)| *operation)
            .collect();
        operations.sort_unstable();
        let before = operations.len();
        operations.dedup();
        assert_eq!(
            before,
            operations.len(),
            "EXCEPTIONS lists an operation more than once"
        );
        for (operation, tolerance, reason) in EXCEPTIONS {
            assert!(
                !reason.is_empty(),
                "{operation} has an exception with no reason"
            );
            assert!(
                tolerance.absolute >= 0.0 && tolerance.relative >= 0.0,
                "{operation} has a negative exception bound"
            );
        }
    }
}
