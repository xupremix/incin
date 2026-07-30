//! Integration tests for the `simd_lanes::<T>()` compile-time constant.
//!
//! PRF-003: `simd_lanes<T>()` compile-time lane-width constant.
//!
//! Checks that:
//!   1. every dtype returns a lane count that is positive and divides 64,
//!   2. every result is a power of two (required for SIMD chunk alignment),
//!   3. the relative ratio between lane counts is internally consistent
//!      regardless of the active vector width,
//!   4. the constant can be used in a `const` context (true `const fn`), and
//!   5. the f32/f64 lane ratio equals 2:1 whenever any vectorization is active.

use half::{bf16, f16};
use incin_backends::simd_lanes;

// ---------------------------------------------------------------------------
// 1. Lane counts are positive and divide 64 (the maximum register width).
// ---------------------------------------------------------------------------

#[test]
fn simd_lanes_positive_and_divide_64() {
    for lanes in [
        simd_lanes::<u8>(),
        simd_lanes::<i8>(),
        simd_lanes::<u16>(),
        simd_lanes::<i16>(),
        simd_lanes::<u32>(),
        simd_lanes::<i32>(),
        simd_lanes::<u64>(),
        simd_lanes::<i64>(),
        simd_lanes::<f32>(),
        simd_lanes::<f64>(),
        simd_lanes::<f16>(),
        simd_lanes::<bf16>(),
    ] {
        assert!(lanes >= 1, "simd_lanes must be at least 1, got {lanes}");
        assert_eq!(
            64 % lanes,
            0,
            "simd_lanes must divide 64 (max register width in bytes), got {lanes}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Lane counts are powers of two (SIMD chunk alignment requirement).
// ---------------------------------------------------------------------------

#[test]
fn simd_lanes_power_of_two() {
    assert!(
        simd_lanes::<f32>().is_power_of_two(),
        "f32 lanes must be a power of two, got {}",
        simd_lanes::<f32>()
    );
    assert!(
        simd_lanes::<f64>().is_power_of_two(),
        "f64 lanes must be a power of two, got {}",
        simd_lanes::<f64>()
    );
    assert!(
        simd_lanes::<f16>().is_power_of_two(),
        "f16 lanes must be a power of two, got {}",
        simd_lanes::<f16>()
    );
    assert!(
        simd_lanes::<bf16>().is_power_of_two(),
        "bf16 lanes must be a power of two, got {}",
        simd_lanes::<bf16>()
    );
    assert!(
        simd_lanes::<u8>().is_power_of_two(),
        "u8 lanes must be a power of two, got {}",
        simd_lanes::<u8>()
    );
    assert!(
        simd_lanes::<i8>().is_power_of_two(),
        "i8 lanes must be a power of two, got {}",
        simd_lanes::<i8>()
    );
    assert!(
        simd_lanes::<u32>().is_power_of_two(),
        "u32 lanes must be a power of two, got {}",
        simd_lanes::<u32>()
    );
    assert!(
        simd_lanes::<i32>().is_power_of_two(),
        "i32 lanes must be a power of two, got {}",
        simd_lanes::<i32>()
    );
}

// ---------------------------------------------------------------------------
// 3. Relative ratios are internally consistent within the active vector width.
//    When the scalar fallback is active (lanes == 1 for every type) the ratios
//    all collapse to 1:1 — that is correct and expected. When vectorization is
//    active the wider the element the fewer lanes fit in one register.
// ---------------------------------------------------------------------------

#[test]
fn simd_lanes_relative_ratios_match_type_sizes() {
    let f32_lanes = simd_lanes::<f32>();
    let f64_lanes = simd_lanes::<f64>();
    let f16_lanes = simd_lanes::<f16>();
    let bf16_lanes = simd_lanes::<bf16>();
    let u8_lanes = simd_lanes::<u8>();
    let i8_lanes = simd_lanes::<i8>();
    let u32_lanes = simd_lanes::<u32>();
    let i32_lanes = simd_lanes::<i32>();
    let u64_lanes = simd_lanes::<u64>();
    let i64_lanes = simd_lanes::<i64>();

    if f32_lanes > 1 {
        // When vectorisation is active, ratios must match type-size ratios.
        assert_eq!(
            f32_lanes,
            f64_lanes * 2,
            "f32 should have twice as many lanes as f64 (f32={f32_lanes}, f64={f64_lanes})"
        );
        assert_eq!(
            f16_lanes,
            f32_lanes * 2,
            "f16 should have twice as many lanes as f32 (f16={f16_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            bf16_lanes,
            f32_lanes * 2,
            "bf16 should have twice as many lanes as f32 (bf16={bf16_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            u8_lanes,
            f32_lanes * 4,
            "u8 should have four times as many lanes as f32 (u8={u8_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            i8_lanes,
            f32_lanes * 4,
            "i8 should have four times as many lanes as f32 (i8={i8_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            u32_lanes, f32_lanes,
            "u32 should have the same lane count as f32 (u32={u32_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            i32_lanes, f32_lanes,
            "i32 should have the same lane count as f32 (i32={i32_lanes}, f32={f32_lanes})"
        );
        assert_eq!(
            u64_lanes, f64_lanes,
            "u64 should have the same lane count as f64 (u64={u64_lanes}, f64={f64_lanes})"
        );
        assert_eq!(
            i64_lanes, f64_lanes,
            "i64 should have the same lane count as f64 (i64={i64_lanes}, f64={f64_lanes})"
        );
    } else {
        // Scalar fallback: all types collapse to 1 lane. That is correct.
        assert_eq!(f32_lanes, 1, "scalar fallback must yield 1 for f32");
        assert_eq!(f64_lanes, 1, "scalar fallback must yield 1 for f64");
        assert_eq!(f16_lanes, 1, "scalar fallback must yield 1 for f16");
        assert_eq!(bf16_lanes, 1, "scalar fallback must yield 1 for bf16");
    }
}

// ---------------------------------------------------------------------------
// 4. `simd_lanes` is a true `const fn`: value must be usable in const context.
// ---------------------------------------------------------------------------

#[test]
fn simd_lanes_usable_in_const_context() {
    const F32_LANES: usize = simd_lanes::<f32>();
    const F64_LANES: usize = simd_lanes::<f64>();
    const U8_LANES: usize = simd_lanes::<u8>();

    // Prove they survive the `const` evaluation gate — use const blocks so
    // the `assertions_on_constants` lint is satisfied.
    const { assert!(F32_LANES >= 1) }
    const { assert!(F64_LANES >= 1) }
    const { assert!(U8_LANES >= 1) }

    // The const values must agree with the runtime call.
    assert_eq!(
        F32_LANES,
        simd_lanes::<f32>(),
        "const and runtime f32 must agree"
    );
    assert_eq!(
        F64_LANES,
        simd_lanes::<f64>(),
        "const and runtime f64 must agree"
    );
    assert_eq!(
        U8_LANES,
        simd_lanes::<u8>(),
        "const and runtime u8 must agree"
    );
}

// ---------------------------------------------------------------------------
// 5. Kernel threshold consistency.
//    PRF-003 gated the AVX2 path on `LANES >= 8` (f32) / `LANES >= 4` (f64).
//    When vectorization is active the f32/f64 ratio must be exactly 2:1 so both
//    thresholds are reached or missed together.
// ---------------------------------------------------------------------------

#[test]
fn simd_lanes_kernel_threshold_consistency() {
    let f32_lanes = simd_lanes::<f32>();
    let f64_lanes = simd_lanes::<f64>();

    if f32_lanes > 1 {
        // Vectorisation is active: ratio must be 2:1.
        assert_eq!(
            f32_lanes,
            f64_lanes * 2,
            "f32/f64 lane ratio must be 2:1 when vectorisation is active \
             (f32={f32_lanes}, f64={f64_lanes}); this ensures the LANES >= 8 \
             and LANES >= 4 AVX2 thresholds are reached consistently"
        );
    } else {
        // Scalar fallback: both should be 1.
        assert_eq!(f32_lanes, 1, "scalar f32 must be 1");
        assert_eq!(f64_lanes, 1, "scalar f64 must be 1");
    }
}
