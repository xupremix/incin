//! Target-feature aware SIMD lane constants and vectorization utilities.
//!
//! Provides compile-time lane width resolution for SIMD optimization without
//! runtime feature detection overhead on supported target feature paths.

use core::mem::size_of;

/// Resolves the compile-time SIMD vector lane count for type `T` based on
/// enabled target features.
///
/// Returns the number of elements of type `T` that fit into a single SIMD vector
/// register on the target CPU architecture. Returns `1` when no vector extension
/// is enabled or when `T` is zero-sized.
#[inline]
pub const fn simd_lanes<T: Sized>() -> usize {
    let size = size_of::<T>();
    if size == 0 {
        return 1;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let lanes = 64 / size;
        if lanes > 0 { lanes } else { 1 }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    {
        let lanes = 32 / size;
        if lanes > 0 { lanes } else { 1 }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.1",
        not(target_feature = "avx2"),
        not(target_feature = "avx512f")
    ))]
    {
        let lanes = 16 / size;
        if lanes > 0 { lanes } else { 1 }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let lanes = 16 / size;
        if lanes > 0 { lanes } else { 1 }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let lanes = 16 / size;
        if lanes > 0 { lanes } else { 1 }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "avx512f"),
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn test_simd_lanes_are_positive_and_power_of_two() {
        assert!(simd_lanes::<f32>() >= 1);
        assert!(simd_lanes::<f64>() >= 1);
        assert!(simd_lanes::<f16>() >= 1);
        assert!(simd_lanes::<bf16>() >= 1);
        assert!(simd_lanes::<u8>() >= 1);
        assert!(simd_lanes::<i32>() >= 1);

        assert!(simd_lanes::<f32>().is_power_of_two());
        assert!(simd_lanes::<f64>().is_power_of_two());
        assert!(simd_lanes::<f16>().is_power_of_two());
        assert!(simd_lanes::<bf16>().is_power_of_two());
    }

    #[test]
    fn test_lane_ratio_matches_type_size() {
        assert_eq!(simd_lanes::<u8>(), simd_lanes::<f32>() * 4);
        assert_eq!(simd_lanes::<f16>(), simd_lanes::<f32>() * 2);
        assert_eq!(simd_lanes::<f64>(), simd_lanes::<f32>() / 2);
    }
}
