use core::mem::size_of;

/// Returns the compile-time SIMD vector width in bytes for the active target architecture.
#[inline]
pub const fn simd_vector_bytes() -> usize {
    #[cfg(target_feature = "avx512f")]
    {
        64
    }
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    {
        32
    }
    #[cfg(all(
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
        any(target_feature = "sse2", target_feature = "sse4.1", target_feature = "neon")
    ))]
    {
        16
    }
    #[cfg(not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "sse2",
        target_feature = "sse4.1",
        target_feature = "neon"
    )))]
    {
        1
    }
}

/// Returns the compile-time SIMD vector lane count for a given type `T`.
///
/// Returns at least `1` for types that are larger than the vector width or zero-sized.
#[inline]
pub const fn simd_lanes<T: Sized>() -> usize {
    let elem_size = size_of::<T>();
    if elem_size == 0 {
        return 1;
    }
    let vector_bytes = simd_vector_bytes();
    let lanes = vector_bytes / elem_size;
    if lanes == 0 {
        1
    } else {
        lanes
    }
}
