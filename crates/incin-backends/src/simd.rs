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
        any(
            target_feature = "sse2",
            target_feature = "sse4.1",
            target_feature = "neon"
        )
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
    if lanes == 0 { 1 } else { lanes }
}

/// Whether this *machine* can execute AVX2, decided once and cached.
///
/// [`simd_vector_bytes`] answers a different question: what the compiler was
/// told to assume. A stock `cargo build` for `x86_64-unknown-linux-gnu` targets
/// the baseline ISA, where `target_feature = "avx2"` is false and every
/// `simd_lanes::<f32>() >= 8` branch is dead code - so the AVX2 kernels in this
/// crate were unreachable in exactly the builds users install, and the CPU
/// backend fell back to a scalar loop. On this machine that was a 9x difference
/// on a 65536-element `f32` add.
///
/// The kernels are annotated `#[target_feature(enable = "avx2")]`, which is
/// what makes them compile on a baseline target and callable once the feature
/// is *proven present*. This is that proof. `is_x86_feature_detected!` caches
/// internally, but it is behind a function call and a branch; caching the
/// answer in a relaxed atomic makes the check a single load on the hot path.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub fn avx2_detected() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    /// `0` not yet probed, `1` absent, `2` present.
    static STATE: AtomicU8 = AtomicU8::new(0);

    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let present = std::arch::is_x86_feature_detected!("avx2");
            STATE.store(if present { 2 } else { 1 }, Ordering::Relaxed);
            present
        }
    }
}

/// AVX2 **and** FMA together, decided once and cached.
///
/// The fused-multiply-add kernels need both, and a CPU with AVX2 but without
/// FMA exists, so they are one question rather than two.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub fn avx2_fma_detected() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    /// `0` not yet probed, `1` absent, `2` present.
    static STATE: AtomicU8 = AtomicU8::new(0);

    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let present = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
            STATE.store(if present { 2 } else { 1 }, Ordering::Relaxed);
            present
        }
    }
}

/// Non-x86_64, or `no_std`: there is no AVX2 to detect.
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
pub fn avx2_detected() -> bool {
    false
}

/// Non-x86_64, or `no_std`: there is no AVX2 or FMA to detect.
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
pub fn avx2_fma_detected() -> bool {
    false
}

/// Chunk size for parallel multi-core SIMD operations.
pub(crate) const SIMD_PARALLEL_CHUNK: usize = 128 * 1024;

/// Reduction operation for the vectorized reduction combinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimdReduceOp {
    Sum,
    Max,
    Min,
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_pd(v: core::arch::x86_64::__m256d) -> f64 {
    use core::arch::x86_64::{
        _mm_add_pd, _mm_cvtsd_f64, _mm_hadd_pd, _mm256_castpd256_pd128, _mm256_extractf128_pd,
    };
    let v_hi = _mm256_extractf128_pd(v, 1);
    let v_lo = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(v_lo, v_hi);
    let sum64 = _mm_hadd_pd(sum128, sum128);
    _mm_cvtsd_f64(sum64)
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmax256_ps(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{
        _mm_cvtss_f32, _mm_max_ps, _mm_movehl_ps, _mm_shuffle_ps, _mm256_castps256_ps128,
        _mm256_extractf128_ps,
    };
    let v_hi = _mm256_extractf128_ps(v, 1);
    let v_lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(v_lo, v_hi);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ps(m64, _mm_shuffle_ps(m64, m64, 0b00000001));
    _mm_cvtss_f32(m32)
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmin256_ps(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{
        _mm_cvtss_f32, _mm_min_ps, _mm_movehl_ps, _mm_shuffle_ps, _mm256_castps256_ps128,
        _mm256_extractf128_ps,
    };
    let v_hi = _mm256_extractf128_ps(v, 1);
    let v_lo = _mm256_castps256_ps128(v);
    let m128 = _mm_min_ps(v_lo, v_hi);
    let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_min_ps(m64, _mm_shuffle_ps(m64, m64, 0b00000001));
    _mm_cvtss_f32(m32)
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_reduce_sum_f32(data: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_castps256_ps128, _mm256_cvtps_pd, _mm256_extractf128_ps,
        _mm256_loadu_ps, _mm256_setzero_pd,
    };

    let len = data.len();
    let ptr = data.as_ptr();
    let unrolled_end = (len / 16) * 16;

    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();

    let mut i = 0;
    while i < unrolled_end {
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v0_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let v0_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v0, 1));
            acc0 = _mm256_add_pd(acc0, v0_lo);
            acc1 = _mm256_add_pd(acc1, v0_hi);

            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let v1_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v1));
            let v1_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v1, 1));
            acc2 = _mm256_add_pd(acc2, v1_lo);
            acc3 = _mm256_add_pd(acc3, v1_hi);
        }
        i += 16;
    }

    let acc_01 = _mm256_add_pd(acc0, acc1);
    let acc_23 = _mm256_add_pd(acc2, acc3);
    let mut acc = _mm256_add_pd(acc_01, acc_23);

    let vec_end = (len / 8) * 8;
    while i < vec_end {
        unsafe {
            let v = _mm256_loadu_ps(ptr.add(i));
            let v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
            let v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
            acc = _mm256_add_pd(acc, _mm256_add_pd(v_lo, v_hi));
        }
        i += 8;
    }

    let mut total = unsafe { hsum256_pd(acc) };
    while i < len {
        total += unsafe { *data.get_unchecked(i) } as f64;
        i += 1;
    }
    total as f32
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_reduce_max_f32(data: &[f32], init: f32) -> f32 {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps};

    let len = data.len();
    if len == 0 {
        return init;
    }
    let ptr = data.as_ptr();
    let unrolled_end = (len / 32) * 32;

    let mut acc0 = _mm256_set1_ps(init);
    let mut acc1 = _mm256_set1_ps(init);
    let mut acc2 = _mm256_set1_ps(init);
    let mut acc3 = _mm256_set1_ps(init);

    let mut i = 0;
    while i < unrolled_end {
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let v3 = _mm256_loadu_ps(ptr.add(i + 24));

            acc0 = _mm256_max_ps(acc0, v0);
            acc1 = _mm256_max_ps(acc1, v1);
            acc2 = _mm256_max_ps(acc2, v2);
            acc3 = _mm256_max_ps(acc3, v3);
        }
        i += 32;
    }

    let mut acc = _mm256_max_ps(_mm256_max_ps(acc0, acc1), _mm256_max_ps(acc2, acc3));

    let vec_end = (len / 8) * 8;
    while i < vec_end {
        unsafe {
            let v = _mm256_loadu_ps(ptr.add(i));
            acc = _mm256_max_ps(acc, v);
        }
        i += 8;
    }

    let mut max_val = unsafe { hmax256_ps(acc) };
    while i < len {
        let val = unsafe { *data.get_unchecked(i) };
        if val > max_val {
            max_val = val;
        }
        i += 1;
    }
    max_val
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_reduce_min_f32(data: &[f32], init: f32) -> f32 {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_min_ps, _mm256_set1_ps};

    let len = data.len();
    if len == 0 {
        return init;
    }
    let ptr = data.as_ptr();
    let unrolled_end = (len / 32) * 32;

    let mut acc0 = _mm256_set1_ps(init);
    let mut acc1 = _mm256_set1_ps(init);
    let mut acc2 = _mm256_set1_ps(init);
    let mut acc3 = _mm256_set1_ps(init);

    let mut i = 0;
    while i < unrolled_end {
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let v3 = _mm256_loadu_ps(ptr.add(i + 24));

            acc0 = _mm256_min_ps(acc0, v0);
            acc1 = _mm256_min_ps(acc1, v1);
            acc2 = _mm256_min_ps(acc2, v2);
            acc3 = _mm256_min_ps(acc3, v3);
        }
        i += 32;
    }

    let mut acc = _mm256_min_ps(_mm256_min_ps(acc0, acc1), _mm256_min_ps(acc2, acc3));

    let vec_end = (len / 8) * 8;
    while i < vec_end {
        unsafe {
            let v = _mm256_loadu_ps(ptr.add(i));
            acc = _mm256_min_ps(acc, v);
        }
        i += 8;
    }

    let mut min_val = unsafe { hmin256_ps(acc) };
    while i < len {
        let val = unsafe { *data.get_unchecked(i) };
        if val < min_val {
            min_val = val;
        }
        i += 1;
    }
    min_val
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_add_into_f32(acc: &mut [f32], src: &[f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    debug_assert_eq!(acc.len(), src.len());
    let len = acc.len();
    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let vec_end = (len / 8) * 8;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let v_acc = _mm256_loadu_ps(acc_ptr.add(i));
            let v_src = _mm256_loadu_ps(src_ptr.add(i));
            let res = _mm256_add_ps(v_acc, v_src);
            _mm256_storeu_ps(acc_ptr.add(i), res);
        }
        i += 8;
    }
    while i < len {
        unsafe {
            *acc.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
        i += 1;
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn parallel_avx2_reduce_sum_f32(data: &[f32]) -> f32 {
    use rayon::prelude::*;
    data.par_chunks(SIMD_PARALLEL_CHUNK)
        .map(|chunk| unsafe { avx2_reduce_sum_f32(chunk) })
        .sum()
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn parallel_avx2_reduce_max_f32(data: &[f32], init: f32) -> f32 {
    use rayon::prelude::*;
    data.par_chunks(SIMD_PARALLEL_CHUNK)
        .map(|chunk| unsafe { avx2_reduce_max_f32(chunk, init) })
        .reduce(|| init, |a, b| if b > a { b } else { a })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn parallel_avx2_reduce_min_f32(data: &[f32], init: f32) -> f32 {
    use rayon::prelude::*;
    data.par_chunks(SIMD_PARALLEL_CHUNK)
        .map(|chunk| unsafe { avx2_reduce_min_f32(chunk, init) })
        .reduce(|| init, |a, b| if b < a { b } else { a })
}

/// Vectorized reduction of an `f32` slice using AVX2 SIMD with multi-core parallelization for large slices.
pub(crate) fn vectorize_reduce_f32(data: &[f32], init: f32, op: SimdReduceOp) -> f32 {
    if data.is_empty() {
        return init;
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_detected() || simd_lanes::<f32>() >= 8 {
            if data.len() < SIMD_PARALLEL_CHUNK {
                return unsafe {
                    match op {
                        SimdReduceOp::Sum => avx2_reduce_sum_f32(data),
                        SimdReduceOp::Max => avx2_reduce_max_f32(data, init),
                        SimdReduceOp::Min => avx2_reduce_min_f32(data, init),
                    }
                };
            }
            return match op {
                SimdReduceOp::Sum => parallel_avx2_reduce_sum_f32(data),
                SimdReduceOp::Max => parallel_avx2_reduce_max_f32(data, init),
                SimdReduceOp::Min => parallel_avx2_reduce_min_f32(data, init),
            };
        }
    }

    // Scalar fallback
    match op {
        SimdReduceOp::Sum => data.iter().copied().fold(init, |acc, x| acc + x),
        SimdReduceOp::Max => data
            .iter()
            .copied()
            .fold(init, |acc, x| if x > acc { x } else { acc }),
        SimdReduceOp::Min => data
            .iter()
            .copied()
            .fold(init, |acc, x| if x < acc { x } else { acc }),
    }
}

/// Vectorized sum of an `f32` slice.
#[inline]
pub(crate) fn vectorize_reduce_sum_f32(data: &[f32]) -> f32 {
    vectorize_reduce_f32(data, 0.0, SimdReduceOp::Sum)
}

/// Vectorized maximum of an `f32` slice.
#[inline]
pub(crate) fn vectorize_reduce_max_f32(data: &[f32], init: f32) -> f32 {
    vectorize_reduce_f32(data, init, SimdReduceOp::Max)
}

/// Vectorized minimum of an `f32` slice.
#[inline]
pub(crate) fn vectorize_reduce_min_f32(data: &[f32], init: f32) -> f32 {
    vectorize_reduce_f32(data, init, SimdReduceOp::Min)
}

/// Vectorized addition of `src` into accumulator `acc`.
pub(crate) fn vectorize_add_into_f32(acc: &mut [f32], src: &[f32]) {
    debug_assert_eq!(acc.len(), src.len());
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_detected() || simd_lanes::<f32>() >= 8 {
            unsafe { avx2_add_into_f32(acc, src) };
            return;
        }
    }
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += s;
    }
}

#[cfg(test)]
mod avx2_detection_tests {
    use super::*;

    /// The answer must not change between calls, since the first call decides
    /// it for the life of the process.
    #[test]
    fn detection_is_stable_across_calls() {
        let first = avx2_detected();
        for _ in 0..1000 {
            assert_eq!(avx2_detected(), first);
        }
    }

    /// A build that was compiled *for* AVX2 must never report it absent: the
    /// compile-time assumption and the runtime probe can disagree in only one
    /// direction, and that direction is a miscompiled binary rather than a
    /// fallback.
    #[test]
    fn a_compile_time_avx2_build_also_detects_it() {
        if simd_lanes::<f32>() >= 8 {
            assert!(
                avx2_detected(),
                "built with AVX2 but the runtime probe denies it"
            );
        }
    }

    /// The regression this function exists for.
    ///
    /// A stock `cargo build` has no compile-time AVX2, so `simd_lanes` reports
    /// 4 and every `>= 8` branch is dead code. If the gate is ever narrowed back
    /// to the constant alone, the AVX2 kernels become unreachable in exactly the
    /// builds users install, and nothing else in the suite notices: the kernels
    /// stay correct, their own tests keep passing, and only a benchmark shows
    /// the 9x. This asserts the gate is runtime-aware on a machine that can run
    /// the instructions.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn hardware_support_is_not_hidden_by_a_baseline_build() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            // No AVX2 to find. The scalar fallback is the correct answer here,
            // and this machine cannot prove or disprove the gate.
            return;
        }
        assert!(
            avx2_detected(),
            "this machine supports AVX2 but the SIMD gate reports otherwise; the \
             kernels in cpu::ops::elementwise_kernel are unreachable"
        );
    }

    #[test]
    fn vectorize_reduce_sum_matches_scalar() {
        for len in [0, 1, 7, 8, 15, 31, 32, 33, 64, 1024, 131_075] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 2.5).collect();
            let scalar_sum: f32 = data.iter().sum();
            let simd_sum = vectorize_reduce_sum_f32(&data);
            let diff = (simd_sum - scalar_sum).abs();
            let rel_diff = if scalar_sum.abs() > 1.0 {
                diff / scalar_sum.abs()
            } else {
                diff
            };
            assert!(
                rel_diff <= 1e-4,
                "len {len}: simd_sum {simd_sum} vs scalar_sum {scalar_sum} diff {diff} rel {rel_diff}"
            );
        }
    }

    #[test]
    fn vectorize_reduce_max_matches_scalar() {
        for len in [0, 1, 7, 8, 15, 31, 32, 33, 64, 1024, 131_075] {
            let data: Vec<f32> = (0..len).map(|i| ((i * 37) % 100) as f32 - 50.0).collect();
            let scalar_max = data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
            let simd_max = vectorize_reduce_max_f32(&data, f32::NEG_INFINITY);
            assert_eq!(simd_max, scalar_max, "len {len}");
        }
    }

    #[test]
    fn vectorize_reduce_min_matches_scalar() {
        for len in [0, 1, 7, 8, 15, 31, 32, 33, 64, 1024, 131_075] {
            let data: Vec<f32> = (0..len).map(|i| ((i * 37) % 100) as f32 - 50.0).collect();
            let scalar_min = data
                .iter()
                .copied()
                .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
            let simd_min = vectorize_reduce_min_f32(&data, f32::INFINITY);
            assert_eq!(simd_min, scalar_min, "len {len}");
        }
    }

    #[test]
    fn vectorize_add_into_matches_scalar() {
        for len in [0, 1, 7, 8, 15, 31, 32, 33, 64, 1024] {
            let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let mut acc_scalar: Vec<f32> = (0..len).map(|i| i as f32 * 1.5).collect();
            let mut acc_simd = acc_scalar.clone();

            for (a, &s) in acc_scalar.iter_mut().zip(src.iter()) {
                *a += s;
            }
            vectorize_add_into_f32(&mut acc_simd, &src);

            assert_eq!(acc_simd, acc_scalar, "len {len}");
        }
    }
}
