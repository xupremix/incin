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
/// `simd_lanes::<f32>() >= 8` branch is dead code — so the AVX2 kernels in this
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
}
