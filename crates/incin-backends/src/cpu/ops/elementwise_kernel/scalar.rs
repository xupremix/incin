use super::*;

pub(super) fn map_binary<T: TypedKernel, F>(lhs: &[T], rhs: &[T], op: &F) -> Vec<T>
where
    F: Fn(T, T) -> T + Send + Sync,
{
    map_binary_typed(lhs, rhs, op)
}

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(unreachable_code)
)]
pub(super) fn map_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    // Either the compiler was told to assume AVX2, or this machine was asked.
    // Only the first is free, and only the second is true of a stock build, so
    // both are consulted: gating on the constant alone left these kernels
    // unreachable in every default `cargo build`.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_f32_available() {
            if lhs.len() < DENSE_PARALLEL_GRAIN {
                // SAFETY: the guard proves AVX2 is available, either as a
                // compile-time target feature or by runtime detection, which is
                // exactly the precondition of a `#[target_feature]` function.
                return unsafe { avx2_binary_f32(op, lhs, rhs) };
            }
            return parallel_avx2_binary_f32(op, lhs, rhs);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if lhs.len() < DENSE_PARALLEL_GRAIN {
            return unsafe { neon_binary_f32(op, lhs, rhs) };
        }
        #[cfg(feature = "std")]
        return parallel_neon_binary_f32(op, lhs, rhs);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { wasm_binary_f32(op, lhs, rhs) };
    }
    map_binary(lhs, rhs, &|lhs, rhs| op.eval_f32(lhs, rhs))
}

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(unreachable_code)
)]
pub(super) fn map_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    // Either the compiler was told to assume AVX2, or this machine was asked.
    // Gating on the constant alone left these kernels unreachable in every
    // default `cargo build`; see `simd::avx2_detected`.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_f64_available() {
            if lhs.len() < DENSE_PARALLEL_GRAIN {
                // SAFETY: the guard proves AVX2 is available, by compile-time
                // target feature or by runtime detection.
                return unsafe { avx2_binary_f64(op, lhs, rhs) };
            }
            return parallel_avx2_binary_f64(op, lhs, rhs);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if lhs.len() < DENSE_PARALLEL_GRAIN {
            return unsafe { neon_binary_f64(op, lhs, rhs) };
        }
        #[cfg(feature = "std")]
        return parallel_neon_binary_f64(op, lhs, rhs);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { wasm_binary_f64(op, lhs, rhs) };
    }
    map_binary(lhs, rhs, &|lhs, rhs| op.eval_f64(lhs, rhs))
}

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(unreachable_code)
)]
pub(super) fn map_scalar_f32(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
) -> Vec<f32> {
    // Either the compiler was told to assume AVX2, or this machine was asked.
    // Gating on the constant alone left these kernels unreachable in every
    // default `cargo build`; see `simd::avx2_detected`.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_f32_available() {
            if dense.len() < DENSE_PARALLEL_GRAIN {
                // SAFETY: the guard proves AVX2 is available, by compile-time
                // target feature or by runtime detection.
                return unsafe { avx2_scalar_f32(op, dense, scalar, scalar_left) };
            }
            return parallel_avx2_scalar_f32(op, dense, scalar, scalar_left);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if dense.len() < DENSE_PARALLEL_GRAIN {
            return unsafe { neon_scalar_f32(op, dense, scalar, scalar_left) };
        }
        #[cfg(feature = "std")]
        return parallel_neon_scalar_f32(op, dense, scalar, scalar_left);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { wasm_scalar_f32(op, dense, scalar, scalar_left) };
    }
    if scalar_left {
        map_scalar_left(scalar, dense, &|lhs, rhs| op.eval_f32(lhs, rhs))
    } else {
        map_scalar_right(dense, scalar, &|lhs, rhs| op.eval_f32(lhs, rhs))
    }
}

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(unreachable_code)
)]
pub(super) fn map_scalar_f64(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
) -> Vec<f64> {
    // Either the compiler was told to assume AVX2, or this machine was asked.
    // Gating on the constant alone left these kernels unreachable in every
    // default `cargo build`; see `simd::avx2_detected`.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if avx2_f64_available() {
            if dense.len() < DENSE_PARALLEL_GRAIN {
                // SAFETY: the guard proves AVX2 is available, by compile-time
                // target feature or by runtime detection.
                return unsafe { avx2_scalar_f64(op, dense, scalar, scalar_left) };
            }
            return parallel_avx2_scalar_f64(op, dense, scalar, scalar_left);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if dense.len() < DENSE_PARALLEL_GRAIN {
            return unsafe { neon_scalar_f64(op, dense, scalar, scalar_left) };
        }
        #[cfg(feature = "std")]
        return parallel_neon_scalar_f64(op, dense, scalar, scalar_left);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { wasm_scalar_f64(op, dense, scalar, scalar_left) };
    }
    if scalar_left {
        map_scalar_left(scalar, dense, &|lhs, rhs| op.eval_f64(lhs, rhs))
    } else {
        map_scalar_right(dense, scalar, &|lhs, rhs| op.eval_f64(lhs, rhs))
    }
}
