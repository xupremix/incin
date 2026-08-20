use super::*;

/// One `[m, k] @ [k, n]` product into a zeroed, contiguous row-major `out`.
///
/// The kernels are tried in decreasing order of specificity, and each one
/// reports whether it applied rather than being selected by a condition
/// duplicated at the call site.
pub(super) fn gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), m * n);

    #[cfg(feature = "cpu-blas")]
    if blocked_gemm(m, k, n, lhs, rhs, out) {
        return;
    }

    if simd_gemm(m, k, n, lhs, rhs, out) {
        return;
    }

    scalar_gemm(m, k, n, lhs, rhs, out);
}

/// Hand the product to `matrixmultiply`'s blocked, register-tiled kernel.
///
/// Declines small problems: the packing this kernel does to get its cache
/// behavior is only worth paying for once the product is large enough to
/// reuse the packed panels, and below that the row-streaming kernels below
/// are faster.
#[cfg(feature = "cpu-blas")]
pub(super) fn blocked_gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) -> bool {
    /// Roughly a 64-cubed product, measured as the point where packing starts
    /// paying for itself rather than chosen for its roundness.
    const MIN_PRODUCT: usize = 64 * 64 * 64;

    if m.saturating_mul(k).saturating_mul(n) < MIN_PRODUCT {
        return false;
    }
    let (Some(lhs_data), Some(rhs_data)) = (lhs.f32_data(), rhs.f32_data()) else {
        return false;
    };
    if !lhs.fits_within(m, k, lhs_data.len()) || !rhs.fits_within(k, n, rhs_data.len()) {
        return false;
    }
    if out.len() != m * n {
        return false;
    }

    // SAFETY: `fits_within` proved every index this kernel reads is inside
    // the corresponding slice, and `out` was just checked to be exactly the
    // `m * n` elements the row stride of `n` addresses. `beta = 0.0` means
    // `out`'s prior contents are overwritten rather than read, so its zeroing
    // is not relied on here.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            lhs_data.as_ptr().add(lhs.offset),
            lhs.row_stride as isize,
            lhs.col_stride as isize,
            rhs_data.as_ptr().add(rhs.offset),
            rhs.row_stride as isize,
            rhs.col_stride as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    true
}

/// Stream whole rows of `rhs` through a SIMD accumulator.
///
/// Requires `rhs`'s columns to be adjacent and both operands to already hold
/// `f32`; the `lhs` requirement is what keeps a wider dtype on the
/// double-accumulating path in `scalar_gemm` rather than silently narrowing
/// it here.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) -> bool {
    let (Some(_), Some(rhs_data)) = (lhs.f32_data(), rhs.f32_data()) else {
        return false;
    };
    if rhs.col_stride != 1 {
        return false;
    }
    // One cached predicate rather than two macro calls. The macros cache too,
    // but keeping every AVX2 decision in `simd` is what stops the elementwise
    // kernels' compile-time-only gate from being reintroduced somewhere else.
    if !crate::simd::avx2_fma_detected() {
        return false;
    }
    // SAFETY: avx2 and fma were just detected on this CPU, and every load
    // below is bounds-checked against `rhs_data` and `out` by the loop
    // arithmetic that `gemm`'s callers derive from the operands' own shapes.
    unsafe { gemm_avx2(m, k, n, lhs, rhs, rhs_data, out) };
    true
}

#[cfg(target_arch = "aarch64")]
fn simd_gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) -> bool {
    let (Some(_), Some(rhs_data)) = (lhs.f32_data(), rhs.f32_data()) else {
        return false;
    };
    if rhs.col_stride != 1 {
        return false;
    }
    // SAFETY: NEON is part of the aarch64 baseline, and every load below is
    // bounds-checked by the loop arithmetic `gemm`'s callers derive from the
    // operands' own shapes.
    unsafe { gemm_neon(m, k, n, lhs, rhs, rhs_data, out) };
    true
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn simd_gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) -> bool {
    let (Some(_), Some(rhs_data)) = (lhs.f32_data(), rhs.f32_data()) else {
        return false;
    };
    if rhs.col_stride != 1 {
        return false;
    }
    // SAFETY: simd128 is a compile-time target feature here, and every load
    // below is bounds-checked by the loop arithmetic `gemm`'s callers derive
    // from the operands' own shapes.
    unsafe { gemm_wasm(m, k, n, lhs, rhs, rhs_data, out) };
    true
}

/// Targets with no vector kernel decline unconditionally and fall through to
/// `scalar_gemm`, which computes the same values.
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn simd_gemm(
    _m: usize,
    _k: usize,
    _n: usize,
    _lhs: MatrixView<'_>,
    _rhs: MatrixView<'_>,
    _out: &mut [f32],
) -> bool {
    false
}

/// True when the product can be written straight into an `f32` buffer.
///
/// Every kernel `gemm` tries writes `f32`, which used to be what the result
/// dtype became: two `f64` operands were read as `f64`, accumulated as `f64`,
/// narrowed once at the end, and handed back labelled `f32`. `matmul` is
/// composed into `scaled_dot_product_attention`, so that mislabel was the
/// whole of why attention returned `f32` for every operand dtype.
pub(super) fn writes_f32(lhs: &CpuStorage, rhs: &CpuStorage) -> bool {
    matches!(
        (&*lhs.buffer, &*rhs.buffer),
        (CpuBuffer::F32(_), CpuBuffer::F32(_))
    )
}

/// The general branch of `scalar_gemm`, kept in `f64` rather than narrowed.
///
/// Only reached when at least one operand is not `f32`, so none of the `f32`
/// kernels above is affected and the hot path is unchanged. The accumulation
/// is the one `scalar_gemm` already performs for a widened operand; all that
/// differs is that the result is not thrown away by a cast on the way out.
pub(super) fn gemm_f64(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), m * n);
    crate::iteration::tile_2d::<64, 64>(m, n, |r0, r1, c0, c1| {
        for row in r0..r1 {
            for col in c0..c1 {
                let mut acc = 0f64;
                for depth in 0..k {
                    acc += lhs.get(row, depth) * rhs.get(depth, col);
                }
                out[row * n + col] = acc;
            }
        }
    });
}

/// The kernel every other one falls back to, and the only one that is always
/// correct for every dtype and layout.
///
/// Two shapes, and the difference between them is deliberate. When `rhs`'s
/// columns are adjacent and already `f32`, whole rows are streamed and
/// accumulated in `f32`. Otherwise each output element is accumulated in
/// `f64` before being narrowed once, which is what a widened dtype gets. The
/// row-streaming branch is guarded on `rhs` actually holding `f32`: reaching
/// it on any other buffer used to leave the output untouched, so a plain
/// contiguous `f64` matmul returned zeros.
pub(super) fn scalar_gemm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    out: &mut [f32],
) {
    match rhs.f32_data() {
        Some(rhs_data) if rhs.col_stride == 1 => {
            crate::iteration::tile_2d::<64, 64>(m, n, |r0, r1, c0, c1| {
                for row in r0..r1 {
                    for depth in 0..k {
                        let scale = lhs.get(row, depth) as f32;
                        let rhs_row = rhs.index(depth, 0);
                        let out_row = row * n;
                        for col in c0..c1 {
                            out[out_row + col] += scale * rhs_data[rhs_row + col];
                        }
                    }
                }
            });
        }
        _ => {
            crate::iteration::tile_2d::<64, 64>(m, n, |r0, r1, c0, c1| {
                for row in r0..r1 {
                    for col in c0..c1 {
                        let mut acc = 0f64;
                        for depth in 0..k {
                            acc += lhs.get(row, depth) * rhs.get(depth, col);
                        }
                        out[row * n + col] = acc as f32;
                    }
                }
            });
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the AVX2/FMA part of a matrix product.
///
/// # Safety
/// The caller must run this only when AVX2 and FMA are available. `rhs_data`
/// and `out` must contain every element addressed by the derived row and
/// column offsets.
unsafe fn gemm_avx2(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    rhs_data: &[f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let n_vec = n - (n % 8);

    for row in 0..m {
        for depth in 0..k {
            let scale = lhs.get(row, depth) as f32;
            let scale_vec = _mm256_set1_ps(scale);

            let rhs_row = rhs.index(depth, 0);
            let out_row = row * n;

            for col in (0..n_vec).step_by(8) {
                // SAFETY: AVX2 and FMA are guaranteed by the caller, and the
                // loop bounds keep each eight-element load and store inside
                // the validated row slices.
                unsafe {
                    let b = _mm256_loadu_ps(rhs_data.as_ptr().add(rhs_row + col));
                    let mut c = _mm256_loadu_ps(out.as_ptr().add(out_row + col));
                    c = _mm256_fmadd_ps(scale_vec, b, c);
                    _mm256_storeu_ps(out.as_mut_ptr().add(out_row + col), c);
                }
            }

            for col in n_vec..n {
                out[out_row + col] += scale * rhs_data[rhs_row + col];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
/// Computes the NEON part of a matrix product.
///
/// # Safety
/// `rhs_data` and `out` must contain every element addressed by the derived
/// row and column offsets. NEON is required by the aarch64 target contract.
unsafe fn gemm_neon(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    rhs_data: &[f32],
    out: &mut [f32],
) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    let n_vec = n - (n % 4);

    for row in 0..m {
        for depth in 0..k {
            let scale = lhs.get(row, depth) as f32;
            // SAFETY: NEON is part of the aarch64 baseline selected for this
            // function by its target architecture.
            let scale_vec = unsafe { vdupq_n_f32(scale) };

            let rhs_row = rhs.index(depth, 0);
            let out_row = row * n;

            for col in (0..n_vec).step_by(4) {
                // SAFETY: the loop bounds keep each four-element load and
                // store inside the validated row slices.
                unsafe {
                    let b = vld1q_f32(rhs_data.as_ptr().add(rhs_row + col));
                    let c = vld1q_f32(out.as_ptr().add(out_row + col));
                    vst1q_f32(
                        out.as_mut_ptr().add(out_row + col),
                        vfmaq_f32(c, scale_vec, b),
                    );
                }
            }

            for col in n_vec..n {
                out[out_row + col] += scale * rhs_data[rhs_row + col];
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
/// Computes the SIMD128 part of a matrix product.
///
/// # Safety
/// `rhs_data` and `out` must contain every element addressed by the derived
/// row and column offsets. The function is compiled only with `simd128`.
unsafe fn gemm_wasm(
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixView<'_>,
    rhs: MatrixView<'_>,
    rhs_data: &[f32],
    out: &mut [f32],
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128_load, v128_store};

    let n_vec = n - (n % 4);

    for row in 0..m {
        for depth in 0..k {
            let scale = lhs.get(row, depth) as f32;
            let scale_vec = f32x4_splat(scale);

            let rhs_row = rhs.index(depth, 0);
            let out_row = row * n;

            for col in (0..n_vec).step_by(4) {
                // SAFETY: the loop bounds keep each four-element load and
                // store inside the validated row slices.
                unsafe {
                    let b = v128_load(rhs_data.as_ptr().add(rhs_row + col).cast());
                    let c = v128_load(out.as_ptr().add(out_row + col).cast());
                    v128_store(
                        out.as_mut_ptr().add(out_row + col).cast(),
                        f32x4_add(c, f32x4_mul(scale_vec, b)),
                    );
                }
            }

            for col in n_vec..n {
                out[out_row + col] += scale * rhs_data[rhs_row + col];
            }
        }
    }
}
