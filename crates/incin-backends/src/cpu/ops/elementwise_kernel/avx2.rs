use super::*;

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(super) fn parallel_avx2_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output = Vec::<f32>::with_capacity(lhs.len());
    output.spare_capacity_mut()[..lhs.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(lhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .zip(rhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|((output, lhs), rhs)| {
            // SAFETY: AVX2 was runtime-checked by the caller. Every output
            // chunk is disjoint and exactly matches its input chunks.
            unsafe { avx2_binary_f32_into(op, lhs, rhs, output.as_mut_ptr().cast::<f32>()) };
        });
    // SAFETY: all disjoint chunks initialized every output slot.
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(super) fn parallel_avx2_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output = Vec::<f64>::with_capacity(lhs.len());
    output.spare_capacity_mut()[..lhs.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(lhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .zip(rhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|((output, lhs), rhs)| {
            // SAFETY: AVX2 was runtime-checked by the caller. Every output
            // chunk is disjoint and exactly matches its input chunks.
            unsafe { avx2_binary_f64_into(op, lhs, rhs, output.as_mut_ptr().cast::<f64>()) };
        });
    // SAFETY: all disjoint chunks initialized every output slot.
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(super) fn parallel_avx2_scalar_f32(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
) -> Vec<f32> {
    let mut output = Vec::<f32>::with_capacity(dense.len());
    output.spare_capacity_mut()[..dense.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(dense.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|(output, dense)| {
            // SAFETY: AVX2 was runtime-checked by the caller. Every output
            // chunk is disjoint and exactly matches its input chunk.
            unsafe {
                avx2_scalar_f32_into(
                    op,
                    dense,
                    scalar,
                    scalar_left,
                    output.as_mut_ptr().cast::<f32>(),
                )
            };
        });
    // SAFETY: all disjoint chunks initialized every output slot.
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(super) fn parallel_avx2_scalar_f64(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
) -> Vec<f64> {
    let mut output = Vec::<f64>::with_capacity(dense.len());
    output.spare_capacity_mut()[..dense.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(dense.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|(output, dense)| {
            // SAFETY: AVX2 was runtime-checked by the caller. Every output
            // chunk is disjoint and exactly matches its input chunk.
            unsafe {
                avx2_scalar_f64_into(
                    op,
                    dense,
                    scalar,
                    scalar_left,
                    output.as_mut_ptr().cast::<f64>(),
                )
            };
        });
    // SAFETY: all disjoint chunks initialized every output slot.
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f32> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    // SAFETY: the caller guarantees AVX2 and output has space for lhs.len().
    unsafe { avx2_binary_f32_into(op, lhs, rhs, output_ptr) };
    // SAFETY: the writer initialized every slot.
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_binary_f32_into(op: BinaryOp, lhs: &[f32], rhs: &[f32], output_ptr: *mut f32) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 8 * 8;
    for index in (0..vectorized).step_by(8) {
        // SAFETY: index..index+8 lies within both inputs and output.
        unsafe {
            let lhs_vector = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let rhs_vector = _mm256_loadu_ps(rhs.as_ptr().add(index));
            let result = match op {
                BinaryOp::Add => _mm256_add_ps(lhs_vector, rhs_vector),
                BinaryOp::Sub => _mm256_sub_ps(lhs_vector, rhs_vector),
                BinaryOp::Mul => _mm256_mul_ps(lhs_vector, rhs_vector),
                BinaryOp::Div => _mm256_div_ps(lhs_vector, rhs_vector),
            };
            _mm256_storeu_ps(output_ptr.add(index), result);
        }
    }
    for index in vectorized..lhs.len() {
        // SAFETY: index is within the allocation and each slot is written once.
        unsafe {
            output_ptr
                .add(index)
                .write(op.eval_f32(lhs[index], rhs[index]))
        };
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f64> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    // SAFETY: the caller guarantees AVX2 and output has space for lhs.len().
    unsafe { avx2_binary_f64_into(op, lhs, rhs, output_ptr) };
    // SAFETY: the writer initialized every slot.
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_binary_f64_into(op: BinaryOp, lhs: &[f64], rhs: &[f64], output_ptr: *mut f64) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd,
        _mm256_sub_pd,
    };

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        // SAFETY: index..index+4 lies within both inputs and output.
        unsafe {
            let lhs_vector = _mm256_loadu_pd(lhs.as_ptr().add(index));
            let rhs_vector = _mm256_loadu_pd(rhs.as_ptr().add(index));
            let result = match op {
                BinaryOp::Add => _mm256_add_pd(lhs_vector, rhs_vector),
                BinaryOp::Sub => _mm256_sub_pd(lhs_vector, rhs_vector),
                BinaryOp::Mul => _mm256_mul_pd(lhs_vector, rhs_vector),
                BinaryOp::Div => _mm256_div_pd(lhs_vector, rhs_vector),
            };
            _mm256_storeu_pd(output_ptr.add(index), result);
        }
    }
    for index in vectorized..lhs.len() {
        // SAFETY: index is within the allocation and each slot is written once.
        unsafe {
            output_ptr
                .add(index)
                .write(op.eval_f64(lhs[index], rhs[index]))
        };
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_scalar_f32(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    // SAFETY: the caller guarantees AVX2 and output has space for dense.len().
    unsafe { avx2_broadcast_scalar_f32(op, dense, scalar, scalar_left, output_ptr) };
    // SAFETY: the writer initialized every slot.
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_broadcast_scalar_f32(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
    output_ptr: *mut f32,
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm256_sub_ps,
    };

    let scalar_vector = _mm256_set1_ps(scalar);
    let vectorized = dense.len() / 8 * 8;
    for index in (0..vectorized).step_by(8) {
        // SAFETY: index..index+8 lies within dense and output.
        unsafe {
            let dense_vector = _mm256_loadu_ps(dense.as_ptr().add(index));
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => _mm256_add_ps(lhs, rhs),
                BinaryOp::Sub => _mm256_sub_ps(lhs, rhs),
                BinaryOp::Mul => _mm256_mul_ps(lhs, rhs),
                BinaryOp::Div => _mm256_div_ps(lhs, rhs),
            };
            _mm256_storeu_ps(output_ptr.add(index), result);
        }
    }
    for (index, &dense_value) in dense.iter().enumerate().skip(vectorized) {
        let value = if scalar_left {
            op.eval_f32(scalar, dense_value)
        } else {
            op.eval_f32(dense_value, scalar)
        };
        // SAFETY: index is within the allocation and each slot is written once.
        unsafe { output_ptr.add(index).write(value) };
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_scalar_f32_into(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
    output_ptr: *mut f32,
) {
    unsafe { avx2_broadcast_scalar_f32(op, dense, scalar, scalar_left, output_ptr) }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_scalar_f64(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    // SAFETY: the caller guarantees AVX2 and output has space for dense.len().
    unsafe { avx2_scalar_f64_into(op, dense, scalar, scalar_left, output_ptr) };
    // SAFETY: the writer initialized every slot.
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_scalar_f64_into(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
    output_ptr: *mut f64,
) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };

    let scalar_vector = _mm256_set1_pd(scalar);
    let vectorized = dense.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        // SAFETY: index..index+4 lies within dense and output.
        unsafe {
            let dense_vector = _mm256_loadu_pd(dense.as_ptr().add(index));
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => _mm256_add_pd(lhs, rhs),
                BinaryOp::Sub => _mm256_sub_pd(lhs, rhs),
                BinaryOp::Mul => _mm256_mul_pd(lhs, rhs),
                BinaryOp::Div => _mm256_div_pd(lhs, rhs),
            };
            _mm256_storeu_pd(output_ptr.add(index), result);
        }
    }
    for (index, &dense_value) in dense.iter().enumerate().skip(vectorized) {
        let value = if scalar_left {
            op.eval_f64(scalar, dense_value)
        } else {
            op.eval_f64(dense_value, scalar)
        };
        // SAFETY: index is within the allocation and each slot is written once.
        unsafe { output_ptr.add(index).write(value) };
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
macro_rules! define_avx2_iteration_kernel {
    ($function:ident, $element:ty, $binary_writer:ident, $scalar_writer:ident) => {
        pub(super) fn $function(
            op: BinaryOp,
            lhs: &[$element],
            rhs: &[$element],
            plan: &IterationPlan,
        ) -> Option<Vec<$element>> {
            if simd_lanes::<$element>() < 8 && !crate::simd::avx2_detected() {
                // Not `avx2_f32_available`: this macro is instantiated for
                // several element types and reads the lane count for each.
                return None;
            }
            if plan.numel == 0 {
                return Some(Vec::new());
            }

            let inner_len = *plan.output_shape.last()?;
            let lhs_inner_stride = *plan.operands[0].strides.last()?;
            let rhs_inner_stride = *plan.operands[1].strides.last()?;
            if !matches!(
                (lhs_inner_stride, rhs_inner_stride),
                (1, 1) | (0, 1) | (1, 0)
            ) {
                return None;
            }

            let mut output = Vec::<$element>::with_capacity(plan.numel);
            let output_spare = &mut output.spare_capacity_mut()[..plan.numel];
            let write_inner =
                |(outer_index, output): (usize, &mut [core::mem::MaybeUninit<$element>])| {
                    let flat_index = outer_index * inner_len;
                    let lhs_index = plan.operands[0].physical_index(flat_index, &plan.output_shape);
                    let rhs_index = plan.operands[1].physical_index(flat_index, &plan.output_shape);
                    let output_ptr = output.as_mut_ptr().cast::<$element>();

                    // SAFETY: AVX2 was checked above. Bounds were validated
                    // when constructing the plan, output chunks are disjoint,
                    // and the supported inner strides prove each dense slice.
                    unsafe {
                        match (lhs_inner_stride, rhs_inner_stride) {
                            (1, 1) => $binary_writer(
                                op,
                                &lhs[lhs_index..lhs_index + inner_len],
                                &rhs[rhs_index..rhs_index + inner_len],
                                output_ptr,
                            ),
                            (0, 1) => $scalar_writer(
                                op,
                                &rhs[rhs_index..rhs_index + inner_len],
                                lhs[lhs_index],
                                true,
                                output_ptr,
                            ),
                            (1, 0) => $scalar_writer(
                                op,
                                &lhs[lhs_index..lhs_index + inner_len],
                                rhs[rhs_index],
                                false,
                                output_ptr,
                            ),
                            _ => unreachable!("inner stride pattern was validated"),
                        }
                    }
                };

            if plan.numel < PARALLEL_GRAIN {
                output_spare
                    .chunks_mut(inner_len)
                    .enumerate()
                    .for_each(write_inner);
            } else {
                output_spare
                    .par_chunks_mut(inner_len)
                    .enumerate()
                    .for_each(write_inner);
            }
            // SAFETY: every disjoint inner chunk initialized all of its slots.
            unsafe { output.set_len(plan.numel) };
            Some(output)
        }
    };
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
define_avx2_iteration_kernel!(
    map_iteration_avx2_f32,
    f32,
    avx2_binary_f32_into,
    avx2_scalar_f32_into
);

#[cfg(all(feature = "std", target_arch = "x86_64"))]
define_avx2_iteration_kernel!(
    map_iteration_avx2_f64,
    f64,
    avx2_binary_f64_into,
    avx2_scalar_f64_into
);
