// Every item below is `#[cfg(target_arch = "aarch64")]`; on any other host
// arch this module compiles empty, so this import is unused there.
#[allow(unused_imports)]
use super::*;

// ======================= ARM NEON (aarch64) =======================

#[cfg(target_arch = "aarch64")]
unsafe fn neon_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f32> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    unsafe { neon_binary_f32_into(op, lhs, rhs, output_ptr) };
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_binary_f32_into(op: BinaryOp, lhs: &[f32], rhs: &[f32], output_ptr: *mut f32) {
    use core::arch::aarch64::{vaddq_f32, vdivq_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        unsafe {
            let lhs_vector = vld1q_f32(lhs.as_ptr().add(index));
            let rhs_vector = vld1q_f32(rhs.as_ptr().add(index));
            let result = match op {
                BinaryOp::Add => vaddq_f32(lhs_vector, rhs_vector),
                BinaryOp::Sub => vsubq_f32(lhs_vector, rhs_vector),
                BinaryOp::Mul => vmulq_f32(lhs_vector, rhs_vector),
                BinaryOp::Div => vdivq_f32(lhs_vector, rhs_vector),
            };
            vst1q_f32(output_ptr.add(index), result);
        }
    }
    for index in vectorized..lhs.len() {
        unsafe {
            output_ptr
                .add(index)
                .write(op.eval_f32(lhs[index], rhs[index]))
        };
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(super) fn parallel_neon_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output = Vec::<f32>::with_capacity(lhs.len());
    output.spare_capacity_mut()[..lhs.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(lhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .zip(rhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|((output, lhs), rhs)| {
            unsafe { neon_binary_f32_into(op, lhs, rhs, output.as_mut_ptr().cast::<f32>()) };
        });
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_scalar_f32(op: BinaryOp, dense: &[f32], scalar: f32, scalar_left: bool) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    unsafe { neon_scalar_f32_into(op, dense, scalar, scalar_left, output_ptr) };
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_scalar_f32_into(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
    output_ptr: *mut f32,
) {
    use core::arch::aarch64::{
        vaddq_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32,
    };

    let scalar_vector = unsafe { vdupq_n_f32(scalar) };
    let vectorized = dense.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        unsafe {
            let dense_vector = vld1q_f32(dense.as_ptr().add(index));
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => vaddq_f32(lhs, rhs),
                BinaryOp::Sub => vsubq_f32(lhs, rhs),
                BinaryOp::Mul => vmulq_f32(lhs, rhs),
                BinaryOp::Div => vdivq_f32(lhs, rhs),
            };
            vst1q_f32(output_ptr.add(index), result);
        }
    }
    for (index, &dense_value) in dense.iter().enumerate().skip(vectorized) {
        let value = if scalar_left {
            op.eval_f32(scalar, dense_value)
        } else {
            op.eval_f32(dense_value, scalar)
        };
        unsafe { output_ptr.add(index).write(value) };
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(super) fn parallel_neon_scalar_f32(
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
            unsafe {
                neon_scalar_f32_into(
                    op,
                    dense,
                    scalar,
                    scalar_left,
                    output.as_mut_ptr().cast::<f32>(),
                )
            };
        });
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f64> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    unsafe { neon_binary_f64_into(op, lhs, rhs, output_ptr) };
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_binary_f64_into(op: BinaryOp, lhs: &[f64], rhs: &[f64], output_ptr: *mut f64) {
    use core::arch::aarch64::{vaddq_f64, vdivq_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 2 * 2;
    for index in (0..vectorized).step_by(2) {
        unsafe {
            let lhs_vector = vld1q_f64(lhs.as_ptr().add(index));
            let rhs_vector = vld1q_f64(rhs.as_ptr().add(index));
            let result = match op {
                BinaryOp::Add => vaddq_f64(lhs_vector, rhs_vector),
                BinaryOp::Sub => vsubq_f64(lhs_vector, rhs_vector),
                BinaryOp::Mul => vmulq_f64(lhs_vector, rhs_vector),
                BinaryOp::Div => vdivq_f64(lhs_vector, rhs_vector),
            };
            vst1q_f64(output_ptr.add(index), result);
        }
    }
    for index in vectorized..lhs.len() {
        unsafe {
            output_ptr
                .add(index)
                .write(op.eval_f64(lhs[index], rhs[index]))
        };
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(super) fn parallel_neon_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output = Vec::<f64>::with_capacity(lhs.len());
    output.spare_capacity_mut()[..lhs.len()]
        .par_chunks_mut(SIMD_PARALLEL_CHUNK)
        .zip(lhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .zip(rhs.par_chunks(SIMD_PARALLEL_CHUNK))
        .for_each(|((output, lhs), rhs)| {
            unsafe { neon_binary_f64_into(op, lhs, rhs, output.as_mut_ptr().cast::<f64>()) };
        });
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_scalar_f64(op: BinaryOp, dense: &[f64], scalar: f64, scalar_left: bool) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    unsafe { neon_scalar_f64_into(op, dense, scalar, scalar_left, output_ptr) };
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_scalar_f64_into(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
    output_ptr: *mut f64,
) {
    use core::arch::aarch64::{
        vaddq_f64, vdivq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64,
    };

    let scalar_vector = unsafe { vdupq_n_f64(scalar) };
    let vectorized = dense.len() / 2 * 2;
    for index in (0..vectorized).step_by(2) {
        unsafe {
            let dense_vector = vld1q_f64(dense.as_ptr().add(index));
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => vaddq_f64(lhs, rhs),
                BinaryOp::Sub => vsubq_f64(lhs, rhs),
                BinaryOp::Mul => vmulq_f64(lhs, rhs),
                BinaryOp::Div => vdivq_f64(lhs, rhs),
            };
            vst1q_f64(output_ptr.add(index), result);
        }
    }
    for (index, &dense_value) in dense.iter().enumerate().skip(vectorized) {
        let value = if scalar_left {
            op.eval_f64(scalar, dense_value)
        } else {
            op.eval_f64(dense_value, scalar)
        };
        unsafe { output_ptr.add(index).write(value) };
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(super) fn parallel_neon_scalar_f64(
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
            unsafe {
                neon_scalar_f64_into(
                    op,
                    dense,
                    scalar,
                    scalar_left,
                    output.as_mut_ptr().cast::<f64>(),
                )
            };
        });
    unsafe { output.set_len(dense.len()) };
    output
}
