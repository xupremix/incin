// Every item below is `#[cfg(target_arch = "wasm32", target_feature =
// "simd128")]`; on any other host/target this module compiles empty, so
// this import is unused there.
#[allow(unused_imports)]
use super::*;

// ======================= WASM SIMD128 (wasm32) =======================

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) unsafe fn wasm_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f32> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    unsafe { wasm_binary_f32_into(op, lhs, rhs, output_ptr) };
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_binary_f32_into(op: BinaryOp, lhs: &[f32], rhs: &[f32], output_ptr: *mut f32) {
    use core::arch::wasm32::{f32x4_add, f32x4_div, f32x4_mul, f32x4_sub, v128_load, v128_store};

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        unsafe {
            let lhs_vector = v128_load(lhs.as_ptr().add(index).cast());
            let rhs_vector = v128_load(rhs.as_ptr().add(index).cast());
            let result = match op {
                BinaryOp::Add => f32x4_add(lhs_vector, rhs_vector),
                BinaryOp::Sub => f32x4_sub(lhs_vector, rhs_vector),
                BinaryOp::Mul => f32x4_mul(lhs_vector, rhs_vector),
                BinaryOp::Div => f32x4_div(lhs_vector, rhs_vector),
            };
            v128_store(output_ptr.add(index).cast(), result);
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) unsafe fn wasm_scalar_f32(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    unsafe { wasm_scalar_f32_into(op, dense, scalar, scalar_left, output_ptr) };
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_scalar_f32_into(
    op: BinaryOp,
    dense: &[f32],
    scalar: f32,
    scalar_left: bool,
    output_ptr: *mut f32,
) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_div, f32x4_mul, f32x4_splat, f32x4_sub, v128_load, v128_store,
    };

    let scalar_vector = f32x4_splat(scalar);
    let vectorized = dense.len() / 4 * 4;
    for index in (0..vectorized).step_by(4) {
        unsafe {
            let dense_vector = v128_load(dense.as_ptr().add(index).cast());
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => f32x4_add(lhs, rhs),
                BinaryOp::Sub => f32x4_sub(lhs, rhs),
                BinaryOp::Mul => f32x4_mul(lhs, rhs),
                BinaryOp::Div => f32x4_div(lhs, rhs),
            };
            v128_store(output_ptr.add(index).cast(), result);
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) unsafe fn wasm_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut output: Vec<f64> = Vec::with_capacity(lhs.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    unsafe { wasm_binary_f64_into(op, lhs, rhs, output_ptr) };
    unsafe { output.set_len(lhs.len()) };
    output
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_binary_f64_into(op: BinaryOp, lhs: &[f64], rhs: &[f64], output_ptr: *mut f64) {
    use core::arch::wasm32::{f64x2_add, f64x2_div, f64x2_mul, f64x2_sub, v128_load, v128_store};

    debug_assert_eq!(lhs.len(), rhs.len());
    let vectorized = lhs.len() / 2 * 2;
    for index in (0..vectorized).step_by(2) {
        unsafe {
            let lhs_vector = v128_load(lhs.as_ptr().add(index).cast());
            let rhs_vector = v128_load(rhs.as_ptr().add(index).cast());
            let result = match op {
                BinaryOp::Add => f64x2_add(lhs_vector, rhs_vector),
                BinaryOp::Sub => f64x2_sub(lhs_vector, rhs_vector),
                BinaryOp::Mul => f64x2_mul(lhs_vector, rhs_vector),
                BinaryOp::Div => f64x2_div(lhs_vector, rhs_vector),
            };
            v128_store(output_ptr.add(index).cast(), result);
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) unsafe fn wasm_scalar_f64(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::with_capacity(dense.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<f64>();
    unsafe { wasm_scalar_f64_into(op, dense, scalar, scalar_left, output_ptr) };
    unsafe { output.set_len(dense.len()) };
    output
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_scalar_f64_into(
    op: BinaryOp,
    dense: &[f64],
    scalar: f64,
    scalar_left: bool,
    output_ptr: *mut f64,
) {
    use core::arch::wasm32::{
        f64x2_add, f64x2_div, f64x2_mul, f64x2_splat, f64x2_sub, v128_load, v128_store,
    };

    let scalar_vector = f64x2_splat(scalar);
    let vectorized = dense.len() / 2 * 2;
    for index in (0..vectorized).step_by(2) {
        unsafe {
            let dense_vector = v128_load(dense.as_ptr().add(index).cast());
            let (lhs, rhs) = if scalar_left {
                (scalar_vector, dense_vector)
            } else {
                (dense_vector, scalar_vector)
            };
            let result = match op {
                BinaryOp::Add => f64x2_add(lhs, rhs),
                BinaryOp::Sub => f64x2_sub(lhs, rhs),
                BinaryOp::Mul => f64x2_mul(lhs, rhs),
                BinaryOp::Div => f64x2_div(lhs, rhs),
            };
            v128_store(output_ptr.add(index).cast(), result);
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
