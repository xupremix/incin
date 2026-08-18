//! Typed CPU pointwise kernel families.
//!
//! Operations are represented once and evaluated using the compute type
//! selected for each storage dtype. F16/BF16 use F32 compute; F32 and F64
//! stay native. The dispatcher specializes contiguous and scalar-broadcast
//! layouts and uses normalized typed iteration for general views.

use core::ops::Range;

use half::{bf16, f16};
use incin_core::error::{Error, Result};
use rayon::prelude::*;

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride;
use crate::cpu::typed_kernel::{TypedKernel, map_binary_typed, map_unary_typed};
use crate::iteration::{IterationPlan, OperandIteration, OperandLayout, UnaryIterationPlan};
use crate::simd_lanes;

// The release microbenchmark shows thread-pool dispatch dominates through
// tens of thousands of elements while large tensors benefit substantially.
// Keep this conservative initial crossover explicit and retune per
// architecture once distributions and thread-count metadata are recorded.
const PARALLEL_GRAIN: usize = 256 * 1024;
// Explicit AVX2 remains faster beyond the generic scalar loop's parallel
// crossover. This separate cutoff is benchmarked by the ignored release test.
const DENSE_PARALLEL_GRAIN: usize = 2 * 1024 * 1024;
const SIMD_PARALLEL_CHUNK: usize = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UnaryOp {
    Neg,
    AddScalar(f64),
    MulScalar(f64),
    Relu,
    Step,
    Mish,
    Elu,
    Gelu,
    Abs,
    Exp,
    Sqrt,
    Log,
    Tanh,
    Sigmoid,
    Swish,
    Powf(f64),
    Clamp(f64, f64),
    Sign,
    Floor,
    Ceil,
    Round,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Erf,
    Rsqrt,
    Trunc,
    Frac,
    TanhBackward,
    SigmoidBackward,
    EluBackward,
    TanBackward,
    AsinBackward,
    AcosBackward,
    AtanBackward,
    AsinhBackward,
    AcoshBackward,
    AtanhBackward,
    ErfBackward,
    RsqrtBackward,
    GeluBackward,
    MishBackward,
}

impl UnaryOp {
    #[inline]
    pub(crate) fn eval_f32(self, value: f32) -> f32 {
        match self {
            Self::Neg => -value,
            Self::AddScalar(scalar) => value + scalar as f32,
            Self::MulScalar(scalar) => value * scalar as f32,
            Self::Relu => value.max(0.0),
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Mish => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                value * softplus.tanh()
            }
            Self::Elu => {
                if value > 0.0 {
                    value
                } else {
                    value.exp() - 1.0
                }
            }
            Self::Gelu => {
                // GELU's analytical backward uses the normal PDF/CDF in F64.
                // Keeping this approximation in F64 avoids exceeding the
                // established finite-difference gradient tolerance.
                let value = f64::from(value);
                (value * 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2))) as f32
            }
            Self::Abs => value.abs(),
            Self::Exp => value.exp(),
            Self::Sqrt => value.sqrt(),
            Self::Log => value.ln(),
            Self::Tanh => value.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Swish => value / (1.0 + (-value).exp()),
            Self::Powf(exp) => value.powf(exp as f32),
            Self::Clamp(min, max) => value.clamp(min as f32, max as f32),
            Self::Sign => {
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => value.floor(),
            Self::Ceil => value.ceil(),
            Self::Round => value.round(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Tan => value.tan(),
            Self::Asin => value.asin(),
            Self::Acos => value.acos(),
            Self::Atan => value.atan(),
            Self::Sinh => value.sinh(),
            Self::Cosh => value.cosh(),
            Self::Asinh => value.asinh(),
            Self::Acosh => value.acosh(),
            Self::Atanh => value.atanh(),
            Self::Erf => erf_approx_f64(f64::from(value)) as f32,
            Self::Rsqrt => 1.0 / value.sqrt(),
            Self::Trunc => value.trunc(),
            Self::Frac => value.fract(),
            Self::TanhBackward => 1.0 - value * value,
            Self::SigmoidBackward => value * (1.0 - value),
            Self::EluBackward => {
                if value > 0.0 {
                    1.0
                } else {
                    value + 1.0
                }
            }
            Self::TanBackward => 1.0 + value.tan().powi(2),
            Self::AsinBackward => 1.0 / (1.0 - value * value).sqrt(),
            Self::AcosBackward => -1.0 / (1.0 - value * value).sqrt(),
            Self::AtanBackward => 1.0 / (1.0 + value * value),
            Self::AsinhBackward => 1.0 / (value * value + 1.0).sqrt(),
            Self::AcoshBackward => 1.0 / (value * value - 1.0).sqrt(),
            Self::AtanhBackward => 1.0 / (1.0 - value * value),
            Self::ErfBackward => (2.0 / core::f32::consts::PI.sqrt()) * (-value * value).exp(),
            Self::RsqrtBackward => -0.5 / (value * value.sqrt()),
            Self::GeluBackward => {
                let value_f64 = f64::from(value);
                let cdf = 0.5 * (1.0 + erf_approx_f64(value_f64 / core::f64::consts::SQRT_2));
                let pdf = (1.0 / (2.0 * core::f64::consts::PI).sqrt())
                    * (-value_f64 * value_f64 / 2.0).exp();
                (cdf + value_f64 * pdf) as f32
            }
            Self::MishBackward => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                let tanh = softplus.tanh();
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                tanh + value * sigmoid * (1.0 - tanh * tanh)
            }
        }
    }

    #[inline]
    pub(crate) fn eval_f64(self, value: f64) -> f64 {
        match self {
            Self::Neg => -value,
            Self::AddScalar(scalar) => value + scalar,
            Self::MulScalar(scalar) => value * scalar,
            Self::Relu => value.max(0.0),
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Mish => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                value * softplus.tanh()
            }
            Self::Elu => {
                if value > 0.0 {
                    value
                } else {
                    value.exp() - 1.0
                }
            }
            Self::Gelu => value * 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2)),
            Self::Abs => value.abs(),
            Self::Exp => value.exp(),
            Self::Sqrt => value.sqrt(),
            Self::Log => value.ln(),
            Self::Tanh => value.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Swish => value / (1.0 + (-value).exp()),
            Self::Powf(exp) => value.powf(exp),
            Self::Clamp(min, max) => value.clamp(min, max),
            Self::Sign => {
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => value.floor(),
            Self::Ceil => value.ceil(),
            Self::Round => value.round(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Tan => value.tan(),
            Self::Asin => value.asin(),
            Self::Acos => value.acos(),
            Self::Atan => value.atan(),
            Self::Sinh => value.sinh(),
            Self::Cosh => value.cosh(),
            Self::Asinh => value.asinh(),
            Self::Acosh => value.acosh(),
            Self::Atanh => value.atanh(),
            Self::Erf => erf_approx_f64(value),
            Self::Rsqrt => 1.0 / value.sqrt(),
            Self::Trunc => value.trunc(),
            Self::Frac => value.fract(),
            Self::TanhBackward => 1.0 - value * value,
            Self::SigmoidBackward => value * (1.0 - value),
            Self::EluBackward => {
                if value > 0.0 {
                    1.0
                } else {
                    value + 1.0
                }
            }
            Self::TanBackward => 1.0 + value.tan().powi(2),
            Self::AsinBackward => 1.0 / (1.0 - value * value).sqrt(),
            Self::AcosBackward => -1.0 / (1.0 - value * value).sqrt(),
            Self::AtanBackward => 1.0 / (1.0 + value * value),
            Self::AsinhBackward => 1.0 / (value * value + 1.0).sqrt(),
            Self::AcoshBackward => 1.0 / (value * value - 1.0).sqrt(),
            Self::AtanhBackward => 1.0 / (1.0 - value * value),
            Self::ErfBackward => (2.0 / core::f64::consts::PI.sqrt()) * (-value * value).exp(),
            Self::RsqrtBackward => -0.5 / (value * value.sqrt()),
            Self::GeluBackward => {
                let cdf = 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2));
                let pdf =
                    (1.0 / (2.0 * core::f64::consts::PI).sqrt()) * (-value * value / 2.0).exp();
                cdf + value * pdf
            }
            Self::MishBackward => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                let tanh = softplus.tanh();
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                tanh + value * sigmoid * (1.0 - tanh * tanh)
            }
        }
    }
}

impl BinaryOp {
    #[inline]
    pub(crate) fn eval_f32(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
        }
    }

    #[inline]
    pub(crate) fn eval_f64(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
        }
    }
}

/// Whether the AVX2 `f32` kernels may be called.
///
/// The decision lives here rather than inline at each call so that there is one
/// place to change and one thing to test. It was inline, and it read
/// `simd_lanes::<f32>() >= 8` alone — true only when the compiler was told to
/// assume AVX2, which a stock `cargo build` never is. See
/// `simd::avx2_detected`.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_f32_available() -> bool {
    const LANES: usize = crate::simd::simd_lanes::<f32>();
    LANES >= 8 || crate::simd::avx2_detected()
}

/// [`avx2_f32_available`] for `f64`, where an AVX2 register holds four lanes.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_f64_available() -> bool {
    const LANES: usize = crate::simd::simd_lanes::<f64>();
    LANES >= 4 || crate::simd::avx2_detected()
}

pub(crate) fn execute_binary(
    op: BinaryOp,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    output_shape: &[usize],
) -> Result<Option<CpuStorage>> {
    if lhs.buffer.dtype_id() != rhs.buffer.dtype_id() {
        return Err(Error::DTypeStorageMismatch {
            expected: lhs.buffer.descriptor(),
            got: rhs.buffer.descriptor(),
        });
    }
    if crate::cpu::validate_cpu_dtype(lhs.buffer.descriptor(), "pointwise").is_err() {
        return Ok(None);
    }

    let output: Option<Result<CpuBuffer>> = match (&*lhs.buffer, &*rhs.buffer) {
        (CpuBuffer::F32(lhs_values), CpuBuffer::F32(rhs_values)) => Some(
            execute_layout_f32(op, lhs, lhs_values, rhs, rhs_values, output_shape)
                .map(CpuBuffer::F32),
        ),
        (CpuBuffer::F64(lhs_values), CpuBuffer::F64(rhs_values)) => Some(
            execute_layout_f64(op, lhs, lhs_values, rhs, rhs_values, output_shape)
                .map(CpuBuffer::F64),
        ),
        (CpuBuffer::F16(lhs_values), CpuBuffer::F16(rhs_values)) => Some(
            execute_layout(lhs, lhs_values, rhs, rhs_values, output_shape, |a, b| {
                f16::from_f32(op.eval_f32(a.to_f32(), b.to_f32()))
            })
            .map(CpuBuffer::F16),
        ),
        (CpuBuffer::BF16(lhs_values), CpuBuffer::BF16(rhs_values)) => Some(
            execute_layout(lhs, lhs_values, rhs, rhs_values, output_shape, |a, b| {
                bf16::from_f32(op.eval_f32(a.to_f32(), b.to_f32()))
            })
            .map(CpuBuffer::BF16),
        ),
        _ => None,
    };

    Ok(output
        .transpose()?
        .map(|buffer| CpuStorage::from_contiguous(buffer, output_shape)))
}

fn execute_layout_f32(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f32],
    rhs: &CpuStorage,
    rhs_values: &[f32],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary_f32(
            op,
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
        ));
    }
    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_f32(
            op,
            &lhs_values[lhs_range],
            rhs_scalar,
            false,
        ));
    }
    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_f32(op, &rhs_values[rhs_range], lhs_scalar, true));
    }
    execute_strided_f32(op, lhs, lhs_values, rhs, rhs_values, output_shape)
}

fn execute_layout_f64(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f64],
    rhs: &CpuStorage,
    rhs_values: &[f64],
    output_shape: &[usize],
) -> Result<Vec<f64>> {
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary_f64(
            op,
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
        ));
    }
    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_f64(
            op,
            &lhs_values[lhs_range],
            rhs_scalar,
            false,
        ));
    }
    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_f64(op, &rhs_values[rhs_range], lhs_scalar, true));
    }
    execute_strided_f64(op, lhs, lhs_values, rhs, rhs_values, output_shape)
}

fn execute_strided_f32(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f32],
    rhs: &CpuStorage,
    rhs_values: &[f32],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if let Some(output) = map_iteration_avx2_f32(op, lhs_values, rhs_values, &plan) {
        return Ok(output);
    }
    Ok(map_binary_strided(
        lhs_values,
        rhs_values,
        &plan,
        &|lhs, rhs| op.eval_f32(lhs, rhs),
    ))
}

fn execute_strided_f64(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f64],
    rhs: &CpuStorage,
    rhs_values: &[f64],
    output_shape: &[usize],
) -> Result<Vec<f64>> {
    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if let Some(output) = map_iteration_avx2_f64(op, lhs_values, rhs_values, &plan) {
        return Ok(output);
    }
    Ok(map_binary_strided(
        lhs_values,
        rhs_values,
        &plan,
        &|lhs, rhs| op.eval_f64(lhs, rhs),
    ))
}

pub(crate) fn execute_unary(op: UnaryOp, input: &CpuStorage) -> Result<Option<CpuStorage>> {
    if crate::cpu::validate_cpu_dtype(input.buffer.descriptor(), "pointwise").is_err() {
        return Ok(None);
    }
    let output: Option<Result<CpuBuffer>> = match &*input.buffer {
        CpuBuffer::F32(values) => Some(
            execute_unary_layout(input, values, |value| op.eval_f32(value)).map(CpuBuffer::F32),
        ),
        CpuBuffer::F64(values) => Some(
            execute_unary_layout(input, values, |value| op.eval_f64(value)).map(CpuBuffer::F64),
        ),
        CpuBuffer::F16(values) => Some(
            execute_unary_layout(input, values, |value| {
                f16::from_f32(op.eval_f32(value.to_f32()))
            })
            .map(CpuBuffer::F16),
        ),
        CpuBuffer::BF16(values) => Some(
            execute_unary_layout(input, values, |value| {
                bf16::from_f32(op.eval_f32(value.to_f32()))
            })
            .map(CpuBuffer::BF16),
        ),
        _ => None,
    };

    Ok(output
        .transpose()?
        .map(|buffer| CpuStorage::from_contiguous(buffer, &input.shape)))
}

fn execute_unary_layout<T>(
    input: &CpuStorage,
    values: &[T],
    op: impl Fn(T) -> T + Send + Sync,
) -> Result<Vec<T>>
where
    T: TypedKernel,
{
    if let Some(range) = dense_range(input, values.len(), &input.shape) {
        return Ok(map_unary(&values[range], &op));
    }

    let plan = UnaryIterationPlan::new(OperandLayout {
        shape: &input.shape,
        strides: &input.strides,
        offset: input.offset_elements,
    })?;
    validate_bounds(&plan.operand, &plan.output_shape, values.len())?;
    Ok(map_unary_strided(values, &plan, &op))
}

fn execute_layout<T>(
    lhs: &CpuStorage,
    lhs_values: &[T],
    rhs: &CpuStorage,
    rhs_values: &[T],
    output_shape: &[usize],
    op: impl Fn(T, T) -> T + Send + Sync,
) -> Result<Vec<T>>
where
    T: TypedKernel,
{
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary(
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
            &op,
        ));
    }

    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_right(&lhs_values[lhs_range], rhs_scalar, &op));
    }

    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_left(lhs_scalar, &rhs_values[rhs_range], &op));
    }

    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    Ok(map_binary_strided(lhs_values, rhs_values, &plan, &op))
}

fn binary_iteration_plan(
    lhs: &CpuStorage,
    lhs_len: usize,
    rhs: &CpuStorage,
    rhs_len: usize,
    output_shape: &[usize],
) -> Result<IterationPlan> {
    let plan = IterationPlan::binary(
        OperandLayout {
            shape: &lhs.shape,
            strides: &lhs.strides,
            offset: lhs.offset_elements,
        },
        OperandLayout {
            shape: &rhs.shape,
            strides: &rhs.strides,
            offset: rhs.offset_elements,
        },
        output_shape,
    )?;
    validate_bounds(&plan.operands[0], &plan.output_shape, lhs_len)?;
    validate_bounds(&plan.operands[1], &plan.output_shape, rhs_len)?;
    Ok(plan)
}

fn map_binary<T: TypedKernel, F>(lhs: &[T], rhs: &[T], op: &F) -> Vec<T>
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
fn map_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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
fn map_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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
fn map_scalar_f32(op: BinaryOp, dense: &[f32], scalar: f32, scalar_left: bool) -> Vec<f32> {
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
fn map_scalar_f64(op: BinaryOp, dense: &[f64], scalar: f64, scalar_left: bool) -> Vec<f64> {
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

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn parallel_avx2_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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
fn parallel_avx2_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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
fn parallel_avx2_scalar_f32(
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
fn parallel_avx2_scalar_f64(
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
unsafe fn avx2_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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
unsafe fn avx2_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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
unsafe fn avx2_scalar_f32(op: BinaryOp, dense: &[f32], scalar: f32, scalar_left: bool) -> Vec<f32> {
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
unsafe fn avx2_scalar_f64(op: BinaryOp, dense: &[f64], scalar: f64, scalar_left: bool) -> Vec<f64> {
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
fn parallel_neon_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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
fn parallel_neon_scalar_f32(
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
fn parallel_neon_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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
fn parallel_neon_scalar_f64(
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

// ======================= WASM SIMD128 (wasm32) =======================

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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
unsafe fn wasm_scalar_f32(op: BinaryOp, dense: &[f32], scalar: f32, scalar_left: bool) -> Vec<f32> {
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
unsafe fn wasm_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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
unsafe fn wasm_scalar_f64(op: BinaryOp, dense: &[f64], scalar: f64, scalar_left: bool) -> Vec<f64> {
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

#[cfg(all(feature = "std", target_arch = "x86_64"))]
macro_rules! define_avx2_iteration_kernel {
    ($function:ident, $element:ty, $binary_writer:ident, $scalar_writer:ident) => {
        fn $function(
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

fn map_unary<T: TypedKernel, F>(input: &[T], op: &F) -> Vec<T>
where
    F: Fn(T) -> T + Send + Sync,
{
    map_unary_typed(input, op)
}

fn map_binary_strided<T, F>(lhs: &[T], rhs: &[T], plan: &IterationPlan, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if plan.numel < PARALLEL_GRAIN {
        return map_binary_strided_serial(lhs, rhs, plan, op);
    }
    let evaluate = |flat_index| {
        let lhs_index = plan.operands[0].physical_index(flat_index, &plan.output_shape);
        let rhs_index = plan.operands[1].physical_index(flat_index, &plan.output_shape);
        op(lhs[lhs_index], rhs[rhs_index])
    };
    (0..plan.numel).into_par_iter().map(evaluate).collect()
}

fn map_binary_strided_serial<T, F>(lhs: &[T], rhs: &[T], plan: &IterationPlan, op: &F) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let mut output = Vec::with_capacity(plan.numel);
    if plan.numel == 0 {
        return output;
    }
    if plan.output_shape.is_empty() {
        output.push(op(
            lhs[plan.operands[0].offset],
            rhs[plan.operands[1].offset],
        ));
        return output;
    }

    let outer_rank = plan.output_shape.len() - 1;
    let inner_len = plan.output_shape[outer_rank];
    let outer_count = plan.numel / inner_len;
    let mut coordinates = vec![0usize; outer_rank];
    let mut lhs_index = plan.operands[0].offset;
    let mut rhs_index = plan.operands[1].offset;
    let lhs_inner_stride = plan.operands[0].strides[outer_rank];
    let rhs_inner_stride = plan.operands[1].strides[outer_rank];

    for outer_index in 0..outer_count {
        for inner_index in 0..inner_len {
            output.push(op(
                lhs[lhs_index + inner_index * lhs_inner_stride],
                rhs[rhs_index + inner_index * rhs_inner_stride],
            ));
        }
        if outer_index + 1 == outer_count {
            break;
        }
        advance_binary(
            &mut coordinates,
            &plan.output_shape[..outer_rank],
            &mut lhs_index,
            &plan.operands[0].strides[..outer_rank],
            &mut rhs_index,
            &plan.operands[1].strides[..outer_rank],
        );
    }
    output
}

fn map_unary_strided<T, F>(input: &[T], plan: &UnaryIterationPlan, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    if plan.numel < PARALLEL_GRAIN {
        return map_unary_strided_serial(input, plan, op);
    }
    let evaluate = |flat_index| {
        let input_index = plan.operand.physical_index(flat_index, &plan.output_shape);
        op(input[input_index])
    };
    (0..plan.numel).into_par_iter().map(evaluate).collect()
}

fn map_unary_strided_serial<T, F>(input: &[T], plan: &UnaryIterationPlan, op: &F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    let mut output = Vec::with_capacity(plan.numel);
    if plan.numel == 0 {
        return output;
    }
    if plan.output_shape.is_empty() {
        output.push(op(input[plan.operand.offset]));
        return output;
    }

    let outer_rank = plan.output_shape.len() - 1;
    let inner_len = plan.output_shape[outer_rank];
    let outer_count = plan.numel / inner_len;
    let mut coordinates = vec![0usize; outer_rank];
    let mut input_index = plan.operand.offset;
    let inner_stride = plan.operand.strides[outer_rank];

    for outer_index in 0..outer_count {
        for inner_index in 0..inner_len {
            output.push(op(input[input_index + inner_index * inner_stride]));
        }
        if outer_index + 1 == outer_count {
            break;
        }
        advance_unary(
            &mut coordinates,
            &plan.output_shape[..outer_rank],
            &mut input_index,
            &plan.operand.strides[..outer_rank],
        );
    }
    output
}

fn advance_binary(
    coordinates: &mut [usize],
    shape: &[usize],
    lhs_index: &mut usize,
    lhs_strides: &[usize],
    rhs_index: &mut usize,
    rhs_strides: &[usize],
) {
    for axis in (0..shape.len()).rev() {
        if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1;
            *lhs_index += lhs_strides[axis];
            *rhs_index += rhs_strides[axis];
            return;
        }
        coordinates[axis] = 0;
        *lhs_index -= lhs_strides[axis] * (shape[axis] - 1);
        *rhs_index -= rhs_strides[axis] * (shape[axis] - 1);
    }
}

fn advance_unary(
    coordinates: &mut [usize],
    shape: &[usize],
    input_index: &mut usize,
    strides: &[usize],
) {
    for axis in (0..shape.len()).rev() {
        if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1;
            *input_index += strides[axis];
            return;
        }
        coordinates[axis] = 0;
        *input_index -= strides[axis] * (shape[axis] - 1);
    }
}

fn map_scalar_right<T, F>(lhs: &[T], rhs: T, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if lhs.len() < PARALLEL_GRAIN {
        lhs.iter().map(|&lhs| op(lhs, rhs)).collect()
    } else {
        lhs.par_iter().map(|&lhs| op(lhs, rhs)).collect()
    }
}

fn map_scalar_left<T, F>(lhs: T, rhs: &[T], op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if rhs.len() < PARALLEL_GRAIN {
        rhs.iter().map(|&rhs| op(lhs, rhs)).collect()
    } else {
        rhs.par_iter().map(|&rhs| op(lhs, rhs)).collect()
    }
}

pub(crate) fn dense_range(
    storage: &CpuStorage,
    buffer_len: usize,
    output_shape: &[usize],
) -> Option<Range<usize>> {
    if storage.shape.dims() != output_shape
        || !stride::is_contiguous(&storage.shape, &storage.strides)
    {
        return None;
    }
    let end = storage
        .offset_elements
        .checked_add(try_numel(output_shape)?)?;
    (end <= buffer_len).then_some(storage.offset_elements..end)
}

fn scalar_value<T: Copy>(storage: &CpuStorage, values: &[T]) -> Option<T> {
    if try_numel(&storage.shape)? != 1 {
        return None;
    }
    values.get(storage.offset_elements).copied()
}

fn validate_bounds(
    operand: &OperandIteration,
    output_shape: &[usize],
    buffer_len: usize,
) -> Result<()> {
    if let Some(max_index) = operand.max_physical_index(output_shape)?
        && max_index >= buffer_len
    {
        return Err(Error::Msg(format!(
            "iteration plan accesses storage index {max_index}, but buffer length is {buffer_len}"
        )));
    }
    Ok(())
}

/// The element count of `shape`, or `None` on overflow.
///
/// This is a fast-path check for two `Option`-returning callers that treat
/// overflow as "this shortcut does not apply" rather than as an error to
/// report; [`crate::bytes::checked_numel`] is the crate's answer to the
/// question "what is this shape's element count, and is it representable".
fn try_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |numel, &dim| numel.checked_mul(dim))
}

fn erf_approx_f64(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let value = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * value);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-value * value).exp())
}

#[cfg(test)]
mod tests {
    /// The AVX2 kernels must be reachable in the build users actually get.
    ///
    /// `avx2_f32_available` was inline at four call sites and read
    /// `simd_lanes::<f32>() >= 8` alone. That constant is false in a stock
    /// `cargo build`, so all four branches were dead code and the CPU backend
    /// fell through to a scalar loop — a 9x difference on `add_f32/65536`,
    /// invisible to every test because the kernels themselves stayed correct.
    ///
    /// This is the assertion that would have caught it. It fails if the gate is
    /// ever narrowed back to a compile-time-only condition.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn the_avx2_gate_opens_on_a_machine_that_supports_avx2() {
        // Deliberately the raw macro and not `simd::avx2_detected`: the point is
        // to ask the hardware independently of the predicate under test. Routing
        // this through the same predicate would make the assertion compare a
        // value with itself and always pass.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        assert!(
            avx2_f32_available(),
            "this machine supports AVX2 but the f32 kernel gate is closed"
        );
        assert!(
            avx2_f64_available(),
            "this machine supports AVX2 but the f64 kernel gate is closed"
        );
    }

    use super::*;

    #[test]
    fn f32_contiguous_add_stays_typed() {
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![3.0, 4.0]), vec![2]);
        let output = execute_binary(BinaryOp::Add, &lhs, &rhs, &[2])
            .unwrap()
            .unwrap();

        assert_eq!(&*output.buffer, &CpuBuffer::F32(vec![4.0, 6.0]));
    }

    #[test]
    fn f64_contiguous_math_keeps_f64_precision() {
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0 + f64::EPSILON]), vec![1]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0]), vec![1]);
        let output = execute_binary(BinaryOp::Sub, &lhs, &rhs, &[1])
            .unwrap()
            .unwrap();

        assert_eq!(&*output.buffer, &CpuBuffer::F64(vec![f64::EPSILON]));
    }

    #[test]
    fn vector_kernels_handle_odd_scalar_tails_for_every_operation() {
        let lhs_f32: Vec<f32> = (1..=19).map(|value| value as f32).collect();
        let rhs_f32: Vec<f32> = (1..=19).map(|value| value as f32 * 0.5).collect();
        let lhs_f64: Vec<f64> = lhs_f32.iter().map(|&value| f64::from(value)).collect();
        let rhs_f64: Vec<f64> = rhs_f32.iter().map(|&value| f64::from(value)).collect();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let actual_f32 = map_binary_f32(op, &lhs_f32, &rhs_f32);
            let expected_f32: Vec<_> = lhs_f32
                .iter()
                .zip(&rhs_f32)
                .map(|(&lhs, &rhs)| op.eval_f32(lhs, rhs))
                .collect();
            assert_eq!(actual_f32, expected_f32);

            let actual_f64 = map_binary_f64(op, &lhs_f64, &rhs_f64);
            let expected_f64: Vec<_> = lhs_f64
                .iter()
                .zip(&rhs_f64)
                .map(|(&lhs, &rhs)| op.eval_f64(lhs, rhs))
                .collect();
            assert_eq!(actual_f64, expected_f64);
        }
    }

    #[test]
    fn scalar_vector_kernels_preserve_order_and_handle_tails() {
        let dense_f32: Vec<f32> = (1..=19).map(|value| value as f32).collect();
        let dense_f64: Vec<f64> = dense_f32.iter().map(|&value| f64::from(value)).collect();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            for scalar_left in [false, true] {
                let actual_f32 = map_scalar_f32(op, &dense_f32, 3.25, scalar_left);
                let expected_f32: Vec<_> = dense_f32
                    .iter()
                    .map(|&dense| {
                        if scalar_left {
                            op.eval_f32(3.25, dense)
                        } else {
                            op.eval_f32(dense, 3.25)
                        }
                    })
                    .collect();
                assert_eq!(actual_f32, expected_f32);

                let actual_f64 = map_scalar_f64(op, &dense_f64, 3.25, scalar_left);
                let expected_f64: Vec<_> = dense_f64
                    .iter()
                    .map(|&dense| {
                        if scalar_left {
                            op.eval_f64(3.25, dense)
                        } else {
                            op.eval_f64(dense, 3.25)
                        }
                    })
                    .collect();
                assert_eq!(actual_f64, expected_f64);
            }
        }
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    // Sized to exceed SIMD_PARALLEL_CHUNK on purpose, which is also what puts it
    // out of reach of the interpreter: miri needs hours for a single crossing.
    // The soundness gate splits the two jobs rather than losing one of them -
    // miri proves the aliasing rules on the small cases, and AddressSanitizer
    // runs this one at native speed, where the chunk count is what matters.
    // See tools/soundness.sh.
    #[cfg_attr(miri, ignore)]
    fn parallel_vector_chunks_preserve_operations_and_tails() {
        if !avx2_f32_available() {
            return;
        }

        let len = SIMD_PARALLEL_CHUNK + 3;
        let lhs_f32: Vec<f32> = (1..=len).map(|value| value as f32).collect();
        let rhs_f32: Vec<f32> = (1..=len).map(|value| value as f32 * 0.5 + 1.0).collect();
        let lhs_f64: Vec<f64> = lhs_f32.iter().map(|&value| f64::from(value)).collect();
        let rhs_f64: Vec<f64> = rhs_f32.iter().map(|&value| f64::from(value)).collect();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let actual_f32 = parallel_avx2_binary_f32(op, &lhs_f32, &rhs_f32);
            let expected_f32: Vec<_> = lhs_f32
                .iter()
                .zip(&rhs_f32)
                .map(|(&lhs, &rhs)| op.eval_f32(lhs, rhs))
                .collect();
            assert_eq!(actual_f32, expected_f32);

            let actual_f64 = parallel_avx2_binary_f64(op, &lhs_f64, &rhs_f64);
            let expected_f64: Vec<_> = lhs_f64
                .iter()
                .zip(&rhs_f64)
                .map(|(&lhs, &rhs)| op.eval_f64(lhs, rhs))
                .collect();
            assert_eq!(actual_f64, expected_f64);

            for scalar_left in [false, true] {
                let actual_f32 = parallel_avx2_scalar_f32(op, &lhs_f32, 3.25, scalar_left);
                let expected_f32: Vec<_> = lhs_f32
                    .iter()
                    .map(|&dense| {
                        if scalar_left {
                            op.eval_f32(3.25, dense)
                        } else {
                            op.eval_f32(dense, 3.25)
                        }
                    })
                    .collect();
                assert_eq!(actual_f32, expected_f32);

                let actual_f64 = parallel_avx2_scalar_f64(op, &lhs_f64, 3.25, scalar_left);
                let expected_f64: Vec<_> = lhs_f64
                    .iter()
                    .map(|&dense| {
                        if scalar_left {
                            op.eval_f64(3.25, dense)
                        } else {
                            op.eval_f64(dense, 3.25)
                        }
                    })
                    .collect();
                assert_eq!(actual_f64, expected_f64);
            }
        }
    }

    #[test]
    fn half_storage_uses_f32_compute() {
        let lhs = CpuStorage::from_contiguous(
            CpuBuffer::F16(vec![f16::from_f32(1.5), f16::from_f32(2.0)]),
            vec![2],
        );
        let rhs = CpuStorage::from_contiguous(
            CpuBuffer::F16(vec![f16::from_f32(2.0), f16::from_f32(4.0)]),
            vec![2],
        );
        let output = execute_binary(BinaryOp::Mul, &lhs, &rhs, &[2])
            .unwrap()
            .unwrap();

        assert_eq!(
            &*output.buffer,
            &CpuBuffer::F16(vec![f16::from_f32(3.0), f16::from_f32(8.0)])
        );
    }

    #[test]
    fn scalar_broadcast_preserves_operand_order() {
        let scalar = CpuStorage::from_contiguous(CpuBuffer::F32(vec![10.0]), vec![]);
        let dense = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);

        let left = execute_binary(BinaryOp::Sub, &scalar, &dense, &[2])
            .unwrap()
            .unwrap();
        let right = execute_binary(BinaryOp::Sub, &dense, &scalar, &[2])
            .unwrap()
            .unwrap();

        assert_eq!(&*left.buffer, &CpuBuffer::F32(vec![9.0, 8.0]));
        assert_eq!(&*right.buffer, &CpuBuffer::F32(vec![-9.0, -8.0]));
    }

    #[test]
    fn unary_family_uses_native_float_compute() {
        let f32_input = CpuStorage::from_contiguous(CpuBuffer::F32(vec![-1.0, 0.0, 2.0]), vec![3]);
        let f32_output = execute_unary(UnaryOp::Relu, &f32_input).unwrap().unwrap();
        assert_eq!(&*f32_output.buffer, &CpuBuffer::F32(vec![0.0, 0.0, 2.0]));

        let f64_input = CpuStorage::from_contiguous(CpuBuffer::F64(vec![0.0, 1.0]), vec![2]);
        let f64_output = execute_unary(UnaryOp::Exp, &f64_input).unwrap().unwrap();
        assert_eq!(
            &*f64_output.buffer,
            &CpuBuffer::F64(vec![1.0, core::f64::consts::E])
        );
    }

    #[test]
    fn general_broadcast_uses_typed_strided_kernel() {
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2, 1]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);

        let output = execute_binary(BinaryOp::Add, &lhs, &rhs, &[2, 3])
            .unwrap()
            .unwrap();
        assert_eq!(
            &*output.buffer,
            &CpuBuffer::F32(vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
        );
    }

    #[test]
    fn dense_broadcast_vector_projection_preserves_order_and_odd_tails() {
        let rows_f32 = vec![2.0, 4.0, 8.0];
        let columns_f32: Vec<f32> = (1..=19).map(|value| value as f32 * 0.5).collect();
        let rows_f64: Vec<f64> = rows_f32.iter().map(|&value| f64::from(value)).collect();
        let columns_f64: Vec<f64> = columns_f32.iter().map(|&value| f64::from(value)).collect();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            for reverse in [false, true] {
                let lhs_f32 = if reverse {
                    CpuStorage::from_contiguous(CpuBuffer::F32(columns_f32.clone()), vec![1, 19])
                } else {
                    CpuStorage::from_contiguous(CpuBuffer::F32(rows_f32.clone()), vec![3, 1])
                };
                let rhs_f32 = if reverse {
                    CpuStorage::from_contiguous(CpuBuffer::F32(rows_f32.clone()), vec![3, 1])
                } else {
                    CpuStorage::from_contiguous(CpuBuffer::F32(columns_f32.clone()), vec![1, 19])
                };
                let actual_f32 = execute_binary(op, &lhs_f32, &rhs_f32, &[3, 19])
                    .unwrap()
                    .unwrap();
                let expected_f32: Vec<_> = rows_f32
                    .iter()
                    .flat_map(|&row| {
                        columns_f32.iter().map(move |&column| {
                            if reverse {
                                op.eval_f32(column, row)
                            } else {
                                op.eval_f32(row, column)
                            }
                        })
                    })
                    .collect();
                assert_eq!(&*actual_f32.buffer, &CpuBuffer::F32(expected_f32));

                let lhs_f64 = if reverse {
                    CpuStorage::from_contiguous(CpuBuffer::F64(columns_f64.clone()), vec![1, 19])
                } else {
                    CpuStorage::from_contiguous(CpuBuffer::F64(rows_f64.clone()), vec![3, 1])
                };
                let rhs_f64 = if reverse {
                    CpuStorage::from_contiguous(CpuBuffer::F64(rows_f64.clone()), vec![3, 1])
                } else {
                    CpuStorage::from_contiguous(CpuBuffer::F64(columns_f64.clone()), vec![1, 19])
                };
                let actual_f64 = execute_binary(op, &lhs_f64, &rhs_f64, &[3, 19])
                    .unwrap()
                    .unwrap();
                let expected_f64: Vec<_> = rows_f64
                    .iter()
                    .flat_map(|&row| {
                        columns_f64.iter().map(move |&column| {
                            if reverse {
                                op.eval_f64(column, row)
                            } else {
                                op.eval_f64(row, column)
                            }
                        })
                    })
                    .collect();
                assert_eq!(&*actual_f64.buffer, &CpuBuffer::F64(expected_f64));
            }
        }
    }

    #[test]
    // 1025 * 257 elements clears PARALLEL_GRAIN deliberately. Same reasoning as
    // parallel_vector_chunks_preserve_operations_and_tails above.
    #[cfg_attr(miri, ignore)]
    fn parallel_dense_broadcast_projection_crosses_chunk_boundaries() {
        let rows = 1_025;
        let columns = 257;
        let row_values: Vec<f32> = (0..rows).map(|value| value as f32).collect();
        let column_values: Vec<f32> = (0..columns).map(|value| value as f32 * 0.25).collect();
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(row_values.clone()), vec![rows, 1]);
        let rhs =
            CpuStorage::from_contiguous(CpuBuffer::F32(column_values.clone()), vec![1, columns]);

        let output = execute_binary(BinaryOp::Sub, &lhs, &rhs, &[rows, columns])
            .unwrap()
            .unwrap();
        let CpuBuffer::F32(values) = &*output.buffer else {
            panic!("expected F32 output");
        };
        for &(row, column) in &[
            (0, 0),
            (0, columns - 1),
            (PARALLEL_GRAIN / columns, PARALLEL_GRAIN % columns),
            (rows - 1, columns - 1),
        ] {
            assert_eq!(
                values[row * columns + column],
                row_values[row] - column_values[column]
            );
        }
    }

    #[test]
    fn broadcast_strided_fast_path_matches_scalar_reference() {
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            // Case 1: [B, T, 1] vs [B, T, C] (layer_norm shape, C not multiple of 8)
            let (b, t, c) = (2, 3, 11);
            let btc_data: Vec<f32> = (1..=b * t * c).map(|x| x as f32 * 0.5 + 1.0).collect();
            let bt1_data: Vec<f32> = (1..=b * t).map(|x| x as f32 * 1.5 + 2.0).collect();

            let full = CpuStorage::from_contiguous(CpuBuffer::F32(btc_data.clone()), vec![b, t, c]);
            let broadcast =
                CpuStorage::from_contiguous(CpuBuffer::F32(bt1_data.clone()), vec![b, t, 1]);

            // Broadcast on RHS: full (op) broadcast
            let out_rhs = execute_binary(op, &full, &broadcast, &[b, t, c])
                .unwrap()
                .unwrap();
            let CpuBuffer::F32(ref vals_rhs) = *out_rhs.buffer else {
                panic!("expected F32 output");
            };
            for bi in 0..b {
                for ti in 0..t {
                    for ci in 0..c {
                        let full_idx = bi * t * c + ti * c + ci;
                        let bcast_idx = bi * t + ti;
                        let expected = op.eval_f32(btc_data[full_idx], bt1_data[bcast_idx]);
                        assert!(
                            (vals_rhs[full_idx] - expected).abs() < 1e-6,
                            "mismatch at b={bi}, t={ti}, c={ci} for op {op:?}"
                        );
                    }
                }
            }

            // Broadcast on LHS: broadcast (op) full
            let out_lhs = execute_binary(op, &broadcast, &full, &[b, t, c])
                .unwrap()
                .unwrap();
            let CpuBuffer::F32(ref vals_lhs) = *out_lhs.buffer else {
                panic!("expected F32 output");
            };
            for bi in 0..b {
                for ti in 0..t {
                    for ci in 0..c {
                        let full_idx = bi * t * c + ti * c + ci;
                        let bcast_idx = bi * t + ti;
                        let expected = op.eval_f32(bt1_data[bcast_idx], btc_data[full_idx]);
                        assert!(
                            (vals_lhs[full_idx] - expected).abs() < 1e-6,
                            "mismatch at b={bi}, t={ti}, c={ci} for op {op:?}"
                        );
                    }
                }
            }

            // Case 2: [C] vs [B, C] (bias-add shape, C not multiple of 8)
            let (b, c) = (4, 13);
            let bc_data: Vec<f32> = (1..=b * c).map(|x| x as f32 * 0.75 + 1.0).collect();
            let c_data: Vec<f32> = (1..=c).map(|x| x as f32 * 2.0 + 3.0).collect();

            let full_bc = CpuStorage::from_contiguous(CpuBuffer::F32(bc_data.clone()), vec![b, c]);
            let bcast_c = CpuStorage::from_contiguous(CpuBuffer::F32(c_data.clone()), vec![c]);

            // Broadcast on RHS: [B, C] (op) [C]
            let out_bias_rhs = execute_binary(op, &full_bc, &bcast_c, &[b, c])
                .unwrap()
                .unwrap();
            let CpuBuffer::F32(ref vals_bias_rhs) = *out_bias_rhs.buffer else {
                panic!("expected F32 output");
            };
            for bi in 0..b {
                for (ci, &c_val) in c_data.iter().enumerate().take(c) {
                    let full_idx = bi * c + ci;
                    let expected = op.eval_f32(bc_data[full_idx], c_val);
                    assert!(
                        (vals_bias_rhs[full_idx] - expected).abs() < 1e-6,
                        "mismatch at b={bi}, c={ci} for op {op:?}"
                    );
                }
            }

            // Broadcast on LHS: [C] (op) [B, C]
            let out_bias_lhs = execute_binary(op, &bcast_c, &full_bc, &[b, c])
                .unwrap()
                .unwrap();
            let CpuBuffer::F32(ref vals_bias_lhs) = *out_bias_lhs.buffer else {
                panic!("expected F32 output");
            };
            for bi in 0..b {
                for (ci, &c_val) in c_data.iter().enumerate().take(c) {
                    let full_idx = bi * c + ci;
                    let expected = op.eval_f32(c_val, bc_data[full_idx]);
                    assert!(
                        (vals_bias_lhs[full_idx] - expected).abs() < 1e-6,
                        "mismatch at b={bi}, c={ci} for op {op:?}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    #[ignore = "microbenchmark: run explicitly with --release --ignored --nocapture"]
    fn benchmark_cpu_binary_kernels() {
        use std::hint::black_box;
        use std::time::Instant;

        println!(
            "execution,layout,dtype,elements,iterations,samples,median_ns_per_element,median_effective_gib_s"
        );
        for &(elements, iterations) in &[
            (1_024usize, 4_000usize),
            (4_096, 2_000),
            (16_384, 500),
            (65_536, 200),
            (262_144, 50),
            (1_048_576, 20),
            (2_097_152, 10),
            (4_194_304, 8),
        ] {
            let lhs =
                CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.25; elements]), vec![elements]);
            let rhs =
                CpuStorage::from_contiguous(CpuBuffer::F32(vec![2.5; elements]), vec![elements]);
            benchmark_case("contiguous", elements, iterations, || {
                execute_binary(BinaryOp::Add, black_box(&lhs), black_box(&rhs), &[elements])
            });

            let scalar = CpuStorage::from_contiguous(CpuBuffer::F32(vec![2.5]), vec![]);
            benchmark_case("scalar_broadcast", elements, iterations, || {
                execute_binary(
                    BinaryOp::Add,
                    black_box(&lhs),
                    black_box(&scalar),
                    &[elements],
                )
            });
            if elements >= DENSE_PARALLEL_GRAIN {
                benchmark_case("contiguous_rayon_reference", elements, iterations, || {
                    let values = map_binary(f32_values(&lhs), f32_values(&rhs), &|a, b| a + b);
                    Ok(Some(CpuStorage::from_contiguous(
                        CpuBuffer::F32(values),
                        vec![elements],
                    )))
                });
                benchmark_case("scalar_rayon_reference", elements, iterations, || {
                    let values = map_scalar_right(f32_values(&lhs), 2.5, &|a, b| a + b);
                    Ok(Some(CpuStorage::from_contiguous(
                        CpuBuffer::F32(values),
                        vec![elements],
                    )))
                });
            }

            let columns = 256;
            let rows = elements / columns;
            let row_values: Vec<f32> = (0..rows).map(|value| value as f32).collect();
            let column_values: Vec<f32> = (0..columns).map(|value| value as f32).collect();
            let rows_storage =
                CpuStorage::from_contiguous(CpuBuffer::F32(row_values), vec![rows, 1]);
            let columns_storage =
                CpuStorage::from_contiguous(CpuBuffer::F32(column_values), vec![1, columns]);
            benchmark_case("dense_broadcast", elements, iterations, || {
                execute_binary(
                    BinaryOp::Add,
                    black_box(&rows_storage),
                    black_box(&columns_storage),
                    &[rows, columns],
                )
            });
            let broadcast_plan = binary_iteration_plan(
                &rows_storage,
                rows,
                &columns_storage,
                columns,
                &[rows, columns],
            )
            .unwrap();
            benchmark_case(
                "dense_broadcast_odometer_reference",
                elements,
                iterations,
                || {
                    let values = map_binary_strided(
                        f32_values(&rows_storage),
                        f32_values(&columns_storage),
                        &broadcast_plan,
                        &|lhs, rhs| lhs + rhs,
                    );
                    Ok(Some(CpuStorage::from_contiguous(
                        CpuBuffer::F32(values),
                        vec![rows, columns],
                    )))
                },
            );
        }

        fn benchmark_case(
            layout: &str,
            elements: usize,
            iterations: usize,
            mut operation: impl FnMut() -> Result<Option<CpuStorage>>,
        ) {
            for _ in 0..5 {
                black_box(operation().unwrap().unwrap());
            }
            const SAMPLES: usize = 7;
            let mut samples = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let started = Instant::now();
                for _ in 0..iterations {
                    black_box(operation().unwrap().unwrap());
                }
                samples.push(started.elapsed().as_secs_f64());
            }
            samples.sort_by(f64::total_cmp);
            let elapsed = samples[SAMPLES / 2];
            let ns_per_element = elapsed * 1e9 / (elements * iterations) as f64;
            let bytes = (elements * iterations * 3 * size_of::<f32>()) as f64;
            let effective_gib_s = bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
            println!(
                "{},{layout},f32,{elements},{iterations},{SAMPLES},{ns_per_element:.4},{effective_gib_s:.3}",
                selected_execution(layout, elements)
            );
        }

        fn selected_execution(layout: &str, elements: usize) -> &'static str {
            if layout.ends_with("_rayon_reference") {
                return "rayon_autovec";
            }
            if layout == "dense_broadcast" {
                #[cfg(all(feature = "std", target_arch = "x86_64"))]
                if avx2_f32_available() {
                    return if elements >= PARALLEL_GRAIN {
                        "rayon_avx2_broadcast"
                    } else {
                        "avx2_broadcast"
                    };
                }
                return if elements >= PARALLEL_GRAIN {
                    "rayon_iterator"
                } else {
                    "serial_odometer"
                };
            }
            if layout == "dense_broadcast_odometer_reference" {
                return if elements >= PARALLEL_GRAIN {
                    "rayon_iterator"
                } else {
                    "serial_odometer"
                };
            }
            if elements >= DENSE_PARALLEL_GRAIN {
                #[cfg(all(feature = "std", target_arch = "x86_64"))]
                if avx2_f32_available() {
                    return "rayon_avx2";
                }
                return "rayon";
            }
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            if avx2_f32_available() {
                return "avx2";
            }
            "scalar"
        }

        fn f32_values(storage: &CpuStorage) -> &[f32] {
            match &*storage.buffer {
                CpuBuffer::F32(values) => values,
                _ => unreachable!("benchmark storage is F32"),
            }
        }
    }
}
