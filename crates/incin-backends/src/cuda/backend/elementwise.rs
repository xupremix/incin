//! Elementwise CUDA operations: tape-tracked binary and unary kernels,
//! activations, and scalar arithmetic.

#![allow(dead_code)]

use super::*;

pub(crate) fn cuda_add_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out =
        crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", lhs, rhs, &out_shape)?;
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            Ok(vec![
                crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)?,
                crate::cuda::tape::unbroadcast(grad_out, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

pub(crate) fn cuda_sub_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out =
        crate::cuda::ops::elementwise::launch_binary_op("sub", "a - b", lhs, rhs, &out_shape)?;
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let neg_grad = crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", grad_out)?;
            Ok(vec![
                crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)?,
                crate::cuda::tape::unbroadcast(&neg_grad, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

pub(crate) fn cuda_mul_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out =
        crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", lhs, rhs, &out_shape)?;
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let grad_lhs_shape =
                crate::layout::broadcast_shape(&grad_out.shape, &rhs_capture.shape)?;
            let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &rhs_capture,
                &grad_lhs_shape,
            )?;
            let grad_rhs_shape =
                crate::layout::broadcast_shape(&grad_out.shape, &lhs_capture.shape)?;
            let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &lhs_capture,
                &grad_rhs_shape,
            )?;
            Ok(vec![
                crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

pub(crate) fn cuda_div_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out =
        crate::cuda::ops::elementwise::launch_binary_op("div", "a / b", lhs, rhs, &out_shape)?;
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let grad_lhs_shape =
                crate::layout::broadcast_shape(&grad_out.shape, &rhs_capture.shape)?;
            let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                "div",
                "a / b",
                grad_out,
                &rhs_capture,
                &grad_lhs_shape,
            )?;
            let rhs_sq_shape =
                crate::layout::broadcast_shape(&rhs_capture.shape, &rhs_capture.shape)?;
            let rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &rhs_capture,
                &rhs_capture,
                &rhs_sq_shape,
            )?;
            let ratio_shape = crate::layout::broadcast_shape(&lhs_capture.shape, &rhs_sq.shape)?;
            let lhs_over_rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                "div",
                "a / b",
                &lhs_capture,
                &rhs_sq,
                &ratio_shape,
            )?;
            let neg_ratio =
                crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &lhs_over_rhs_sq)?;
            let grad_rhs_shape = crate::layout::broadcast_shape(&grad_out.shape, &neg_ratio.shape)?;
            let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &neg_ratio,
                &grad_rhs_shape,
            )?;
            Ok(vec![
                crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

impl<D: Device> CudaBackendImpl<D> {
    pub(crate) fn add<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        cuda_add_storage(lhs, rhs)
    }
}

pub(crate) fn push_unary_tape_entry(
    t_id: crate::cuda::storage::TensorId,
    out_id: crate::cuda::storage::TensorId,
    grad_fn: impl Fn(&CudaStorage) -> Result<CudaStorage> + Send + Sync + 'static,
) {
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| grad_fn(grad_out).map(|grad| vec![grad])),
    });
}

/// Computes `grad_out * f'(x)` in one kernel, when the IR can express `f`.
///
/// The fallback path below this one evaluates `f'(x)` into a fresh buffer the
/// size of the input and then launches a second kernel to multiply it by the
/// incoming gradient. Both kernels are pointwise over the same shape, so the
/// intermediate exists only because the two halves are written as separate
/// strings. `codegen::catalog::unary_fused_backward` differentiates the forward
/// symbolically and performs the multiply inside the IR, which collapses the
/// pair into a single binary kernel and removes one launch and one `numel`
/// allocation per operation per backward pass.
///
/// Returns `Ok(None)` for an operation the IR has no definition for, leaving the
/// caller on its hand-written path. That is what keeps this an incremental
/// change: an operation is fused when the catalog covers it and is otherwise
/// untouched.
///
/// Only valid where the derivative is a function of the *input*. The
/// `unary_wrt_output` family deliberately captures its output instead of its
/// input to avoid keeping the input alive, and its hand-written derivative is
/// written in terms of that output; `diff` produces a derivative in terms of the
/// input, so those operations are not eligible and do not call this.
#[cfg(feature = "cuda")]
fn fused_unary_backward(
    op_name: &'static str,
    fused_name: &'static str,
    grad_out: &CudaStorage,
    input: &CudaStorage,
) -> Result<Option<CudaStorage>> {
    let Some(fused) = crate::codegen::unary_fused_backward(op_name) else {
        return Ok(None);
    };
    let dtype = crate::cuda::backend::require_cuda_builtin_dtype(input.buffer.dtype, op_name)?;
    let body = crate::kernel::lower_binary_body(&fused, dtype)?;
    crate::cuda::ops::elementwise::launch_binary_body(
        fused_name,
        &body,
        grad_out,
        input,
        &grad_out.shape,
    )
    .map(Some)
}

macro_rules! cuda_pointwise {
    (
        $(
            unary_wrt_input: $fn_name:ident ($op_name:literal, $fwd_expr:literal, $deriv_expr:literal);
        )*
        $(
            unary_wrt_output: $fn_name_out:ident ($op_name_out:literal, $fwd_expr_out:literal, $deriv_expr_out:literal);
        )*
        $(
            unary_no_grad: $fn_name_ng:ident ($op_name_ng:literal, $fwd_expr_ng:literal);
        )*
        $(
            binary: $fn_name_bin:ident ($op_name_bin:literal, $fwd_expr_bin:literal, $deriv_lhs_expr:literal, $deriv_rhs_expr:literal);
        )*
    ) => {
        $(
            pub(crate) fn $fn_name(t: &CudaStorage) -> Result<CudaStorage> {
                let out = crate::cuda::ops::elementwise::launch_unary_op($op_name, $fwd_expr, t)?;
                let t_capture = t.clone();
                push_unary_tape_entry(t.id, out.id, move |grad_out| {
                    if let Some(grad) = fused_unary_backward(
                        $op_name,
                        concat!($op_name, "_fused_grad"),
                        grad_out,
                        &t_capture,
                    )? {
                        return Ok(grad);
                    }
                    let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                        concat!($op_name, "_grad"),
                        $deriv_expr,
                        &t_capture,
                    )?;
                    crate::cuda::ops::elementwise::launch_binary_op(
                        "mul",
                        "a * b",
                        grad_out,
                        &deriv,
                        &grad_out.shape,
                    )
                });
                Ok(out)
            }
        )*
        $(
            pub(crate) fn $fn_name_out(t: &CudaStorage) -> Result<CudaStorage> {
                let out = crate::cuda::ops::elementwise::launch_unary_op($op_name_out, $fwd_expr_out, t)?;
                let out_capture = out.clone();
                push_unary_tape_entry(t.id, out.id, move |grad_out| {
                    let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                        concat!($op_name_out, "_grad"),
                        $deriv_expr_out,
                        &out_capture,
                    )?;
                    crate::cuda::ops::elementwise::launch_binary_op(
                        "mul",
                        "a * b",
                        grad_out,
                        &deriv,
                        &grad_out.shape,
                    )
                });
                Ok(out)
            }
        )*
        $(
            pub(crate) fn $fn_name_ng(t: &CudaStorage) -> Result<CudaStorage> {
                crate::cuda::ops::elementwise::launch_unary_op($op_name_ng, $fwd_expr_ng, t)
            }
        )*
        $(
            pub(crate) fn $fn_name_bin(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
                let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
                let out = crate::cuda::ops::elementwise::launch_binary_op($op_name_bin, $fwd_expr_bin, lhs, rhs, &out_shape)?;
                let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
                let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
                let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
                crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
                    output_id: out_id,
                    input_ids: vec![lhs_id, rhs_id],
                    backward: Box::new(move |grad_out: &CudaStorage| {
                        let deriv_lhs_shape = crate::layout::broadcast_shape(&lhs_capture.shape, &rhs_capture.shape)?;
                        let deriv_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                            concat!($op_name_bin, "_grad_lhs"),
                            $deriv_lhs_expr,
                            &lhs_capture,
                            &rhs_capture,
                            &deriv_lhs_shape,
                        )?;
                        let grad_lhs_shape = crate::layout::broadcast_shape(&grad_out.shape, &deriv_lhs.shape)?;
                        let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                            "mul",
                            "a * b",
                            grad_out,
                            &deriv_lhs,
                            &grad_lhs_shape,
                        )?;

                        let deriv_rhs_shape = crate::layout::broadcast_shape(&lhs_capture.shape, &rhs_capture.shape)?;
                        let deriv_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                            concat!($op_name_bin, "_grad_rhs"),
                            $deriv_rhs_expr,
                            &lhs_capture,
                            &rhs_capture,
                            &deriv_rhs_shape,
                        )?;
                        let grad_rhs_shape = crate::layout::broadcast_shape(&grad_out.shape, &deriv_rhs.shape)?;
                        let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                            "mul",
                            "a * b",
                            grad_out,
                            &deriv_rhs,
                            &grad_rhs_shape,
                        )?;

                        Ok(vec![
                            crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                            crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                        ])
                    }),
                });
                Ok(out)
            }
        )*
    };
}

cuda_pointwise! {
    unary_wrt_input: cuda_relu_storage("relu", "x > 0.0f ? x : 0.0f", "x > 0.0f ? 1.0f : 0.0f");
    unary_wrt_input: cuda_mish_storage("mish", "x * tanhf(log1pf(expf(x)))", "tanhf(log1pf(expf(x))) + x * (1.0f / (1.0f + expf(-x))) * (1.0f - tanhf(log1pf(expf(x))) * tanhf(log1pf(expf(x))))");
    unary_wrt_input: cuda_elu_storage("elu", "x >= 0.0f ? x : (expf(x) - 1.0f)", "x >= 0.0f ? 1.0f : expf(x)");
    unary_wrt_input: cuda_gelu_storage("gelu", "0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))", "0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) + 0.5f * x * (1.0f - tanhf(0.7978845608f * (x + 0.044715f * x * x * x)) * tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x)");
    unary_wrt_input: cuda_abs_storage("abs", "fabsf(x)", "x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)");
    unary_wrt_input: cuda_neg_storage("neg", "-x", "-1.0f");
    unary_wrt_input: cuda_log_storage("log", "logf(x)", "1.0f / x");
    unary_wrt_input: cuda_swish_storage("swish", "x / (1.0f + expf(-x))", "(1.0f / (1.0f + expf(-x))) * (1.0f + x * (1.0f - (1.0f / (1.0f + expf(-x)))))");
    unary_wrt_input: cuda_sin_storage("sin", "sinf(x)", "cosf(x)");
    unary_wrt_input: cuda_cos_storage("cos", "cosf(x)", "-sinf(x)");
    unary_wrt_input: cuda_tan_storage("tan", "tanf(x)", "1.0f / (cosf(x) * cosf(x))");
    unary_wrt_input: cuda_asin_storage("asin", "asinf(x)", "1.0f / sqrtf(1.0f - x * x)");
    unary_wrt_input: cuda_acos_storage("acos", "acosf(x)", "-1.0f / sqrtf(1.0f - x * x)");
    unary_wrt_input: cuda_atan_storage("atan", "atanf(x)", "1.0f / (1.0f + x * x)");
    unary_wrt_input: cuda_sinh_storage("sinh", "sinhf(x)", "coshf(x)");
    unary_wrt_input: cuda_cosh_storage("cosh", "coshf(x)", "sinhf(x)");
    unary_wrt_input: cuda_asinh_storage("asinh", "asinhf(x)", "1.0f / sqrtf(x * x + 1.0f)");
    unary_wrt_input: cuda_acosh_storage("acosh", "acoshf(x)", "1.0f / sqrtf(x * x - 1.0f)");
    unary_wrt_input: cuda_atanh_storage("atanh", "atanhf(x)", "1.0f / (1.0f - x * x)");
    unary_wrt_input: cuda_erf_storage("erf", "erff(x)", "1.12837916709551257390f * expf(-x * x)");
    unary_wrt_input: cuda_log2_storage("log2", "log2f(x)", "1.0f / (x * 0.6931471805599453f)");
    unary_wrt_input: cuda_log10_storage("log10", "log10f(x)", "1.0f / (x * 2.302585092994046f)");

    unary_wrt_output: cuda_exp_storage("exp", "expf(x)", "x");
    unary_wrt_output: cuda_sqrt_storage("sqrt", "sqrtf(x)", "0.5f / x");
    unary_wrt_output: cuda_rsqrt_storage("rsqrt", "rsqrtf(x)", "-0.5f * x * x * x");
    unary_wrt_output: cuda_tanh_storage("tanh", "tanhf(x)", "1.0f - x * x");
    unary_wrt_output: cuda_sigmoid_storage("sigmoid", "1.0f / (1.0f + expf(-x))", "x * (1.0f - x)");

    unary_no_grad: cuda_step_storage("step", "x > 0.0f ? 1.0f : 0.0f");
    unary_no_grad: cuda_sign_storage("sign", "x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)");
    unary_no_grad: cuda_floor_storage("floor", "floorf(x)");
    unary_no_grad: cuda_ceil_storage("ceil", "ceilf(x)");
    unary_no_grad: cuda_round_storage("round", "roundf(x)");
    unary_no_grad: cuda_trunc_storage("trunc", "truncf(x)");
    unary_no_grad: cuda_frac_storage("frac", "x - truncf(x)");

    binary: cuda_maximum_storage("maximum", "a > b ? a : b", "a >= b ? 1.0f : 0.0f", "a < b ? 1.0f : 0.0f");
    binary: cuda_minimum_storage("minimum", "a < b ? a : b", "a <= b ? 1.0f : 0.0f", "a > b ? 1.0f : 0.0f");
    binary: cuda_abs_diff_storage("abs_diff", "fabsf(a - b)", "a >= b ? 1.0f : -1.0f", "a >= b ? -1.0f : 1.0f");
}

pub(crate) fn cuda_powf_storage(t: &CudaStorage, exp: f64) -> Result<CudaStorage> {
    let expr = format!("powf(x, {exp:.17})");
    let out = crate::cuda::ops::elementwise::launch_unary_op("powf", &expr, t)?;
    let t_capture = t.clone();
    let exp_minus_1 = exp - 1.0;
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let deriv_expr = format!("({exp:.17}) * powf(x, {exp_minus_1:.17})");
        let deriv =
            crate::cuda::ops::elementwise::launch_unary_op("powf_grad", &deriv_expr, &t_capture)?;
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &deriv,
            &grad_out.shape,
        )
    });
    Ok(out)
}

pub(crate) fn cuda_clamp_storage(t: &CudaStorage, min: f64, max: f64) -> Result<CudaStorage> {
    let expr = format!("x < {min:.17} ? {min:.17} : (x > {max:.17} ? {max:.17} : x)");
    let out = crate::cuda::ops::elementwise::launch_unary_op("clamp", &expr, t)?;
    let t_capture = t.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let deriv_expr = format!("(x >= {min:.17} && x <= {max:.17}) ? 1.0f : 0.0f");
        let mask =
            crate::cuda::ops::elementwise::launch_unary_op("clamp_grad", &deriv_expr, &t_capture)?;
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &mask,
            &grad_out.shape,
        )
    });
    Ok(out)
}

pub(crate) fn cuda_atan2_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    crate::cuda::ops::elementwise::launch_binary_op("atan2", "atan2f(a, b)", lhs, rhs, &out_shape)
}

pub(crate) fn cuda_fmod_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    crate::cuda::ops::elementwise::launch_binary_op("fmod", "fmodf(a, b)", lhs, rhs, &out_shape)
}

pub(crate) fn cuda_remainder_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    crate::cuda::ops::elementwise::launch_binary_op(
        "remainder",
        "remainderf(a, b)",
        lhs,
        rhs,
        &out_shape,
    )
}

pub(crate) fn cuda_lerp_storage(
    start: &CudaStorage,
    end: &CudaStorage,
    weight: f64,
) -> Result<CudaStorage> {
    let diff = cuda_sub_storage(end, start)?;
    let scaled = CudaBackendImpl::<Cuda>::mul_scalar_float::<f32>(&diff, weight)?;
    cuda_add_storage(start, &scaled)
}

/// `exp(x - max(x)) / sum(exp(x - max(x)))` along `axis`, composed entirely
/// from already tape-tracked primitives. `max_keepdim` is not itself
/// tape-tracked, which is exactly right here: softmax is invariant to a
/// constant shift, so the true gradient through the stabilizing max is zero,
/// and an untracked leaf gives that for free instead of needing a
/// hand-written zero. Shared by `Execute<op::Softmax>` and
/// `scaled_dot_product_attention`, which both need this exact composition.
pub(crate) fn cuda_softmax<D: Device>(input: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    let max_val = CudaBackendImpl::<D>::max_keepdim::<f32>(input, axis)?;
    let shifted = cuda_sub_storage(input, &max_val)?;
    let exp_vals = cuda_exp_storage(&shifted)?;
    let sum_val = CudaBackendImpl::<D>::sum_keepdim::<f32>(&exp_vals, axis)?;
    cuda_div_storage(&exp_vals, &sum_val)
}

pub(crate) fn cuda_log_softmax<D: Device>(input: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    let max_val = CudaBackendImpl::<D>::max_keepdim::<f32>(input, axis)?;
    let shifted = cuda_sub_storage(input, &max_val)?;
    let exp_vals = cuda_exp_storage(&shifted)?;
    let sum_val = CudaBackendImpl::<D>::sum_keepdim::<f32>(&exp_vals, axis)?;
    let log_sum = cuda_log_storage(&sum_val)?;
    cuda_sub_storage(&shifted, &log_sum)
}

#[allow(clippy::extra_unused_type_parameters)]
impl<D: Device> CudaBackendImpl<D> {
    pub(crate) fn relu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_relu_storage(t)
    }
    pub(crate) fn step<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_step_storage(t)
    }
    pub(crate) fn mish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_mish_storage(t)
    }
    pub(crate) fn elu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_elu_storage(t)
    }
    pub(crate) fn gelu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_gelu_storage(t)
    }
    pub(crate) fn abs<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_abs_storage(t)
    }
    pub(crate) fn exp<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_exp_storage(t)
    }
    pub(crate) fn neg<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_neg_storage(t)
    }
    pub(crate) fn sqrt<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sqrt_storage(t)
    }
    pub(crate) fn log<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log_storage(t)
    }
    pub(crate) fn tanh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_tanh_storage(t)
    }
    pub(crate) fn sigmoid<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sigmoid_storage(t)
    }
    pub(crate) fn swish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_swish_storage(t)
    }
    pub(crate) fn sign<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sign_storage(t)
    }
    pub(crate) fn floor<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_floor_storage(t)
    }
    pub(crate) fn ceil<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_ceil_storage(t)
    }
    pub(crate) fn round<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_round_storage(t)
    }
    pub(crate) fn log2<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log2_storage(t)
    }
    pub(crate) fn log10<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log10_storage(t)
    }
    pub(crate) fn sin<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sin_storage(t)
    }
    pub(crate) fn cos<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_cos_storage(t)
    }
    pub(crate) fn tan<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_tan_storage(t)
    }
    pub(crate) fn asin<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_asin_storage(t)
    }
    pub(crate) fn acos<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_acos_storage(t)
    }
    pub(crate) fn atan<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_atan_storage(t)
    }
    pub(crate) fn sinh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sinh_storage(t)
    }
    pub(crate) fn cosh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_cosh_storage(t)
    }
    pub(crate) fn asinh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_asinh_storage(t)
    }
    pub(crate) fn acosh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_acosh_storage(t)
    }
    pub(crate) fn atanh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_atanh_storage(t)
    }
    pub(crate) fn erf<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_erf_storage(t)
    }
    pub(crate) fn rsqrt<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_rsqrt_storage(t)
    }
    pub(crate) fn trunc<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_trunc_storage(t)
    }
    pub(crate) fn frac<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_frac_storage(t)
    }
    pub(crate) fn powf<K: DType>(t: &CudaStorage, exp: f64) -> Result<CudaStorage> {
        cuda_powf_storage(t, exp)
    }
    pub(crate) fn clamp<K: DType>(t: &CudaStorage, min: f64, max: f64) -> Result<CudaStorage> {
        cuda_clamp_storage(t, min, max)
    }
    pub(crate) fn atan2<K: DType>(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
        cuda_atan2_storage(lhs, rhs)
    }
    pub(crate) fn fmod<K: DType>(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
        cuda_fmod_storage(lhs, rhs)
    }
    pub(crate) fn remainder<K: DType>(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
        cuda_remainder_storage(lhs, rhs)
    }

    /// The literal is emitted unsuffixed (full `f64` precision, not narrowed
    /// to `f32` first) so the `f64` compute-type family actually computes at
    /// `f64` precision instead of silently narrowing - see `sub_scalar_float`
    /// below, which this now matches instead of contradicting.
    pub(crate) fn add_scalar_float<K: DType>(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
        let expr = format!("x + ({scalar:.17})");
        let out = crate::cuda::ops::elementwise::launch_unary_op("add_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, |grad_out| Ok(grad_out.clone()));
        Ok(out)
    }

    pub(crate) fn mul_scalar_float<K: DType>(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
        let expr = format!("x * ({scalar:.17})");
        let out = crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let expr = format!("x * ({scalar:.17})");
            crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, grad_out)
        });
        Ok(out)
    }

    /// `x - val`. The literal is emitted unsuffixed (full `f64` precision,
    /// not narrowed to `f32` first) so the `f64` compute-type family
    /// actually computes at `f64` precision instead of silently narrowing -
    /// the same distinction `exp`/`sqrt`/`log`/`tanh` above draw against
    /// their float-suffixed intrinsics.
    pub(crate) fn sub_scalar_float<K: DType>(t: &CudaStorage, val: f64) -> Result<CudaStorage> {
        let expr = format!("x - ({val:.17})");
        let out = crate::cuda::ops::elementwise::launch_unary_op("sub_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, |grad_out| Ok(grad_out.clone()));
        Ok(out)
    }

    /// `x / val`. Backward is `grad_out / val`, the same scalar division run
    /// on the incoming gradient.
    pub(crate) fn div_scalar_float<K: DType>(t: &CudaStorage, val: f64) -> Result<CudaStorage> {
        let expr = format!("x / ({val:.17})");
        let out = crate::cuda::ops::elementwise::launch_unary_op("div_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let expr = format!("x / ({val:.17})");
            crate::cuda::ops::elementwise::launch_unary_op("div_scalar", &expr, grad_out)
        });
        Ok(out)
    }
}

pub(crate) fn cuda_add_scalar_float(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::add_scalar_float::<f32>(t, scalar)
}

pub(crate) fn cuda_mul_scalar_float(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::mul_scalar_float::<f32>(t, scalar)
}

pub(crate) fn cuda_sub_scalar_float(t: &CudaStorage, val: f64) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::sub_scalar_float::<f32>(t, val)
}

pub(crate) fn cuda_div_scalar_float(t: &CudaStorage, val: f64) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::div_scalar_float::<f32>(t, val)
}
