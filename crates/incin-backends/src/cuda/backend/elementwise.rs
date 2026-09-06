//! Elementwise CUDA operations: tape-tracked binary and unary kernels,
//! activations, and scalar arithmetic.

#![allow(dead_code)]

use super::*;

pub(crate) fn cuda_add_storage(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    spec: crate::kernel::KernelSpecialization,
) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out = crate::cuda::ops::elementwise::launch_binary_body(
        "add",
        &crate::codegen::ScalarFragment::literal("a + b"),
        lhs,
        rhs,
        &out_shape,
        spec,
    )?;
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

pub(crate) fn cuda_sub_storage(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    spec: crate::kernel::KernelSpecialization,
) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out = crate::cuda::ops::elementwise::launch_binary_body(
        "sub",
        &crate::codegen::ScalarFragment::literal("a - b"),
        lhs,
        rhs,
        &out_shape,
        spec,
    )?;
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

pub(crate) fn cuda_mul_storage(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    spec: crate::kernel::KernelSpecialization,
) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out = crate::cuda::ops::elementwise::launch_binary_body(
        "mul",
        &crate::codegen::ScalarFragment::literal("a * b"),
        lhs,
        rhs,
        &out_shape,
        spec,
    )?;
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

pub(crate) fn cuda_div_storage(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    spec: crate::kernel::KernelSpecialization,
) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let out = crate::cuda::ops::elementwise::launch_binary_body(
        "div",
        &crate::codegen::ScalarFragment::literal("a / b"),
        lhs,
        rhs,
        &out_shape,
        spec,
    )?;
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
        cuda_add_storage(lhs, rhs, crate::kernel::KernelSpecialization::NONE)
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
        // A gradient's geometry comes from the tape at runtime, not from a
        // shape type, so the backward pass has nothing proven to specialize on.
        crate::kernel::KernelSpecialization::NONE,
    )
    .map(Some)
}

/// Whether `t` needs the double-precision spelling of a hand-written
/// pointwise literal (see `f64_unary_fwd`).
fn uses_f64_compute(t: &CudaStorage) -> bool {
    t.buffer.dtype.builtin_id() == Some(DTypeId::F64)
}

/// Double-precision forward spelling for the hand-written literal `cuda_pointwise!`
/// emits under `op_name`.
///
/// The literals in the macro invocation below are written for the `float`
/// compute type (`expf`, `tanhf`, `1.0f`, ...). F16/BF16 tensors also compute
/// in `float`, so those spellings are right for them, but passing an f64 `x`
/// to `expf` narrows it to `float` first: an f64 tensor would silently compute
/// at f32 precision. The generated functions select this table's spelling when
/// the input's dtype is f64.
///
/// An op missing here fails loudly instead of running the float spelling on
/// doubles, so adding a new float-suffixed literal to the macro without a row
/// here is a compile-visible error at runtime rather than a silent narrowing.
fn f64_unary_fwd(op_name: &'static str) -> Result<&'static str> {
    match op_name {
        "relu" => Ok("x > 0.0 ? x : 0.0"),
        "mish" => Ok("x * tanh(log1p(exp(x)))"),
        "elu" => Ok("x >= 0.0 ? x : (exp(x) - 1.0)"),
        "gelu" => Ok("0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))"),
        "abs" => Ok("fabs(x)"),
        "neg" => Ok("-x"),
        "log" => Ok("log(x)"),
        "swish" => Ok("x / (1.0 + exp(-x))"),
        "sin" => Ok("sin(x)"),
        "cos" => Ok("cos(x)"),
        "tan" => Ok("tan(x)"),
        "asin" => Ok("asin(x)"),
        "acos" => Ok("acos(x)"),
        "atan" => Ok("atan(x)"),
        "sinh" => Ok("sinh(x)"),
        "cosh" => Ok("cosh(x)"),
        "asinh" => Ok("asinh(x)"),
        "acosh" => Ok("acosh(x)"),
        "atanh" => Ok("atanh(x)"),
        "erf" => Ok("erf(x)"),
        "log2" => Ok("log2(x)"),
        "log10" => Ok("log10(x)"),
        "exp" => Ok("exp(x)"),
        "sqrt" => Ok("sqrt(x)"),
        "rsqrt" => Ok("rsqrt(x)"),
        "tanh" => Ok("tanh(x)"),
        "sigmoid" => Ok("1.0 / (1.0 + exp(-x))"),
        "step" => Ok("x > 0.0 ? 1.0 : 0.0"),
        "sign" => Ok("x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)"),
        "floor" => Ok("floor(x)"),
        "ceil" => Ok("ceil(x)"),
        "round" => Ok("round(x)"),
        "trunc" => Ok("trunc(x)"),
        "frac" => Ok("x - trunc(x)"),
        // No float-suffixed intrinsic: the f32 spelling is already exact in
        // `double`, so these rows are identity entries that keep the resolver
        // total rather than special-casing the call sites.
        "maximum" => Ok("a > b ? a : b"),
        "minimum" => Ok("a < b ? a : b"),
        "abs_diff" => Ok("fabs(a - b)"),
        _ => Err(Error::UnsupportedDType {
            dtype: DTypeId::F64.descriptor(),
            backend: "Cuda",
            op: "f64 pointwise forward without a double-precision spelling",
        }),
    }
}

/// Double-precision derivative spelling for the hand-written derivative
/// `cuda_pointwise!` emits under `op_name`. Same contract as `f64_unary_fwd`:
/// the IR-lowered fused path (`fused_unary_backward`) is already
/// dtype-correct, so only this hand-written fallback needs the table.
fn f64_unary_deriv(op_name: &'static str) -> Result<&'static str> {
    match op_name {
        "relu" => Ok("x > 0.0 ? 1.0 : 0.0"),
        "mish" => Ok(
            "tanh(log1p(exp(x))) + x * (1.0 / (1.0 + exp(-x))) * (1.0 - tanh(log1p(exp(x))) * tanh(log1p(exp(x))))",
        ),
        "elu" => Ok("x >= 0.0 ? 1.0 : exp(x)"),
        "gelu" => Ok(
            "0.5 * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) + 0.5 * x * (1.0 - tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)) * tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)",
        ),
        "abs" => Ok("x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)"),
        "neg" => Ok("-1.0"),
        "log" => Ok("1.0 / x"),
        "swish" => Ok("(1.0 / (1.0 + exp(-x))) * (1.0 + x * (1.0 - (1.0 / (1.0 + exp(-x)))))"),
        "sin" => Ok("cos(x)"),
        "cos" => Ok("-sin(x)"),
        "tan" => Ok("1.0 / (cos(x) * cos(x))"),
        "asin" => Ok("1.0 / sqrt(1.0 - x * x)"),
        "acos" => Ok("-1.0 / sqrt(1.0 - x * x)"),
        "atan" => Ok("1.0 / (1.0 + x * x)"),
        "sinh" => Ok("cosh(x)"),
        "cosh" => Ok("sinh(x)"),
        "asinh" => Ok("1.0 / sqrt(x * x + 1.0)"),
        "acosh" => Ok("1.0 / sqrt(x * x - 1.0)"),
        "atanh" => Ok("1.0 / (1.0 - x * x)"),
        "erf" => Ok("1.1283791670955126 * exp(-x * x)"),
        "log2" => Ok("1.0 / (x * 0.6931471805599453)"),
        "log10" => Ok("1.0 / (x * 2.302585092994046)"),
        "exp" => Ok("x"),
        "sqrt" => Ok("0.5 / x"),
        "rsqrt" => Ok("-0.5 * x * x * x"),
        "tanh" => Ok("1.0 - x * x"),
        "sigmoid" => Ok("x * (1.0 - x)"),
        // The binary derivatives are bare `1.0f`/`0.0f` selections, exact in
        // `double`, so these rows repeat the f32 spelling unsuffixed.
        "maximum_lhs" => Ok("a >= b ? 1.0 : 0.0"),
        "maximum_rhs" => Ok("a < b ? 1.0 : 0.0"),
        "minimum_lhs" => Ok("a <= b ? 1.0 : 0.0"),
        "minimum_rhs" => Ok("a > b ? 1.0 : 0.0"),
        "abs_diff_lhs" => Ok("a >= b ? 1.0 : -1.0"),
        "abs_diff_rhs" => Ok("a >= b ? -1.0 : 1.0"),
        _ => Err(Error::UnsupportedDType {
            dtype: DTypeId::F64.descriptor(),
            backend: "Cuda",
            op: "f64 pointwise derivative without a double-precision spelling",
        }),
    }
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
            pub(crate) fn $fn_name(
                t: &CudaStorage,
                spec: crate::kernel::KernelSpecialization,
            ) -> Result<CudaStorage> {
                // The literal below is the `float` spelling; f64 inputs take
                // the double-precision one from `f64_unary_fwd` instead of
                // silently narrowing through the float intrinsics.
                let f64_compute = uses_f64_compute(t);
                let fwd: &str = if f64_compute {
                    f64_unary_fwd($op_name)?
                } else {
                    $fwd_expr
                };
                let out = crate::cuda::ops::elementwise::launch_unary_body(
                    $op_name,
                    &crate::codegen::ScalarFragment::literal(fwd),
                    t,
                    spec,
                )?;
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
                    let deriv: &str = if f64_compute {
                        f64_unary_deriv($op_name)?
                    } else {
                        $deriv_expr
                    };
                    let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                        concat!($op_name, "_grad"),
                        deriv,
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
            pub(crate) fn $fn_name_out(
                t: &CudaStorage,
                spec: crate::kernel::KernelSpecialization,
            ) -> Result<CudaStorage> {
                // Same float/double selection as the `_wrt_input` arm above:
                // the literal is the `float` spelling.
                let f64_compute = uses_f64_compute(t);
                let fwd: &str = if f64_compute {
                    f64_unary_fwd($op_name_out)?
                } else {
                    $fwd_expr_out
                };
                let out = crate::cuda::ops::elementwise::launch_unary_body(
                    $op_name_out,
                    &crate::codegen::ScalarFragment::literal(fwd),
                    t,
                    spec,
                )?;
                let out_capture = out.clone();
                push_unary_tape_entry(t.id, out.id, move |grad_out| {
                    let deriv: &str = if f64_compute {
                        f64_unary_deriv($op_name_out)?
                    } else {
                        $deriv_expr_out
                    };
                    let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                        concat!($op_name_out, "_grad"),
                        deriv,
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
            pub(crate) fn $fn_name_ng(
                t: &CudaStorage,
                spec: crate::kernel::KernelSpecialization,
            ) -> Result<CudaStorage> {
                // Grad-less but not precision-less: `floorf` on a `double`
                // still narrows first, so the forward takes the double
                // spelling for f64 inputs too.
                let fwd: &str = if uses_f64_compute(t) {
                    f64_unary_fwd($op_name_ng)?
                } else {
                    $fwd_expr_ng
                };
                crate::cuda::ops::elementwise::launch_unary_body(
                    $op_name_ng,
                    &crate::codegen::ScalarFragment::literal(fwd),
                    t,
                    spec,
                )
            }
        )*
        $(
            pub(crate) fn $fn_name_bin(
                lhs: &CudaStorage,
                rhs: &CudaStorage,
                spec: crate::kernel::KernelSpecialization,
            ) -> Result<CudaStorage> {
                let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
                // The launch already refuses mixed dtypes, so either side
                // names the compute type. Only `abs_diff`'s forward carries a
                // float-suffixed intrinsic; the `1.0f`/`0.0f` derivative
                // selections are exact in `double`, so the derivatives keep
                // one spelling via `f64_unary_deriv`'s `_lhs`/`_rhs` rows.
                let f64_compute = uses_f64_compute(lhs);
                let fwd: &str = if f64_compute {
                    f64_unary_fwd($op_name_bin)?
                } else {
                    $fwd_expr_bin
                };
                let out = crate::cuda::ops::elementwise::launch_binary_body(
                    $op_name_bin,
                    &crate::codegen::ScalarFragment::literal(fwd),
                    lhs,
                    rhs,
                    &out_shape,
                    spec,
                )?;
                let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
                let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
                let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
                crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
                    output_id: out_id,
                    input_ids: vec![lhs_id, rhs_id],
                    backward: Box::new(move |grad_out: &CudaStorage| {
                        let deriv_lhs_shape = crate::layout::broadcast_shape(&lhs_capture.shape, &rhs_capture.shape)?;
                        let deriv_lhs_expr: &str = if f64_compute {
                            f64_unary_deriv(concat!($op_name_bin, "_lhs"))?
                        } else {
                            $deriv_lhs_expr
                        };
                        let deriv_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                            concat!($op_name_bin, "_grad_lhs"),
                            deriv_lhs_expr,
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
                        let deriv_rhs_expr: &str = if f64_compute {
                            f64_unary_deriv(concat!($op_name_bin, "_rhs"))?
                        } else {
                            $deriv_rhs_expr
                        };
                        let deriv_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                            concat!($op_name_bin, "_grad_rhs"),
                            deriv_rhs_expr,
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
    // `powf` narrows a `double` x to `float` first; f64 inputs take `pow`.
    let f64_compute = uses_f64_compute(t);
    let kernel = if f64_compute { "pow" } else { "powf" };
    let expr = format!("{kernel}(x, {exp:.17})");
    let out = crate::cuda::ops::elementwise::launch_unary_op("powf", &expr, t)?;
    let t_capture = t.clone();
    let exp_minus_1 = exp - 1.0;
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let deriv_expr = format!("({exp:.17}) * {kernel}(x, {exp_minus_1:.17})");
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
    // `atan2f` narrows `double` operands first; f64 inputs take `atan2`.
    let expr = if uses_f64_compute(lhs) {
        "atan2(a, b)"
    } else {
        "atan2f(a, b)"
    };
    crate::cuda::ops::elementwise::launch_binary_op("atan2", expr, lhs, rhs, &out_shape)
}

pub(crate) fn cuda_fmod_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let expr = if uses_f64_compute(lhs) {
        "fmod(a, b)"
    } else {
        "fmodf(a, b)"
    };
    crate::cuda::ops::elementwise::launch_binary_op("fmod", expr, lhs, rhs, &out_shape)
}

pub(crate) fn cuda_remainder_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let expr = if uses_f64_compute(lhs) {
        "remainder(a, b)"
    } else {
        "remainderf(a, b)"
    };
    crate::cuda::ops::elementwise::launch_binary_op("remainder", expr, lhs, rhs, &out_shape)
}

pub(crate) fn cuda_lerp_storage(
    start: &CudaStorage,
    end: &CudaStorage,
    weight: f64,
) -> Result<CudaStorage> {
    let diff = cuda_sub_storage(end, start, crate::kernel::KernelSpecialization::NONE)?;
    let scaled = CudaBackendImpl::<Cuda>::mul_scalar_float::<f32>(&diff, weight)?;
    cuda_add_storage(start, &scaled, crate::kernel::KernelSpecialization::NONE)
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
    let shifted = cuda_sub_storage(input, &max_val, crate::kernel::KernelSpecialization::NONE)?;
    let exp_vals = cuda_exp_storage(&shifted, crate::kernel::KernelSpecialization::NONE)?;
    let sum_val = CudaBackendImpl::<D>::sum_keepdim::<f32>(&exp_vals, axis)?;
    cuda_div_storage(
        &exp_vals,
        &sum_val,
        crate::kernel::KernelSpecialization::NONE,
    )
}

pub(crate) fn cuda_log_softmax<D: Device>(input: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    let max_val = CudaBackendImpl::<D>::max_keepdim::<f32>(input, axis)?;
    let shifted = cuda_sub_storage(input, &max_val, crate::kernel::KernelSpecialization::NONE)?;
    let exp_vals = cuda_exp_storage(&shifted, crate::kernel::KernelSpecialization::NONE)?;
    let sum_val = CudaBackendImpl::<D>::sum_keepdim::<f32>(&exp_vals, axis)?;
    let log_sum = cuda_log_storage(&sum_val, crate::kernel::KernelSpecialization::NONE)?;
    cuda_sub_storage(
        &shifted,
        &log_sum,
        crate::kernel::KernelSpecialization::NONE,
    )
}

#[allow(clippy::extra_unused_type_parameters)]
impl<D: Device> CudaBackendImpl<D> {
    pub(crate) fn relu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_relu_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn step<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_step_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn mish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_mish_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn elu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_elu_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn gelu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_gelu_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn abs<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_abs_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn exp<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_exp_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn neg<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_neg_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn sqrt<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sqrt_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn log<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn tanh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_tanh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn sigmoid<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sigmoid_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn swish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_swish_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn sign<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sign_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn floor<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_floor_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn ceil<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_ceil_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn round<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_round_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn log2<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log2_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn log10<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_log10_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn sin<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sin_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn cos<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_cos_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn tan<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_tan_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn asin<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_asin_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn acos<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_acos_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn atan<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_atan_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn sinh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_sinh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn cosh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_cosh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn asinh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_asinh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn acosh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_acosh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn atanh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_atanh_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn erf<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_erf_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn rsqrt<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_rsqrt_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn trunc<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_trunc_storage(t, crate::kernel::KernelSpecialization::NONE)
    }
    pub(crate) fn frac<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        cuda_frac_storage(t, crate::kernel::KernelSpecialization::NONE)
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

#[cfg(test)]
mod f64_precision_tests {
    use super::*;

    const FWD_OPS: &[&str] = &[
        "relu", "mish", "elu", "gelu", "abs", "neg", "log", "swish", "sin", "cos", "tan", "asin",
        "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "log2", "log10", "exp",
        "sqrt", "rsqrt", "tanh", "sigmoid", "step", "sign", "floor", "ceil", "round", "trunc",
        "frac", "maximum", "minimum", "abs_diff",
    ];

    const DERIV_OPS: &[&str] = &[
        "relu",
        "mish",
        "elu",
        "gelu",
        "abs",
        "neg",
        "log",
        "swish",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
        "log2",
        "log10",
        "exp",
        "sqrt",
        "rsqrt",
        "tanh",
        "sigmoid",
        "maximum_lhs",
        "maximum_rhs",
        "minimum_lhs",
        "minimum_rhs",
        "abs_diff_lhs",
        "abs_diff_rhs",
    ];

    #[test]
    fn every_hand_written_literal_has_a_double_spelling() {
        for op in FWD_OPS {
            f64_unary_fwd(op).unwrap_or_else(|e| panic!("{op} has no f64 forward spelling: {e:?}"));
        }
        for op in DERIV_OPS {
            f64_unary_deriv(op)
                .unwrap_or_else(|e| panic!("{op} has no f64 derivative spelling: {e:?}"));
        }
    }

    /// A double spelling that still names a float intrinsic (or carries a
    /// float-suffixed literal) would narrow exactly like the bug it replaces.
    /// Every `f` in these spellings must therefore belong to an identifier
    /// (`fabs`, `floor`, `fmax`-free) rather than to a suffix: no `f` may
    /// directly follow an ASCII digit, and no float-suffixed intrinsic name
    /// may appear at all.
    #[test]
    fn double_spellings_name_no_float_intrinsics_or_suffixed_literals() {
        const FORBIDDEN_INTRINSICS: &[&str] = &[
            "expf", "logf", "log1pf", "log2f", "log10f", "tanhf", "fabsf", "sinf", "cosf", "tanf",
            "asinf", "acosf", "atanf", "atan2f", "sinhf", "coshf", "asinhf", "acoshf", "atanhf",
            "erff", "sqrtf", "rsqrtf", "powf", "fmodf", "floorf", "ceilf", "roundf", "truncf",
        ];
        let mut spellings = Vec::new();
        for op in FWD_OPS {
            spellings.push(f64_unary_fwd(op).unwrap());
        }
        for op in DERIV_OPS {
            spellings.push(f64_unary_deriv(op).unwrap());
        }
        assert!(!spellings.is_empty());
        for spelling in spellings {
            for forbidden in FORBIDDEN_INTRINSICS {
                assert!(
                    !spelling.contains(forbidden),
                    "double spelling {spelling:?} still names float intrinsic {forbidden}"
                );
            }
            let bytes = spelling.as_bytes();
            for (i, &b) in bytes.iter().enumerate() {
                if b == b'f' && i > 0 && bytes[i - 1].is_ascii_digit() {
                    panic!("double spelling {spelling:?} carries a float-suffixed literal");
                }
            }
        }
    }

    #[test]
    fn an_unmapped_op_fails_loudly_instead_of_running_the_float_spelling() {
        assert!(f64_unary_fwd("not_an_op").is_err());
        assert!(f64_unary_deriv("not_an_op").is_err());
        // Grad-less spellings have no derivative row by construction.
        assert!(f64_unary_deriv("floor").is_err());
    }
}
