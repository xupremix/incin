//! Elementwise CUDA operations: tape-tracked binary and unary kernels,
//! activations, and scalar arithmetic.

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

/// `max(x, 0)`. The mask is recomputed from the saved input rather than the
/// output, because the two agree everywhere except at `x == 0`, where the
/// subgradient is conventionally taken to be zero either way.
pub(crate) fn cuda_relu_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out = crate::cuda::ops::elementwise::launch_unary_op("relu", "x > 0.0f ? x : 0.0f", t)?;
    let t_capture = t.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let mask = crate::cuda::ops::elementwise::launch_unary_op(
            "relu_mask",
            "x > 0.0f ? 1.0f : 0.0f",
            &t_capture,
        )?;
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

/// `exp(x)`. Its own value is its derivative, so the backward closure only
/// has to keep the forward output around.
pub(crate) fn cuda_exp_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out = crate::cuda::ops::elementwise::launch_unary_op("exp", "exp(x)", t)?;
    let out_capture = out.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &out_capture,
            &grad_out.shape,
        )
    });
    Ok(out)
}

/// `sqrt(x)`. `d/dx sqrt(x) = 1 / (2 sqrt(x))`, computed from the forward
/// output rather than a fresh division so the closure needs no copy of `x`.
pub(crate) fn cuda_sqrt_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out = crate::cuda::ops::elementwise::launch_unary_op("sqrt", "sqrt(x)", t)?;
    let out_capture = out.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let half_over_out =
            crate::cuda::ops::elementwise::launch_unary_op("sqrt_grad", "0.5f / x", &out_capture)?;
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &half_over_out,
            &grad_out.shape,
        )
    });
    Ok(out)
}

/// `ln(x)`. `d/dx ln(x) = 1 / x`.
pub(crate) fn cuda_log_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out = crate::cuda::ops::elementwise::launch_unary_op("log", "log(x)", t)?;
    let t_capture = t.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        crate::cuda::ops::elementwise::launch_binary_op(
            "div",
            "a / b",
            grad_out,
            &t_capture,
            &grad_out.shape,
        )
    });
    Ok(out)
}

/// `tanh(x)`. `d/dx tanh(x) = 1 - tanh(x)^2`, computed from the forward
/// output.
pub(crate) fn cuda_tanh_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out = crate::cuda::ops::elementwise::launch_unary_op("tanh", "tanh(x)", t)?;
    let out_capture = out.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let derivative = crate::cuda::ops::elementwise::launch_unary_op(
            "tanh_grad",
            "1.0f - x * x",
            &out_capture,
        )?;
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &derivative,
            &grad_out.shape,
        )
    });
    Ok(out)
}

/// `1 / (1 + exp(-x))`. `d/dx sigmoid(x) = sigmoid(x) (1 - sigmoid(x))`,
/// computed from the forward output.
pub(crate) fn cuda_sigmoid_storage(t: &CudaStorage) -> Result<CudaStorage> {
    let out =
        crate::cuda::ops::elementwise::launch_unary_op("sigmoid", "1.0 / (1.0 + exp(-x))", t)?;
    let out_capture = out.clone();
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        let derivative = crate::cuda::ops::elementwise::launch_unary_op(
            "sigmoid_grad",
            "x * (1.0f - x)",
            &out_capture,
        )?;
        crate::cuda::ops::elementwise::launch_binary_op(
            "mul",
            "a * b",
            grad_out,
            &derivative,
            &grad_out.shape,
        )
    });
    Ok(out)
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

#[allow(clippy::extra_unused_type_parameters)]
impl<D: Device> CudaBackendImpl<D> {
    // No CUDA kernel is launched for these yet. They are declared rather than
    // inherited so the gap is visible from the backend that has it.
    crate::unsupported::unsupported_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    /// The literal is emitted unsuffixed (full `f64` precision, not narrowed
    /// to `f32` first) so the `f64` compute-type family actually computes at
    /// `f64` precision instead of silently narrowing — see `sub_scalar_float`
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
    /// actually computes at `f64` precision instead of silently narrowing —
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
