//! Elementwise WGPU operations: tape-tracked binary and unary kernels,
//! activations, and scalar arithmetic.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
//   (add, sub, mul, div)
// ─────────────────────────────────────────────────────────────────────────────
/// Materialize `t` at `shape`, recording the broadcast on the tape.
///
/// The body of [`::broadcast_as`], lifted to a free function so the
/// elementwise path below can reach it too. It has to be free rather than a
/// method: `binary_op` takes bare storage and has no `Self` to call through,
/// and duplicating the dispatch here would mean two broadcasts that could
/// drift in how they push their tape entry.
pub(crate) fn broadcast_storage(t: &WgpuStorage, shape: &[usize]) -> Result<WgpuStorage> {
    let out_elements = num_elements(shape)?;
    let out_n = checked_u32(out_elements, "WGPU broadcast output element count")?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Storage)?;

    let params = dispatch::prepare_shape_params(
        3, // op_mode = broadcast
        out_n,
        shape,
        &t.shape,
        &[],
    )?;
    dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
    let out = WgpuStorage::new(out_buf, shape.to_vec());

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
            Ok(vec![crate::wgpu::tape::unbroadcast(
                grad_out,
                &original_shape,
            )?])
        }),
    });
    Ok(out)
}

/// One elementwise binary operation, broadcasting its operands when they
/// disagree.
///
/// The kernel itself is strictly elementwise - it walks two equal-length
/// buffers - so an operand that needs stretching is materialized at the
/// broadcast shape first rather than the shader growing a stride argument.
/// That costs an allocation for the stretched operand, which is why the
/// equal-shape case still goes straight to the kernel.
///
/// This used to refuse any shape disagreement outright, which made
/// `broadcast_add` and friends fail on WGPU even though the frontend had
/// already resolved the output shape at the type level, and made
/// `Linear::forward` - a matmul plus a rank-one bias add - unusable on this
/// backend for every model. The backward pass never had that gap: both tape
/// Op modes of `shaders/binary.wgsl`, named where a gradient path needs one.
/// The shader's header comment is the full list; these are the ones reached
/// from a backward closure, where a bare integer would be unreadable.
const SUB: u32 = 1;
const MUL: u32 = 2;
const CMP_LT: u32 = 9;
const CMP_GT: u32 = 11;

/// entries here have always called `unbroadcast`, which only does anything
/// when a broadcast actually happened.
#[allow(clippy::extra_unused_type_parameters)]
fn binary_op<T: DType>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    op_mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    let (lhs_owned, rhs_owned);
    let (lhs, rhs) = if lhs.shape == rhs.shape {
        (lhs, rhs)
    } else {
        // `broadcast_shape` reports the mismatch itself when the two cannot
        // align, so an incompatible pair still fails here - with the axis
        // named, rather than with this function's old blanket message.
        let target = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape).map_err(|_| {
            Error::ShapeMismatch {
                op: op_name,
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "operands do not broadcast against each other".to_string(),
            }
        })?;
        lhs_owned = if lhs.shape[..] == target[..] {
            None
        } else {
            Some(broadcast_storage(lhs, &target)?)
        };
        rhs_owned = if rhs.shape[..] == target[..] {
            None
        } else {
            Some(broadcast_storage(rhs, &target)?)
        };
        (
            lhs_owned.as_ref().unwrap_or(lhs),
            rhs_owned.as_ref().unwrap_or(rhs),
        )
    };

    let n = checked_u32(num_elements(&lhs.shape)?, "WGPU binary element count")?;
    let out_buf = WgpuBuffer::new_zeros(lhs.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_binary(&lhs.buffer, &rhs.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, lhs.shape.to_vec()))
}

impl<D: Device> WgpuBackendImpl<D> {
    /// `add`.
    pub(crate) fn add<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op::<K>(lhs, rhs, 0, "add")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(grad_out, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
    /// `sub`.
    pub(crate) fn sub<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op::<K>(lhs, rhs, 1, "sub")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let neg_grad = unary_op::<K>(grad_out, 5)?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&neg_grad, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
    /// `mul`.
    pub(crate) fn mul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op::<K>(lhs, rhs, 2, "mul")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let grad_lhs = binary_op::<K>(grad_out, &rhs_capture, 2, "mul_grad")?;
                let grad_rhs = binary_op::<K>(grad_out, &lhs_capture, 2, "mul_grad")?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
    /// `div`.
    pub(crate) fn div<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op::<K>(lhs, rhs, 3, "div")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs = binary_op::<K>(grad_out, &rhs_capture, 3, "div_grad_lhs")?;
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq = binary_op::<K>(&rhs_capture, &rhs_capture, 2, "div_grad_rhs_sq")?;
                let lhs_over_rhs_sq =
                    binary_op::<K>(&lhs_capture, &rhs_sq, 3, "div_grad_rhs_ratio")?;
                let neg_ratio = unary_op::<K>(&lhs_over_rhs_sq, 5)?;
                let grad_rhs = binary_op::<K>(grad_out, &neg_ratio, 2, "div_grad_rhs")?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    /// `maximum`, elementwise, with the cotangent routed to the winning side.
    ///
    /// The shader's fused mode is the forward. The gradient cannot come from
    /// it, so the tape entry rebuilds the selection with the comparison mode
    /// facing the same way the CPU reference does: `cpu::canonical`'s
    /// `Execute<op::Maximum>` masks on `a > b` and selects `where(mask, lhs,
    /// rhs)`, so a tie resolves to the right operand and its whole cotangent
    /// goes there. Masking on `>=` here would agree on every value and disagree
    /// on every tie, which is the kind of divergence an oracle finds late.
    pub(crate) fn maximum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::select_binary::<K>(lhs, rhs, 15, CMP_GT, "maximum")
    }

    /// `minimum`. See [`maximum`](Self::maximum); the mask faces `a < b`,
    /// which is what the CPU reference compares on.
    pub(crate) fn minimum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::select_binary::<K>(lhs, rhs, 16, CMP_LT, "minimum")
    }

    /// The shared body of `maximum`/`minimum`: one fused forward, one tape
    /// entry that splits the cotangent by a 0/1 mask.
    ///
    /// `grad_rhs` is `grad_out - grad_lhs` rather than a second masked
    /// multiply, because the two masks are complements by construction and
    /// subtracting keeps them that way exactly. Building the complement with a
    /// separate comparison would let a NaN operand answer `false` to both.
    fn select_binary<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
        forward_mode: u32,
        mask_mode: u32,
        op_name: &'static str,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op::<K>(lhs, rhs, forward_mode, op_name)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let mask = binary_op::<K>(&lhs_capture, &rhs_capture, mask_mode, op_name)?;
                let grad_lhs = binary_op::<K>(grad_out, &mask, MUL, op_name)?;
                let grad_rhs = binary_op::<K>(grad_out, &grad_lhs, SUB, op_name)?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    /// `abs_diff`, composed exactly as the CPU reference composes it.
    ///
    /// The shader has a fused mode for this too, and it is deliberately not
    /// used: `sub` and `abs` each push their own tape entry, so composing them
    /// gets the gradient for free and gets the same one the CPU path produces,
    /// including at `a == b` where the subgradient is a convention rather than
    /// a derivative. A fused forward would need that convention restated here,
    /// in a second place, where it could drift.
    pub(crate) fn abs_diff<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = Self::sub::<K>(lhs, rhs)?;
        Self::abs::<K>(&diff)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//   (scalar + unary activations)
// ─────────────────────────────────────────────────────────────────────────────
/// `unary_op`.
#[allow(clippy::extra_unused_type_parameters)]
fn unary_op<T: DType>(t: &WgpuStorage, op_mode: u32) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU unary element count")?;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_unary(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.to_vec()))
}

/// `scalar_op`.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn scalar_op<T: DType>(
    t: &WgpuStorage,
    scalar: f64,
    op_mode: u32,
) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU scalar element count")?;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let scalar_bits = (scalar as f32).to_bits();
    let params = [op_mode, n, scalar_bits];
    dispatch::dispatch_scalar(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.to_vec()))
}

/// Push a single-input `TapeEntry` whose backward closure is `grad_fn`.
/// Shared by every unary `` impl below to avoid repeating the
/// `TapeEntry { output_id, input_ids: vec![t.id], backward: ... }`
/// boilerplate at each of the ~10 call sites.
fn push_unary_tape_entry(
    t_id: crate::wgpu::storage::TensorId,
    out_id: crate::wgpu::storage::TensorId,
    grad_fn: impl Fn(&WgpuStorage) -> Result<WgpuStorage> + Send + Sync + 'static,
) {
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &WgpuStorage| grad_fn(grad_out).map(|grad| vec![grad])),
    });
}

impl<D: Device> WgpuBackendImpl<D> {
    // No WGSL kernel exists for these yet. They are declared rather than
    // inherited so the shader gap is visible from the backend that has it.
    crate::unsupported::unsupported_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    /// `add_scalar_float`.
    pub(crate) fn add_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op::<K>(t, scalar, 0)?;
        // Gradient passes through unchanged (same shape, no unbroadcast
        // needed - scalar ops don't change shape).
        push_unary_tape_entry(t.id, out.id, |grad_out| Ok(grad_out.clone()));
        Ok(out)
    }
    /// `mul_scalar_float`.
    pub(crate) fn mul_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op::<K>(t, scalar, 1)?;
        // Gradient scales by the same constant.
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            scalar_op::<K>(grad_out, scalar, 1)
        });
        Ok(out)
    }
    /// `relu`.
    pub(crate) fn relu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 0)?;
        // relu'(x) = step(x) (1 if x>0 else 0) - input-based.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let deriv = unary_op::<K>(&t_capture, 10)?;
            binary_op::<K>(grad_out, &deriv, 2, "relu_grad")
        });
        Ok(out)
    }
    /// `step`.
    pub(crate) fn step<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 10)?;
        // step'(x) = 0 almost everywhere.
        push_unary_tape_entry(t.id, out.id, |grad_out| scalar_op::<K>(grad_out, 0.0, 1));
        Ok(out)
    }
    /// `elu`.
    pub(crate) fn elu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 12)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<K>(grad_out, &t_capture, 5, "elu_grad")
        });
        Ok(out)
    }
    pub(crate) fn gelu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 1)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<K>(grad_out, &t_capture, 4, "gelu_grad")
        });
        Ok(out)
    }
    pub(crate) fn mish<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 11)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<K>(grad_out, &t_capture, 6, "mish_grad")
        });
        Ok(out)
    }
    /// `tanh`.
    pub(crate) fn tanh<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 2)?;
        // tanh'(x) = 1 - out^2 (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let out_sq = binary_op::<K>(&out_capture, &out_capture, 2, "tanh_grad_sq")?;
            let neg_out_sq = unary_op::<K>(&out_sq, 5)?;
            let deriv = scalar_op::<K>(&neg_out_sq, 1.0, 0)?;
            binary_op::<K>(grad_out, &deriv, 2, "tanh_grad")
        });
        Ok(out)
    }
    /// `sigmoid`.
    pub(crate) fn sigmoid<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 3)?;
        // sigmoid'(x) = out*(1-out) (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_out = unary_op::<K>(&out_capture, 5)?;
            let one_minus_out = scalar_op::<K>(&neg_out, 1.0, 0)?;
            let deriv = binary_op::<K>(&out_capture, &one_minus_out, 2, "sigmoid_grad_deriv")?;
            binary_op::<K>(grad_out, &deriv, 2, "sigmoid_grad")
        });
        Ok(out)
    }
    /// `abs`.
    pub(crate) fn abs<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 4)?;
        // abs'(x) = sign(x) (input-based), computed as step(x) - step(-x):
        // 1 if x>0, -1 if x<0, 0 if x==0 - matches the CPU backend exactly.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_t = unary_op::<K>(&t_capture, 5)?;
            let step_pos = unary_op::<K>(&t_capture, 10)?;
            let step_neg = unary_op::<K>(&neg_t, 10)?;
            let neg_step_neg = unary_op::<K>(&step_neg, 5)?;
            let sign = binary_op::<K>(&step_pos, &neg_step_neg, 0, "abs_grad_sign")?;
            binary_op::<K>(grad_out, &sign, 2, "abs_grad")
        });
        Ok(out)
    }
    /// `neg`.
    pub(crate) fn neg<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 5)?;
        // neg'(x) = -1 (constant; no input capture needed).
        push_unary_tape_entry(t.id, out.id, |grad_out| unary_op::<K>(grad_out, 5));
        Ok(out)
    }
    /// `sqrt`.
    pub(crate) fn sqrt<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 6)?;
        // sqrt'(x) = 1/(2*out) (output-based) -> grad = grad_out/out * 0.5.
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let ratio = binary_op::<K>(grad_out, &out_capture, 3, "sqrt_grad_ratio")?;
            scalar_op::<K>(&ratio, 0.5, 1)
        });
        Ok(out)
    }
    /// `exp`.
    pub(crate) fn exp<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 7)?;
        // exp'(x) = out (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<K>(grad_out, &out_capture, 2, "exp_grad")
        });
        Ok(out)
    }
    /// `softmax(x, axis) = exp(log_softmax(x, axis))`, composed exactly the way
    /// CPU's kernel composes it, which in turn matches `candle-nn`'s:
    ///
    /// ```text
    /// max     = max_keepdim(x, axis)
    /// diff    = x - max                     // subtracting the max is what
    /// exp_d   = exp(diff)                   // keeps exp from overflowing
    /// sum_exp = sum_keepdim(exp_d, axis)
    /// log_sm  = diff - log(sum_exp)
    /// out     = exp(log_sm)
    /// ```
    ///
    /// Every step is already a tape-tracked primitive, so this writes no
    /// backward code at all: the composite's gradient is the replay of the six
    /// entries below. `softmax` advertises `training = true`, and that claim is
    /// only honest because each link in this chain records its own entry.
    ///
    /// Going through `log_softmax` rather than the shorter `exp(diff) /
    /// sum_exp` is deliberate: it is the numerically stable form, and it is the
    /// form the CPU reference uses, so the two backends agree bit for bit on
    /// the same input rather than merely agreeing to a tolerance.
    ///
    /// The subtractions broadcast: `max` and `sum_exp` keep the reduced axis at
    /// extent 1, and `binary_op` aligns them against the full shape.
    pub(crate) fn softmax<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let max = Self::max_keepdim::<K>(t, axis)?;
        let diff = Self::sub::<K>(t, &max)?;
        let exp_diff = Self::exp::<K>(&diff)?;
        let sum_exp = Self::sum_keepdim::<K>(&exp_diff, axis)?;
        let log_sum_exp = Self::log::<K>(&sum_exp)?;
        let log_softmax = Self::sub::<K>(&diff, &log_sum_exp)?;
        Self::exp::<K>(&log_softmax)
    }

    /// `log`.
    pub(crate) fn log<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 8)?;
        // log'(x) = 1/x (input-based, NOT output-based).
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<K>(grad_out, &t_capture, 3, "log_grad")
        });
        Ok(out)
    }
    /// `swish`.
    pub(crate) fn swish<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = unary_op::<K>(t, 9)?;
        // swish(x) = x*sigmoid(x); swish'(x) = out + sigmoid(x)*(1-out).
        let t_capture = t.clone();
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sig = unary_op::<K>(&t_capture, 3)?;
            let neg_out = unary_op::<K>(&out_capture, 5)?;
            let one_minus_out = scalar_op::<K>(&neg_out, 1.0, 0)?;
            let sig_term = binary_op::<K>(&sig, &one_minus_out, 2, "swish_grad_sig_term")?;
            let deriv = binary_op::<K>(&out_capture, &sig_term, 0, "swish_grad_deriv")?;
            binary_op::<K>(grad_out, &deriv, 2, "swish_grad")
        });
        Ok(out)
    }
}
