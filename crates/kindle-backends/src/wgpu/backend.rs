use crate::wgpu::dispatch;
use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
use kindle_core::prelude::*;

/// WebGPU compute backend for Kindle.
/// This backend evaluates tensor operations by compiling WGSL compute shaders
/// and dispatching them to the user's primary GPU adapter via `wgpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuBackend<T, D>(core::marker::PhantomData<(T, D)>);

#[derive(Clone)]
/// Implementation of `WgpuVar` for the respective backend..
pub struct WgpuVar {
    /// Core abstraction for `storage` within the Kindle framework..
    pub storage: WgpuStorage,
}

pub type WgpuGrads = crate::wgpu::tape::WgpuGrads;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compute flat element count from shape
// ─────────────────────────────────────────────────────────────────────────────
/// Core abstraction for `num_elements` within the Kindle framework..
pub(crate) fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend core trait
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> Backend for WgpuBackend<T, D> {
    /// Core abstraction for `Device` within the Kindle framework..
    type Device = D;
    /// Core abstraction for `FloatElem` within the Kindle framework..
    type FloatElem = T;
    /// Core abstraction for `IntElem` within the Kindle framework..
    type IntElem = i64;
    /// Core abstraction for `BackendWithDevice` within the Kindle framework..
    type BackendWithDevice<NewD: Device> = WgpuBackend<T, NewD>;

    /// Core abstraction for `Storage` within the Kindle framework..
    type Storage<K: DType> = WgpuStorage;
    /// Core abstraction for `RawVar` within the Kindle framework..
    type RawVar = WgpuVar;
    /// Core abstraction for `Grads` within the Kindle framework..
    type Grads = WgpuGrads;
    /// Core abstraction for `InnerBackend` within the Kindle framework..
    type InnerBackend = Self;

    /// Core abstraction for `shape` within the Kindle framework..
    fn shape<K: DType>(t: &Self::Storage<K>) -> Vec<usize> {
        t.shape.clone()
    }

    /// Core abstraction for `format_tensor_display` within the Kindle framework..
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> String {
        "WgpuTensor(...)".to_string()
    }

    /// Core abstraction for `format_tensor_debug` within the Kindle framework..
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> String {
        format!("WgpuTensor(shape={:?})", t.shape)
    }

    /// Core abstraction for `var_as_tensor` within the Kindle framework..
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    /// Core abstraction for `var_from_tensor` within the Kindle framework..
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(WgpuVar { storage: t.clone() })
    }

    /// Core abstraction for `var_to_device` within the Kindle framework..
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        Ok(WgpuVar {
            storage: var.storage.clone(),
        })
    }

    /// Core abstraction for `assign_var` within the Kindle framework..
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }

    /// Core abstraction for `backward` within the Kindle framework..
    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::wgpu::tape::backward(loss)
    }

    /// Core abstraction for `backward_with_nan_check` within the Kindle framework..
    fn backward_with_nan_check<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::wgpu::tape::backward_with_nan_check(loss)
    }

    /// Core abstraction for `get_grad` within the Kindle framework..
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    /// Core abstraction for `to_bytes` within the Kindle framework..
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        Ok(t.buffer.to_vec::<u8>())
    }

    /// Core abstraction for `from_bytes` within the Kindle framework..
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<Self::Storage<K>> {
        let buffer = WgpuBuffer::from_slice(bytes);
        Ok(WgpuStorage::new(buffer, shape.to_vec()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CreationOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> CreationOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `zeros` within the Kindle framework..
    fn zeros<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let n = num_elements(shape);
        let data: Vec<f32> = vec![0.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Core abstraction for `ones` within the Kindle framework..
    fn ones<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let n = num_elements(shape);
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Core abstraction for `rand` within the Kindle framework..
    fn rand<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape);
        // Simple LCG for now – GPU-side random generation would need more infrastructure
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32)
            })
            .collect();
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Core abstraction for `randn` within the Kindle framework..
    fn randn<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape);
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let lcg = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f32) / (u32::MAX as f32)
        };
        // Box-Muller transform
        let data: Vec<f32> = (0..n.div_ceil(2))
            .flat_map(|_| {
                let u1 = lcg(&mut state).max(1e-7);
                let u2 = lcg(&mut state);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                [r * theta.cos(), r * theta.sin()]
            })
            .take(n)
            .collect();
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Core abstraction for `var_zeros` within the Kindle framework..
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Core abstraction for `var_ones` within the Kindle framework..
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Core abstraction for `var_rand` within the Kindle framework..
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Core abstraction for `var_randn` within the Kindle framework..
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Core abstraction for `tensor_to_device` within the Kindle framework..
    fn tensor_to_device<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // WGPU buffers are already on the GPU
        Ok(t.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NumericOps  (add, sub, mul, div)
// ─────────────────────────────────────────────────────────────────────────────
/// Core abstraction for `binary_op` within the Kindle framework..
#[allow(clippy::extra_unused_type_parameters)]
fn binary_op<T: DType, D: Device>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    op_mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    if lhs.shape != rhs.shape {
        return Err(Error::ShapeMismatch {
            op: op_name,
            expected: lhs.shape.clone(),
            got: rhs.shape.clone(),
            msg: "shapes must match for elementwise op".to_string(),
        });
    }
    let n = num_elements(&lhs.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(lhs.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_binary(&lhs.buffer, &rhs.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, lhs.shape.clone()))
}

impl<T: DType, D: Device> NumericOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `add` within the Kindle framework..
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T, D>(lhs, rhs, 0, "add")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (add)"),
                    crate::wgpu::tape::unbroadcast(grad_out, &rhs_shape)
                        .expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `sub` within the Kindle framework..
    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T, D>(lhs, rhs, 1, "sub")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let neg_grad = unary_op::<T, D>(grad_out, 5).expect("neg (sub backward)");
                vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (sub)"),
                    crate::wgpu::tape::unbroadcast(&neg_grad, &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `mul` within the Kindle framework..
    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T, D>(lhs, rhs, 2, "mul")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let grad_lhs = binary_op::<T, D>(grad_out, &rhs_capture, 2, "mul_grad")
                    .expect("mul backward (lhs)");
                let grad_rhs = binary_op::<T, D>(grad_out, &lhs_capture, 2, "mul_grad")
                    .expect("mul backward (rhs)");
                vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (mul)"),
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `div` within the Kindle framework..
    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T, D>(lhs, rhs, 3, "div")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs = binary_op::<T, D>(grad_out, &rhs_capture, 3, "div_grad_lhs")
                    .expect("div backward (lhs)");
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq = binary_op::<T, D>(&rhs_capture, &rhs_capture, 2, "div_grad_rhs_sq")
                    .expect("rhs^2 (div backward)");
                let lhs_over_rhs_sq =
                    binary_op::<T, D>(&lhs_capture, &rhs_sq, 3, "div_grad_rhs_ratio")
                        .expect("lhs/rhs^2 (div backward)");
                let neg_ratio = unary_op::<T, D>(&lhs_over_rhs_sq, 5).expect("neg (div backward)");
                let grad_rhs = binary_op::<T, D>(grad_out, &neg_ratio, 2, "div_grad_rhs")
                    .expect("div backward (rhs)");
                vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (div)"),
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FloatOps  (scalar + unary activations)
// ─────────────────────────────────────────────────────────────────────────────
/// Core abstraction for `unary_op` within the Kindle framework..
#[allow(clippy::extra_unused_type_parameters)]
fn unary_op<T: DType, D: Device>(t: &WgpuStorage, op_mode: u32) -> Result<WgpuStorage> {
    let n = num_elements(&t.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_unary(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.clone()))
}

/// Core abstraction for `scalar_op` within the Kindle framework..
#[allow(clippy::extra_unused_type_parameters)]
fn scalar_op<T: DType, D: Device>(
    t: &WgpuStorage,
    scalar: f64,
    op_mode: u32,
) -> Result<WgpuStorage> {
    let n = num_elements(&t.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let scalar_bits = (scalar as f32).to_bits();
    let params = [op_mode, n, scalar_bits];
    dispatch::dispatch_scalar(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.clone()))
}

/// Push a single-input `TapeEntry` whose backward closure is `grad_fn`.
/// Shared by every unary `FloatOps` impl below to avoid repeating the
/// `TapeEntry { output_id, input_ids: vec![t.id], backward: ... }`
/// boilerplate at each of the ~10 call sites.
fn push_unary_tape_entry(
    t_id: crate::wgpu::storage::TensorId,
    out_id: crate::wgpu::storage::TensorId,
    grad_fn: impl Fn(&WgpuStorage) -> WgpuStorage + Send + Sync + 'static,
) {
    crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &WgpuStorage| vec![grad_fn(grad_out)]),
    });
}

impl<T: DType, D: Device> FloatOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `add_scalar_float` within the Kindle framework..
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = scalar_op::<T, D>(t, scalar, 0)?;
        // Gradient passes through unchanged (same shape, no unbroadcast
        // needed — scalar ops don't change shape).
        push_unary_tape_entry(t.id, out.id, |grad_out| grad_out.clone());
        Ok(out)
    }
    /// Core abstraction for `mul_scalar_float` within the Kindle framework..
    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = scalar_op::<T, D>(t, scalar, 1)?;
        // Gradient scales by the same constant.
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            scalar_op::<T, D>(grad_out, scalar, 1).expect("mul_scalar_float backward")
        });
        Ok(out)
    }
    /// Core abstraction for `relu` within the Kindle framework..
    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 0)?;
        // relu'(x) = step(x) (1 if x>0 else 0) — input-based.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let deriv = unary_op::<T, D>(&t_capture, 10).expect("step (relu backward)");
            binary_op::<T, D>(grad_out, &deriv, 2, "relu_grad").expect("relu backward")
        });
        Ok(out)
    }
    /// Core abstraction for `step` within the Kindle framework..
    fn step<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 10)?;
        // step'(x) = 0 almost everywhere.
        push_unary_tape_entry(t.id, out.id, |grad_out| {
            scalar_op::<T, D>(grad_out, 0.0, 1).expect("step backward (zero grad)")
        });
        Ok(out)
    }
    /// Core abstraction for `mish` within the Kindle framework..
    fn mish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        // NOT tape-wired yet: mish'(x) needs a composition this backend's
        // primitives don't cover in one pass (see ROADMAP.md C-3 follow-up).
        unary_op::<T, D>(t, 11)
    }
    /// Core abstraction for `elu` within the Kindle framework..
    fn elu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        // NOT tape-wired yet — see ROADMAP.md C-3 follow-up.
        unary_op::<T, D>(t, 12)
    }
    /// Core abstraction for `gelu` within the Kindle framework..
    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        // NOT tape-wired yet: exact erf-based gelu' has no GPU primitive here
        // — see ROADMAP.md C-3 follow-up.
        unary_op::<T, D>(t, 1)
    }
    /// Core abstraction for `tanh` within the Kindle framework..
    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 2)?;
        // tanh'(x) = 1 - out^2 (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let out_sq = binary_op::<T, D>(&out_capture, &out_capture, 2, "tanh_grad_sq")
                .expect("out^2 (tanh backward)");
            let neg_out_sq = unary_op::<T, D>(&out_sq, 5).expect("neg (tanh backward)");
            let deriv = scalar_op::<T, D>(&neg_out_sq, 1.0, 0).expect("1 - out^2 (tanh backward)");
            binary_op::<T, D>(grad_out, &deriv, 2, "tanh_grad").expect("tanh backward")
        });
        Ok(out)
    }
    /// Core abstraction for `sigmoid` within the Kindle framework..
    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 3)?;
        // sigmoid'(x) = out*(1-out) (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_out = unary_op::<T, D>(&out_capture, 5).expect("neg (sigmoid backward)");
            let one_minus_out =
                scalar_op::<T, D>(&neg_out, 1.0, 0).expect("1 - out (sigmoid backward)");
            let deriv = binary_op::<T, D>(&out_capture, &one_minus_out, 2, "sigmoid_grad_deriv")
                .expect("out*(1-out) (sigmoid backward)");
            binary_op::<T, D>(grad_out, &deriv, 2, "sigmoid_grad").expect("sigmoid backward")
        });
        Ok(out)
    }
    /// Core abstraction for `abs` within the Kindle framework..
    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 4)?;
        // abs'(x) = sign(x) (input-based), computed as step(x) - step(-x):
        // 1 if x>0, -1 if x<0, 0 if x==0 — matches the CPU backend exactly.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_t = unary_op::<T, D>(&t_capture, 5).expect("neg (abs backward)");
            let step_pos = unary_op::<T, D>(&t_capture, 10).expect("step(x) (abs backward)");
            let step_neg = unary_op::<T, D>(&neg_t, 10).expect("step(-x) (abs backward)");
            let neg_step_neg = unary_op::<T, D>(&step_neg, 5).expect("neg (abs backward)");
            let sign = binary_op::<T, D>(&step_pos, &neg_step_neg, 0, "abs_grad_sign")
                .expect("sign(x) (abs backward)");
            binary_op::<T, D>(grad_out, &sign, 2, "abs_grad").expect("abs backward")
        });
        Ok(out)
    }
    /// Core abstraction for `neg` within the Kindle framework..
    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 5)?;
        // neg'(x) = -1 (constant; no input capture needed).
        push_unary_tape_entry(t.id, out.id, |grad_out| {
            unary_op::<T, D>(grad_out, 5).expect("neg backward")
        });
        Ok(out)
    }
    /// Core abstraction for `sqrt` within the Kindle framework..
    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 6)?;
        // sqrt'(x) = 1/(2*out) (output-based) -> grad = grad_out/out * 0.5.
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let ratio = binary_op::<T, D>(grad_out, &out_capture, 3, "sqrt_grad_ratio")
                .expect("sqrt backward");
            scalar_op::<T, D>(&ratio, 0.5, 1).expect("sqrt backward (halve)")
        });
        Ok(out)
    }
    /// Core abstraction for `exp` within the Kindle framework..
    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 7)?;
        // exp'(x) = out (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T, D>(grad_out, &out_capture, 2, "exp_grad").expect("exp backward")
        });
        Ok(out)
    }
    /// Core abstraction for `log` within the Kindle framework..
    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 8)?;
        // log'(x) = 1/x (input-based, NOT output-based).
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T, D>(grad_out, &t_capture, 3, "log_grad").expect("log backward")
        });
        Ok(out)
    }
    /// Core abstraction for `swish` within the Kindle framework..
    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T, D>(t, 9)?;
        // swish(x) = x*sigmoid(x); swish'(x) = out + sigmoid(x)*(1-out).
        let t_capture = t.clone();
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sig = unary_op::<T, D>(&t_capture, 3).expect("sigmoid(x) (swish backward)");
            let neg_out = unary_op::<T, D>(&out_capture, 5).expect("neg (swish backward)");
            let one_minus_out =
                scalar_op::<T, D>(&neg_out, 1.0, 0).expect("1 - out (swish backward)");
            let sig_term = binary_op::<T, D>(&sig, &one_minus_out, 2, "swish_grad_sig_term")
                .expect("sigmoid(x)*(1-out) (swish backward)");
            let deriv = binary_op::<T, D>(&out_capture, &sig_term, 0, "swish_grad_deriv")
                .expect("swish backward deriv");
            binary_op::<T, D>(grad_out, &deriv, 2, "swish_grad").expect("swish backward")
        });
        Ok(out)
    }

    /// Core abstraction for `softmax` within the Kindle framework..
    fn softmax<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // NOT tape-wired yet: this is a single monolithic GPU kernel (not
        // composed from log_softmax like the CPU backend), so its backward
        // needs its own dedicated implementation — see ROADMAP.md C-3
        // follow-up.
        let shape = &t.shape;
        // Flatten to [batch, n] where n = shape[dim..] product
        let n: usize = shape[dim..].iter().product();
        let batch: usize = shape[..dim].iter().product::<usize>().max(1);
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
        dispatch::dispatch_softmax(&t.buffer, &out_buf, batch as u32, n as u32);
        Ok(WgpuStorage::new(out_buf, shape.clone()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorOps  (reshape, transpose, matmul, narrow, flatten, squeeze, stack, concat, etc.)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> TensorOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `matmul` within the Kindle framework..
    fn matmul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: vec![2],
                got: vec![lhs.shape.len(), rhs.shape.len()],
                msg: "matmul requires at least 2D inputs".to_string(),
            });
        }

        let lhs_rank = lhs.shape.len();
        let rhs_rank = rhs.shape.len();

        let m = lhs.shape[lhs_rank - 2] as u32;
        let k = lhs.shape[lhs_rank - 1] as u32;
        let n = rhs.shape[rhs_rank - 1] as u32;

        if k as usize != rhs.shape[rhs_rank - 2] {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.clone(),
                got: rhs.shape.clone(),
                msg: "matmul inner dims must match".to_string(),
            });
        }

        // Compute batch dims
        let mut lhs_batch = 1;
        for i in 0..lhs_rank - 2 {
            lhs_batch *= lhs.shape[i];
        }
        let mut rhs_batch = 1;
        for i in 0..rhs_rank - 2 {
            rhs_batch *= rhs.shape[i];
        }

        let batch = core::cmp::max(lhs_batch, rhs_batch);
        if lhs_batch != 1 && rhs_batch != 1 && lhs_batch != rhs_batch {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.clone(),
                got: rhs.shape.clone(),
                msg: "matmul batch dims incompatible".to_string(),
            });
        }

        let lhs_stride_b = if lhs_batch == 1 { 0 } else { m * k };
        let rhs_stride_b = if rhs_batch == 1 { 0 } else { k * n };

        // Output shape matches the larger batched input
        let mut out_shape = if lhs_batch > 1 {
            lhs.shape[..lhs_rank - 2].to_vec()
        } else {
            rhs.shape[..rhs_rank - 2].to_vec()
        };
        if out_shape.is_empty() && batch > 1 {
            out_shape.push(batch);
        }
        out_shape.push(m as usize);
        out_shape.push(n as usize);

        let state = crate::wgpu::device::get_device_state();
        let shader = include_str!("shaders/matmul.wgsl");
        let pipeline = crate::wgpu::pipeline::get_or_create_pipeline("matmul", shader, "main");

        let out_buf =
            WgpuBuffer::new_zeros((batch as u32 * m * n) as usize * core::mem::size_of::<f32>());
        let shape_data = [m, k, n, batch as u32, lhs_stride_b, rhs_stride_b];
        let shape_buf = WgpuBuffer::from_slice(&shape_data);

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = state
            .device
            .create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("Matmul BG"),
                layout: &bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lhs.buffer.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shape_buf.buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = state
            .device
            .create_command_encoder(&::wgpu::CommandEncoderDescriptor {
                label: Some("Matmul"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
                label: Some("Matmul"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), batch as u32);
        }
        state.queue.submit(core::iter::once(encoder.finish()));
        let out = WgpuStorage::new(out_buf, out_shape);

        // Backward: grad_lhs = grad_out @ rhs^T, grad_rhs = lhs^T @ grad_out,
        // composed from Self::matmul + Self::transpose recursion (mirrors the
        // CPU backend's batched_matmul_impl exactly) rather than a bespoke
        // kernel. Self::matmul already broadcasts a batch=1 operand against
        // the other's batch shape internally (lhs_stride_b/rhs_stride_b=0
        // above), so `grad_out @ rhs^T`/`lhs^T @ grad_out` naturally come out
        // at the OUTPUT batch shape; `unbroadcast` then reduces back down to
        // each operand's own original (possibly batch=1) shape.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let rhs_rank = rhs_capture.shape.len();
                let rhs_t = Self::transpose::<K>(&rhs_capture, rhs_rank - 2, rhs_rank - 1)
                    .expect("rhs^T (matmul backward)");
                let grad_lhs_full = Self::matmul::<K>(grad_out, &rhs_t)
                    .expect("grad_out @ rhs^T (matmul backward)");

                let lhs_rank = lhs_capture.shape.len();
                let lhs_t = Self::transpose::<K>(&lhs_capture, lhs_rank - 2, lhs_rank - 1)
                    .expect("lhs^T (matmul backward)");
                let grad_rhs_full = Self::matmul::<K>(&lhs_t, grad_out)
                    .expect("lhs^T @ grad_out (matmul backward)");

                vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs_full, &lhs_shape)
                        .expect("unbroadcast lhs (matmul backward)"),
                    crate::wgpu::tape::unbroadcast(&grad_rhs_full, &rhs_shape)
                        .expect("unbroadcast rhs (matmul backward)"),
                ]
            }),
        });
        Ok(out)
    }

    /// Core abstraction for `reshape` within the Kindle framework..
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        if num_elements(&t.shape) != num_elements(shape) {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: t.shape.clone(),
                got: shape.to_vec(),
                msg: "total elements must match".to_string(),
            });
        }
        let out = WgpuStorage::new(t.buffer.clone(), shape.to_vec());
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![Self::reshape::<K>(grad_out, &original_shape).expect("reshape backward")]
            }),
        });
        Ok(out)
    }

    /// Core abstraction for `transpose` within the Kindle framework..
    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let mut new_shape = shape.clone();
        new_shape.swap(dim1, dim2);

        let out_n = num_elements(&new_shape) as u32;
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);

        let mut aux = (0..shape.len()).collect::<Vec<_>>();
        aux.swap(dim1, dim2);

        let params = dispatch::prepare_shape_params(
            2, // op_mode = transpose
            out_n, &new_shape, shape, &aux,
        );

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, new_shape);

        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![Self::transpose::<K>(grad_out, dim1, dim2).expect("transpose backward")]
            }),
        });
        Ok(out)
    }

    /// Core abstraction for `flatten` within the Kindle framework..
    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let flat_size: usize = shape[start_dim..=end_dim].iter().product();
        let mut new_shape: Vec<usize> = shape[..start_dim].to_vec();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&shape[end_dim + 1..]);
        Self::reshape::<K>(t, &new_shape)
    }

    /// Core abstraction for `squeeze` within the Kindle framework..
    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut new_shape = t.shape.clone();
        if new_shape[dim] == 1 {
            new_shape.remove(dim);
        }
        Self::reshape::<K>(t, &new_shape)
    }

    fn narrow<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let mut new_shape = shape.clone();
        new_shape[dim] = len;

        let out_n = num_elements(&new_shape) as u32;
        let out_buf = WgpuBuffer::new_zeros(out_n as usize * 4);

        let mut aux = vec![0usize; shape.len()];
        aux[dim] = start;

        let params = dispatch::prepare_shape_params(
            0, // op_mode = slice
            out_n, &new_shape, shape, &aux,
        );

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, new_shape);

        let original_shape = t.shape.clone();
        let mut region_start = vec![0usize; original_shape.len()];
        region_start[dim] = start;
        let (t_id, out_id) = (t.id, out.id);

        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![crate::wgpu::storage::scatter_into_zeros(
                    &original_shape,
                    &region_start,
                    grad_out,
                )]
            }),
        });
        Ok(out)
    }

    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_n = num_elements(shape) as u32;
        let out_buf = WgpuBuffer::new_zeros(out_n as usize * 4);

        let params = dispatch::prepare_shape_params(
            3, // op_mode = broadcast
            out_n,
            shape,
            &t.shape,
            &[],
        );

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, shape.to_vec());

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &original_shape)
                        .expect("broadcast_as backward"),
                ]
            }),
        });
        Ok(out)
    }

    /// Core abstraction for `broadcast_left` within the Kindle framework..
    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = shape.to_vec();
        target_shape.extend_from_slice(&t.shape);
        Self::broadcast_as::<K>(t, &target_shape)
    }

    /// Core abstraction for `slice` within the Kindle framework..
    fn slice<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut out = t.clone();
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            out = Self::narrow::<K>(&out, dim, start, end - start)?;
        }
        Ok(out)
    }

    /// Core abstraction for `stack` within the Kindle framework..
    fn stack<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::Msg("stack: empty tensor list".to_string()));
        }
        // Unsqueeze each tensor at `dim` then concat
        let mut unsqueezed = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let mut target_shape = t.shape.clone();
            target_shape.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &target_shape)?);
        }
        let refs: Vec<&<Self as Backend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// Core abstraction for `concat` within the Kindle framework..
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::Msg("concat: empty tensor list".to_string()));
        }
        let rank = tensors[0].shape.len();
        let mut out_shape = tensors[0].shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        let out_n = num_elements(&out_shape);
        let out_buf = WgpuBuffer::new_zeros(out_n * 4);

        let mut current_offset = 0usize;
        for t in tensors {
            let in_n = num_elements(&t.shape) as u32;
            let mut aux = vec![0usize; rank];
            aux[dim] = current_offset;

            let params = dispatch::prepare_shape_params(
                1, // op_mode = paste
                in_n, &out_shape, &t.shape, &aux,
            );
            dispatch::dispatch_shape(&t.buffer, &out_buf, &params);

            current_offset += t.shape[dim];
        }
        let out = WgpuStorage::new(out_buf, out_shape);

        // Calculate cumulative offsets for backward
        let mut cumulative_offsets = Vec::with_capacity(tensors.len());
        let mut running = 0usize;
        for t in tensors.iter() {
            cumulative_offsets.push(running);
            running += t.shape[dim];
        }

        let out_id = out.id;
        let input_ids: Vec<_> = tensors.iter().map(|t| t.id).collect();
        let input_dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape[dim]).collect();
        let offsets = cumulative_offsets;
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids,
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                offsets
                    .iter()
                    .zip(input_dim_sizes.iter())
                    .map(|(&offset, &len)| {
                        Self::narrow::<K>(grad_out, dim, offset, len).expect("concat backward")
                    })
                    .collect()
            }),
        });

        Ok(out)
    }

    /// Core abstraction for `float_to_scalar` within the Kindle framework..
    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.first().copied().unwrap_or(0.0) as f64)
    }

    /// Core abstraction for `float_to_vec1` within the Kindle framework..
    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<f64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    /// Core abstraction for `int_to_scalar` within the Kindle framework..
    fn int_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.first().copied().unwrap_or(0.0) as i64)
    }

    /// Core abstraction for `int_to_vec1` within the Kindle framework..
    fn int_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<i64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.iter().map(|&x| x as i64).collect())
    }

    /// Core abstraction for `tensor_to_dtype` within the Kindle framework..
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as Backend>::Storage<K>,
        _dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        // Simple passthrough (all stored as f32 internally)
        Ok(WgpuStorage {
            buffer: t.buffer.clone(),
            shape: t.shape.clone(),
            strides: t.strides.clone(),
            id: crate::wgpu::storage::TensorId::next(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReductionOps
// ─────────────────────────────────────────────────────────────────────────────
/// Core abstraction for `reduce_all_to_storage` within the Kindle framework..
fn reduce_all_to_storage(t: &WgpuStorage, mode: u32) -> WgpuStorage {
    let n = num_elements(&t.shape) as u32;
    let out = dispatch::dispatch_reduce_all(&t.buffer, n, mode);
    WgpuStorage::new(out, vec![1])
}

/// Core abstraction for `reduce_dim_to_storage` within the Kindle framework..
fn reduce_dim_to_storage(t: &WgpuStorage, dim: usize, mode: u32, keepdim: bool) -> WgpuStorage {
    let shape = &t.shape;
    let mut out_shape = shape.clone();
    out_shape[dim] = 1;
    let out_n = num_elements(&out_shape);

    let dim_size = shape[dim] as u32;
    let mut inner_stride = 1usize;
    for d in (dim + 1..shape.len()).rev() {
        inner_stride *= shape[d];
    }

    // mode mapping: CPU reduce_dim mode (0=sum, 1=max, 2=min) maps directly
    // to my shader ops (0=sum, 2=max, 3=min).
    let op_mode = match mode {
        0 => 0u32, // sum
        1 => 2u32, // max
        2 => 3u32, // min
        _ => panic!("Unknown reduce dim mode"),
    };

    let out_buf = WgpuBuffer::new_zeros(out_n * 4);
    dispatch::dispatch_reduce_dim(
        &t.buffer,
        &out_buf,
        op_mode,
        dim_size,
        inner_stride as u32,
        out_n as u32,
    );

    let final_shape = if keepdim {
        out_shape
    } else {
        let mut s = shape.clone();
        s.remove(dim);
        s
    };
    WgpuStorage::new(out_buf, final_shape)
}

impl<T: DType, D: Device> ReductionOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `sum_all` within the Kindle framework..
    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 0);
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![
                    Self::broadcast_as::<K>(grad_out, &original_shape)
                        .expect("sum_all backward failed"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `mean_all` within the Kindle framework..
    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_all_to_storage(t, 0);
        let n = num_elements(&t.shape) as f64;
        let out = scalar_op::<T, D>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let scaled = scalar_op::<T, D>(grad_out, 1.0 / n, 1).unwrap();
                vec![
                    Self::broadcast_as::<K>(&scaled, &original_shape)
                        .expect("mean_all backward failed"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `max_all` within the Kindle framework..
    fn max_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_all_to_storage(t, 1))
    }
    /// Core abstraction for `min_all` within the Kindle framework..
    fn min_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_all_to_storage(t, 2))
    }

    /// Core abstraction for `sum_dim` within the Kindle framework..
    fn sum_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, false);
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.clone();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape).unwrap();
                vec![
                    Self::broadcast_as::<K>(&keepdim, &original_shape)
                        .expect("sum_dim backward failed"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `sum_keepdim` within the Kindle framework..
    fn sum_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, true);
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                vec![
                    Self::broadcast_as::<K>(grad_out, &original_shape)
                        .expect("sum_keepdim backward failed"),
                ]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `mean_dim` within the Kindle framework..
    fn mean_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, false);
        let n = t.shape[dim] as f64;
        let out = scalar_op::<T, D>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.clone();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape).unwrap();
                let expanded = Self::broadcast_as::<K>(&keepdim, &original_shape).unwrap();
                vec![scalar_op::<T, D>(&expanded, 1.0 / n, 1).unwrap()]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `mean_keepdim` within the Kindle framework..
    fn mean_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, true);
        let n = t.shape[dim] as f64;
        let out = scalar_op::<T, D>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let expanded = Self::broadcast_as::<K>(grad_out, &original_shape).unwrap();
                vec![scalar_op::<T, D>(&expanded, 1.0 / n, 1).unwrap()]
            }),
        });
        Ok(out)
    }
    /// Core abstraction for `max_dim` within the Kindle framework..
    fn max_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 1, false))
    }
    /// Core abstraction for `max_keepdim` within the Kindle framework..
    fn max_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 1, true))
    }
    /// Core abstraction for `min_dim` within the Kindle framework..
    fn min_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 2, false))
    }
    /// Core abstraction for `min_keepdim` within the Kindle framework..
    fn min_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 2, true))
    }

    /// Core abstraction for `argmax` within the Kindle framework..
    fn argmax<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let buf = WgpuBuffer::from_slice(&[idx as u32]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                let mut out_shape = shape.clone();
                out_shape[d] = 1;
                let out_n = num_elements(&out_shape);

                let dim_size = shape[d] as u32;
                let mut inner_stride = 1usize;
                for dd in (d + 1..shape.len()).rev() {
                    inner_stride *= shape[dd];
                }

                let out_buf = WgpuBuffer::new_zeros(out_n * 4);
                dispatch::dispatch_reduce_dim(
                    &t.buffer,
                    &out_buf,
                    4, // argmax
                    dim_size,
                    inner_stride as u32,
                    out_n as u32,
                );

                let mut final_shape = shape.clone();
                final_shape.remove(d);
                Ok(WgpuStorage::new(out_buf, final_shape))
            }
        }
    }

    /// Core abstraction for `argmin` within the Kindle framework..
    fn argmin<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let buf = WgpuBuffer::from_slice(&[idx as u32]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                let mut out_shape = shape.clone();
                out_shape[d] = 1;
                let out_n = num_elements(&out_shape);

                let dim_size = shape[d] as u32;
                let mut inner_stride = 1usize;
                for dd in (d + 1..shape.len()).rev() {
                    inner_stride *= shape[dd];
                }

                let out_buf = WgpuBuffer::new_zeros(out_n * 4);
                dispatch::dispatch_reduce_dim(
                    &t.buffer,
                    &out_buf,
                    5, // argmin
                    dim_size,
                    inner_stride as u32,
                    out_n as u32,
                );

                let mut final_shape = shape.clone();
                final_shape.remove(d);
                Ok(WgpuStorage::new(out_buf, final_shape))
            }
        }
    }

    /// Core abstraction for `topk` within the Kindle framework..
    fn topk<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        <Self as Backend>::Storage<K>,
        <Self as Backend>::Storage<KInt>,
    )> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "topk",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("topk: axis {} out of range", dim),
            });
        }
        let k = k.min(shape[dim]);
        let data: Vec<f32> = t.buffer.to_vec::<f32>();

        let mut out_shape = shape.clone();
        out_shape[dim] = k;
        let mut base_shape = shape.clone();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape);
        let mut out_vals = vec![0.0f32; num_elements(&out_shape)];
        let mut out_indices = vec![0u32; num_elements(&out_shape)];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += coords[dd] * stride;
                    stride *= shape[dd];
                }
                slice_vals.push((data[flat], j as u32));
            }

            if largest {
                slice_vals
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }

            let mut out_coords = coords.clone();
            for (j, &(val, idx)) in slice_vals.iter().enumerate().take(k) {
                out_coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..out_shape.len()).rev() {
                    flat += out_coords[dd] * stride;
                    stride *= out_shape[dd];
                }
                out_vals[flat] = val;
                out_indices[flat] = idx;
            }
        }
        let buf_vals = WgpuBuffer::from_slice(&out_vals);
        let buf_indices = WgpuBuffer::from_slice(&out_indices);
        Ok((
            WgpuStorage::new(buf_vals, out_shape.clone()),
            WgpuStorage::new(buf_indices, out_shape),
        ))
    }

    /// Core abstraction for `argsort` within the Kindle framework..
    fn argsort<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "argsort",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("argsort: axis {} out of range", dim),
            });
        }
        let data: Vec<f32> = t.buffer.to_vec::<f32>();

        let mut base_shape = shape.clone();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape);
        let mut out = vec![0u32; num_elements(shape)];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += coords[dd] * stride;
                    stride *= shape[dd];
                }
                slice_vals.push((data[flat], j as u32));
            }

            if descending {
                slice_vals
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }

            let mut out_coords = coords.clone();
            for (j, &(_, idx)) in slice_vals.iter().enumerate() {
                out_coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += out_coords[dd] * stride;
                    stride *= shape[dd];
                }
                out[flat] = idx;
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.clone()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModuleOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> ModuleOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `embedding` within the Kindle framework..
    fn embedding<K: DType, KInt: DType>(
        indices: &<Self as Backend>::Storage<KInt>,
        weight: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let embed_dim = weight.shape[1];
        let vocab_size = weight.shape[0];
        let seq_len = num_elements(&indices.shape);
        let out_buf = WgpuBuffer::new_zeros(seq_len * embed_dim * 4);

        dispatch::dispatch_embedding(
            &indices.buffer,
            &weight.buffer,
            &out_buf,
            seq_len as u32,
            embed_dim as u32,
            vocab_size as u32,
        );

        Ok(WgpuStorage::new(out_buf, vec![seq_len, embed_dim]))
    }

    /// Core abstraction for `layer_norm` within the Kindle framework..
    fn layer_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let rank = t.shape.len();
        let last_dim = rank - 1;

        let mean = Self::mean_keepdim::<K>(t, last_dim)?;
        let mean_b = Self::broadcast_as::<K>(&mean, &t.shape)?;
        let centered = Self::sub::<K>(t, &mean_b)?;
        let sq = Self::mul::<K>(&centered, &centered)?;
        let variance = Self::mean_keepdim::<K>(&sq, last_dim)?;
        let var_plus_eps = Self::add_scalar_float::<K>(&variance, eps as f64)?;
        let std = Self::sqrt::<K>(&var_plus_eps)?;
        let std_b = Self::broadcast_as::<K>(&std, &t.shape)?;
        let normalized = Self::div::<K>(&centered, &std_b)?;

        let mut w_shape = vec![1; rank];
        w_shape[last_dim] = weight.shape[0];
        let w_reshaped = Self::reshape::<K>(weight, &w_shape)?;
        let weight_b = Self::broadcast_as::<K>(&w_reshaped, &t.shape)?;
        let scaled = Self::mul::<K>(&normalized, &weight_b)?;

        match bias {
            Some(b) => {
                let b_reshaped = Self::reshape::<K>(b, &w_shape)?;
                let bias_b = Self::broadcast_as::<K>(&b_reshaped, &t.shape)?;
                Self::add::<K>(&scaled, &bias_b)
            }
            None => {
                let n = num_elements(&t.shape);
                let zeros = WgpuStorage::new(WgpuBuffer::new_zeros(n * 4), t.shape.clone());
                Self::add::<K>(&scaled, &zeros)
            }
        }
    }

    /// Core abstraction for `batch_norm` within the Kindle framework..
    fn batch_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: Option<&<Self as Backend>::Storage<K>>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        running_mean: Option<&<Self as Backend>::Storage<K>>,
        running_var: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let rank = t.shape.len();
        let c = if rank > 1 { t.shape[1] } else { t.shape[0] };

        let mut param_shape = vec![1; rank];
        if rank > 1 {
            param_shape[1] = c;
        } else {
            param_shape[0] = c;
        }

        let rm = match running_mean {
            Some(m) => Self::reshape::<K>(m, &param_shape)?,
            None => WgpuStorage::new(WgpuBuffer::new_zeros(c * 4), param_shape.clone()),
        };
        let rv = match running_var {
            Some(v) => Self::reshape::<K>(v, &param_shape)?,
            None => {
                let out = WgpuStorage::new(WgpuBuffer::new_zeros(c * 4), param_shape.clone());
                Self::add_scalar_float::<K>(&out, 1.0)?
            }
        };

        let w = match weight {
            Some(w) => Self::reshape::<K>(w, &param_shape)?,
            None => {
                let out = WgpuStorage::new(WgpuBuffer::new_zeros(c * 4), param_shape.clone());
                Self::add_scalar_float::<K>(&out, 1.0)?
            }
        };

        let b = match bias {
            Some(b) => Self::reshape::<K>(b, &param_shape)?,
            None => WgpuStorage::new(WgpuBuffer::new_zeros(c * 4), param_shape.clone()),
        };

        let rv_plus_eps = Self::add_scalar_float::<K>(&rv, eps as f64)?;
        let std = Self::sqrt::<K>(&rv_plus_eps)?;

        let rm_b = Self::broadcast_as::<K>(&rm, &t.shape)?;
        let std_b = Self::broadcast_as::<K>(&std, &t.shape)?;
        let w_b = Self::broadcast_as::<K>(&w, &t.shape)?;
        let b_b = Self::broadcast_as::<K>(&b, &t.shape)?;

        let centered = Self::sub::<K>(t, &rm_b)?;
        let normalized = Self::div::<K>(&centered, &std_b)?;
        let scaled = Self::mul::<K>(&normalized, &w_b)?;
        Self::add::<K>(&scaled, &b_b)
    }

    /// Core abstraction for `adaptive_avg_pool2d` within the Kindle framework..
    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape; // [N, C, H, W]
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = output_size;
        let out_buf = WgpuBuffer::new_zeros(n * c * oh * ow * 4);

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 0, // mode 0 = adaptive_avg
            n as u32, c as u32, h as u32, w as u32, oh as u32, ow as u32, 0, 0, 0, 0, 0, 0, 0,
            0, // unused kernel params
        );

        Ok(WgpuStorage::new(out_buf, vec![n, c, oh, ow]))
    }

    /// Core abstraction for `avg_pool2d` within the Kindle framework..
    fn avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        let out_buf = WgpuBuffer::new_zeros(n * c * oh * ow * 4);

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 1, // mode 1 = avg
            n as u32, c as u32, h as u32, w as u32, oh as u32, ow as u32, kh as u32, kw as u32,
            sh as u32, sw as u32, ph as u32, pw as u32, 1, 1, // dilation = 1
        );

        Ok(WgpuStorage::new(out_buf, vec![n, c, oh, ow]))
    }

    /// Core abstraction for `max_pool2d` within the Kindle framework..
    fn max_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;
        let oh = (h + 2 * ph - eff_kh) / sh + 1;
        let ow = (w + 2 * pw - eff_kw) / sw + 1;

        let out_buf = WgpuBuffer::new_zeros(n * c * oh * ow * 4);

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 2, // mode 2 = max
            n as u32, c as u32, h as u32, w as u32, oh as u32, ow as u32, kh as u32, kw as u32,
            sh as u32, sw as u32, ph as u32, pw as u32, dh as u32, dw as u32,
        );

        Ok(WgpuStorage::new(out_buf, vec![n, c, oh, ow]))
    }

    /// Core abstraction for `conv1d` within the Kindle framework..
    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Implement as conv2d over a fake spatial H=1 dimension
        // Input:  [N, C_in, L]       -> [N, C_in, 1, L]
        // Weight: [C_out, C_in, Kl]  -> [C_out, C_in, 1, Kl]
        let inp_shape = &t.shape; // [N, C_in, L]
        let w_shape = &weight.shape; // [C_out, C_in/groups, Kl]
        let (n, c_in, l_in) = (inp_shape[0], inp_shape[1], inp_shape[2]);
        let (c_out, _, kl) = (w_shape[0], w_shape[1], w_shape[2]);

        let inp4d = WgpuStorage {
            buffer: t.buffer.clone(),
            shape: vec![n, c_in, 1, l_in],
            strides: vec![],
            id: crate::wgpu::storage::TensorId::next(),
        };
        let w4d = WgpuStorage {
            buffer: weight.buffer.clone(),
            shape: vec![c_out, w_shape[1], 1, kl],
            strides: vec![],
            id: crate::wgpu::storage::TensorId::next(),
        };
        let bias4d = bias;

        let out = Self::conv2d::<K>(&inp4d, &w4d, bias4d, stride, padding, dilation, groups)?;
        // out: [N, C_out, 1, L_out]  -> [N, C_out, L_out]
        let l_out = out.shape[3];
        Ok(WgpuStorage::new(out.buffer, vec![n, c_out, l_out]))
    }

    /// Core abstraction for `conv2d` within the Kindle framework..
    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // im2col + batched matmul (groups=1 fast path; groups>1 loop)
        let shape = &t.shape; // [N, C_in, H, W]
        let ws = &weight.shape; // [C_out, C_in/groups, Kh, Kw]
        if shape.len() != 4 || ws.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![4],
                got: vec![shape.len()],
                msg: "expected 4D input and weight".into(),
            });
        }
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, c_in_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let g = groups;
        let c_in_g = c_in / g;
        assert_eq!(c_in_g, c_in_per_g, "groups mismatch");

        let h_out = (h_in + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;

        // ── im2col ────────────────────────────────────────────────────────────
        // col: [N, C_in * Kh * Kw, H_out * W_out]
        let col_channels = c_in * kh * kw;
        let col_spatial = h_out * w_out;
        let col_buf = WgpuBuffer::new_zeros(batch * col_channels * col_spatial * 4);

        let params: [u32; 14] = [
            batch as u32,
            c_in as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            padding as u32,
            padding as u32,
            dilation as u32,
            dilation as u32,
        ];
        dispatch::dispatch_im2col(&t.buffer, &col_buf, &params);

        // ── matmul per batch: weight [C_out/g, C_in/g * Kh * Kw] x col_slice -> out_slice ──
        // For g=1 this is a single batched matmul.
        // For g>1 we slice and loop.
        let _w_data: Vec<f32> = weight.buffer.to_vec::<f32>();
        let _col_data: Vec<f32> = col_buf.to_vec::<f32>();
        let k_size = c_in_g * kh * kw;

        if g == 1 {
            // GPU batched matmul fast path
            let w_storage = WgpuStorage::new(weight.buffer.clone(), vec![c_out, k_size]);
            let col_storage = WgpuStorage::new(col_buf, vec![batch, k_size, col_spatial]);
            let out_storage = Self::matmul::<K>(&w_storage, &col_storage)?;

            // Apply bias on GPU (if present)
            if let Some(b_storage) = bias {
                dispatch::dispatch_bias_add(
                    &out_storage.buffer,
                    &b_storage.buffer,
                    batch as u32,
                    c_out as u32,
                    col_spatial as u32,
                );
            }

            return Ok(WgpuStorage::new(
                out_storage.buffer,
                vec![batch, c_out, h_out, w_out],
            ));
        }

        // ── Direct convolution per batch for groups > 1 ──
        let out_buf = WgpuBuffer::new_zeros(batch * c_out * h_out * w_out * 4);
        let conv_params: [u32; 16] = [
            batch as u32,
            c_in as u32,
            h_in as u32,
            w_in as u32,
            c_out as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            padding as u32,
            padding as u32,
            dilation as u32,
            dilation as u32,
            groups as u32,
        ];

        dispatch::dispatch_conv2d_direct(&t.buffer, &weight.buffer, &out_buf, &conv_params);

        if let Some(b_storage) = bias {
            let spatial = h_out * w_out;
            dispatch::dispatch_bias_add(
                &out_buf,
                &b_storage.buffer,
                batch as u32,
                c_out as u32,
                spatial as u32,
            );
        }

        Ok(WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]))
    }

    /// Core abstraction for `conv_transpose2d` within the Kindle framework..
    fn conv_transpose2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape; // [N, C_in, H_in, W_in]
        let ws = &weight.shape; // [C_in, C_out/groups, kH, kW]

        if shape.len() != 4 || ws.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![4],
                got: vec![shape.len()],
                msg: "expected 4D input and weight".into(),
            });
        }

        let batch = shape[0];
        let c_in = shape[1];
        let h_in = shape[2];
        let w_in = shape[3];

        let w_c_in = ws[0];
        let c_out_per_group = ws[1];
        let kh = ws[2];
        let kw = ws[3];

        let c_out = c_out_per_group * groups;
        assert_eq!(c_in, w_c_in, "Input channels must match weight in_channels");

        let h_out = (h_in - 1) * stride + dilation * (kh - 1) + output_padding + 1;
        let h_out = h_out.saturating_sub(2 * padding);
        let w_out = (w_in - 1) * stride + dilation * (kw - 1) + output_padding + 1;
        let w_out = w_out.saturating_sub(2 * padding);

        let out_buf = WgpuBuffer::new_zeros(batch * c_out * h_out * w_out * 4);

        let params: [u32; 16] = [
            batch as u32,
            c_in as u32,
            c_out as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            padding as u32,
            padding as u32,
            dilation as u32,
            dilation as u32,
            groups as u32,
        ];

        dispatch::dispatch_conv_transpose2d(&t.buffer, &weight.buffer, &out_buf, &params);
        let out_storage = WgpuStorage::new(out_buf.clone(), vec![batch, c_out, h_out, w_out]);

        if let Some(b_storage) = bias {
            let spatial = h_out * w_out;
            dispatch::dispatch_bias_add(
                &out_buf,
                &b_storage.buffer,
                batch as u32,
                c_out as u32,
                spatial as u32,
            );
        }

        Ok(out_storage)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LossOps (cross_entropy delegated to base trait which composes from float/reduce ops)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> LossOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `cross_entropy_loss` within the Kindle framework..
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<KInt>,
        reduction: kindle_core::prelude::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Compute softmax then nll
        let softmax = <Self as FloatOps<Self>>::softmax::<K>(pred, pred.shape.len() - 1)?;
        let log_sm = <Self as FloatOps<Self>>::log::<K>(&softmax)?;

        let batch = num_elements(&target.shape);
        let n_classes = pred.shape.last().copied().unwrap_or(1);

        let nll_buf = WgpuBuffer::new_zeros(batch * 4);
        dispatch::dispatch_nll_loss(
            &log_sm.buffer,
            &target.buffer,
            &nll_buf,
            batch as u32,
            n_classes as u32,
        );

        match reduction {
            kindle_core::prelude::Reduction::None => Ok(WgpuStorage::new(nll_buf, vec![batch])),
            kindle_core::prelude::Reduction::Mean => {
                let out_buf = WgpuBuffer::new_zeros(4);
                dispatch::dispatch_reduce_dim(
                    &nll_buf,
                    &out_buf,
                    1, // mean
                    batch as u32,
                    1,
                    1,
                );
                Ok(WgpuStorage::new(out_buf, vec![1]))
            }
            kindle_core::prelude::Reduction::Sum => {
                let out_buf = WgpuBuffer::new_zeros(4);
                dispatch::dispatch_reduce_dim(
                    &nll_buf,
                    &out_buf,
                    0, // sum
                    batch as u32,
                    1,
                    1,
                );
                Ok(WgpuStorage::new(out_buf, vec![1]))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedOps  (Q8_0: CPU-side encode/decode, GPU matmul via dequant)
// ─────────────────────────────────────────────────────────────────────────────
//
// WgpuStorage stores raw bytes in a WgpuBuffer. For Q8_0 quantized tensors,
// the buffer holds packed BlockQ8_0 structs (34 bytes each):
//   [0..1]  = f16 scale `d` (little-endian)
//   [2..33] = 32 × i8 quantized weights
//
// This mirrors the NativeBackend's `BlockQ8_0` layout, allowing byte-level
// interoperability.  The encode/decode runs on the CPU (WgpuBuffer::to_vec /
// from_slice); a GPU-native WGSL kernel is deferred post-0.1.0.
impl<T: DType, D: Device> QuantizedOps<Self> for WgpuBackend<T, D> {
    /// Quantize a contiguous f32 tensor to Q8_0 format.
    ///
    /// Only `K = f32` and `Q = Q8_0` are supported; any other combination
    /// returns `UnsupportedBackendOperation`.
    fn quantize<K: FloatDType, Q: QuantDType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<Q>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>()
            || core::any::TypeId::of::<K>() != core::any::TypeId::of::<f32>()
        {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "WGPU (only F32 to Q8_0 supported)",
            });
        }

        let f32_data: Vec<f32> = t.buffer.to_vec::<f32>();
        let n = f32_data.len();
        if !n.is_multiple_of(32) {
            return Err(Error::Msg(alloc::format!(
                "WGPU quantize Q8_0: length must be a multiple of 32, got {}",
                n
            )));
        }

        // Encode as packed BlockQ8_0 bytes: [f16_le(d), i8×32] per block.
        let blocks = n / 32;
        let block_bytes = 2 + 32; // sizeof(f16) + 32 × sizeof(i8)
        let mut out_bytes: Vec<u8> = Vec::with_capacity(blocks * block_bytes);

        for chunk in f32_data.chunks_exact(32) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = max_abs / 127.0;
            let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

            // Write scale as f16 little-endian.
            let d_f16 = kindle_core::prelude::f16::from_f32(d);
            let d_bits = d_f16.to_bits();
            out_bytes.push((d_bits & 0xFF) as u8);
            out_bytes.push((d_bits >> 8) as u8);

            // Write 32 quantized i8 values.
            for &v in chunk {
                let q = (v * inv_d).round().clamp(-128.0, 127.0) as i8;
                out_bytes.push(q as u8);
            }
        }

        let buf = WgpuBuffer::from_slice::<u8>(&out_bytes);
        Ok(WgpuStorage::new(buf, t.shape.clone()))
    }

    /// Dequantize a Q8_0 tensor back to f32.
    fn dequantize<Q: QuantDType, K: FloatDType>(
        t: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>()
            || core::any::TypeId::of::<K>() != core::any::TypeId::of::<f32>()
        {
            return Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "WGPU (only Q8_0 to F32 supported)",
            });
        }

        let raw: Vec<u8> = t.buffer.to_vec::<u8>();
        let block_bytes = 34usize; // 2-byte f16 + 32 × i8
        if !raw.len().is_multiple_of(block_bytes) {
            return Err(Error::Msg(alloc::format!(
                "WGPU dequantize: raw buffer length {} is not a multiple of 34",
                raw.len()
            )));
        }

        let blocks = raw.len() / block_bytes;
        let mut f32_data: Vec<f32> = Vec::with_capacity(blocks * 32);

        for block in raw.chunks_exact(block_bytes) {
            let d_bits = (block[0] as u16) | ((block[1] as u16) << 8);
            let d = kindle_core::prelude::f16::from_bits(d_bits).to_f32();
            for i in 0..32 {
                let q = block[2 + i] as i8;
                f32_data.push(q as f32 * d);
            }
        }

        let buf = WgpuBuffer::from_slice::<f32>(&f32_data);
        Ok(WgpuStorage::new(buf, t.shape.clone()))
    }

    /// Quantized matmul: dequantize both operands to f32 then dispatch to
    /// the GPU matmul shader.  A native WGSL Q8_0 matmul kernel is deferred.
    fn quantized_matmul<Q: QuantDType>(
        lhs: &<Self as Backend>::Storage<Q>,
        rhs: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<f32>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>() {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "WGPU (only Q8_0 supported)",
            });
        }
        // Dequantize both operands, then run the existing GPU matmul.
        let lhs_f32 = Self::dequantize::<Q, f32>(lhs)?;
        let rhs_f32 = Self::dequantize::<Q, f32>(rhs)?;
        Self::matmul::<f32>(&lhs_f32, &rhs_f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OptimizerOps (AdamW)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> OptimizerOps<Self> for WgpuBackend<T, D> {
    /// Core abstraction for `adamw_step` within the Kindle framework..
    fn adamw_step<K: DType>(
        var: &mut <Self as Backend>::RawVar,
        grad: &<Self as Backend>::Storage<K>,
        m: &mut <Self as Backend>::Storage<K>,
        v: &mut <Self as Backend>::Storage<K>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<()> {
        let n = num_elements(&var.storage.shape) as u32;
        let bc1 = (1.0 - beta1.powi(step as i32)) as f32;
        let bc2 = (1.0 - beta2.powi(step as i32)) as f32;

        // Pack all hyperparams as f32 bits in a u32 metadata buffer
        let meta: [u32; 8] = [
            n,
            (lr as f32).to_bits(),
            (beta1 as f32).to_bits(),
            (beta2 as f32).to_bits(),
            (eps as f32).to_bits(),
            (weight_decay as f32).to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ];

        dispatch::dispatch_adamw(
            &var.storage.buffer,
            &grad.buffer,
            &m.buffer,
            &v.buffer,
            &meta,
        );
        Ok(())
    }
}
