use crate::dtype_policy::{BackendFamily, OperationKind, resolve_dtype_policy};
use crate::wgpu::dispatch;
use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
use incin_core::backend_authoring::*;
use incin_core::prelude::*;

/// WebGPU compute backend for Incin. Type alias for `IncinBackend<T, D>`.
#[derive(Clone)]
pub struct WgpuBackendImpl<T = f32, D = Wgpu>(core::marker::PhantomData<(T, D)>);

impl<T, D> WgpuBackendImpl<T, D> {
    /// Construct the stateless WGPU executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<T, D> Default for WgpuBackendImpl<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DType, D: Device> SupportsDType<f32> for WgpuBackendImpl<T, D> {
    fn resolve_dtype(field: &<f32 as DType>::Field, _device: &DeviceId) -> Result<DTypeId> {
        Ok(<f32 as DType>::to_incin(field))
    }
}

impl<T: DType, D: Device> SupportsDType<Dyn> for WgpuBackendImpl<T, D> {
    fn resolve_dtype(field: &DTypeId, _device: &DeviceId) -> Result<DTypeId> {
        resolve_dtype_policy(
            BackendFamily::Wgpu,
            OperationKind::Storage,
            *field,
            "storage",
        )
        .map(|_| *field)
    }
}

#[derive(Clone)]
/// Implementation of `WgpuVar` for the respective backend..
pub struct WgpuVar {
    /// `storage`.
    pub storage: WgpuStorage,
}

pub type WgpuGrads = crate::wgpu::tape::WgpuGrads;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compute flat element count from shape
// ─────────────────────────────────────────────────────────────────────────────
/// `num_elements`.
pub(crate) fn num_elements(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Into::into)
}

pub(crate) fn checked_u32(value: usize, expression: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        incin_core::prelude::ShapeError::ArithmeticOverflow {
            operation: OperationKind::Storage,
            expression,
        }
        .into()
    })
}

fn checked_u32_array<const N: usize>(
    values: [usize; N],
    expression: &'static str,
) -> Result<[u32; N]> {
    let mut checked = [0; N];
    for (target, value) in checked.iter_mut().zip(values) {
        *target = checked_u32(value, expression)?;
    }
    Ok(checked)
}

fn validate_wgpu(
    dtype: DTypeId,
    device: &DeviceId,
    family: OperationKind,
    op: &'static str,
) -> Result<()> {
    if device.kind() != DeviceKind::Wgpu {
        return Err(Error::DeviceInitializationError {
            expected: "wgpu".to_string(),
            got: format!("{:?}", device.kind()),
        });
    }
    if device.ordinal() != 0 {
        return Err(Error::InvalidDeviceOrdinal {
            backend: "Wgpu",
            ordinal: device.ordinal(),
        });
    }
    resolve_dtype_policy(BackendFamily::Wgpu, family, dtype, op).map(|_| ())
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend core trait
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> Backend for WgpuBackendImpl<T, D> {
    /// `Device`.
    type Device = D;
    /// `FloatElem`.
    type FloatElem = T;
    /// `IntElem`.
    type IntElem = i64;
    /// `Storage`.
    type Storage<K: DType> = WgpuStorage;
    /// `RawVar`.
    type RawVar = WgpuVar;
    /// `Grads`.
    type Grads = WgpuGrads;
    /// `InnerBackend`.
    type InnerBackend = Self;

    /// `shape`.
    fn shape<K: DType>(t: &Self::Storage<K>) -> Vec<usize> {
        t.shape.to_vec()
    }

    fn storage_dtype<K: DType>(t: &Self::Storage<K>) -> Option<DTypeId> {
        Some(t.dtype)
    }

    fn storage_device<K: DType>(t: &Self::Storage<K>) -> Option<DeviceId> {
        Some(t.device)
    }

    /// `format_tensor_display`.
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> String {
        "WgpuTensor(...)".to_string()
    }

    /// `format_tensor_debug`.
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> String {
        format!("WgpuTensor(shape={:?})", t.shape)
    }

    /// `var_as_tensor`.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    /// `var_from_tensor`.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(WgpuVar { storage: t.clone() })
    }

    /// `assign_var`.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }

    /// `backward`.
    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::wgpu::tape::backward(loss)
    }

    /// `get_grad`.
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    /// `to_bytes`.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        t.buffer.to_vec::<u8>()
    }

    /// `from_bytes`.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Storage, "from_bytes")?;
        let expected = num_elements(shape)?
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(incin_core::prelude::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "WGPU element count * element byte width",
            })?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        let buffer = WgpuBuffer::try_from_slice(bytes)?;
        Ok(WgpuStorage::new(buffer, shape.to_vec()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CreationOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> CreationOps<Self> for WgpuBackendImpl<T, D> {
    /// `full`. WGPU storage is always physically f32 (`zeros`/`ones` above
    /// build a `Vec<f32>` regardless of the requested `dtype`, which
    /// `validate_wgpu` restricts to what the dtype policy allows), so this
    /// fills a host-side `Vec<f32>` and uploads it exactly like they do.
    fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "full")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![val as f32; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }
    /// `arange`.
    fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "arange")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }
    /// `linspace`.
    fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "linspace")?;
        let n = num_elements(shape)?;
        let step = if n > 1 {
            (end - start) / ((n - 1) as f64)
        } else {
            0.0
        };
        let data: Vec<f32> = (0..n)
            .map(|i| if i == n - 1 { end } else { start + (i as f64) * step } as f32)
            .collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `zeros`.
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "zeros")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![0.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `ones`.
    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "ones")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `rand`.
    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Random, "rand")?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape)?;
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
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `randn`.
    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Random, "randn")?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape)?;
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
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `var_zeros`.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// `var_ones`.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// `var_rand`.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// `var_randn`.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NumericOps  (add, sub, mul, div)
// ─────────────────────────────────────────────────────────────────────────────
/// `binary_op`.
#[allow(clippy::extra_unused_type_parameters)]
fn binary_op<T: DType>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    op_mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    if lhs.shape != rhs.shape {
        return Err(Error::ShapeMismatch {
            op: op_name,
            expected: lhs.shape.to_vec(),
            got: rhs.shape.to_vec(),
            msg: "shapes must match for elementwise op".to_string(),
        });
    }
    let n = checked_u32(num_elements(&lhs.shape)?, "WGPU binary element count")?;
    let out_buf = WgpuBuffer::new_zeros(lhs.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_binary(&lhs.buffer, &rhs.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, lhs.shape.to_vec()))
}

impl<T: DType, D: Device> NumericOps<Self> for WgpuBackendImpl<T, D> {
    /// `add`.
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T>(lhs, rhs, 0, "add")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
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
    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T>(lhs, rhs, 1, "sub")?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let neg_grad = unary_op::<T>(grad_out, 5)?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(grad_out, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&neg_grad, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
    /// `mul`.
    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T>(lhs, rhs, 2, "mul")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let grad_lhs = binary_op::<T>(grad_out, &rhs_capture, 2, "mul_grad")?;
                let grad_rhs = binary_op::<T>(grad_out, &lhs_capture, 2, "mul_grad")?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
    /// `div`.
    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = binary_op::<T>(lhs, rhs, 3, "div")?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs = binary_op::<T>(grad_out, &rhs_capture, 3, "div_grad_lhs")?;
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq = binary_op::<T>(&rhs_capture, &rhs_capture, 2, "div_grad_rhs_sq")?;
                let lhs_over_rhs_sq =
                    binary_op::<T>(&lhs_capture, &rhs_sq, 3, "div_grad_rhs_ratio")?;
                let neg_ratio = unary_op::<T>(&lhs_over_rhs_sq, 5)?;
                let grad_rhs = binary_op::<T>(grad_out, &neg_ratio, 2, "div_grad_rhs")?;
                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FloatOps  (scalar + unary activations)
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
fn scalar_op<T: DType>(t: &WgpuStorage, scalar: f64, op_mode: u32) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU scalar element count")?;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let scalar_bits = (scalar as f32).to_bits();
    let params = [op_mode, n, scalar_bits];
    dispatch::dispatch_scalar(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.to_vec()))
}

/// Push a single-input `TapeEntry` whose backward closure is `grad_fn`.
/// Shared by every unary `FloatOps` impl below to avoid repeating the
/// `TapeEntry { output_id, input_ids: vec![t.id], backward: ... }`
/// boilerplate at each of the ~10 call sites.
fn push_unary_tape_entry(
    t_id: crate::wgpu::storage::TensorId,
    out_id: crate::wgpu::storage::TensorId,
    grad_fn: impl Fn(&WgpuStorage) -> Result<WgpuStorage> + Send + Sync + 'static,
) {
    crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &WgpuStorage| grad_fn(grad_out).map(|grad| vec![grad])),
    });
}

impl<T: DType, D: Device> FloatOps<Self> for WgpuBackendImpl<T, D> {
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
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = scalar_op::<T>(t, scalar, 0)?;
        // Gradient passes through unchanged (same shape, no unbroadcast
        // needed — scalar ops don't change shape).
        push_unary_tape_entry(t.id, out.id, |grad_out| Ok(grad_out.clone()));
        Ok(out)
    }
    /// `mul_scalar_float`.
    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = scalar_op::<T>(t, scalar, 1)?;
        // Gradient scales by the same constant.
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            scalar_op::<T>(grad_out, scalar, 1)
        });
        Ok(out)
    }
    /// `relu`.
    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 0)?;
        // relu'(x) = step(x) (1 if x>0 else 0) — input-based.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let deriv = unary_op::<T>(&t_capture, 10)?;
            binary_op::<T>(grad_out, &deriv, 2, "relu_grad")
        });
        Ok(out)
    }
    /// `step`.
    fn step<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 10)?;
        // step'(x) = 0 almost everywhere.
        push_unary_tape_entry(t.id, out.id, |grad_out| scalar_op::<T>(grad_out, 0.0, 1));
        Ok(out)
    }
    /// `elu`.
    fn elu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 12)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T>(grad_out, &t_capture, 5, "elu_grad")
        });
        Ok(out)
    }
    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 1)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T>(grad_out, &t_capture, 4, "gelu_grad")
        });
        Ok(out)
    }
    fn mish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 11)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T>(grad_out, &t_capture, 6, "mish_grad")
        });
        Ok(out)
    }
    /// `tanh`.
    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 2)?;
        // tanh'(x) = 1 - out^2 (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let out_sq = binary_op::<T>(&out_capture, &out_capture, 2, "tanh_grad_sq")?;
            let neg_out_sq = unary_op::<T>(&out_sq, 5)?;
            let deriv = scalar_op::<T>(&neg_out_sq, 1.0, 0)?;
            binary_op::<T>(grad_out, &deriv, 2, "tanh_grad")
        });
        Ok(out)
    }
    /// `sigmoid`.
    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 3)?;
        // sigmoid'(x) = out*(1-out) (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_out = unary_op::<T>(&out_capture, 5)?;
            let one_minus_out = scalar_op::<T>(&neg_out, 1.0, 0)?;
            let deriv = binary_op::<T>(&out_capture, &one_minus_out, 2, "sigmoid_grad_deriv")?;
            binary_op::<T>(grad_out, &deriv, 2, "sigmoid_grad")
        });
        Ok(out)
    }
    /// `abs`.
    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 4)?;
        // abs'(x) = sign(x) (input-based), computed as step(x) - step(-x):
        // 1 if x>0, -1 if x<0, 0 if x==0 — matches the CPU backend exactly.
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_t = unary_op::<T>(&t_capture, 5)?;
            let step_pos = unary_op::<T>(&t_capture, 10)?;
            let step_neg = unary_op::<T>(&neg_t, 10)?;
            let neg_step_neg = unary_op::<T>(&step_neg, 5)?;
            let sign = binary_op::<T>(&step_pos, &neg_step_neg, 0, "abs_grad_sign")?;
            binary_op::<T>(grad_out, &sign, 2, "abs_grad")
        });
        Ok(out)
    }
    /// `neg`.
    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 5)?;
        // neg'(x) = -1 (constant; no input capture needed).
        push_unary_tape_entry(t.id, out.id, |grad_out| unary_op::<T>(grad_out, 5));
        Ok(out)
    }
    /// `sqrt`.
    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 6)?;
        // sqrt'(x) = 1/(2*out) (output-based) -> grad = grad_out/out * 0.5.
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let ratio = binary_op::<T>(grad_out, &out_capture, 3, "sqrt_grad_ratio")?;
            scalar_op::<T>(&ratio, 0.5, 1)
        });
        Ok(out)
    }
    /// `exp`.
    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 7)?;
        // exp'(x) = out (output-based).
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T>(grad_out, &out_capture, 2, "exp_grad")
        });
        Ok(out)
    }
    /// `log`.
    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 8)?;
        // log'(x) = 1/x (input-based, NOT output-based).
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            binary_op::<T>(grad_out, &t_capture, 3, "log_grad")
        });
        Ok(out)
    }
    /// `swish`.
    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op::<T>(t, 9)?;
        // swish(x) = x*sigmoid(x); swish'(x) = out + sigmoid(x)*(1-out).
        let t_capture = t.clone();
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sig = unary_op::<T>(&t_capture, 3)?;
            let neg_out = unary_op::<T>(&out_capture, 5)?;
            let one_minus_out = scalar_op::<T>(&neg_out, 1.0, 0)?;
            let sig_term = binary_op::<T>(&sig, &one_minus_out, 2, "swish_grad_sig_term")?;
            let deriv = binary_op::<T>(&out_capture, &sig_term, 0, "swish_grad_deriv")?;
            binary_op::<T>(grad_out, &deriv, 2, "swish_grad")
        });
        Ok(out)
    }

    /// `softmax`.
    fn softmax<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let ls = log_softmax::<T, K>(t, dim)?;
        Self::exp::<K>(&ls)
    }
}

/// Helper function to compute log_softmax composed from primitives.
pub(crate) fn log_softmax<T: DType, K: DType>(t: &WgpuStorage, dim: usize) -> Result<WgpuStorage> {
    let max = WgpuBackendImpl::<T>::max_keepdim::<K>(t, dim)?;
    let max_b = WgpuBackendImpl::<T>::broadcast_as::<K>(&max, &t.shape)?;
    let diff = WgpuBackendImpl::<T>::sub::<K>(t, &max_b)?;
    let exp_diff = WgpuBackendImpl::<T>::exp::<K>(&diff)?;
    let sum_exp = WgpuBackendImpl::<T>::sum_keepdim::<K>(&exp_diff, dim)?;
    let sum_exp_b = WgpuBackendImpl::<T>::broadcast_as::<K>(&sum_exp, &t.shape)?;
    let log_sum = WgpuBackendImpl::<T>::log::<K>(&sum_exp_b)?;
    WgpuBackendImpl::<T>::sub::<K>(&diff, &log_sum)
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorOps  (reshape, transpose, matmul, narrow, flatten, squeeze, stack, concat, etc.)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> TensorOps<Self> for WgpuBackendImpl<T, D> {
    // No WGSL kernels exist for these yet. They were reachable before only
    // because the trait answered for them; declaring them here keeps the gap
    // visible and makes the compiler name any operation added later.
    crate::unsupported::unsupported_tensor_ops! {
        where_cond, gather, scatter,
        unfold, pixel_shuffle, group_norm, instance_norm,
    }

    /// `scaled_dot_product_attention`. Composed from the already tape-wired
    /// `transpose`/`matmul`/`mul_scalar_float`/`add`/`softmax`, matching
    /// CPU's own composition exactly (down to the `1/sqrt(d_k)` default
    /// scale), so gradients flow through `q`/`k`/`v`/`mask` the same way they
    /// do on CPU rather than dead-ending on the tape.
    fn scaled_dot_product_attention<K: DType>(
        q: &<Self as Backend>::Storage<K>,
        k: &<Self as Backend>::Storage<K>,
        v: &<Self as Backend>::Storage<K>,
        mask: Option<&<Self as Backend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let k_rank = k.shape.len();
        let k_t = if k_rank >= 2 {
            <Self as TensorOps<Self>>::transpose::<K>(k, k_rank - 2, k_rank - 1)?
        } else {
            k.clone()
        };
        let scores = <Self as TensorOps<Self>>::matmul::<K>(q, &k_t)?;
        let d_k = *q.shape.last().unwrap_or(&1) as f64;
        let s = scale.unwrap_or_else(|| 1.0 / d_k.sqrt());
        let scaled_scores = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&scores, s)?;
        let masked_scores = if let Some(m) = mask {
            <Self as NumericOps<Self>>::add::<K>(&scaled_scores, m)?
        } else {
            scaled_scores
        };
        let attn = <Self as FloatOps<Self>>::softmax::<K>(&masked_scores, scores.shape.len() - 1)?;
        <Self as TensorOps<Self>>::matmul::<K>(&attn, v)
    }

    /// `index_select`. Same host-readback/upload pattern as `repeat`; `index`
    /// is `WgpuStorage` regardless of `KInt` (WGPU has one physical storage
    /// representation for every dtype), so its values are read back as f32
    /// and truncated to a position like CPU's `index.get(&idx) as usize`
    /// does. Not autograd-wired, matching CPU.
    fn index_select<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        index: &<Self as Backend>::Storage<KInt>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let index_data = index.buffer.to_vec::<f32>()?;
        let mut out_shape = t.shape.to_vec();
        out_shape[dim] = index_data.len();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let selected = index_data[out_idx[dim]] as usize;
            let mut src_idx = out_idx.clone();
            src_idx[dim] = selected;
            out.push(data[checked_flat_index(&src_idx, &t.shape, OperationKind::Reshape)?]);
            if !out_shape.is_empty() {
                increment_multi_index(&mut out_idx, &out_shape);
            }
        }
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, out_shape))
    }

    /// `masked_fill`. Same host-readback/upload pattern as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn masked_fill<K: DType, KMask: DType>(
        t: &<Self as Backend>::Storage<K>,
        mask: &<Self as Backend>::Storage<KMask>,
        value: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if t.shape != mask.shape {
            return Err(Error::ShapeMismatch {
                op: "masked_fill",
                expected: t.shape.to_vec(),
                got: mask.shape.to_vec(),
                msg: "mask must match the operand's shape exactly".to_string(),
            });
        }
        let data = t.buffer.to_vec::<f32>()?;
        let mask_data = mask.buffer.to_vec::<f32>()?;
        let out: Vec<f32> = data
            .iter()
            .zip(mask_data.iter())
            .map(|(&v, &m)| if m != 0.0 { value as f32 } else { v })
            .collect();
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, t.shape.to_vec()))
    }

    /// `repeat`. WGPU storage has no shader for this yet, so it reads the
    /// operand back to the host, repeats it with the same row-major walk
    /// CPU's own `repeat` uses, and re-uploads — the same host-compute
    /// pattern `full`/`arange`/`linspace` above already use. Not
    /// autograd-wired, matching CPU.
    fn repeat<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let out_shape: Vec<usize> = t.shape.iter().zip(repeats).map(|(a, b)| a * b).collect();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let src_idx: Vec<usize> = idx
                .iter()
                .zip(t.shape.iter())
                .map(|(&s, &dim)| s % dim)
                .collect();
            out.push(data[checked_flat_index(&src_idx, &t.shape, OperationKind::Reshape)?]);
            if !out_shape.is_empty() {
                increment_multi_index(&mut idx, &out_shape);
            }
        }
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, out_shape))
    }

    /// `pad`. Same host-readback/upload pattern as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn pad<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let out_shape: Vec<usize> = t
            .shape
            .iter()
            .zip(padding)
            .map(|(&s, &(before, after))| s + before + after)
            .collect();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let mut inside = true;
            let mut src_idx = Vec::with_capacity(idx.len());
            for (axis, &p) in idx.iter().enumerate() {
                let (before, _) = padding[axis];
                if p < before || p >= before + t.shape[axis] {
                    inside = false;
                    break;
                }
                src_idx.push(p - before);
            }
            out.push(if inside {
                data[checked_flat_index(&src_idx, &t.shape, OperationKind::Reshape)?]
            } else {
                val as f32
            });
            if !out_shape.is_empty() {
                increment_multi_index(&mut idx, &out_shape);
            }
        }
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, out_shape))
    }

    /// `triu`. Same host-readback/upload pattern as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn triu<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let rank = t.shape.len();
        let total = num_elements(&t.shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for &value in data.iter().take(total) {
            let (r, c) = if rank >= 2 {
                (idx[rank - 2] as i64, idx[rank - 1] as i64)
            } else {
                (0, idx[0] as i64)
            };
            out.push(if c >= r + k { value } else { 0.0 });
            if !t.shape.is_empty() {
                increment_multi_index(&mut idx, &t.shape);
            }
        }
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, t.shape.to_vec()))
    }

    /// `tril`. Same host-readback/upload pattern as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn tril<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let rank = t.shape.len();
        let total = num_elements(&t.shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for &value in data.iter().take(total) {
            let (r, c) = if rank >= 2 {
                (idx[rank - 2] as i64, idx[rank - 1] as i64)
            } else {
                (0, idx[0] as i64)
            };
            out.push(if c <= r + k { value } else { 0.0 });
            if !t.shape.is_empty() {
                increment_multi_index(&mut idx, &t.shape);
            }
        }
        let buf = WgpuBuffer::try_from_slice(&out)?;
        Ok(WgpuStorage::new(buf, t.shape.to_vec()))
    }

    /// `diag`. Same host-readback/upload pattern as `repeat`, matching CPU's
    /// two cases: a 1D operand builds a 2D matrix with that operand on its
    /// `k`-th diagonal, an operand of rank 2+ extracts its `k`-th diagonal
    /// into a 1D result. Not autograd-wired, matching CPU.
    fn diag<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data = t.buffer.to_vec::<f32>()?;
        let rank = t.shape.len();
        if rank == 1 {
            let n = t.shape[0];
            let k_abs = k.unsigned_abs() as usize;
            let out_dim = n.checked_add(k_abs).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "WGPU diagonal output dimension",
            })?;
            let out_total = num_elements(&[out_dim, out_dim])?;
            let mut out = vec![0.0f32; out_total];
            for (i, &value) in data.iter().enumerate().take(n) {
                let r = if k >= 0 { i } else { i + k_abs };
                let c = if k >= 0 { i + k_abs } else { i };
                if r < out_dim && c < out_dim {
                    out[r * out_dim + c] = value;
                }
            }
            let buf = WgpuBuffer::try_from_slice(&out)?;
            Ok(WgpuStorage::new(buf, vec![out_dim, out_dim]))
        } else {
            let r_len = t.shape[rank - 2];
            let c_len = t.shape[rank - 1];
            let mut diag_vals = Vec::new();
            for r in 0..r_len {
                let c = (r as i64 + k) as usize;
                if c < c_len {
                    diag_vals.push(data[r * c_len + c]);
                }
            }
            let out_len = diag_vals.len();
            let buf = WgpuBuffer::try_from_slice(&diag_vals)?;
            Ok(WgpuStorage::new(buf, vec![out_len]))
        }
    }

    /// `addmm`. `beta * mat + alpha * (mat1 @ mat2)`, composed from the
    /// already tape-wired `matmul`/`mul_scalar_float`/`add`, matching CPU's
    /// own composition — and so, like CPU, differentiable through all three
    /// operands rather than a dead end on the tape.
    fn addmm<K: DType>(
        mat: &<Self as Backend>::Storage<K>,
        mat1: &<Self as Backend>::Storage<K>,
        mat2: &<Self as Backend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mm = <Self as TensorOps<Self>>::matmul::<K>(mat1, mat2)?;
        let mm_alpha = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&mm, alpha)?;
        let mat_beta = <Self as FloatOps<Self>>::mul_scalar_float::<K>(mat, beta)?;
        <Self as NumericOps<Self>>::add::<K>(&mat_beta, &mm_alpha)
    }
    /// `bmm`. `matmul` already handles the batch dimensions, matching CPU.
    fn bmm<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        <Self as TensorOps<Self>>::matmul::<K>(lhs, rhs)
    }

    /// `unsqueeze`. Metadata-only, like `reshape` (which it delegates to and
    /// so inherits gradient wiring from): inserts a size-1 axis.
    fn unsqueeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = t.shape.to_vec();
        if dim <= target_shape.len() {
            target_shape.insert(dim, 1);
        } else {
            target_shape.push(1);
        }
        Self::reshape::<K>(t, &target_shape)
    }

    /// `cmp_eq`. Matches the CPU backend: same-dtype output encoding
    /// true/false as 1.0/0.0, and (like CPU) not autograd-wired since the
    /// output is not a differentiable function of the operands.
    fn cmp_eq<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 7, "cmp_eq")
    }
    /// `cmp_ne`.
    fn cmp_ne<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 8, "cmp_ne")
    }
    /// `cmp_lt`.
    fn cmp_lt<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 9, "cmp_lt")
    }
    /// `cmp_le`.
    fn cmp_le<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 10, "cmp_le")
    }
    /// `cmp_gt`.
    fn cmp_gt<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 11, "cmp_gt")
    }
    /// `cmp_ge`.
    fn cmp_ge<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 12, "cmp_ge")
    }

    /// `logical_and`. Not autograd-wired, matching CPU.
    fn logical_and<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 13, "logical_and")
    }
    /// `logical_or`.
    fn logical_or<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 14, "logical_or")
    }
    /// `logical_not`.
    fn logical_not<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T>(t, 13)
    }

    /// `sub_scalar`. Not autograd-wired: matches CPU, whose `TensorOps`
    /// scalar/comparison methods (as opposed to `FloatOps`'s
    /// `add_scalar_float`/`mul_scalar_float`) carry no backward closure.
    fn sub_scalar<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        scalar_op::<T>(t, val, 2)
    }
    /// `div_scalar`.
    fn div_scalar<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        scalar_op::<T>(t, val, 3)
    }

    /// `maximum`. Not autograd-wired, matching CPU.
    fn maximum<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 15, "maximum")
    }
    /// `minimum`.
    fn minimum<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 16, "minimum")
    }
    /// `abs_diff`.
    fn abs_diff<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T>(lhs, rhs, 17, "abs_diff")
    }

    /// `lerp`. `start + weight * (end - start)`, composed from existing
    /// primitives; not autograd-wired, matching CPU.
    fn lerp<K: DType>(
        start: &<Self as Backend>::Storage<K>,
        end: &<Self as Backend>::Storage<K>,
        weight: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let diff = binary_op::<T>(end, start, 1, "lerp_diff")?;
        let scaled = scalar_op::<T>(&diff, weight, 1)?;
        binary_op::<T>(start, &scaled, 0, "lerp_add")
    }

    /// `matmul`.
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

        let m = lhs.shape[lhs_rank - 2];
        let k = lhs.shape[lhs_rank - 1];
        let n = rhs.shape[rhs_rank - 1];

        if k != rhs.shape[rhs_rank - 2] {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "matmul inner dims must match".to_string(),
            });
        }

        // Compute batch dims
        let lhs_batch = ShapeBuf::from_slice(&lhs.shape[..lhs_rank - 2])
            .checked_numel(OperationKind::MatMul)?;
        let rhs_batch = ShapeBuf::from_slice(&rhs.shape[..rhs_rank - 2])
            .checked_numel(OperationKind::MatMul)?;

        let batch = core::cmp::max(lhs_batch, rhs_batch);
        if lhs_batch != 1 && rhs_batch != 1 && lhs_batch != rhs_batch {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "matmul batch dims incompatible".to_string(),
            });
        }

        let lhs_stride_b = if lhs_batch == 1 {
            0
        } else {
            m.checked_mul(k).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::MatMul,
                expression: "WGPU matmul lhs batch stride",
            })?
        };
        let rhs_stride_b = if rhs_batch == 1 {
            0
        } else {
            k.checked_mul(n).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::MatMul,
                expression: "WGPU matmul rhs batch stride",
            })?
        };

        // Output shape matches the larger batched input
        let mut out_shape = if lhs_batch > 1 {
            lhs.shape[..lhs_rank - 2].to_vec()
        } else {
            rhs.shape[..rhs_rank - 2].to_vec()
        };
        if out_shape.is_empty() && batch > 1 {
            out_shape.push(batch);
        }
        out_shape.push(m);
        out_shape.push(n);

        let state = crate::wgpu::device::get_device_state();
        let shader = include_str!("shaders/matmul.wgsl");
        let pipeline = crate::wgpu::pipeline::get_or_create_pipeline("matmul", shader, "main");

        let out_n = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::MatMul)?;
        let [
            m_u32,
            k_u32,
            n_u32,
            batch_u32,
            lhs_stride_u32,
            rhs_stride_u32,
        ] = checked_u32_array(
            [m, k, n, batch, lhs_stride_b, rhs_stride_b],
            "WGPU matmul kernel parameter",
        )?;
        let shape_data = [
            m_u32,
            k_u32,
            n_u32,
            batch_u32,
            lhs_stride_u32,
            rhs_stride_u32,
        ];
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
            cpass.dispatch_workgroups(n_u32.div_ceil(16), m_u32.div_ceil(16), batch_u32);
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
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let rhs_rank = rhs_capture.shape.len();
                let rhs_t = Self::transpose::<K>(&rhs_capture, rhs_rank - 2, rhs_rank - 1)?;
                let grad_lhs_full = Self::matmul::<K>(grad_out, &rhs_t)?;

                let lhs_rank = lhs_capture.shape.len();
                let lhs_t = Self::transpose::<K>(&lhs_capture, lhs_rank - 2, lhs_rank - 1)?;
                let grad_rhs_full = Self::matmul::<K>(&lhs_t, grad_out)?;

                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs_full, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs_full, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    /// `reshape`.
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        if num_elements(&t.shape)? != num_elements(shape)? {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: t.shape.to_vec(),
                got: shape.to_vec(),
                msg: "total elements must match".to_string(),
            });
        }
        let out = WgpuStorage::new(t.buffer.clone(), shape.to_vec());
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::reshape::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }

    /// `transpose`.
    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let mut new_shape = shape.to_vec();
        new_shape.swap(dim1, dim2);

        let out_n = checked_u32(
            num_elements(&new_shape)?,
            "WGPU transpose output element count",
        )?;
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);

        let mut aux = (0..shape.len()).collect::<Vec<_>>();
        aux.swap(dim1, dim2);

        let params = dispatch::prepare_shape_params(
            2, // op_mode = transpose
            out_n, &new_shape, shape, &aux,
        )?;

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, new_shape);

        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::transpose::<K>(grad_out, dim1, dim2)?])
            }),
        });
        Ok(out)
    }

    /// `flatten`.
    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let flat_size: usize =
            incin_core::prelude::ShapeBuf::from_slice(&(shape[start_dim..=end_dim]))
                .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let mut new_shape: Vec<usize> = shape[..start_dim].to_vec();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&shape[end_dim + 1..]);
        Self::reshape::<K>(t, &new_shape)
    }

    /// `squeeze`.
    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut new_shape = t.shape.to_vec();
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
        let mut new_shape = shape.to_vec();
        new_shape[dim] = len;

        let out_elements = num_elements(&new_shape)?;
        let out_n = checked_u32(out_elements, "WGPU narrow output element count")?;
        let out_buf =
            WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Storage)?;

        let mut aux = vec![0usize; shape.len()];
        aux[dim] = start;

        let params = dispatch::prepare_shape_params(
            0, // op_mode = slice
            out_n, &new_shape, shape, &aux,
        )?;

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, new_shape);

        let original_shape = t.shape.to_vec();
        let mut region_start = vec![0usize; original_shape.len()];
        region_start[dim] = start;
        let (t_id, out_id) = (t.id, out.id);

        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![crate::wgpu::storage::scatter_into_zeros(
                    &original_shape,
                    &region_start,
                    grad_out,
                )?])
            }),
        });
        Ok(out)
    }

    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_elements = num_elements(shape)?;
        let out_n = checked_u32(out_elements, "WGPU broadcast output element count")?;
        let out_buf =
            WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Storage)?;

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
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
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

    /// `broadcast_left`.
    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = shape.to_vec();
        target_shape.extend_from_slice(&t.shape);
        Self::broadcast_as::<K>(t, &target_shape)
    }

    /// `slice`.
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

    /// `stack`.
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
            let mut target_shape = t.shape.to_vec();
            target_shape.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &target_shape)?);
        }
        let refs: Vec<&<Self as Backend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// `concat`.
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::Msg("concat: empty tensor list".to_string()));
        }
        let rank = tensors[0].shape.len();
        let mut out_shape = tensors[0].shape.to_vec();
        out_shape[dim] = tensors.iter().try_fold(0usize, |total, tensor| {
            total
                .checked_add(tensor.shape[dim])
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Concat,
                    expression: "WGPU concat output dimension",
                })
        })?;

        let out_n = num_elements(&out_shape)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;

        let mut current_offset = 0usize;
        for t in tensors {
            let in_n = checked_u32(num_elements(&t.shape)?, "WGPU concat input element count")?;
            let mut aux = vec![0usize; rank];
            aux[dim] = current_offset;

            let params = dispatch::prepare_shape_params(
                1, // op_mode = paste
                in_n, &out_shape, &t.shape, &aux,
            )?;
            dispatch::dispatch_shape(&t.buffer, &out_buf, &params);

            current_offset =
                current_offset
                    .checked_add(t.shape[dim])
                    .ok_or(ShapeError::ArithmeticOverflow {
                        operation: OperationKind::Concat,
                        expression: "WGPU concat cumulative offset",
                    })?;
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
            // Collecting an iterator of `Result` straight into `Result<Vec<_>>`
            // is the whole conversion here, as on the CPU side.
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                offsets
                    .iter()
                    .zip(input_dim_sizes.iter())
                    .map(|(&offset, &len)| Self::narrow::<K>(grad_out, dim, offset, len))
                    .collect()
            }),
        });

        Ok(out)
    }

    /// `float_to_scalar`.
    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        let numel = ShapeBuf::from_slice(&t.shape).checked_numel(OperationKind::Storage)?;
        if numel != 1 {
            return Err(Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::Storage,
                parameter: "float_to_scalar element count",
                value: numel,
            }));
        }
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        let value = data.first().copied().ok_or(Error::InternalInvariant {
            operation: "wgpu_float_to_scalar",
            reason: "validated one-element storage read back no bytes",
        })?;
        Ok(f64::from(value))
    }

    /// `float_to_vec1`.
    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<f64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    /// `int_to_scalar`.
    fn int_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        let value = data.first().copied().ok_or(Error::InvalidByteLength {
            expected: core::mem::size_of::<f32>(),
            got: 0,
        })?;
        incin_core::prelude::convert_f64_to_i64(
            "int_to_scalar",
            t.dtype,
            f64::from(value),
            incin_core::prelude::FloatToIntPolicy::Exact,
        )
    }

    /// `int_to_vec1`.
    fn int_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<i64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        data.into_iter()
            .map(|value| {
                incin_core::prelude::convert_f64_to_i64(
                    "int_to_vec1",
                    t.dtype,
                    f64::from(value),
                    incin_core::prelude::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }

    /// `tensor_to_dtype`.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as Backend>::Storage<K>,
        _dtype: DTypeId,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        // Simple passthrough (all stored as f32 internally)
        WgpuStorage::try_new(t.buffer.clone(), t.shape.to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReductionOps
// ─────────────────────────────────────────────────────────────────────────────
/// `reduce_all_to_storage`.
fn reduce_all_to_storage(t: &WgpuStorage, mode: u32) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU reduction element count")?;
    let out = dispatch::dispatch_reduce_all(&t.buffer, n, mode)?;
    Ok(WgpuStorage::new(out, vec![]))
}

/// `reduce_dim_to_storage`.
fn reduce_dim_to_storage(
    t: &WgpuStorage,
    dim: usize,
    mode: u32,
    keepdim: bool,
) -> Result<WgpuStorage> {
    let shape = &t.shape;
    if dim >= shape.len() {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        }
        .into());
    }
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out_n = num_elements(&out_shape)?;

    let dim_size = checked_u32(shape[dim], "WGPU reduction dimension")?;
    let inner_stride =
        ShapeBuf::from_slice(&shape[dim + 1..]).checked_numel(OperationKind::Reduction)?;

    // mode mapping: CPU reduce_dim mode (0=sum, 1=max, 2=min, 3=product) maps
    // to my shader ops (0=sum, 2=max, 3=min, 6=product).
    let op_mode = match mode {
        0 => 0u32, // sum
        1 => 2u32, // max
        2 => 3u32, // min
        3 => 6u32, // product
        _ => {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Reduction,
                reason: "unknown WGPU reduction mode",
            }));
        }
    };

    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
    dispatch::dispatch_reduce_dim(
        &t.buffer,
        &out_buf,
        op_mode,
        dim_size,
        checked_u32(inner_stride, "WGPU reduction inner stride")?,
        checked_u32(out_n, "WGPU reduction output element count")?,
    );

    let final_shape = if keepdim {
        out_shape
    } else {
        let mut s = shape.to_vec();
        s.remove(dim);
        s
    };
    Ok(WgpuStorage::new(out_buf, final_shape))
}

/// Splits a contiguous shape into `(outer, axis, inner)` element counts
/// around `dim`, i.e. `shape[..dim]`, `shape[dim]`, `shape[dim+1..]`
/// products. For a contiguous row-major tensor this is enough to address
/// any element directly (`outer_idx*(axis*inner) + axis_idx*inner + inner_idx`)
/// without a general N-dimensional odometer — WGPU storage has no
/// non-contiguous view support, so this always applies.
fn axis_reduce_dims(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    let outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(shape[..dim]))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let axis = shape[dim];
    let inner: usize = incin_core::prelude::ShapeBuf::from_slice(&(shape[dim + 1..]))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    Ok((outer, axis, inner))
}

fn checked_flat_index(
    coords: &[usize],
    shape: &[usize],
    operation: OperationKind,
) -> Result<usize> {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&coord, &dimension) in coords.iter().zip(shape).rev() {
        flat = coord
            .checked_mul(stride)
            .and_then(|offset| flat.checked_add(offset))
            .ok_or(ShapeError::ArithmeticOverflow {
                operation,
                expression: "WGPU flat index",
            })?;
        stride = stride
            .checked_mul(dimension)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation,
                expression: "WGPU index stride",
            })?;
    }
    Ok(flat)
}

/// Advances a row-major multi-index by one position (odometer increment),
/// least-significant axis first. Mirrors the CPU backend's
/// `cpu::storage::increment_index`; WGPU storage is always contiguous so the
/// same row-major walk applies directly to a flat host-side readback.
fn increment_multi_index(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}

/// Backward for `max_dim`/`min_dim`: recomputes each output position's
/// winning (first-encountered, strict `>`/`<`) source position from the
/// captured input, then scatters `grad_out`'s value there with a bare `=`
/// (never `+=` — unlike pooling, a plain axis reduction never has two output
/// positions sharing the same winning source element). Mirrors the CPU
/// backend's `max_axis_with_indices`/`min_axis_with_indices` +
/// `scatter_axis_grad` (`cpu/ops/reduce.rs`) exactly. Not used for
/// `max_keepdim`/`min_keepdim` — see their doc comments.
fn push_extremum_dim_tape_entry(t: &WgpuStorage, out: &WgpuStorage, dim: usize, is_max: bool) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
            let (outer, axis, inner) = axis_reduce_dims(&input_shape, dim)?;
            let input_data = t_capture.buffer.to_vec::<f32>()?;
            let grad_data = grad_out.buffer.to_vec::<f32>()?;
            let grad_elements = ShapeBuf::from_slice(&[outer, axis, inner])
                .checked_numel(OperationKind::Reduction)?;
            let mut grad_input = vec![0.0f32; grad_elements];
            for o in 0..outer {
                for i in 0..inner {
                    let mut best_val = if is_max {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    let mut best_flat = o * (axis * inner) + i;
                    for a in 0..axis {
                        let flat = o * (axis * inner) + a * inner + i;
                        let v = input_data[flat];
                        if (is_max && v > best_val) || (!is_max && v < best_val) {
                            best_val = v;
                            best_flat = flat;
                        }
                    }
                    let flat_out = o * inner + i;
                    grad_input[best_flat] = grad_data[flat_out];
                }
            }
            Ok(vec![WgpuStorage::new(
                WgpuBuffer::from_slice(&grad_input),
                input_shape.clone(),
            )])
        }),
    });
}

/// Backward for `max_all`/`min_all`: the whole-tensor special case of
/// `push_extremum_dim_tape_entry` (`outer = inner = 1`, `axis = numel`).
fn push_extremum_all_tape_entry(t: &WgpuStorage, out: &WgpuStorage, is_max: bool) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
            let input_data = t_capture.buffer.to_vec::<f32>()?;
            let grad_values = grad_out.buffer.to_vec::<f32>()?;
            let grad_val = *grad_values.first().ok_or(Error::InternalInvariant {
                operation: "wgpu_extremum_backward",
                reason: "scalar extremum gradient read back no value",
            })?;
            let mut best_val = if is_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            let mut best_flat = 0usize;
            for (flat, &v) in input_data.iter().enumerate() {
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_flat = flat;
                }
            }
            let mut grad_input = vec![0.0f32; input_data.len()];
            grad_input[best_flat] = grad_val;
            Ok(vec![WgpuStorage::new(
                WgpuBuffer::from_slice(&grad_input),
                input_shape.clone(),
            )])
        }),
    });
}

impl<T: DType, D: Device> ReductionOps<Self> for WgpuBackendImpl<T, D> {
    // No prefix-scan shader exists yet.
    crate::unsupported::unsupported_reduction_ops! {
        all: ;
        dim: cumsum;
    }

    /// `prod_all`. Not autograd-wired, matching CPU.
    fn prod_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        reduce_all_to_storage(t, 3)
    }
    /// `prod_dim`. Not autograd-wired, matching CPU.
    fn prod_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        reduce_dim_to_storage(t, dim, 3, false)
    }

    /// `sum_all`.
    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 0)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_all`.
    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_all_to_storage(t, 0)?;
        let n = num_elements(&t.shape)? as f64;
        let out = scalar_op::<T>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let scaled = scalar_op::<T>(grad_out, 1.0 / n, 1)?;
                Ok(vec![Self::broadcast_as::<K>(&scaled, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `max_all`.
    fn max_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 1)?;
        push_extremum_all_tape_entry(t, &out, true);
        Ok(out)
    }
    /// `min_all`.
    fn min_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 2)?;
        push_extremum_all_tape_entry(t, &out, false);
        Ok(out)
    }

    /// `sum_dim`.
    fn sum_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, false)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.to_vec();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape)?;
                Ok(vec![Self::broadcast_as::<K>(&keepdim, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `sum_keepdim`.
    fn sum_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, true)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_dim`.
    fn mean_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, false)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<T>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.to_vec();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape)?;
                let expanded = Self::broadcast_as::<K>(&keepdim, &original_shape)?;
                Ok(vec![scalar_op::<T>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }
    /// `mean_keepdim`.
    fn mean_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, true)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<T>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let expanded = Self::broadcast_as::<K>(grad_out, &original_shape)?;
                Ok(vec![scalar_op::<T>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }
    /// `max_dim`.
    fn max_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 1, false)?;
        push_extremum_dim_tape_entry(t, &out, dim, true);
        Ok(out)
    }
    /// `max_keepdim`.
    ///
    /// Autograd-wired the same way as `max_dim`. `log_softmax` calls this to
    /// subtract a per-row max `M` for numerical stability:
    /// `log_softmax(x) = (x - M) - log(sum(exp(x - M)))`. This is not merely
    /// numerically close but ALGEBRAICALLY IDENTICAL to `x - log(sum(exp(x)))`
    /// for any `M` (the `M` terms cancel exactly:
    /// `-M - log(exp(-M)*sum(exp(x))) = -M - (-M + log(sum(exp(x)))) =
    /// -log(sum(exp(x)))`), so the two expressions are literally the same
    /// function of `x` on their whole domain, not just equal at one point —
    /// their gradients must therefore be identical too, whether or not `M`
    /// is treated as differentiable. Wiring a real gradient here does NOT
    /// need `log_softmax` to detach `M`. Matches the CPU backend, whose
    /// `max_keepdim` is fully wired and whose composed `log_softmax` (same
    /// formula) passes `softmax_gradcheck`/`log_softmax_gradcheck`.
    fn max_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 1, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, true);
        Ok(out)
    }
    /// `min_dim`.
    fn min_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, false)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
    }
    /// `min_keepdim`.
    ///
    /// Autograd-wired the same way as `min_dim` — see `max_keepdim`'s doc
    /// comment above.
    fn min_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
    }

    /// `argmax`.
    fn argmax<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .ok_or(ShapeError::InvalidParameter {
                        operation: OperationKind::Reduction,
                        parameter: "non-empty input",
                        value: 0,
                    })?;
                let buf = WgpuBuffer::from_slice(&[checked_u32(idx, "WGPU argmax index")?]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                if d >= shape.len() {
                    return Err(ShapeError::InvalidParameter {
                        operation: OperationKind::Reduction,
                        parameter: "axis",
                        value: d,
                    }
                    .into());
                }
                let mut out_shape = shape.to_vec();
                out_shape[d] = 1;
                let out_n = num_elements(&out_shape)?;

                let dim_size = checked_u32(shape[d], "WGPU argmax dimension")?;
                let inner_stride = ShapeBuf::from_slice(&shape[d + 1..])
                    .checked_numel(OperationKind::Reduction)?;

                let out_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
                dispatch::dispatch_reduce_dim(
                    &t.buffer,
                    &out_buf,
                    4, // argmax
                    dim_size,
                    checked_u32(inner_stride, "WGPU argmax inner stride")?,
                    checked_u32(out_n, "WGPU argmax output element count")?,
                );

                let mut final_shape = shape.to_vec();
                final_shape.remove(d);
                Ok(WgpuStorage::new(out_buf, final_shape))
            }
        }
    }

    /// `argmin`.
    fn argmin<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .ok_or(ShapeError::InvalidParameter {
                        operation: OperationKind::Reduction,
                        parameter: "non-empty input",
                        value: 0,
                    })?;
                let buf = WgpuBuffer::from_slice(&[checked_u32(idx, "WGPU argmin index")?]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                if d >= shape.len() {
                    return Err(ShapeError::InvalidParameter {
                        operation: OperationKind::Reduction,
                        parameter: "axis",
                        value: d,
                    }
                    .into());
                }
                let mut out_shape = shape.to_vec();
                out_shape[d] = 1;
                let out_n = num_elements(&out_shape)?;

                let dim_size = checked_u32(shape[d], "WGPU argmin dimension")?;
                let inner_stride = ShapeBuf::from_slice(&shape[d + 1..])
                    .checked_numel(OperationKind::Reduction)?;

                let out_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
                dispatch::dispatch_reduce_dim(
                    &t.buffer,
                    &out_buf,
                    5, // argmin
                    dim_size,
                    checked_u32(inner_stride, "WGPU argmin inner stride")?,
                    checked_u32(out_n, "WGPU argmin output element count")?,
                );

                let mut final_shape = shape.to_vec();
                final_shape.remove(d);
                Ok(WgpuStorage::new(out_buf, final_shape))
            }
        }
    }

    /// `topk`.
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
                expected: shape.to_vec(),
                got: vec![dim],
                msg: format!("topk: axis {} out of range", dim),
            });
        }
        let k = k.min(shape[dim]);
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;

        let mut out_shape = shape.to_vec();
        out_shape[dim] = k;
        let mut base_shape = shape.to_vec();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape)?;
        let mut out_vals = vec![0.0f32; num_elements(&out_shape)?];
        let mut out_indices = vec![0u32; num_elements(&out_shape)?];

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
                let flat = checked_flat_index(&coords, shape, OperationKind::Reduction)?;
                slice_vals.push((data[flat], checked_u32(j, "WGPU topk index")?));
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
                let flat = checked_flat_index(&out_coords, &out_shape, OperationKind::Reduction)?;
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

    /// `argsort`.
    fn argsort<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "argsort",
                expected: shape.to_vec(),
                got: vec![dim],
                msg: format!("argsort: axis {} out of range", dim),
            });
        }
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;

        let mut base_shape = shape.to_vec();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape)?;
        let mut out = vec![0u32; num_elements(shape)?];

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
                let flat = checked_flat_index(&coords, shape, OperationKind::Reduction)?;
                slice_vals.push((data[flat], checked_u32(j, "WGPU argsort index")?));
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
                let flat = checked_flat_index(&out_coords, shape, OperationKind::Reduction)?;
                out[flat] = idx;
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conv backward helpers (CPU-side, used inside backward closures that
// read data back from WgpuBuffer and compute gradients in host memory).
// The same im2col / col2im logic as `crates/incin-backends/src/cpu/ops/conv.rs`
// but operating on plain `Vec<f32>` instead of `CpuStorage`.
// ─────────────────────────────────────────────────────────────────────────────

/// Gather a `[B, Cin, H, W]` buffer (row-major) into a
/// `[B, H_out*W_out, Cin*Kh*Kw]` column matrix. Out-of-bounds positions
/// (i.e. positions in the padded region) contribute 0.0.
fn im2col_2d_cpu(
    input: &[f32],
    b: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let h_out = pool_output_dim(h, kh, stride, padding, dilation)?;
    let w_out = pool_output_dim(w, kw, stride, padding, dilation)?;
    let col_len = ShapeBuf::from_slice(&[cin, kh, kw]).checked_numel(OperationKind::Conv2d)?;
    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let out_elements =
        ShapeBuf::from_slice(&[b, spatial, col_len]).checked_numel(OperationKind::Conv2d)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let o_flat = oh * w_out + ow;
                for ci in 0..cin {
                    for ki_h in 0..kh {
                        for ki_w in 0..kw {
                            let src_h = oh * stride + ki_h * dilation;
                            let src_w = ow * stride + ki_w * dilation;
                            let val = if src_h >= padding
                                && src_h - padding < h
                                && src_w >= padding
                                && src_w - padding < w
                            {
                                let ih = src_h - padding;
                                let iw = src_w - padding;
                                input[bi * (cin * h * w) + ci * (h * w) + ih * w + iw]
                            } else {
                                0.0
                            };
                            let col_idx = ci * kh * kw + ki_h * kw + ki_w;
                            out[bi * (spatial * col_len) + o_flat * col_len + col_idx] = val;
                        }
                    }
                }
            }
        }
    }
    Ok((out, h_out, w_out))
}

/// Scatter-ADD a `[B, H_out*W_out, Cin*Kh*Kw]` gradient back into a
/// zero-initialized `[B, Cin, H, W]` buffer.
fn col2im_2d_cpu(
    cols_grad: &[f32],
    b: usize,
    cin: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Vec<f32>> {
    let col_len = ShapeBuf::from_slice(&[cin, kh, kw]).checked_numel(OperationKind::Conv2d)?;
    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let out_elements =
        ShapeBuf::from_slice(&[b, cin, h, w]).checked_numel(OperationKind::Conv2d)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let o_flat = oh * w_out + ow;
                for ci in 0..cin {
                    for ki_h in 0..kh {
                        for ki_w in 0..kw {
                            let src_h = oh * stride + ki_h * dilation;
                            let src_w = ow * stride + ki_w * dilation;
                            if src_h >= padding
                                && src_h - padding < h
                                && src_w >= padding
                                && src_w - padding < w
                            {
                                let ih = src_h - padding;
                                let iw = src_w - padding;
                                let col_idx = ci * kh * kw + ki_h * kw + ki_w;
                                out[bi * (cin * h * w) + ci * (h * w) + ih * w + iw] += cols_grad
                                    [bi * (spatial * col_len) + o_flat * col_len + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Batched matrix multiply on CPU: lhs `[B, M, K]` × rhs `[B, K, N]` → `[B, M, N]`.
/// Used inside conv backward closures.
fn cpu_bmm(lhs: &[f32], rhs: &[f32], b: usize, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[b, m, n]).checked_numel(OperationKind::MatMul)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += lhs[bi * (m * k) + mi * k + ki] * rhs[bi * (k * n) + ki * n + ni];
                }
                out[bi * (m * n) + mi * n + ni] = acc;
            }
        }
    }
    Ok(out)
}

/// Transpose the last two dimensions of a `[B, M, N]` tensor → `[B, N, M]`.
fn cpu_transpose_last2(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[b, m, n]).checked_numel(OperationKind::Transpose)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                out[bi * (n * m) + ni * m + mi] = src[bi * (m * n) + mi * n + ni];
            }
        }
    }
    Ok(out)
}

/// Sum a `[B, M, N]` buffer over its leading batch axis → `[M, N]`.
fn cpu_sum_batch(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[m, n]).checked_numel(OperationKind::Reduction)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                out[mi * n + ni] += src[bi * (m * n) + mi * n + ni];
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// WgpuBackendImpl inherent helpers
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> WgpuBackendImpl<T, D> {
    /// Forward-only conv2d (no tape entry). Used by both `conv1d` and `conv2d`
    /// so they can push exactly ONE clean tape entry each for their respective
    /// grad shapes, rather than having nested entries from the internal matmul.
    fn conv2d_no_tape<K: DType>(
        t: &WgpuStorage,
        weight: &WgpuStorage,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<WgpuStorage> {
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
        let g = groups.max(1);
        let c_in_g = c_in / g;
        assert_eq!(c_in_g, c_in_per_g, "groups mismatch");

        let h_out =
            (h_in + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
        let w_out =
            (w_in + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;

        let col_channels =
            ShapeBuf::from_slice(&[c_in, kh, kw]).checked_numel(OperationKind::Conv2d)?;
        let col_spatial =
            ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
        let col_elements = ShapeBuf::from_slice(&[batch, col_channels, col_spatial])
            .checked_numel(OperationKind::Conv2d)?;
        let col_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, col_elements, OperationKind::Conv2d)?;

        let params = checked_u32_array(
            [
                batch, c_in, h_in, w_in, h_out, w_out, kh, kw, stride, stride, padding, padding,
                dilation, dilation,
            ],
            "WGPU im2col kernel parameter",
        )?;
        dispatch::dispatch_im2col(&t.buffer, &col_buf, &params)?;

        let k_size =
            ShapeBuf::from_slice(&[c_in_g, kh, kw]).checked_numel(OperationKind::Conv2d)?;

        if g == 1 {
            let w_storage = WgpuStorage::new(weight.buffer.clone(), vec![c_out, k_size]);
            let col_storage = WgpuStorage::new(col_buf, vec![batch, k_size, col_spatial]);
            let out_storage = Self::matmul::<K>(&w_storage, &col_storage)?;
            return Ok(WgpuStorage::new(
                out_storage.buffer,
                vec![batch, c_out, h_out, w_out],
            ));
        }

        // groups > 1: direct kernel
        let out_elements = ShapeBuf::from_slice(&[batch, c_out, h_out, w_out])
            .checked_numel(OperationKind::Conv2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Conv2d)?;
        let conv_params = checked_u32_array(
            [
                batch, c_in, h_in, w_in, c_out, h_out, w_out, kh, kw, stride, stride, padding,
                padding, dilation, dilation, groups,
            ],
            "WGPU convolution kernel parameter",
        )?;
        dispatch::dispatch_conv2d_direct(&t.buffer, &weight.buffer, &out_buf, &conv_params)?;
        Ok(WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]))
    }
}

/// Row-major contiguous strides for a rank-4 `[N, C, H, W]` shape. WGPU
/// storage has no non-contiguous view support (`WgpuStorage::new` always
/// derives strides from shape), so pooling backward closures — which read
/// buffers back to a flat host `Vec` — can compute this directly instead of
/// pulling in `cpu::stride`.
fn contiguous_strides_4d(shape: &[usize]) -> Result<[usize; 4]> {
    let strides = StrideBuf::contiguous_for(&ShapeBuf::from_slice(shape), OperationKind::Storage)?;
    strides
        .strides()
        .try_into()
        .map_err(|_| Error::Msg("WGPU pooling expected rank-four storage".into()))
}

/// Per-axis adaptive-pooling window bounds: `start = floor(i*input_size/output_size)`,
/// `end = ceil((i+1)*input_size/output_size)`. Matches both the CPU backend's
/// `adaptive_window_bounds` (`cpu/ops/pool.rs`) and the WGSL forward kernel's
/// own `h_start`/`h_end` computation (`shaders/pool2d.wgsl`, mode 0) exactly —
/// never derives an equivalent fixed kernel_size/stride, which is wrong
/// whenever `input_size` doesn't evenly divide `output_size`.
fn adaptive_window_bounds(
    input_size: usize,
    output_size: usize,
    i: usize,
) -> Result<(usize, usize)> {
    if output_size == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::AdaptiveAvgPool2d,
            parameter: "output size",
            value: output_size,
        }
        .into());
    }
    let start = i
        .checked_mul(input_size)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::AdaptiveAvgPool2d,
            expression: "adaptive-pooling start index",
        })?
        / output_size;
    let end = i
        .checked_add(1)
        .and_then(|next| next.checked_mul(input_size))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::AdaptiveAvgPool2d,
            expression: "adaptive-pooling end index",
        })?
        .div_ceil(output_size);
    Ok((start, end))
}

fn pool_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| input.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling padded input dimension",
        })?;
    let effective_kernel = dilation
        .checked_mul(kernel - 1)
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling effective kernel dimension",
        })?;
    if effective_kernel > padded {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "effective kernel exceeds padded input",
            value: effective_kernel,
        }
        .into());
    }
    (padded - effective_kernel)
        .checked_div(stride)
        .and_then(|steps| steps.checked_add(1))
        .ok_or_else(|| {
            ShapeError::ArithmeticOverflow {
                operation: OperationKind::Pool2d,
                expression: "pooling output dimension",
            }
            .into()
        })
}

fn conv_transpose_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_padding: usize,
) -> Result<usize> {
    if input == 0 || kernel == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Conv2d,
            parameter: "transposed-convolution dimensions must be nonzero",
            value: 0,
        }
        .into());
    }
    let unpadded = (input - 1)
        .checked_mul(stride)
        .and_then(|span| {
            dilation
                .checked_mul(kernel - 1)
                .and_then(|effective| span.checked_add(effective))
        })
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "transposed-convolution output dimension",
        })?;
    let twice_padding = padding
        .checked_mul(2)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "transposed-convolution padding",
        })?;
    unpadded
        .checked_sub(twice_padding)
        .and_then(|natural| natural.checked_add(output_padding))
        .ok_or_else(|| {
            ShapeError::InvalidParameter {
                operation: OperationKind::Conv2d,
                parameter: "transposed-convolution padding/output padding",
                value: padding,
            }
            .into()
        })
}

// ─────────────────────────────────────────────────────────────────────────────
// ModuleOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> ModuleOps<Self> for WgpuBackendImpl<T, D> {
    /// `embedding`.
    fn embedding<K: DType, KInt: DType>(
        indices: &<Self as Backend>::Storage<KInt>,
        weight: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let embed_dim = weight.shape[1];
        let vocab_size = weight.shape[0];
        let seq_len = num_elements(&indices.shape)?;
        let out_elements =
            ShapeBuf::from_slice(&[seq_len, embed_dim]).checked_numel(OperationKind::Embedding)?;
        let out_buf =
            WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Storage)?;

        let [seq_len_u32, embed_dim_u32, vocab_size_u32] = checked_u32_array(
            [seq_len, embed_dim, vocab_size],
            "WGPU embedding kernel parameter",
        )?;

        dispatch::dispatch_embedding(
            &indices.buffer,
            &weight.buffer,
            &out_buf,
            seq_len_u32,
            embed_dim_u32,
            vocab_size_u32,
        )?;

        let out = WgpuStorage::new(out_buf, vec![seq_len, embed_dim]);

        let (indices_capture, weight_shape) = (indices.clone(), weight.shape.to_vec());
        let (weight_id, out_id) = (weight.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![weight_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                // Index storage is physically F32 bytes (the WGSL forward
                // kernel declares `indices: array<f32>` and does a real
                // `u32(indices[i])` value conversion) — `to_vec::<u32>()`
                // would bit-reinterpret those bytes instead of converting
                // the value, corrupting every index except 0.0 (whose bit
                // pattern happens to equal integer 0).
                let indices_data = indices_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let weight_grad_elements = ShapeBuf::from_slice(&[vocab_size, embed_dim])
                    .checked_numel(OperationKind::Embedding)?;
                let mut weight_grad = vec![0.0f32; weight_grad_elements];

                for (i, &idx) in indices_data.iter().enumerate() {
                    let idx = idx as usize;
                    if idx < vocab_size {
                        for d in 0..embed_dim {
                            weight_grad[idx * embed_dim + d] += grad_data[i * embed_dim + d];
                        }
                    }
                }

                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&weight_grad),
                    weight_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// `layer_norm`.
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
                let n = num_elements(&t.shape)?;
                let zeros = WgpuStorage::new(
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n, OperationKind::Storage)?,
                    t.shape.to_vec(),
                );
                Self::add::<K>(&scaled, &zeros)
            }
        }
    }

    /// `batch_norm`.
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
            None => WgpuStorage::new(
                WgpuBuffer::new_zeros_for(DTypeId::F32, c, OperationKind::Storage)?,
                param_shape.clone(),
            ),
        };
        let rv = match running_var {
            Some(v) => Self::reshape::<K>(v, &param_shape)?,
            None => {
                let out = WgpuStorage::new(
                    WgpuBuffer::new_zeros_for(DTypeId::F32, c, OperationKind::Storage)?,
                    param_shape.clone(),
                );
                Self::add_scalar_float::<K>(&out, 1.0)?
            }
        };

        let w = match weight {
            Some(w) => Self::reshape::<K>(w, &param_shape)?,
            None => {
                let out = WgpuStorage::new(
                    WgpuBuffer::new_zeros_for(DTypeId::F32, c, OperationKind::Storage)?,
                    param_shape.clone(),
                );
                Self::add_scalar_float::<K>(&out, 1.0)?
            }
        };

        let b = match bias {
            Some(b) => Self::reshape::<K>(b, &param_shape)?,
            None => WgpuStorage::new(
                WgpuBuffer::new_zeros_for(DTypeId::F32, c, OperationKind::Storage)?,
                param_shape.clone(),
            ),
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

    /// `adaptive_avg_pool2d`.
    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape; // [N, C, H, W]
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = output_size;
        let out_elements = ShapeBuf::from_slice(&[n, c, oh, ow])
            .checked_numel(OperationKind::AdaptiveAvgPool2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Pool2d)?;
        let [n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32] = checked_u32_array(
            [n, c, h, w, oh, ow],
            "WGPU adaptive-pooling kernel parameter",
        )?;

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 0, // mode 0 = adaptive_avg
            n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32, 0, 0, 0, 0, 0, 0, 0, 0,
        )?;

        let out = WgpuStorage::new(out_buf, vec![n, c, oh, ow]);

        // Backward: distributes grad_out's per-position value uniformly
        // (divided by that position's actual window element count — windows
        // vary in size when input_size doesn't evenly divide output_size)
        // into every input position the window covered, `+=`-accumulating
        // across overlapping windows. Mirrors the CPU backend's
        // `adaptive_avg_pool2d_impl` (`cpu/ops/pool.rs`) exactly, including
        // its `adaptive_window_bounds` formula
        // (`start = floor(i*input_size/output_size)`,
        // `end = ceil((i+1)*input_size/output_size)`), which the WGSL
        // forward kernel (`shaders/pool2d.wgsl`, mode 0) also already uses.
        let input_shape = t.shape.to_vec();
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let _ = &t_capture; // input values aren't needed, only its shape
                let (b, c, h, w) = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_elements = ShapeBuf::from_slice(&input_shape)
                    .checked_numel(OperationKind::AdaptiveAvgPool2d)?;
                let mut grad_input = vec![0.0f32; grad_elements];
                let in_strides = contiguous_strides_4d(&input_shape)?;
                let h_out = grad_out.shape[2];
                let w_out = grad_out.shape[3];
                for bi in 0..b {
                    for ci in 0..c {
                        for oh in 0..h_out {
                            let (h_start, h_end) = adaptive_window_bounds(h, h_out, oh)?;
                            for ow in 0..w_out {
                                let (w_start, w_end) = adaptive_window_bounds(w, w_out, ow)?;
                                let count = ((h_end - h_start) * (w_end - w_start)) as f32;
                                let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                                let g = grad_data[flat_out] / count;
                                for ih in h_start..h_end {
                                    for iw in w_start..w_end {
                                        let flat = bi * in_strides[0]
                                            + ci * in_strides[1]
                                            + ih * in_strides[2]
                                            + iw * in_strides[3];
                                        grad_input[flat] += g;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_input),
                    input_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// `avg_pool2d`.
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
        let oh = pool_output_dim(h, kh, sh, ph, 1)?;
        let ow = pool_output_dim(w, kw, sw, pw, 1)?;

        let out_elements =
            ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Pool2d)?;
        let [
            n_u32,
            c_u32,
            h_u32,
            w_u32,
            oh_u32,
            ow_u32,
            kh_u32,
            kw_u32,
            sh_u32,
            sw_u32,
            ph_u32,
            pw_u32,
        ] = checked_u32_array(
            [n, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw],
            "WGPU average-pooling kernel parameter",
        )?;

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 1, // mode 1 = avg
            n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32, kh_u32, kw_u32, sh_u32, sw_u32, ph_u32,
            pw_u32, 1, 1,
        )?;

        let out = WgpuStorage::new(out_buf, vec![n, c, oh, ow]);

        // Backward: distributes grad_out's per-position value uniformly
        // (divided by the FIXED kh*kw divisor — count_include_pad=True,
        // PyTorch's default, matching this op's forward, which sums the
        // padded region as 0.0 but still divides by kh*kw) into every input
        // position the window covered (padded positions are skipped, never
        // written), `+=`-accumulating across overlapping windows. Mirrors
        // the CPU backend's `avg_pool2d_impl` exactly.
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        let window_count = (kh * kw) as f32;
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let (b, c, h, w) = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_elements =
                    ShapeBuf::from_slice(&input_shape).checked_numel(OperationKind::Pool2d)?;
                let mut grad_input = vec![0.0f32; grad_elements];
                let in_strides = contiguous_strides_4d(&input_shape)?;
                let h_out = grad_out.shape[2];
                let w_out = grad_out.shape[3];
                for bi in 0..b {
                    for ci in 0..c {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                                let g = grad_data[flat_out] / window_count;
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let src_h = oh * sh + khi;
                                        let src_w = ow * sw + kwi;
                                        if src_h >= ph
                                            && src_h - ph < h
                                            && src_w >= pw
                                            && src_w - pw < w
                                        {
                                            let ih = src_h - ph;
                                            let iw = src_w - pw;
                                            let flat = bi * in_strides[0]
                                                + ci * in_strides[1]
                                                + ih * in_strides[2]
                                                + iw * in_strides[3];
                                            grad_input[flat] += g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_input),
                    input_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// `max_pool2d`.
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
        let oh = pool_output_dim(h, kh, sh, ph, dh)?;
        let ow = pool_output_dim(w, kw, sw, pw, dw)?;

        let out_elements =
            ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Pool2d)?;
        let [
            n_u32,
            c_u32,
            h_u32,
            w_u32,
            oh_u32,
            ow_u32,
            kh_u32,
            kw_u32,
            sh_u32,
            sw_u32,
            ph_u32,
            pw_u32,
            dh_u32,
            dw_u32,
        ] = checked_u32_array(
            [n, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw],
            "WGPU max-pooling kernel parameter",
        )?;

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 2, // mode 2 = max
            n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32, kh_u32, kw_u32, sh_u32, sw_u32, ph_u32,
            pw_u32, dh_u32, dw_u32,
        )?;

        let out = WgpuStorage::new(out_buf, vec![n, c, oh, ow]);

        // Backward: recomputes each output position's winning (first-argmax,
        // strict `>`) source position from the captured input (padded
        // positions are never candidates, never substituted with 0.0 —
        // matches the WGSL forward's `-FLT_MAX` init and its bounds-checked
        // skip), then `+=`-accumulates grad_out's value there — never `=`,
        // since overlapping windows (stride < kernel_size) can share a
        // winning input position. Mirrors the CPU backend's
        // `max_window_2d`/`scatter_pool_grad_2d` exactly.
        let input_shape = t.shape.to_vec();
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let (b, c, h, w) = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                let input_data = t_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_elements =
                    ShapeBuf::from_slice(&input_shape).checked_numel(OperationKind::Pool2d)?;
                let mut grad_input = vec![0.0f32; grad_elements];
                let in_strides = contiguous_strides_4d(&input_shape)?;
                let h_out = grad_out.shape[2];
                let w_out = grad_out.shape[3];
                for bi in 0..b {
                    for ci in 0..c {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let mut best_val = f32::NEG_INFINITY;
                                let mut best_flat = 0usize;
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let src_h = oh * sh + khi * dh;
                                        let src_w = ow * sw + kwi * dw;
                                        if src_h < ph
                                            || src_h - ph >= h
                                            || src_w < pw
                                            || src_w - pw >= w
                                        {
                                            continue;
                                        }
                                        let ih = src_h - ph;
                                        let iw = src_w - pw;
                                        let flat = bi * in_strides[0]
                                            + ci * in_strides[1]
                                            + ih * in_strides[2]
                                            + iw * in_strides[3];
                                        let v = input_data[flat];
                                        if v > best_val {
                                            best_val = v;
                                            best_flat = flat;
                                        }
                                    }
                                }
                                let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                                grad_input[best_flat] += grad_data[flat_out];
                            }
                        }
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_input),
                    input_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// `conv1d`.
    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Delegate to conv2d over a fake H=1 spatial dimension, then reshape
        // back to 3D. We capture IDs before delegating so we can wire our OWN
        // tape entry that maps grad_out [N, C_out, L_out] → grads for the
        // original 3D tensors, hiding the internal 4D reshape from callers.
        let inp_shape = &t.shape; // [N, C_in, L]
        let w_shape = &weight.shape; // [C_out, C_in/groups, Kl]
        let (n, c_in, l_in) = (inp_shape[0], inp_shape[1], inp_shape[2]);
        let (c_out, c_in_g, kl) = (w_shape[0], w_shape[1], w_shape[2]);
        let c_out_g = c_out / groups.max(1);

        // Forward: CPU im2col + batched matmul via GPU conv2d path.
        let inp4d = WgpuStorage::new(t.buffer.clone(), vec![n, c_in, 1, l_in]);
        let w4d = WgpuStorage::new(weight.buffer.clone(), vec![c_out, c_in_g, 1, kl]);
        // conv2d push its own internal tape entries; we want ONE clean entry
        // that owns the conv1d backward, so we deliberately skip bias inside
        // conv2d and add it via the already-tape-tracked add below.
        let out4d = Self::conv2d_no_tape::<K>(&inp4d, &w4d, stride, padding, dilation, groups)?;
        let l_out = out4d.shape[3];
        let out = WgpuStorage::new(out4d.buffer, vec![n, c_out, l_out]);

        // Push tape entry for conv1d.
        let (inp_capture, w_capture) = (t.clone(), weight.clone());
        let (inp_id, w_id, out_id) = (t.id, weight.id, out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![inp_id, w_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                // grad_out: [N, C_out, L_out]
                let input_data = inp_capture.buffer.to_vec::<f32>()?;
                let weight_data = w_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;

                // Treat as [N, C_in, 1, L] / [N, C_out, 1, L_out] for the 2D helpers.
                let (h, h_out) = (1usize, 1usize);

                let mut grad_input_groups: Vec<Vec<f32>> = Vec::with_capacity(groups);
                let mut grad_weight_groups: Vec<Vec<f32>> = Vec::with_capacity(groups);

                for g in 0..groups {
                    // Slice per-group input/weight/grad_out
                    let gin_size = ShapeBuf::from_slice(&[n, c_in_g, l_in])
                        .checked_numel(OperationKind::Conv1d)?;
                    let mut input_g = vec![0.0f32; gin_size];
                    for bi in 0..n {
                        for ci in 0..c_in_g {
                            for li in 0..l_in {
                                input_g[bi * c_in_g * l_in + ci * l_in + li] =
                                    input_data[bi * c_in * l_in + (g * c_in_g + ci) * l_in + li];
                            }
                        }
                    }

                    let gwt_size = ShapeBuf::from_slice(&[c_out_g, c_in_g, kl])
                        .checked_numel(OperationKind::Conv1d)?;
                    let mut weight_g = vec![0.0f32; gwt_size];
                    for co in 0..c_out_g {
                        for ci in 0..c_in_g {
                            for ki in 0..kl {
                                weight_g[co * c_in_g * kl + ci * kl + ki] =
                                    weight_data[(g * c_out_g + co) * c_in_g * kl + ci * kl + ki];
                            }
                        }
                    }

                    let ggo_size = ShapeBuf::from_slice(&[n, c_out_g, l_out])
                        .checked_numel(OperationKind::Conv1d)?;
                    let mut grad_out_g = vec![0.0f32; ggo_size];
                    for bi in 0..n {
                        for co in 0..c_out_g {
                            for li in 0..l_out {
                                grad_out_g[bi * c_out_g * l_out + co * l_out + li] =
                                    grad_data[bi * c_out * l_out + (g * c_out_g + co) * l_out + li];
                            }
                        }
                    }

                    // im2col on input_g treated as [N, C_in_g, 1, L]
                    let (cols, ..) = im2col_2d_cpu(
                        &input_g, n, c_in_g, h, l_in, h, kl, stride, padding, dilation,
                    )?;
                    // cols: [N, L_out, C_in_g*kl]
                    // weight_mat_t: [C_in_g*kl, C_out_g] (transposed for grad_input)
                    let weight_mat_t = cpu_transpose_last2(&weight_g, 1, c_out_g, c_in_g * kl)?;
                    // grad_out_g: [N, C_out_g, L_out] -> [N, L_out, C_out_g]
                    let go_elements = ShapeBuf::from_slice(&[n, l_out, c_out_g])
                        .checked_numel(OperationKind::Conv1d)?;
                    let mut go_t = vec![0.0f32; go_elements];
                    for bi in 0..n {
                        for li in 0..l_out {
                            for co in 0..c_out_g {
                                go_t[bi * l_out * c_out_g + li * c_out_g + co] =
                                    grad_out_g[bi * c_out_g * l_out + co * l_out + li];
                            }
                        }
                    }
                    // grad_cols = go_t @ weight_mat_t: [N, L_out, C_out_g] @ [C_out_g, C_in_g*kl]
                    let grad_cols = cpu_bmm(&go_t, &weight_mat_t, n, l_out, c_out_g, c_in_g * kl)?;
                    // col2im: [N, L_out, C_in_g*kl] -> [N, C_in_g, 1, L]
                    let grad_input_g = col2im_2d_cpu(
                        &grad_cols, n, c_in_g, h, l_in, h_out, l_out, h, kl, stride, padding,
                        dilation,
                    )?;
                    grad_input_groups.push(grad_input_g);

                    // grad_weight: go_t^T @ cols: [N, C_out_g, L_out] @ [N, L_out, C_in_g*kl]
                    let go_t2 = cpu_transpose_last2(&go_t, n, l_out, c_out_g)?;
                    let gw_mat = cpu_bmm(&go_t2, &cols, n, c_out_g, l_out, c_in_g * kl)?;
                    // sum over batch: [N, C_out_g, C_in_g*kl] -> [C_out_g, C_in_g*kl]
                    let gw_summed = cpu_sum_batch(&gw_mat, n, c_out_g, c_in_g * kl)?;
                    // reshape to [C_out_g, C_in_g, Kl]
                    grad_weight_groups.push(gw_summed);
                }

                // Reassemble gradient tensors
                let grad_input_elements =
                    ShapeBuf::from_slice(&[n, c_in, l_in]).checked_numel(OperationKind::Conv1d)?;
                let mut grad_input_data = vec![0.0f32; grad_input_elements];
                for g in 0..groups {
                    let g_data = &grad_input_groups[g];
                    for bi in 0..n {
                        for ci in 0..c_in_g {
                            for li in 0..l_in {
                                grad_input_data
                                    [bi * c_in * l_in + (g * c_in_g + ci) * l_in + li] +=
                                    g_data[bi * c_in_g * l_in + ci * l_in + li];
                            }
                        }
                    }
                }

                let grad_weight_elements = ShapeBuf::from_slice(&[c_out, c_in_g, kl])
                    .checked_numel(OperationKind::Conv1d)?;
                let mut grad_weight_data = vec![0.0f32; grad_weight_elements];
                for g in 0..groups {
                    let g_data = &grad_weight_groups[g];
                    for co in 0..c_out_g {
                        for rest in 0..c_in_g * kl {
                            grad_weight_data[(g * c_out_g + co) * c_in_g * kl + rest] +=
                                g_data[co * c_in_g * kl + rest];
                        }
                    }
                }

                Ok(vec![
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_input_data),
                        inp_capture.shape.to_vec(),
                    ),
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_weight_data),
                        w_capture.shape.to_vec(),
                    ),
                ])
            }),
        });

        // Same defect and same repair as `conv2d` below: WGPU's `add` requires
        // equal shapes, so the bias is stretched to the output shape first.
        // grad_bias flows back through the broadcast entry.
        match bias {
            Some(b) => {
                let b_shaped = Self::reshape::<K>(b, &[1, c_out, 1])?;
                let b_stretched = Self::broadcast_as::<K>(&b_shaped, &out.shape)?;
                Self::add::<K>(&out, &b_stretched)
            }
            None => Ok(out),
        }
    }

    /// `conv2d`.
    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let conv_out = Self::conv2d_no_tape::<K>(t, weight, stride, padding, dilation, groups)?;
        let shape = &t.shape; // [N, C_in, H, W]
        let ws = &weight.shape; // [C_out, C_in/groups, Kh, Kw]
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, c_in_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let c_out_g = c_out / groups.max(1);
        let h_out = conv_out.shape[2];
        let w_out = conv_out.shape[3];

        // Wire autograd tape entry.
        let (inp_capture, w_capture) = (t.clone(), weight.clone());
        let (inp_id, w_id, out_id) = (t.id, weight.id, conv_out.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![inp_id, w_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = inp_capture.buffer.to_vec::<f32>()?;
                let weight_data = w_capture.buffer.to_vec::<f32>()?;
                // grad_out: [N, C_out, H_out, W_out] → flatten to row-major vec
                let grad_data = grad_out.buffer.to_vec::<f32>()?;

                let grad_input_elements = ShapeBuf::from_slice(&[batch, c_in, h_in, w_in])
                    .checked_numel(OperationKind::Conv2d)?;
                let grad_weight_elements = ShapeBuf::from_slice(&[c_out, c_in_g, kh, kw])
                    .checked_numel(OperationKind::Conv2d)?;
                let mut grad_input_data = vec![0.0f32; grad_input_elements];
                let mut grad_weight_data = vec![0.0f32; grad_weight_elements];

                for g in 0..groups {
                    // Slice input group [N, C_in_g, H, W]
                    let input_group_elements = ShapeBuf::from_slice(&[batch, c_in_g, h_in, w_in])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut input_g = vec![0.0f32; input_group_elements];
                    for bi in 0..batch {
                        for ci in 0..c_in_g {
                            for hi in 0..h_in {
                                for wi in 0..w_in {
                                    input_g[bi * (c_in_g * h_in * w_in)
                                        + ci * (h_in * w_in)
                                        + hi * w_in
                                        + wi] = input_data[bi * (c_in * h_in * w_in)
                                        + (g * c_in_g + ci) * (h_in * w_in)
                                        + hi * w_in
                                        + wi];
                                }
                            }
                        }
                    }

                    // Slice weight group [C_out_g, C_in_g, Kh, Kw]
                    let weight_group_elements = ShapeBuf::from_slice(&[c_out_g, c_in_g, kh, kw])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut weight_g = vec![0.0f32; weight_group_elements];
                    for co in 0..c_out_g {
                        let src_co = g * c_out_g + co;
                        for rest in 0..c_in_g * kh * kw {
                            weight_g[co * c_in_g * kh * kw + rest] =
                                weight_data[src_co * c_in_g * kh * kw + rest];
                        }
                    }

                    // Slice grad_out group [N, C_out_g, H_out, W_out]
                    let go_group_elements = ShapeBuf::from_slice(&[batch, c_out_g, h_out, w_out])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut go_g = vec![0.0f32; go_group_elements];
                    for bi in 0..batch {
                        for co in 0..c_out_g {
                            for hi in 0..h_out {
                                for wi in 0..w_out {
                                    go_g[bi * (c_out_g * h_out * w_out)
                                        + co * (h_out * w_out)
                                        + hi * w_out
                                        + wi] = grad_data[bi * (c_out * h_out * w_out)
                                        + (g * c_out_g + co) * (h_out * w_out)
                                        + hi * w_out
                                        + wi];
                                }
                            }
                        }
                    }

                    let (cols, ..) = im2col_2d_cpu(
                        &input_g, batch, c_in_g, h_in, w_in, kh, kw, stride, padding, dilation,
                    )?;
                    // cols: [N, H_out*W_out, C_in_g*Kh*Kw]
                    // go_g: [N, C_out_g, H_out*W_out] → [N, H_out*W_out, C_out_g]
                    let spatial = ShapeBuf::from_slice(&[h_out, w_out])
                        .checked_numel(OperationKind::Conv2d)?;
                    let go_transposed_elements = ShapeBuf::from_slice(&[batch, spatial, c_out_g])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut go_t = vec![0.0f32; go_transposed_elements];
                    for bi in 0..batch {
                        for co in 0..c_out_g {
                            for s in 0..spatial {
                                go_t[bi * spatial * c_out_g + s * c_out_g + co] =
                                    go_g[bi * c_out_g * spatial + co * spatial + s];
                            }
                        }
                    }

                    // grad_cols = go_t @ weight_g: [N, spatial, C_out_g] @ [C_out_g, C_in_g*Kh*Kw]
                    let grad_cols =
                        cpu_bmm(&go_t, &weight_g, batch, spatial, c_out_g, c_in_g * kh * kw)?;
                    // col2im → grad for input_g [N, C_in_g, H, W]
                    let grad_input_g = col2im_2d_cpu(
                        &grad_cols, batch, c_in_g, h_in, w_in, h_out, w_out, kh, kw, stride,
                        padding, dilation,
                    )?;

                    // Accumulate into grad_input_data
                    for bi in 0..batch {
                        for ci in 0..c_in_g {
                            for hi in 0..h_in {
                                for wi in 0..w_in {
                                    grad_input_data[bi * (c_in * h_in * w_in)
                                        + (g * c_in_g + ci) * (h_in * w_in)
                                        + hi * w_in
                                        + wi] += grad_input_g[bi * (c_in_g * h_in * w_in)
                                        + ci * (h_in * w_in)
                                        + hi * w_in
                                        + wi];
                                }
                            }
                        }
                    }

                    // grad_weight_g: go_t^T @ cols → [N, C_out_g, C_in_g*Kh*Kw] → sum over batch
                    let go_t2 = cpu_transpose_last2(&go_t, batch, spatial, c_out_g)?;
                    let gw_mat = cpu_bmm(&go_t2, &cols, batch, c_out_g, spatial, c_in_g * kh * kw)?;
                    let gw_summed = cpu_sum_batch(&gw_mat, batch, c_out_g, c_in_g * kh * kw)?;

                    for co in 0..c_out_g {
                        for rest in 0..c_in_g * kh * kw {
                            grad_weight_data[(g * c_out_g + co) * c_in_g * kh * kw + rest] +=
                                gw_summed[co * c_in_g * kh * kw + rest];
                        }
                    }
                }

                Ok(vec![
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_input_data),
                        inp_capture.shape.to_vec(),
                    ),
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_weight_data),
                        w_capture.shape.to_vec(),
                    ),
                ])
            }),
        });

        // The bias is stretched to the output shape *before* the add: WGPU's
        // elementwise kernels require equal shapes and do not broadcast, so
        // handing `add` a `[1, C_out, 1, 1]` operand fails for every biased
        // convolution. Both steps are tape-tracked, so grad_bias still flows
        // back through the broadcast.
        match bias {
            Some(b) => {
                let b_shaped = Self::reshape::<K>(b, &[1, c_out, 1, 1])?;
                let b_stretched = Self::broadcast_as::<K>(&b_shaped, &conv_out.shape)?;
                Self::add::<K>(&conv_out, &b_stretched)
            }
            None => Ok(conv_out),
        }
    }

    /// `conv_transpose2d`.
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

        let h_nat = conv_transpose_output_dim(h_in, kh, stride, padding, dilation, 0)?;
        let w_nat = conv_transpose_output_dim(w_in, kw, stride, padding, dilation, 0)?;
        let h_out = h_nat
            .checked_add(output_padding)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Conv2d,
                expression: "transposed-convolution height plus output padding",
            })?;
        let w_out = w_nat
            .checked_add(output_padding)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Conv2d,
                expression: "transposed-convolution width plus output padding",
            })?;

        let out_elements = ShapeBuf::from_slice(&[batch, c_out, h_out, w_out])
            .checked_numel(OperationKind::Conv2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Conv2d)?;

        let params = checked_u32_array(
            [
                batch, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw, stride, stride, padding,
                padding, dilation, dilation, groups,
            ],
            "WGPU transposed-convolution kernel parameter",
        )?;

        dispatch::dispatch_conv_transpose2d(&t.buffer, &weight.buffer, &out_buf, &params)?;
        let out_storage = WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]);

        // Wire autograd tape (groups==1 only; matches CPU backend's documented scope).
        let (inp_capture, w_capture) = (t.clone(), weight.clone());
        let (inp_id, w_id, out_id) = (t.id, weight.id, out_storage.id);
        crate::wgpu::tape::push(crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![inp_id, w_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = inp_capture.buffer.to_vec::<f32>()?;
                let weight_data = w_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;

                // Narrow away output_padding rows/cols from grad_out.
                let go_nat: Vec<f32> = if output_padding == 0 {
                    grad_data.clone()
                } else {
                    let natural_elements = ShapeBuf::from_slice(&[batch, c_out, h_nat, w_nat])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut nat = vec![0.0f32; natural_elements];
                    for bi in 0..batch {
                        for co in 0..c_out {
                            for hi in 0..h_nat {
                                for wi in 0..w_nat {
                                    nat[bi * (c_out * h_nat * w_nat)
                                        + co * (h_nat * w_nat)
                                        + hi * w_nat
                                        + wi] = grad_data[bi * (c_out * h_out * w_out)
                                        + co * (h_out * w_out)
                                        + hi * w_out
                                        + wi];
                                }
                            }
                        }
                    }
                    nat
                };

                // grad_input: apply conv2d forward formula to go_nat with weight.
                // im2col on go_nat [N, C_out, H_nat, W_nat] using same kernel geometry.
                let (go_cols, ..) = im2col_2d_cpu(
                    &go_nat, batch, c_out, h_nat, w_nat, kh, kw, stride, padding, dilation,
                )?;
                // go_cols: [N, H_in*W_in, C_out*Kh*Kw]
                // weight_mat_t: [C_out*Kh*Kw, C_in] (transposed for grad_input: grad = go_cols @ W^T)
                let weight_mat_t = cpu_transpose_last2(&weight_data, 1, c_in, c_out * kh * kw)?;
                // grad_input_flat = go_cols @ weight_mat_t: [N, H_in*W_in, C_out*Kh*Kw] @ [C_out*Kh*Kw, C_in]
                let spatial_in = h_in * w_in;
                let grad_input_flat = cpu_bmm(
                    &go_cols,
                    &weight_mat_t,
                    batch,
                    spatial_in,
                    c_out * kh * kw,
                    c_in,
                )?;
                // [N, H_in*W_in, C_in] -> [N, C_in, H_in, W_in]
                let grad_input_elements = ShapeBuf::from_slice(&[batch, c_in, h_in, w_in])
                    .checked_numel(OperationKind::Conv2d)?;
                let mut grad_input_data = vec![0.0f32; grad_input_elements];
                for bi in 0..batch {
                    for ci in 0..c_in {
                        for s in 0..spatial_in {
                            let hi = s / w_in;
                            let wi = s % w_in;
                            grad_input_data
                                [bi * (c_in * h_in * w_in) + ci * (h_in * w_in) + hi * w_in + wi] =
                                grad_input_flat[bi * (spatial_in * c_in) + s * c_in + ci];
                        }
                    }
                }

                // grad_weight: same swap as CPU conv_transpose2d backward:
                // input_t^T @ go_cols -> [N, C_in, C_out*Kh*Kw] -> sum_batch -> [C_in, C_out, Kh, Kw]
                let input_flat_elements = ShapeBuf::from_slice(&[batch, spatial_in, c_in])
                    .checked_numel(OperationKind::Conv2d)?;
                let mut input_flat_t = vec![0.0f32; input_flat_elements];
                for bi in 0..batch {
                    for ci in 0..c_in {
                        for s in 0..spatial_in {
                            let hi = s / w_in;
                            let wi_idx = s % w_in;
                            input_flat_t[bi * (spatial_in * c_in) + s * c_in + ci] = input_data[bi
                                * (c_in * h_in * w_in)
                                + ci * (h_in * w_in)
                                + hi * w_in
                                + wi_idx];
                        }
                    }
                }
                let input_t2 = cpu_transpose_last2(&input_flat_t, batch, spatial_in, c_in)?;
                let gw_mat = cpu_bmm(
                    &input_t2,
                    &go_cols,
                    batch,
                    c_in,
                    spatial_in,
                    c_out * kh * kw,
                )?;
                let gw_summed = cpu_sum_batch(&gw_mat, batch, c_in, c_out * kh * kw)?;

                Ok(vec![
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_input_data),
                        inp_capture.shape.to_vec(),
                    ),
                    WgpuStorage::new(WgpuBuffer::from_slice(&gw_summed), w_capture.shape.to_vec()),
                ])
            }),
        });

        // Bias via already-tape-tracked add.
        match bias {
            Some(b) => {
                let b_shaped = Self::reshape::<K>(b, &[1, c_out, 1, 1])?;
                Self::add::<K>(&out_storage, &b_shaped)
            }
            None => Ok(out_storage),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LossOps (cross_entropy delegated to base trait which composes from float/reduce ops)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> LossOps<Self> for WgpuBackendImpl<T, D> {
    /// `cross_entropy_loss`.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<KInt>,
        reduction: incin_core::prelude::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let batch = pred.shape[0];
        let classes = pred.shape[1];

        // 1. log_probs = log_softmax(pred, 1)
        let log_probs = log_softmax::<T, K>(pred, 1)?;

        // 2. Build one_hot constant WgpuStorage. Target storage is
        // physically F32 bytes (this backend has no genuine integer
        // storage), so read it as F32 and convert the value — a raw
        // `to_vec::<u32>()` bit-reinterpret would corrupt every class index
        // except 0.0 (whose bit pattern happens to equal integer 0).
        let target_data = target.buffer.to_vec::<f32>()?;
        let one_hot_elements =
            ShapeBuf::from_slice(&[batch, classes]).checked_numel(OperationKind::Storage)?;
        let mut one_hot_data = vec![0.0f32; one_hot_elements];
        for b_idx in 0..batch {
            let class_idx = target_data[b_idx] as usize;
            if class_idx < classes {
                one_hot_data[b_idx * classes + class_idx] = 1.0;
            }
        }
        let one_hot_buf = WgpuBuffer::from_slice(&one_hot_data);
        let one_hot = WgpuStorage::new(one_hot_buf, vec![batch, classes]);

        // 3. picked = log_probs * one_hot
        let picked = Self::mul::<K>(&log_probs, &one_hot)?;

        // 4. per_nll = -sum_dim(picked, 1)
        let sum_picked = Self::sum_dim::<K>(&picked, 1)?;
        let per_nll = Self::neg::<K>(&sum_picked)?;

        // 5. Dispatch reduction
        match reduction {
            incin_core::prelude::Reduction::Mean => Self::mean_all::<K>(&per_nll),
            incin_core::prelude::Reduction::Sum => Self::sum_all::<K>(&per_nll),
            incin_core::prelude::Reduction::None => Ok(per_nll),
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
impl<T: DType, D: Device> QuantizedOps<Self> for WgpuBackendImpl<T, D> {
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

        let f32_data: Vec<f32> = t.buffer.to_vec::<f32>()?;
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
            let d_f16 = incin_core::prelude::f16::from_f32(d);
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
        WgpuStorage::try_new_packed_q8(buf, t.shape.to_vec())
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

        let raw: Vec<u8> = t.buffer.to_vec::<u8>()?;
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
            let d = incin_core::prelude::f16::from_bits(d_bits).to_f32();
            for i in 0..32 {
                let q = block[2 + i] as i8;
                f32_data.push(q as f32 * d);
            }
        }

        let buf = WgpuBuffer::from_slice::<f32>(&f32_data);
        Ok(WgpuStorage::new(buf, t.shape.to_vec()))
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
impl<T: DType, D: Device> OptimizerOps<Self> for WgpuBackendImpl<T, D> {
    /// `adamw_step`.
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
        let n = checked_u32(
            num_elements(&var.storage.shape)?,
            "WGPU AdamW parameter element count",
        )?;
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
