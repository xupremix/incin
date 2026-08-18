pub(crate) use crate::wgpu::capability::validate_wgpu_dtype;
use crate::wgpu::dispatch;
use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
use incin_core::backend_authoring::*;
use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::{OperationKind, ShapeError, StrideBuf};
use incin_core::tensor::device::{Device, DeviceId, DeviceKind, Wgpu};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

/// WebGPU compute backend implementation for Incin.
#[derive(Clone)]
pub struct WgpuBackendImpl<D = Wgpu>(core::marker::PhantomData<D>);

impl<D> WgpuBackendImpl<D> {
    /// Construct the stateless WGPU executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for WgpuBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// A trainable-parameter slot: the one deliberate interior-mutability
/// boundary in the WGPU backend, mirroring `CpuVar`.
///
/// The shared handle is load bearing, not a style choice. An optimizer holds
/// its own map of these and commits an update through
/// [`VariableBackend::assign_var`]; the model holds the *same* parameters. With a
/// plain owned `WgpuStorage` here, `assign_var` replaced the optimizer's copy
/// and the model never saw it, so `optimizer.step()` was a no-op and training
/// loss sat at exactly its initial value forever — the failure looks like a
/// bad learning rate rather than a broken write, which is why it survived.
#[derive(Clone)]
pub struct WgpuVar {
    /// Boxed behind `Rc<RefCell<_>>` so every clone of this parameter slot
    /// observes an assignment. Private: handing out the cell would let a
    /// caller hold a live borrow across an `assign_var` and panic on the
    /// reentrant mutable borrow, which is the same hazard `cpu::var`
    /// documents.
    storage: alloc::rc::Rc<core::cell::RefCell<WgpuStorage>>,
}

impl WgpuVar {
    /// Wrap `storage` in a fresh parameter slot.
    #[must_use]
    pub(crate) fn new(storage: WgpuStorage) -> Self {
        Self {
            storage: alloc::rc::Rc::new(core::cell::RefCell::new(storage)),
        }
    }

    /// The current value, cloned out.
    ///
    /// Never returns the `Ref` guard: holding one across a later
    /// `assign_var` on the same slot would panic on a reentrant mutable
    /// borrow.
    #[must_use]
    pub(crate) fn value(&self) -> WgpuStorage {
        self.storage.borrow().clone()
    }

    /// Replace the wrapped value, visible through every clone of this slot.
    pub(crate) fn assign(&self, storage: WgpuStorage) {
        *self.storage.borrow_mut() = storage;
    }
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
        incin_core::shapes::ShapeError::ArithmeticOverflow {
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
    dtype: DTypeDescriptor,
    device: &DeviceId,
    _family: OperationKind,
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
    validate_wgpu_dtype(dtype, op)
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend core trait
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> incin_core::backend_authoring::StorageBackend for WgpuBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Wgpu";
    type Storage<K: DType> = WgpuStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        let t: &WgpuStorage = t;
        &t.meta
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl incin_core::backend_authoring::StorageOutput for WgpuStorage {}

impl<D: Device> Backend for WgpuBackendImpl<D> {
    /// `Grads`.
    /// `InnerBackend`.
    type InnerBackend = Self;

    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.
}

impl<D: Device> incin_core::backend_authoring::HostReadback for WgpuBackendImpl<D> {
    fn float_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<f64>> {
        let t: &WgpuStorage = t;
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    fn int_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<i64>> {
        let t: &WgpuStorage = t;
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        data.into_iter()
            .map(|value| {
                incin_core::error::convert_f64_to_i64(
                    "int_to_vec1",
                    t.dtype,
                    f64::from(value),
                    incin_core::error::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for WgpuBackendImpl<D> {
    /// `to_bytes`.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        let t: &WgpuStorage = t;
        t.buffer.to_vec::<u8>()
    }
    /// `from_bytes`.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Storage, "from_bytes")?;
        let expected = num_elements(shape)?
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(incin_core::shapes::ShapeError::ArithmeticOverflow {
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
// Concrete creation helpers
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `full`. WGPU storage is always physically f32 (`zeros`/`ones` above
    /// build a `Vec<f32>` regardless of the requested `dtype`, which
    /// `validate_wgpu` restricts to what the dtype policy allows), so this
    /// fills a host-side `Vec<f32>` and uploads it exactly like they do.
    pub(crate) fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "full")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![val as f32; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }
    /// `arange`.
    pub(crate) fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "arange")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `linspace`.
    pub(crate) fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
    pub(crate) fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "zeros")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![0.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `ones`.
    pub(crate) fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "ones")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `rand`.
    pub(crate) fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
    pub(crate) fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
    pub(crate) fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_ones`.
    pub(crate) fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_rand`.
    pub(crate) fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_randn`.
    pub(crate) fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }
}

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
fn broadcast_storage(t: &WgpuStorage, shape: &[usize]) -> Result<WgpuStorage> {
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
/// The kernel itself is strictly elementwise — it walks two equal-length
/// buffers — so an operand that needs stretching is materialized at the
/// broadcast shape first rather than the shader growing a stride argument.
/// That costs an allocation for the stretched operand, which is why the
/// equal-shape case still goes straight to the kernel.
///
/// This used to refuse any shape disagreement outright, which made
/// `broadcast_add` and friends fail on WGPU even though the frontend had
/// already resolved the output shape at the type level, and made
/// `Linear::forward` — a matmul plus a rank-one bias add — unusable on this
/// backend for every model. The backward pass never had that gap: both tape
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
        // align, so an incompatible pair still fails here — with the axis
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
fn scalar_op<T: DType>(t: &WgpuStorage, scalar: f64, op_mode: u32) -> Result<WgpuStorage> {
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
        // needed — scalar ops don't change shape).
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
        // relu'(x) = step(x) (1 if x>0 else 0) — input-based.
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
        // 1 if x>0, -1 if x<0, 0 if x==0 — matches the CPU backend exactly.
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

// ─────────────────────────────────────────────────────────────────────────────
//   (reshape, transpose, matmul, narrow, flatten, squeeze, stack, concat, etc.)
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `matmul`.
    pub(crate) fn matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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
    pub(crate) fn reshape<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::reshape::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }

    /// `transpose`.
    pub(crate) fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::transpose::<K>(grad_out, dim1, dim2)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn broadcast_as<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        broadcast_storage(t, shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//
// ─────────────────────────────────────────────────────────────────────────────
/// `reduce_all_to_storage`.
pub(crate) fn reduce_all_to_storage(t: &WgpuStorage, mode: u32) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU reduction element count")?;
    let out = dispatch::dispatch_reduce_all(&t.buffer, n, mode)?;
    Ok(WgpuStorage::new(out, vec![]))
}

/// `reduce_dim_to_storage`.
pub(crate) fn reduce_dim_to_storage(
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
pub(crate) fn axis_reduce_dims(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    let outer: usize = incin_core::shapes::ShapeBuf::from_slice(&(shape[..dim]))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let axis = shape[dim];
    let inner: usize = incin_core::shapes::ShapeBuf::from_slice(&(shape[dim + 1..]))
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    Ok((outer, axis, inner))
}

/// Backward for `max_dim`/`min_dim`: recomputes each output position's
/// winning (first-encountered, strict `>`/`<`) source position from the
/// captured input, then scatters `grad_out`'s value there with a bare `=`
/// (never `+=` — unlike pooling, a plain axis reduction never has two output
/// positions sharing the same winning source element). Mirrors the CPU
/// backend's `max_axis_with_indices`/`min_axis_with_indices` +
/// `scatter_axis_grad` (`cpu/ops/reduce.rs`) exactly. Not used for
/// `max_keepdim`/`min_keepdim` — see their doc comments.
pub(crate) fn push_extremum_dim_tape_entry(
    t: &WgpuStorage,
    out: &WgpuStorage,
    dim: usize,
    is_max: bool,
) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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
pub(crate) fn push_extremum_all_tape_entry(t: &WgpuStorage, out: &WgpuStorage, is_max: bool) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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

impl<D: Device> WgpuBackendImpl<D> {
    /// `prod_all`. Not autograd-wired, matching CPU.
    pub(crate) fn prod_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        reduce_all_to_storage(t, 3)
    }
    /// `prod_dim`. Not autograd-wired, matching CPU.
    pub(crate) fn prod_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        reduce_dim_to_storage(t, dim, 3, false)
    }

    /// `sum_all`.
    pub(crate) fn sum_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 0)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_all`.
    pub(crate) fn mean_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_all_to_storage(t, 0)?;
        let n = num_elements(&t.shape)? as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let scaled = scalar_op::<K>(grad_out, 1.0 / n, 1)?;
                Ok(vec![Self::broadcast_as::<K>(&scaled, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `max_all`.
    pub(crate) fn max_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 1)?;
        push_extremum_all_tape_entry(t, &out, true);
        Ok(out)
    }
    /// `min_all`.
    pub(crate) fn min_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 2)?;
        push_extremum_all_tape_entry(t, &out, false);
        Ok(out)
    }

    /// `sum_dim`.
    pub(crate) fn sum_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, false)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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
    pub(crate) fn sum_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, true)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_dim`.
    pub(crate) fn mean_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, false)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.to_vec();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape)?;
                let expanded = Self::broadcast_as::<K>(&keepdim, &original_shape)?;
                Ok(vec![scalar_op::<K>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }
    /// `mean_keepdim`.
    pub(crate) fn mean_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, true)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let expanded = Self::broadcast_as::<K>(grad_out, &original_shape)?;
                Ok(vec![scalar_op::<K>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }
    /// `max_dim`.
    pub(crate) fn max_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
    /// function of `x` on their whole domain, not just equal at one point --
    /// their gradients must therefore be identical too, whether or not `M`
    /// is treated as differentiable. Wiring a real gradient here does NOT
    /// need `log_softmax` to detach `M`. Matches the CPU backend, whose
    /// `max_keepdim` is fully wired and whose composed `log_softmax` (same
    /// formula) passes `softmax_gradcheck`/`log_softmax_gradcheck`.
    pub(crate) fn max_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 1, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, true);
        Ok(out)
    }
    /// `min_dim`.
    pub(crate) fn min_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, false)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
    }
    /// `min_keepdim`.
    ///
    /// Autograd-wired the same way as `min_dim` -- see `max_keepdim`'s doc
    /// comment above.
    pub(crate) fn min_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col_2d_cpu(
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn col2im_2d_cpu(
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
pub(crate) fn cpu_bmm(
    lhs: &[f32],
    rhs: &[f32],
    b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
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
pub(crate) fn cpu_transpose_last2(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
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
pub(crate) fn cpu_sum_batch(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
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
impl<D: Device> WgpuBackendImpl<D> {
    /// Forward-only conv2d (no tape entry). Used by both `conv1d` and `conv2d`
    /// so they can push exactly ONE clean tape entry each for their respective
    /// grad shapes, rather than having nested entries from the internal matmul.
    pub(crate) fn conv2d_no_tape<K: DType>(
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
pub(crate) fn contiguous_strides_4d(shape: &[usize]) -> Result<[usize; 4]> {
    let strides = StrideBuf::contiguous_for(&ShapeBuf::from_slice(shape), OperationKind::Storage)?;
    strides
        .strides()
        .try_into()
        .map_err(|_| Error::Msg("WGPU pooling expected rank-four storage".into()))
}

pub(crate) fn pool_output_dim(
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

// ─────────────────────────────────────────────────────────────────────────────
//
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `avg_pool2d`.
    pub(crate) fn avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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
    pub(crate) fn max_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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

    /// `conv2d`.
    pub(crate) fn conv2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss helper (cross_entropy is composed from float/reduce operations).
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Quantization helpers (Q8_0: CPU-side encode/decode, GPU matmul via dequant)
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

impl<D: Device> incin_core::backend_authoring::AutogradBackend for WgpuBackendImpl<D> {
    type Grads = WgpuGrads;

    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::wgpu::tape::backward(loss)
    }

    fn backward_with<K: DType>(
        loss: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        crate::wgpu::tape::backward_with(loss, seed)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    fn set_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()> {
        grads.set(t.id, value);
        Ok(())
    }
}
impl<D: Device> VariableBackend for WgpuBackendImpl<D> {
    /// `Var<K>`.
    type Var<K: DType> = WgpuVar;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Ok(var.value())
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
        let t: &WgpuStorage = t;
        Ok(WgpuVar::new(t.clone()))
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
        var.assign(tensor.clone());
        Ok(())
    }
}
