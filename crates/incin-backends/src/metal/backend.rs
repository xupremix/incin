//! Metal compute backend for Incin on Apple Silicon and macOS.

use core::marker::PhantomData;

use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::exec::TensorMeta;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::{Device, DeviceId, DeviceKind, Metal};
use incin_core::tensor::dtype::{DType, DTypeDescriptor};

pub(crate) use crate::metal::capability::validate_metal_storage_dtype;
use crate::metal::storage::{MetalStorage, MetalStorageMode};
use crate::metal::tape::MetalGrads;

/// Metal compute backend implementation for Incin.
#[derive(Clone)]
pub struct MetalBackendImpl<D = Metal>(PhantomData<D>);

impl<D> MetalBackendImpl<D> {
    /// Constructs a stateless Metal executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D> Default for MetalBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
/// A trainable variable whose bytes live in Metal memory.
pub struct MetalVar {
    /// Metal-resident storage backing this variable.
    pub storage: MetalStorage,
}

fn validate_metal(
    dtype: DTypeDescriptor,
    device: &DeviceId,
    _family: OperationKind,
    op: &'static str,
) -> Result<()> {
    if device.kind() != DeviceKind::Metal {
        return Err(Error::DeviceInitializationError {
            expected: "metal".to_string(),
            got: format!("{:?}", device.kind()),
        });
    }
    validate_metal_storage_dtype(dtype, op)
}

fn num_elements(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Into::into)
}

fn unsupported(op: &'static str) -> Error {
    Error::UnsupportedBackendOperation {
        op,
        backend: "Metal",
    }
}

// ─── Backend ────────────────────────────────────────────────────────────────

impl<D: Device> incin_core::backend_authoring::StorageBackend for MetalBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Metal";
    type Storage<K: DType> = MetalStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        t.metadata()
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl incin_core::backend_authoring::StorageOutput for MetalStorage {}

// A variable is a trainable handle, not storage, so it carries `ExecuteOutput`
// directly rather than through `StorageOutput`. `CpuVar` does the same; the
// `var_*` creation executors are what need it.
impl incin_core::backend_authoring::ExecuteOutput for MetalVar {}

impl<D: Device> Backend for MetalBackendImpl<D> {
    type InnerBackend = Self;

    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.
}

impl<D: Device> incin_core::backend_authoring::HostReadback for MetalBackendImpl<D> {
    fn float_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<f64>> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    fn int_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<i64>> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        data.iter()
            .map(|&value| {
                incin_core::error::convert_f64_to_i64(
                    "int_to_vec1",
                    t.metadata().dtype(),
                    f64::from(value),
                    incin_core::error::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for MetalBackendImpl<D> {
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        t.as_bytes().map(<[u8]>::to_vec)
    }
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Storage, "from_bytes")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let numel = num_elements(shape)?;
        let meta =
            TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), numel)?;
        MetalStorage::from_bytes(
            bytes.to_vec(),
            meta,
            MetalStorageMode::Shared,
            device.ordinal(),
        )
    }
}

// ─── Concrete creation helpers ──────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    /// `full`. Same host-fill-then-upload pattern `ones` above already
    /// uses.
    pub(crate) fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "full")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![val as f32; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }
    /// `arange`.
    pub(crate) fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "arange")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let data: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }
    /// `linspace`.
    pub(crate) fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "linspace")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let step = if n > 1 {
            (end - start) / ((n - 1) as f64)
        } else {
            0.0
        };
        let data: Vec<f32> = (0..n)
            .map(|i| if i == n - 1 { end } else { start + (i as f64) * step } as f32)
            .collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    pub(crate) fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "zeros")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        MetalStorage::zeros(
            &shape_buf,
            dtype,
            MetalStorageMode::Shared,
            device.ordinal(),
        )
    }

    pub(crate) fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "ones")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    pub(crate) fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Random, "rand")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![0.5; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    pub(crate) fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Random, "randn")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![0.0; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn unbroadcast(grad: &MetalStorage, target_shape: &[usize]) -> Result<MetalStorage> {
    if grad.metadata().shape().dims() == target_shape {
        return Ok(grad.clone());
    }
    let grad_dims = grad.metadata().shape().dims();
    let ndim_diff = grad_dims.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0)?;
    }

    let cur_dims = result.metadata().shape().dims().to_vec();
    for (i, &t_dim) in target_shape.iter().enumerate() {
        if t_dim == 1 && cur_dims[i] != 1 {
            result = sum_dim_keepdim(&result, i)?;
        }
    }

    Ok(result)
}

fn sum_dim_squeeze(storage: &MetalStorage, axis: usize) -> Result<MetalStorage> {
    let reduced = sum_dim_keepdim(storage, axis)?;
    let mut new_dims = reduced.metadata().shape().dims().to_vec();
    new_dims.remove(axis);
    reshape_metal(&reduced, &new_dims)
}

fn sum_dim_keepdim(storage: &MetalStorage, axis: usize) -> Result<MetalStorage> {
    sum_dim_impl(storage, axis, true)
}

fn binary_op_metal(
    lhs: &MetalStorage,
    rhs: &MetalStorage,
    op_name: &'static str,
    f: impl Fn(f32, f32) -> f32,
) -> Result<MetalStorage> {
    let a_dims = lhs.metadata().shape().dims();
    let b_dims = rhs.metadata().shape().dims();

    if a_dims == b_dims {
        let a_bytes = lhs.as_bytes()?;
        let b_bytes = rhs.as_bytes()?;
        let a_slice: &[f32] = bytemuck::cast_slice(a_bytes);
        let b_slice: &[f32] = bytemuck::cast_slice(b_bytes);
        let out_data: Vec<f32> = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        let shape_buf = ShapeBuf::from_slice(a_dims);
        let meta = TensorMeta::contiguous(
            shape_buf,
            lhs.metadata().dtype(),
            lhs.device(),
            MetalStorage::alignment(),
            out_data.len(),
        )?;
        MetalStorage::from_bytes(out_bytes, meta, lhs.mode(), lhs.device_ordinal())
    } else {
        let out_shape = incin_core::shapes::broadcast::broadcast_dim_slices(a_dims, b_dims)
            .map_err(|error| Error::ShapeMismatch {
                op: op_name,
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
                msg: error.to_string(),
            })?;
        let max_rank = out_shape.len();
        let total: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_shape))
            .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
        let a_bytes = lhs.as_bytes()?;
        let b_bytes = rhs.as_bytes()?;
        let a_slice: &[f32] = bytemuck::cast_slice(a_bytes);
        let b_slice: &[f32] = bytemuck::cast_slice(b_bytes);

        let mut out_data = Vec::with_capacity(total);
        for idx in 0..total {
            let mut curr = idx;
            let mut multi = vec![0; max_rank];
            for i in (0..max_rank).rev() {
                multi[i] = curr % out_shape[i];
                curr /= out_shape[i];
            }
            let mut idx_a = 0;
            let mut stride_a = 1;
            for i in (0..a_dims.len()).rev() {
                let m_idx = multi[i + max_rank - a_dims.len()];
                let a_axis_idx = if a_dims[i] == 1 { 0 } else { m_idx };
                idx_a += a_axis_idx * stride_a;
                stride_a *= a_dims[i];
            }
            let mut idx_b = 0;
            let mut stride_b = 1;
            for i in (0..b_dims.len()).rev() {
                let m_idx = multi[i + max_rank - b_dims.len()];
                let b_axis_idx = if b_dims[i] == 1 { 0 } else { m_idx };
                idx_b += b_axis_idx * stride_b;
                stride_b *= b_dims[i];
            }
            out_data.push(f(a_slice[idx_a], b_slice[idx_b]));
        }
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        let shape_buf = ShapeBuf::from_slice(&out_shape);
        let meta = TensorMeta::contiguous(
            shape_buf,
            lhs.metadata().dtype(),
            lhs.device(),
            MetalStorage::alignment(),
            out_data.len(),
        )?;
        MetalStorage::from_bytes(out_bytes, meta, lhs.mode(), lhs.device_ordinal())
    }
}

fn unary_op_metal(t: &MetalStorage, f: impl Fn(f32) -> f32) -> Result<MetalStorage> {
    let bytes = t.as_bytes()?;
    let slice: &[f32] = bytemuck::cast_slice(bytes);
    let out_data: Vec<f32> = slice.iter().map(|&x| f(x)).collect();
    let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
    let meta = TensorMeta::contiguous(
        t.metadata().shape().clone(),
        t.metadata().dtype(),
        t.device(),
        MetalStorage::alignment(),
        out_data.len(),
    )?;
    MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())
}

fn scalar_op_metal(
    t: &MetalStorage,
    scalar: f64,
    f: impl Fn(f32, f32) -> f32,
) -> Result<MetalStorage> {
    let s_f32 = scalar as f32;
    unary_op_metal(t, move |x| f(x, s_f32))
}

fn sum_dim_impl(t: &MetalStorage, axis: usize, keepdim: bool) -> Result<MetalStorage> {
    let dims = t.metadata().shape().dims();
    if axis >= dims.len() {
        return Err(Error::ShapeMismatch {
            op: "sum_dim",
            expected: vec![dims.len()],
            got: vec![axis],
            msg: "axis out of bounds".to_string(),
        });
    }
    let mut out_dims = dims.to_vec();
    if keepdim {
        out_dims[axis] = 1;
    } else {
        out_dims.remove(axis);
    }
    let out_numel: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_dims))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let mut out_data = vec![0.0f32; out_numel];

    let bytes = t.as_bytes()?;
    let in_slice: &[f32] = bytemuck::cast_slice(bytes);
    let outer: usize = incin_core::shapes::ShapeBuf::from_slice(&(dims[..axis]))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let axis_len = dims[axis];
    let inner: usize = incin_core::shapes::ShapeBuf::from_slice(&(dims[axis + 1..]))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;

    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                let in_idx = o * axis_len * inner + a * inner + i;
                let out_idx = o * inner + i;
                out_data[out_idx] += in_slice[in_idx];
            }
        }
    }

    let shape_buf = ShapeBuf::from_slice(&out_dims);
    let meta = TensorMeta::contiguous(
        shape_buf,
        t.metadata().dtype(),
        t.device(),
        MetalStorage::alignment(),
        out_numel,
    )?;
    let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
    MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())
}

fn mean_dim_impl(t: &MetalStorage, axis: usize, keepdim: bool) -> Result<MetalStorage> {
    let dims = t.metadata().shape().dims();
    if axis >= dims.len() {
        return Err(Error::ShapeMismatch {
            op: "mean_dim",
            expected: vec![dims.len()],
            got: vec![axis],
            msg: "axis out of bounds".to_string(),
        });
    }
    let count = dims[axis] as f32;
    let sum = sum_dim_impl(t, axis, keepdim)?;
    scalar_op_metal(&sum, (1.0 / count) as f64, |x, s| x * s)
}

fn matmul_metal(lhs: &MetalStorage, rhs: &MetalStorage) -> Result<MetalStorage> {
    let lhs_dims = lhs.metadata().shape().dims();
    let rhs_dims = rhs.metadata().shape().dims();
    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![2],
            got: vec![lhs_dims.len(), rhs_dims.len()],
            msg: "matmul requires at least 2D inputs".to_string(),
        });
    }

    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let rhs_k = rhs_dims[rhs_dims.len() - 2];
    let n = rhs_dims[rhs_dims.len() - 1];

    if k != rhs_k {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: lhs_dims.to_vec(),
            got: rhs_dims.to_vec(),
            msg: "matmul inner dims must match".to_string(),
        });
    }

    let lhs_batch: usize =
        incin_core::shapes::ShapeBuf::from_slice(&(lhs_dims[..lhs_dims.len() - 2]))
            .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let rhs_batch: usize =
        incin_core::shapes::ShapeBuf::from_slice(&(rhs_dims[..rhs_dims.len() - 2]))
            .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    let batch = lhs_batch.max(rhs_batch).max(1);

    let mut out_shape = if lhs_batch >= rhs_batch && lhs_dims.len() > 2 {
        lhs_dims[..lhs_dims.len() - 2].to_vec()
    } else if rhs_dims.len() > 2 {
        rhs_dims[..rhs_dims.len() - 2].to_vec()
    } else {
        vec![]
    };
    out_shape.push(m);
    out_shape.push(n);

    let out_numel: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    let mut out_data = vec![0.0f32; out_numel];

    let a_bytes = lhs.as_bytes()?;
    let b_bytes = rhs.as_bytes()?;
    let a_slice: &[f32] = bytemuck::cast_slice(a_bytes);
    let b_slice: &[f32] = bytemuck::cast_slice(b_bytes);

    let lhs_b_stride = if lhs_batch == 1 { 0 } else { m * k };
    let rhs_b_stride = if rhs_batch == 1 { 0 } else { k * n };

    for b_idx in 0..batch {
        let a_offset = b_idx * lhs_b_stride;
        let b_offset = b_idx * rhs_b_stride;
        let out_offset = b_idx * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_slice[a_offset + i * k + kk] * b_slice[b_offset + kk * n + j];
                }
                out_data[out_offset + i * n + j] = sum;
            }
        }
    }

    let shape_buf = ShapeBuf::from_slice(&out_shape);
    let meta = TensorMeta::contiguous(
        shape_buf,
        lhs.metadata().dtype(),
        lhs.device(),
        MetalStorage::alignment(),
        out_numel,
    )?;
    let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
    MetalStorage::from_bytes(out_bytes, meta, lhs.mode(), lhs.device_ordinal())
}

fn reshape_metal(storage: &MetalStorage, shape: &[usize]) -> Result<MetalStorage> {
    let numel = storage
        .metadata()
        .shape()
        .checked_numel(OperationKind::Storage)?;
    let new_numel: usize = incin_core::shapes::ShapeBuf::from_slice(shape)
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    if numel != new_numel {
        return Err(Error::ShapeMismatch {
            op: "reshape",
            expected: vec![numel],
            got: vec![new_numel],
            msg: "reshape total element count mismatch".to_string(),
        });
    }
    let shape_buf = ShapeBuf::from_slice(shape);
    let meta = TensorMeta::contiguous(
        shape_buf,
        storage.metadata().dtype(),
        storage.device(),
        MetalStorage::alignment(),
        numel,
    )?;
    MetalStorage::from_bytes(
        storage.as_bytes()?.to_vec(),
        meta,
        storage.mode(),
        storage.device_ordinal(),
    )
}

fn transpose_metal(storage: &MetalStorage, dim0: usize, dim1: usize) -> Result<MetalStorage> {
    let dims = storage.metadata().shape().dims();
    if dim0 >= dims.len() || dim1 >= dims.len() {
        return Err(Error::ShapeMismatch {
            op: "transpose",
            expected: vec![dims.len()],
            got: vec![dim0, dim1],
            msg: "transpose dimensions out of bounds".to_string(),
        });
    }
    let mut out_dims = dims.to_vec();
    out_dims.swap(dim0, dim1);

    let numel: usize = incin_core::shapes::ShapeBuf::from_slice(&out_dims)
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    let bytes = storage.as_bytes()?;
    let in_slice: &[f32] = bytemuck::cast_slice(bytes);
    let mut out_data = vec![0.0f32; numel];

    for (idx, output) in out_data.iter_mut().enumerate().take(numel) {
        let mut curr = idx;
        let mut multi = vec![0; out_dims.len()];
        for i in (0..out_dims.len()).rev() {
            multi[i] = curr % out_dims[i];
            curr /= out_dims[i];
        }
        multi.swap(dim0, dim1);
        let mut in_idx = 0;
        let mut stride = 1;
        for i in (0..dims.len()).rev() {
            in_idx += multi[i] * stride;
            stride *= dims[i];
        }
        *output = in_slice[in_idx];
    }

    let shape_buf = ShapeBuf::from_slice(&out_dims);
    let meta = TensorMeta::contiguous(
        shape_buf,
        storage.metadata().dtype(),
        storage.device(),
        MetalStorage::alignment(),
        numel,
    )?;
    let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
    MetalStorage::from_bytes(out_bytes, meta, storage.mode(), storage.device_ordinal())
}

// ───  ─────────────────────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    pub(crate) fn add<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "add", |x, y| x + y)?;
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![
                    unbroadcast(grad_out, &lhs_dims)?,
                    unbroadcast(grad_out, &rhs_dims)?,
                ])
            }),
        });
        Ok(out)
    }

    pub(crate) fn sub<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "sub", |x, y| x - y)?;
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let neg_grad = unary_op_metal(grad_out, |x| -x)?;
                Ok(vec![
                    unbroadcast(grad_out, &lhs_dims)?,
                    unbroadcast(&neg_grad, &rhs_dims)?,
                ])
            }),
        });
        Ok(out)
    }

    pub(crate) fn mul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "mul", |x, y| x * y)?;
        let (lhs_cap, rhs_cap) = (lhs.clone(), rhs.clone());
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let ga = binary_op_metal(grad_out, &rhs_cap, "mul_grad", |x, y| x * y)?;
                let gb = binary_op_metal(grad_out, &lhs_cap, "mul_grad", |x, y| x * y)?;
                Ok(vec![
                    unbroadcast(&ga, &lhs_dims)?,
                    unbroadcast(&gb, &rhs_dims)?,
                ])
            }),
        });
        Ok(out)
    }

    pub(crate) fn div<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "div", |x, y| x / y)?;
        let (lhs_cap, rhs_cap) = (lhs.clone(), rhs.clone());
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let ga = binary_op_metal(grad_out, &rhs_cap, "div_grad_lhs", |x, y| x / y)?;
                let rhs_sq = binary_op_metal(&rhs_cap, &rhs_cap, "div_grad_rhs_sq", |x, y| x * y)?;
                let neg_lhs = unary_op_metal(&lhs_cap, |x| -x)?;
                let num = binary_op_metal(grad_out, &neg_lhs, "div_grad_num", |x, y| x * y)?;
                let gb = binary_op_metal(&num, &rhs_sq, "div_grad_rhs", |x, y| x / y)?;
                Ok(vec![
                    unbroadcast(&ga, &lhs_dims)?,
                    unbroadcast(&gb, &rhs_dims)?,
                ])
            }),
        });
        Ok(out)
    }
}

// ───  ───────────────────────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    crate::unsupported::unsupported_float_ops! {
        unary:
            sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
            atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    pub(crate) fn add_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op_metal(t, scalar, |x, s| x + s)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| Ok(vec![grad_out.clone()])),
        });
        Ok(out)
    }

    pub(crate) fn mul_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op_metal(t, scalar, |x, s| x * s)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![scalar_op_metal(grad_out, scalar, |x, s| x * s)?])
            }),
        });
        Ok(out)
    }
}

// ───  ───────────────────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    pub(crate) fn sum_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = sum_dim_impl(t, dim, false)?;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![unbroadcast(grad_out, &t_dims)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn sum_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = sum_dim_impl(t, dim, true)?;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![unbroadcast(grad_out, &t_dims)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn mean_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = mean_dim_impl(t, dim, false)?;
        let count = t.metadata().shape().dims()[dim] as f64;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let scaled = scalar_op_metal(grad_out, 1.0 / count, |x, s| x * s)?;
                Ok(vec![unbroadcast(&scaled, &t_dims)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn mean_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = mean_dim_impl(t, dim, true)?;
        let count = t.metadata().shape().dims()[dim] as f64;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let scaled = scalar_op_metal(grad_out, 1.0 / count, |x, s| x * s)?;
                Ok(vec![unbroadcast(&scaled, &t_dims)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn sum_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let bytes = t.as_bytes()?;
        let slice: &[f32] = bytemuck::cast_slice(bytes);
        let sum: f32 = slice.iter().sum();
        let shape_buf = ShapeBuf::from_slice(&[]);
        let meta = TensorMeta::contiguous(
            shape_buf,
            t.metadata().dtype(),
            t.device(),
            MetalStorage::alignment(),
            1,
        )?;
        let out = MetalStorage::from_bytes(
            bytemuck::cast_slice(&[sum]).to_vec(),
            meta,
            t.mode(),
            t.device_ordinal(),
        )?;
        let t_dims = dims.to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![unbroadcast(grad_out, &t_dims)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn mean_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let total: usize = incin_core::shapes::ShapeBuf::from_slice(dims)
            .checked_numel(incin_core::shapes::OperationKind::Storage)?;
        let sum = Self::sum_all::<K>(t)?;
        scalar_op_metal(&sum, 1.0 / (total as f64), |x, s| x * s)
    }
}

// ───  ──────────────────────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    pub(crate) fn matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = matmul_metal(lhs, rhs)?;
        let (lhs_cap, rhs_cap) = (lhs.clone(), rhs.clone());
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let rhs_rank = rhs_dims.len();
                let lhs_rank = lhs_dims.len();
                let rhs_t = transpose_metal(&rhs_cap, rhs_rank - 2, rhs_rank - 1)?;
                let lhs_t = transpose_metal(&lhs_cap, lhs_rank - 2, lhs_rank - 1)?;
                let ga = matmul_metal(grad_out, &rhs_t)?;
                let gb = matmul_metal(&lhs_t, grad_out)?;
                Ok(vec![
                    unbroadcast(&ga, &lhs_dims)?,
                    unbroadcast(&gb, &rhs_dims)?,
                ])
            }),
        });
        Ok(out)
    }

    pub(crate) fn reshape<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reshape_metal(t, shape)?;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                reshape_metal(grad_out, &t_dims).map(|g| vec![g])
            }),
        });
        Ok(out)
    }

    pub(crate) fn broadcast_as<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let zeros = MetalStorage::zeros(
            &ShapeBuf::from_slice(shape),
            t.metadata().dtype(),
            t.mode(),
            t.device_ordinal(),
        )?;
        let out = binary_op_metal(t, &zeros, "broadcast_as", |x, _| x)?;
        let t_dims = t.metadata().shape().dims().to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                unbroadcast(grad_out, &t_dims).map(|g| vec![g])
            }),
        });
        Ok(out)
    }
}

// ───  ──────────────────────────────────────────────────────────────

impl<D: Device> MetalBackendImpl<D> {
    pub(crate) fn conv2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _w: &<Self as StorageBackend>::Storage<K>,
        _b: Option<&<Self as StorageBackend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("conv2d"))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "matches the backend operation contract shared by CPU, CUDA, and Metal"
    )]
    pub(crate) fn max_pool2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("max_pool2d"))
    }

    pub(crate) fn avg_pool2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("avg_pool2d"))
    }
}

// ─── Loss helper ────────────────────────────────────────────────────────────

// ─── Quantization helpers ───────────────────────────────────────────────────

impl<D: Device> incin_core::backend_authoring::AutogradBackend for MetalBackendImpl<D> {
    type Grads = MetalGrads;

    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::metal::tape::backward(loss)
    }

    fn backward_with<K: DType>(
        loss: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        crate::metal::tape::backward_with(loss, seed)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id()).cloned())
    }

    fn set_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()> {
        grads.set(t.id(), value);
        Ok(())
    }
}
impl<D: Device> VariableBackend for MetalBackendImpl<D> {
    type Var<K: DType> = MetalVar;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
        Ok(MetalVar { storage: t.clone() })
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }
}
