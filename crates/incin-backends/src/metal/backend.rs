//! Metal compute backend for Incin on Apple Silicon and macOS.

use core::marker::PhantomData;

use incin_core::exec::TensorMeta;
use incin_core::backend_authoring::*;
use incin_core::prelude::*;
use incin_core::shapes::ShapeBuf;

use crate::dtype_policy::{BackendFamily, OperationKind, resolve_dtype_policy};
use crate::metal::storage::{MetalStorage, MetalStorageMode};
use crate::metal::tape::MetalGrads;

/// Metal compute backend implementation for Incin.
#[derive(Clone)]
pub struct MetalBackendImpl<T = f32, D = Metal>(PhantomData<(T, D)>);

impl<T, D> MetalBackendImpl<T, D> {
    /// Constructs a stateless Metal executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, D> Default for MetalBackendImpl<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DType, D: Device> SupportsDType<f32> for MetalBackendImpl<T, D> {
    fn resolve_dtype(field: &<f32 as DType>::Field, _device: &DeviceId) -> Result<DTypeId> {
        Ok(<f32 as DType>::to_incin(field))
    }
}

impl<T: DType, D: Device> SupportsDType<Dyn> for MetalBackendImpl<T, D> {
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
pub struct MetalVar {
    pub storage: MetalStorage,
}

fn validate_metal(
    dtype: DTypeId,
    device: &DeviceId,
    family: OperationKind,
    op: &'static str,
) -> Result<()> {
    if device.kind() != DeviceKind::Metal {
        return Err(Error::DeviceInitializationError {
            expected: "metal".to_string(),
            got: format!("{:?}", device.kind()),
        });
    }
    resolve_dtype_policy(BackendFamily::Wgpu, family, dtype, op).map(|_| ())
}

fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unsupported(op: &'static str) -> Error {
    Error::UnsupportedBackendOperation {
        op,
        backend: "Metal",
    }
}

// ─── Backend ────────────────────────────────────────────────────────────────

impl<T: DType, D: Device> Backend for MetalBackendImpl<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = i64;
    type Storage<K: DType> = MetalStorage;
    type RawVar = MetalVar;
    type Grads = MetalGrads;
    type InnerBackend = Self;

    fn shape<K: DType>(t: &Self::Storage<K>) -> Vec<usize> {
        t.metadata().shape().dims().to_vec()
    }

    fn storage_dtype<K: DType>(t: &Self::Storage<K>) -> Option<DTypeId> {
        Some(t.metadata().dtype())
    }

    fn storage_device<K: DType>(t: &Self::Storage<K>) -> Option<DeviceId> {
        Some(t.device())
    }

    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> String {
        "MetalTensor(...)".to_string()
    }

    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> String {
        format!("MetalTensor(shape={:?})", t.metadata().shape().dims())
    }

    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(MetalVar { storage: t.clone() })
    }

    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }

    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::metal::tape::backward(loss)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id()).cloned())
    }

    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        t.as_bytes().map(<[u8]>::to_vec)
    }

    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Storage, "from_bytes")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let numel = num_elements(shape);
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

// ─── CreationOps ────────────────────────────────────────────────────────────

impl<T: DType, D: Device> CreationOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_creation_ops! {
        fill: full;
        sequence: arange, linspace;
    }

    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "zeros")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        MetalStorage::zeros(
            &shape_buf,
            dtype,
            MetalStorageMode::Shared,
            device.ordinal(),
        )
    }

    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Fill, "ones")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape);
        let data: Vec<f32> = vec![1.0; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Random, "rand")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape);
        let data: Vec<f32> = vec![0.5; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::Storage<K>> {
        validate_metal(dtype, device, OperationKind::Random, "randn")?;
        let shape_buf = incin_core::shapes::ShapeBuf::from_slice(shape);
        let n = num_elements(shape);
        let data: Vec<f32> = vec![0.0; n];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = TensorMeta::contiguous(shape_buf, dtype, *device, MetalStorage::alignment(), n)?;
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, device.ordinal())
    }

    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
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
        let max_rank = a_dims.len().max(b_dims.len());
        let mut out_shape = vec![0; max_rank];
        for i in 0..max_rank {
            let d_a = if i < max_rank - a_dims.len() {
                1
            } else {
                a_dims[i - (max_rank - a_dims.len())]
            };
            let d_b = if i < max_rank - b_dims.len() {
                1
            } else {
                b_dims[i - (max_rank - b_dims.len())]
            };
            if d_a == d_b {
                out_shape[i] = d_a;
            } else if d_a == 1 {
                out_shape[i] = d_b;
            } else if d_b == 1 {
                out_shape[i] = d_a;
            } else {
                return Err(Error::ShapeMismatch {
                    op: op_name,
                    expected: a_dims.to_vec(),
                    got: b_dims.to_vec(),
                    msg: "incompatible shapes for broadcast".to_string(),
                });
            }
        }
        let total: usize = out_shape.iter().product();
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
    let out_numel: usize = out_dims.iter().product();
    let mut out_data = vec![0.0f32; out_numel];

    let bytes = t.as_bytes()?;
    let in_slice: &[f32] = bytemuck::cast_slice(bytes);
    let outer: usize = dims[..axis].iter().product();
    let axis_len = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();

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

    let lhs_batch: usize = lhs_dims[..lhs_dims.len() - 2].iter().product();
    let rhs_batch: usize = rhs_dims[..rhs_dims.len() - 2].iter().product();
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

    let out_numel: usize = out_shape.iter().product();
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
    let new_numel: usize = shape.iter().product();
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

    let numel: usize = out_dims.iter().product();
    let bytes = storage.as_bytes()?;
    let in_slice: &[f32] = bytemuck::cast_slice(bytes);
    let mut out_data = vec![0.0f32; numel];

    for idx in 0..numel {
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
        out_data[idx] = in_slice[in_idx];
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

// ─── NumericOps ─────────────────────────────────────────────────────────────

impl<T: DType, D: Device> NumericOps<Self> for MetalBackendImpl<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

// ─── FloatOps ───────────────────────────────────────────────────────────────

impl<T: DType, D: Device> FloatOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_float_ops! {
        unary:
            sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
            atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = scalar_op_metal(t, scalar, |x, s| x + s)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| Ok(vec![grad_out.clone()])),
        });
        Ok(out)
    }

    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| if x > 0.0 { x } else { 0.0 })?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| if x > 0.0 { 1.0 } else { 0.0 })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "relu_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn step<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| if x > 0.0 { 1.0 } else { 0.0 })?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![scalar_op_metal(grad_out, 0.0, |_, _| 0.0)?])
            }),
        });
        Ok(out)
    }

    fn elu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| if x > 0.0 { x } else { x.exp() - 1.0 })?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| if x > 0.0 { 1.0 } else { x.exp() })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "elu_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| {
            0.5 * x
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        })?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| {
                    let c = (2.0 / std::f32::consts::PI).sqrt();
                    let inner = c * (x + 0.044715 * x.powi(3));
                    let tanh_inner = inner.tanh();
                    let dtanh = 1.0 - tanh_inner * tanh_inner;
                    0.5 * (1.0 + tanh_inner) + 0.5 * x * dtanh * c * (1.0 + 3.0 * 0.044715 * x * x)
                })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "gelu_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn mish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x * (1.0 + x.exp()).ln().tanh())?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| {
                    let sp = (1.0 + x.exp()).ln();
                    let tsp = sp.tanh();
                    let sig = 1.0 / (1.0 + (-x).exp());
                    tsp + x * sig * (1.0 - tsp * tsp)
                })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "mish_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x.tanh())?;
        let out_cap = out.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&out_cap, |y| 1.0 - y * y)?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "tanh_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| 1.0 / (1.0 + (-x).exp()))?;
        let out_cap = out.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&out_cap, |y| y * (1.0 - y))?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "sigmoid_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x.abs())?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "abs_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| -x)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![unary_op_metal(grad_out, |x| -x)?])
            }),
        });
        Ok(out)
    }

    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x.sqrt())?;
        let out_cap = out.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&out_cap, |y| 1.0 / (2.0 * y))?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "sqrt_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x.exp())?;
        let out_cap = out.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![binary_op_metal(
                    grad_out,
                    &out_cap,
                    "exp_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x.ln())?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| 1.0 / x)?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "log_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = unary_op_metal(t, |x| x / (1.0 + (-x).exp()))?;
        let t_cap = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let deriv = unary_op_metal(&t_cap, |x| {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    sig + x * sig * (1.0 - sig)
                })?;
                Ok(vec![binary_op_metal(
                    grad_out,
                    &deriv,
                    "swish_grad",
                    |x, y| x * y,
                )?])
            }),
        });
        Ok(out)
    }

    fn softmax<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if dim >= dims.len() {
            return Err(Error::ShapeMismatch {
                op: "softmax",
                expected: vec![dims.len()],
                got: vec![dim],
                msg: "softmax dim out of bounds".to_string(),
            });
        }
        let exp_t = unary_op_metal(t, |x| x.exp())?;
        let sum_exp = sum_dim_impl(&exp_t, dim, true)?;
        let out = binary_op_metal(&exp_t, &sum_exp, "softmax_div", |x, y| x / y)?;

        let out_cap = out.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let g_times_s = binary_op_metal(grad_out, &out_cap, "sm_grad_prod", |x, y| x * y)?;
                let sum_g_times_s = sum_dim_impl(&g_times_s, dim, true)?;
                let sub = binary_op_metal(grad_out, &sum_g_times_s, "sm_grad_sub", |x, y| x - y)?;
                Ok(vec![binary_op_metal(&out_cap, &sub, "sm_grad", |x, y| {
                    x * y
                })?])
            }),
        });
        Ok(out)
    }
}

// ─── ReductionOps ───────────────────────────────────────────────────────────

impl<T: DType, D: Device> ReductionOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_reduction_ops! {
        all: max_all, min_all, prod_all;
        dim:
            max_dim, min_dim,
            max_keepdim, min_keepdim,
            prod_dim, cumsum;
    }

    fn sum_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn sum_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn mean_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn mean_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let total: usize = dims.iter().product();
        let sum = Self::sum_all::<K>(t)?;
        scalar_op_metal(&sum, 1.0 / (total as f64), |x, s| x * s)
    }

    fn argmax<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        Err(unsupported("argmax"))
    }

    fn argmin<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        Err(unsupported("argmin"))
    }

    fn topk<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _k: usize,
        _dim: usize,
        _largest: bool,
    ) -> Result<(
        <Self as Backend>::Storage<K>,
        <Self as Backend>::Storage<KInt>,
    )> {
        Err(unsupported("topk"))
    }

    fn argsort<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
        _descending: bool,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        Err(unsupported("argsort"))
    }
}

// ─── TensorOps ──────────────────────────────────────────────────────────────

impl<T: DType, D: Device> TensorOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_tensor_ops! {
        where_cond, gather, scatter, index_select, masked_fill, unsqueeze,
        repeat, pad, triu, tril, diag,
        cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge,
        logical_and, logical_or, logical_not,
        sub_scalar, div_scalar, maximum, minimum, abs_diff, lerp,
        addmm, bmm, scaled_dot_product_attention,
        unfold, pixel_shuffle, group_norm, instance_norm,
        float_to_scalar, float_to_vec1, int_to_scalar, int_to_vec1,
        tensor_to_dtype,
    }

    fn matmul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim0: usize,
        dim1: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = transpose_metal(t, dim0, dim1)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                transpose_metal(grad_out, dim0, dim1).map(|g| vec![g])
            }),
        });
        Ok(out)
    }

    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
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

    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Self::broadcast_as::<K>(t, shape)
    }

    fn narrow<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if dim >= dims.len() || start + len > dims[dim] {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: vec![dims[dim]],
                got: vec![start + len],
                msg: "narrow bounds out of range".to_string(),
            });
        }
        let mut out_dims = dims.to_vec();
        out_dims[dim] = len;
        let out_numel: usize = out_dims.iter().product();
        let mut out_data = Vec::with_capacity(out_numel);

        let bytes = t.as_bytes()?;
        let in_slice: &[f32] = bytemuck::cast_slice(bytes);

        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();

        for o in 0..outer {
            for i in start..start + len {
                let start_idx = o * dims[dim] * inner + i * inner;
                out_data.extend_from_slice(&in_slice[start_idx..start_idx + inner]);
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
        let out = MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())?;
        let (t_id, out_id) = (t.id(), out.id());
        let t_dims = dims.to_vec();
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let full_grad = MetalStorage::zeros(
                    &ShapeBuf::from_slice(&t_dims),
                    grad_out.metadata().dtype(),
                    grad_out.mode(),
                    grad_out.device_ordinal(),
                )?;
                let g_bytes = grad_out.as_bytes()?;
                let g_slice: &[f32] = bytemuck::cast_slice(g_bytes);
                let fg_bytes = full_grad.as_bytes()?;
                let mut fg_data: Vec<f32> = bytemuck::cast_slice(fg_bytes).to_vec();
                let g_outer: usize = t_dims[..dim].iter().product();
                let g_inner: usize = t_dims[dim + 1..].iter().product();
                for o in 0..g_outer {
                    for i in 0..len {
                        let src_start = o * len * g_inner + i * g_inner;
                        let dst_start = o * t_dims[dim] * g_inner + (start + i) * g_inner;
                        fg_data[dst_start..dst_start + g_inner]
                            .copy_from_slice(&g_slice[src_start..src_start + g_inner]);
                    }
                }
                let meta = TensorMeta::contiguous(
                    ShapeBuf::from_slice(&t_dims),
                    grad_out.metadata().dtype(),
                    grad_out.device(),
                    MetalStorage::alignment(),
                    fg_data.len(),
                )?;
                MetalStorage::from_bytes(
                    bytemuck::cast_slice(&fg_data).to_vec(),
                    meta,
                    grad_out.mode(),
                    grad_out.device_ordinal(),
                )
                .map(|g| vec![g])
            }),
        });
        Ok(out)
    }

    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if start_dim > end_dim || end_dim >= dims.len() {
            return Err(Error::ShapeMismatch {
                op: "flatten",
                expected: vec![dims.len()],
                got: vec![start_dim, end_dim],
                msg: "invalid flatten dimensions".to_string(),
            });
        }
        let mut new_dims = Vec::new();
        new_dims.extend_from_slice(&dims[..start_dim]);
        let folded: usize = dims[start_dim..=end_dim].iter().product();
        new_dims.push(folded);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        Self::reshape::<K>(t, &new_dims)
    }

    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if dim >= dims.len() || dims[dim] != 1 {
            return Err(Error::ShapeMismatch {
                op: "squeeze",
                expected: vec![dims.len()],
                got: vec![dim],
                msg: "cannot squeeze non-1 dimension".to_string(),
            });
        }
        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);
        Self::reshape::<K>(t, &new_dims)
    }

    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::UnsupportedBackendOperation {
                op: "concat",
                backend: "Metal (empty tensors slice)",
            });
        }
        let first_dims = tensors[0].metadata().shape().dims();
        if dim >= first_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: vec![first_dims.len()],
                got: vec![dim],
                msg: "concat dimension out of bounds".to_string(),
            });
        }
        let mut out_dims = first_dims.to_vec();
        let mut total_concat_len = 0;
        for t in tensors {
            let d = t.metadata().shape().dims();
            if d.len() != first_dims.len() {
                return Err(Error::ShapeMismatch {
                    op: "concat",
                    expected: vec![first_dims.len()],
                    got: vec![d.len()],
                    msg: "concat input ranks must match".to_string(),
                });
            }
            for (i, (&d1, &d2)) in first_dims.iter().zip(d.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    return Err(Error::ShapeMismatch {
                        op: "concat",
                        expected: vec![d1],
                        got: vec![d2],
                        msg: "concat non-concat dimensions must match".to_string(),
                    });
                }
            }
            total_concat_len += d[dim];
        }
        out_dims[dim] = total_concat_len;

        let outer: usize = first_dims[..dim].iter().product();
        let inner: usize = first_dims[dim + 1..].iter().product();
        let out_numel: usize = out_dims.iter().product();
        let mut out_data = Vec::with_capacity(out_numel);

        for o in 0..outer {
            for t in tensors {
                let t_bytes = t.as_bytes()?;
                let t_slice: &[f32] = bytemuck::cast_slice(t_bytes);
                let t_len = t.metadata().shape().dims()[dim];
                let start = o * t_len * inner;
                out_data.extend_from_slice(&t_slice[start..start + t_len * inner]);
            }
        }

        let shape_buf = ShapeBuf::from_slice(&out_dims);
        let meta = TensorMeta::contiguous(
            shape_buf,
            tensors[0].metadata().dtype(),
            tensors[0].device(),
            MetalStorage::alignment(),
            out_numel,
        )?;
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        let out = MetalStorage::from_bytes(
            out_bytes,
            meta,
            tensors[0].mode(),
            tensors[0].device_ordinal(),
        )?;

        let input_ids: Vec<_> = tensors.iter().map(|t| t.id()).collect();
        let slice_lens: Vec<_> = tensors
            .iter()
            .map(|t| t.metadata().shape().dims()[dim])
            .collect();
        let out_id = out.id();
        let first_mode = tensors[0].mode();
        let first_ord = tensors[0].device_ordinal();

        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids,
            backward: Box::new(move |grad_out: &MetalStorage| {
                let mut grads = Vec::with_capacity(slice_lens.len());
                let mut start = 0;
                for &len in &slice_lens {
                    let mut sub_dims = out_dims.clone();
                    sub_dims[dim] = len;
                    let sub_numel: usize = sub_dims.iter().product();
                    let mut sub_data = Vec::with_capacity(sub_numel);

                    let g_bytes = grad_out.as_bytes()?;
                    let g_slice: &[f32] = bytemuck::cast_slice(g_bytes);

                    for o in 0..outer {
                        let src_offset = o * out_dims[dim] * inner + start * inner;
                        sub_data.extend_from_slice(&g_slice[src_offset..src_offset + len * inner]);
                    }

                    let shape_buf = ShapeBuf::from_slice(&sub_dims);
                    let meta = TensorMeta::contiguous(
                        shape_buf,
                        grad_out.metadata().dtype(),
                        grad_out.device(),
                        MetalStorage::alignment(),
                        sub_numel,
                    )?;
                    let sub_bytes: Vec<u8> = bytemuck::cast_slice(&sub_data).to_vec();
                    grads.push(MetalStorage::from_bytes(
                        sub_bytes, meta, first_mode, first_ord,
                    )?);
                    start += len;
                }
                Ok(grads)
            }),
        });
        Ok(out)
    }

    fn stack<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::UnsupportedBackendOperation {
                op: "stack",
                backend: "Metal (empty tensors slice)",
            });
        }
        let mut unsqueezed = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mut u_dims = t.metadata().shape().dims().to_vec();
            u_dims.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &u_dims)?);
        }
        let refs: Vec<&<Self as Backend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    fn slice<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut curr = t.clone();
        for (dim, &(start, len)) in ranges.iter().enumerate() {
            curr = Self::narrow::<K>(&curr, dim, start, len)?;
        }
        Ok(curr)
    }
}

// ─── ModuleOps ──────────────────────────────────────────────────────────────

impl<T: DType, D: Device> ModuleOps<Self> for MetalBackendImpl<T, D> {
    fn layer_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let last_dim = dims[dims.len() - 1];
        let outer: usize = dims[..dims.len() - 1].iter().product();

        let bytes = t.as_bytes()?;
        let in_slice: &[f32] = bytemuck::cast_slice(bytes);
        let w_bytes = weight.as_bytes()?;
        let w_slice: &[f32] = bytemuck::cast_slice(w_bytes);

        let b_slice: Option<&[f32]> = match bias {
            Some(b) => Some(bytemuck::cast_slice(b.as_bytes()?)),
            None => None,
        };

        let mut out_data = vec![0.0f32; in_slice.len()];

        for o in 0..outer {
            let offset = o * last_dim;
            let slice = &in_slice[offset..offset + last_dim];
            let mean: f32 = slice.iter().sum::<f32>() / (last_dim as f32);
            let var: f32 =
                slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / (last_dim as f32);
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..last_dim {
                let norm = (slice[i] - mean) * inv_std;
                let b_val = b_slice.map(|b| b[i]).unwrap_or(0.0);
                out_data[offset + i] = norm * w_slice[i] + b_val;
            }
        }

        let shape_buf = ShapeBuf::from_slice(dims);
        let meta = TensorMeta::contiguous(
            shape_buf,
            t.metadata().dtype(),
            t.device(),
            MetalStorage::alignment(),
            in_slice.len(),
        )?;
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())
    }

    fn batch_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: Option<&<Self as Backend>::Storage<K>>,
        b: Option<&<Self as Backend>::Storage<K>>,
        _rm: Option<&<Self as Backend>::Storage<K>>,
        _rv: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if dims.len() < 2 {
            return Err(Error::ShapeMismatch {
                op: "batch_norm",
                expected: vec![2],
                got: vec![dims.len()],
                msg: "batch_norm requires at least 2D input".to_string(),
            });
        }
        let batch = dims[0];
        let channels = dims[1];
        let spatial: usize = dims[2..].iter().product();

        let bytes = t.as_bytes()?;
        let in_slice: &[f32] = bytemuck::cast_slice(bytes);

        let w_slice: Option<&[f32]> = match w {
            Some(w_st) => Some(bytemuck::cast_slice(w_st.as_bytes()?)),
            None => None,
        };
        let b_slice: Option<&[f32]> = match b {
            Some(b_st) => Some(bytemuck::cast_slice(b_st.as_bytes()?)),
            None => None,
        };

        let mut out_data = vec![0.0f32; in_slice.len()];
        let total_per_channel = batch * spatial;

        for c in 0..channels {
            let mut sum = 0.0f32;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    sum += in_slice[idx];
                }
            }
            let mean = sum / (total_per_channel as f32);

            let mut var_sum = 0.0f32;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    let diff = in_slice[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / (total_per_channel as f32);
            let inv_std = 1.0 / (var + eps).sqrt();

            let scale = w_slice.map(|ws| ws[c]).unwrap_or(1.0);
            let shift = b_slice.map(|bs| bs[c]).unwrap_or(0.0);

            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    out_data[idx] = (in_slice[idx] - mean) * inv_std * scale + shift;
                }
            }
        }

        let shape_buf = ShapeBuf::from_slice(dims);
        let meta = TensorMeta::contiguous(
            shape_buf,
            t.metadata().dtype(),
            t.device(),
            MetalStorage::alignment(),
            in_slice.len(),
        )?;
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())
    }

    fn embedding<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<KInt>,
        _w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("embedding"))
    }

    fn conv1d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _w: &<Self as Backend>::Storage<K>,
        _b: Option<&<Self as Backend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("conv1d"))
    }

    fn conv2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _w: &<Self as Backend>::Storage<K>,
        _b: Option<&<Self as Backend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("conv2d"))
    }

    fn conv_transpose2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _w: &<Self as Backend>::Storage<K>,
        _b: Option<&<Self as Backend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _output_padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("conv_transpose2d"))
    }

    fn max_pool2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("max_pool2d"))
    }

    fn avg_pool2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("avg_pool2d"))
    }

    fn adaptive_avg_pool2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("adaptive_avg_pool2d"))
    }
}

// ─── LossOps ────────────────────────────────────────────────────────────────

impl<T: DType, D: Device> LossOps<Self> for MetalBackendImpl<T, D> {
    fn cross_entropy_loss<K: DType, KInt: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<KInt>,
        _reduction: incin_core::nn::loss::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("cross_entropy_loss"))
    }
}

// ─── QuantizedOps ───────────────────────────────────────────────────────────

impl<T: DType, D: Device> QuantizedOps<Self> for MetalBackendImpl<T, D> {
    fn quantize<K: FloatDType, Q: QuantDType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<Q>> {
        Err(unsupported("quantize"))
    }

    fn dequantize<Q: QuantDType, K: FloatDType>(
        _t: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("dequantize"))
    }

    fn quantized_matmul<Q: QuantDType>(
        _lhs: &<Self as Backend>::Storage<Q>,
        _rhs: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<f32>> {
        Err(unsupported("quantized_matmul"))
    }
}

// ─── OptimizerOps ───────────────────────────────────────────────────────────
// Uses default adamw_step composed from NumericOps/FloatOps (via trait default).
impl<T: DType, D: Device> OptimizerOps<Self> for MetalBackendImpl<T, D> {}
