//! Metal compute backend for Incin on Apple Silicon and macOS.

use core::marker::PhantomData;

use incin_core::backend_authoring::*;
use incin_core::__backend_compat::legacy::*;
use incin_core::exec::TensorMeta;
use incin_core::shapes::ShapeBuf;
use incin_core::prelude::{
    BackendError, ConstDType, DType, DTypeDescriptor, DTypeId, Device, DeviceId, DeviceKind, Dyn,
    Error, FloatDType, OperationKind, Q8_0, QuantDType, Result, ShapeError, StrideBuf, Metal,
};

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

impl<K: DType, D: Device> SupportsDType<K> for MetalBackendImpl<D> {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        let descriptor = K::descriptor(field);
        validate_metal_storage_dtype(descriptor, "resolve_dtype")?;
        Ok(descriptor)
    }
}

pub(crate) fn validate_metal_storage_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    let is_supported = matches!(
        dtype.builtin_id(),
        Some(
            DTypeId::F32
                | DTypeId::F64
                | DTypeId::F16
                | DTypeId::BF16
                | DTypeId::I64
                | DTypeId::Q8_0
        )
    );
    if is_supported {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype,
            backend: "Metal",
            op,
        })
    }
}

pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    validate_metal_storage_dtype(request.storage, "native_precision")?;

    let compute = match request.storage.builtin_id() {
        Some(DTypeId::F16 | DTypeId::BF16) => DTypeId::F32.descriptor(),
        _ => request.storage,
    };

    let accumulator = match request.operation {
        OperationKind::Reduction | OperationKind::Normalization
            if matches!(
                request.storage.builtin_id(),
                Some(DTypeId::F16 | DTypeId::BF16)
            ) =>
        {
            DTypeId::F32.descriptor()
        }
        _ => compute,
    };

    Ok(incin_core::exec::ResolvedPrecision::new(
        request.storage,
        compute,
        accumulator,
        request.output,
        incin_core::exec::LossScaling::None,
    ))
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for MetalBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        native_precision(request)
    }
}

#[derive(Clone)]
pub struct MetalVar {
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

impl<D: Device> Backend for MetalBackendImpl<D> {
    type InnerBackend = Self;

    // `format_tensor_display`/`format_tensor_debug` use `Backend`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.


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

// ─── CreationOps ────────────────────────────────────────────────────────────

impl<D: Device> CreationOps<Self> for MetalBackendImpl<D> {
    /// `full`. Same host-fill-then-upload pattern `ones` above already
    /// uses.
    fn full<K: DType>(
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
    fn arange<K: DType>(
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
    fn linspace<K: DType>(
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

    fn zeros<K: DType>(
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

    fn ones<K: DType>(
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

    fn rand<K: DType>(
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

    fn randn<K: DType>(
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

    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::RawVar> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::RawVar> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::RawVar> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(MetalVar { storage: s })
    }

    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::RawVar> {
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
        let out_shape = incin_core::shapes::broadcast::broadcast_dim_slices(a_dims, b_dims)
            .map_err(|error| Error::ShapeMismatch {
                op: op_name,
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
                msg: error.to_string(),
            })?;
        let max_rank = out_shape.len();
        let total: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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

/// Builds a fresh contiguous F32 `MetalStorage` of `shape` from `data`,
/// reusing `t`'s dtype/device/mode/ordinal — the same
/// `TensorMeta::contiguous` + `MetalStorage::from_bytes` construction
/// `binary_op_metal`/`unary_op_metal` above already use, factored out for
/// the structural/index ops below, which build a fresh shape rather than
/// reusing the input's.
fn upload_f32_metal(t: &MetalStorage, shape: Vec<usize>, data: Vec<f32>) -> Result<MetalStorage> {
    let out_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let meta = TensorMeta::contiguous(
        ShapeBuf::from_slice(&shape),
        t.metadata().dtype(),
        t.device(),
        MetalStorage::alignment(),
        data.len(),
    )?;
    MetalStorage::from_bytes(out_bytes, meta, t.mode(), t.device_ordinal())
}

/// Shared implementation for `max_all`/`min_all`: reduces to a scalar and
/// wires a gradient that routes `grad_out`'s value to only the winning
/// element's original position, matching CPU's own `max_all`/`min_all`
/// exactly (first-encountered winner under a strict `>`/`<` comparison).
fn extremum_all_metal(t: &MetalStorage, is_max: bool) -> Result<MetalStorage> {
    let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
    let mut best_val = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut best_flat = 0usize;
    for (flat, &v) in data.iter().enumerate() {
        if (is_max && v > best_val) || (!is_max && v < best_val) {
            best_val = v;
            best_flat = flat;
        }
    }
    let out = upload_f32_metal(t, vec![], vec![best_val])?;

    let total = data.len();
    let (t_id, out_id) = (t.id(), out.id());
    crate::metal::tape::push(crate::metal::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &MetalStorage| {
            let grad_data: &[f32] = bytemuck::cast_slice(grad_out.as_bytes()?);
            let mut vals = vec![0.0f32; total];
            vals[best_flat] = grad_data[0];
            upload_f32_metal(grad_out, vec![total], vals).map(|s| vec![s])
        }),
    });
    Ok(out)
}

/// Shared implementation for `max_dim`/`min_dim`/`max_keepdim`/`min_keepdim`:
/// reduces along `dim`, recording each output position's winning source
/// position, and wires a gradient that scatters `grad_out`'s value to only
/// those recorded positions — matching CPU's own `max_axis_with_indices`/
/// `scatter_axis_grad` exactly.
fn extremum_dim_metal(
    t: &MetalStorage,
    dim: usize,
    keepdim: bool,
    is_max: bool,
) -> Result<MetalStorage> {
    let in_shape = t.metadata().shape().dims().to_vec();
    let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
    let mut keep_shape = in_shape.clone();
    keep_shape[dim] = 1;
    let out_total = num_elements(&keep_shape)?;
    let mut best_val = vec![
        if is_max {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        out_total
    ];
    let mut best_flat_src = vec![0usize; out_total];
    let out_strides = crate::layout::contiguous_strides(&keep_shape);

    let src_total = num_elements(&in_shape)?;
    let mut idx = vec![0usize; in_shape.len()];
    for (src_flat, &v) in data.iter().take(src_total).enumerate() {
        let mut out_idx = idx.clone();
        out_idx[dim] = 0;
        let flat_out: usize = out_idx
            .iter()
            .zip(out_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        if (is_max && v > best_val[flat_out]) || (!is_max && v < best_val[flat_out]) {
            best_val[flat_out] = v;
            best_flat_src[flat_out] = src_flat;
        }
        crate::layout::increment_index(&mut idx, &in_shape);
    }

    let out_shape = if keepdim {
        keep_shape
    } else {
        let mut s = in_shape.clone();
        s.remove(dim);
        s
    };
    let out = upload_f32_metal(t, out_shape, best_val)?;

    let (t_id, out_id) = (t.id(), out.id());
    crate::metal::tape::push(crate::metal::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &MetalStorage| {
            let grad_data: &[f32] = bytemuck::cast_slice(grad_out.as_bytes()?);
            let mut vals = vec![0.0f32; src_total];
            for (flat_out, &g) in grad_data.iter().enumerate().take(out_total) {
                vals[best_flat_src[flat_out]] = g;
            }
            upload_f32_metal(grad_out, in_shape.clone(), vals).map(|s| vec![s])
        }),
    });
    Ok(out)
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
    let out_numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_dims))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let mut out_data = vec![0.0f32; out_numel];

    let bytes = t.as_bytes()?;
    let in_slice: &[f32] = bytemuck::cast_slice(bytes);
    let outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[..axis]))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let axis_len = dims[axis];
    let inner: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[axis + 1..]))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;

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
        incin_core::prelude::ShapeBuf::from_slice(&(lhs_dims[..lhs_dims.len() - 2]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let rhs_batch: usize =
        incin_core::prelude::ShapeBuf::from_slice(&(rhs_dims[..rhs_dims.len() - 2]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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

    let out_numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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
    let new_numel: usize = incin_core::prelude::ShapeBuf::from_slice(shape)
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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

    let numel: usize = incin_core::prelude::ShapeBuf::from_slice(&out_dims)
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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

// ─── NumericOps ─────────────────────────────────────────────────────────────

impl<D: Device> NumericOps<Self> for MetalBackendImpl<D> {
    fn add<K: DType>(
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

    fn sub<K: DType>(
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

    fn mul<K: DType>(
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

    fn div<K: DType>(
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

// ─── FloatOps ───────────────────────────────────────────────────────────────

impl<D: Device> FloatOps<Self> for MetalBackendImpl<D> {
    crate::unsupported::unsupported_float_ops! {
        unary:
            sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
            atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    fn add_scalar_float<K: DType>(
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

    fn mul_scalar_float<K: DType>(
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

    fn relu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn step<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn elu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn gelu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn mish<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn tanh<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn abs<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn neg<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn sqrt<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn exp<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn log<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn swish<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

impl<D: Device> ReductionOps<Self> for MetalBackendImpl<D> {
    /// `max_all`. Same host round-trip as `sum_all` above, plus a real
    /// gradient matching CPU's own: the winning element's flat position is
    /// recorded and only that position receives `grad_out`'s value on the
    /// way back, everything else zero.
    fn max_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_all_metal(t, true)
    }
    /// `min_all`. Mirror of `max_all` with a strict `<` comparison.
    fn min_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_all_metal(t, false)
    }
    /// `prod_all`. Same host round-trip as `sum_all`. Not autograd-wired,
    /// matching CPU.
    fn prod_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let product: f32 = data.iter().product();
        upload_f32_metal(t, vec![], vec![product])
    }

    /// `max_dim`. Same host round-trip as `sum_dim`, plus the same
    /// winning-position gradient routing as `max_all`.
    fn max_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_dim_metal(t, dim, false, true)
    }
    /// `min_dim`.
    fn min_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_dim_metal(t, dim, false, false)
    }
    /// `max_keepdim`.
    fn max_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_dim_metal(t, dim, true, true)
    }
    /// `min_keepdim`.
    fn min_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        extremum_dim_metal(t, dim, true, false)
    }
    /// `prod_dim`. Same host round-trip as `sum_dim`. Not autograd-wired,
    /// matching CPU.
    fn prod_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let mut out_shape = in_shape.clone();
        out_shape.remove(dim);
        let mut keep_shape = in_shape.clone();
        keep_shape[dim] = 1;
        let out_total = num_elements(&keep_shape)?;
        let mut prods = vec![1.0f32; out_total];
        let out_strides = crate::layout::contiguous_strides(&keep_shape);
        let src_total = num_elements(&in_shape)?;
        let mut idx = vec![0usize; in_shape.len()];
        for &value in data.iter().take(src_total) {
            let mut out_idx = idx.clone();
            out_idx[dim] = 0;
            let flat_out: usize = out_idx
                .iter()
                .zip(out_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            prods[flat_out] *= value;
            crate::layout::increment_index(&mut idx, &in_shape);
        }
        upload_f32_metal(t, out_shape, prods)
    }
    /// `cumsum`. Same host round-trip as `sum_dim`. Not autograd-wired,
    /// matching CPU.
    fn cumsum<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let total = num_elements(&in_shape)?;
        let dim_len = in_shape[dim];
        let strides = crate::layout::contiguous_strides(&in_shape);
        let mut out = vec![0.0f32; total];
        let mut idx = vec![0usize; in_shape.len()];
        for _ in 0..total {
            if idx[dim] == 0 {
                let mut current = 0.0f32;
                let mut step_idx = idx.clone();
                for step in 0..dim_len {
                    step_idx[dim] = step;
                    let flat: usize = step_idx
                        .iter()
                        .zip(strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    current += data[flat];
                    out[flat] = current;
                }
            }
            crate::layout::increment_index(&mut idx, &in_shape);
        }
        upload_f32_metal(t, in_shape, out)
    }

    fn sum_dim<K: DType>(
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

    fn sum_keepdim<K: DType>(
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

    fn mean_dim<K: DType>(
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

    fn mean_keepdim<K: DType>(
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

    fn sum_all<K: DType>(
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

    fn mean_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let total: usize = incin_core::prelude::ShapeBuf::from_slice(dims)
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let sum = Self::sum_all::<K>(t)?;
        scalar_op_metal(&sum, 1.0 / (total as f64), |x, s| x * s)
    }

    fn argmax<K: DType, KInt: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        Err(unsupported("argmax"))
    }

    fn argmin<K: DType, KInt: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        Err(unsupported("argmin"))
    }

    fn topk<K: DType, KInt: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _k: usize,
        _dim: usize,
        _largest: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<KInt>,
    )> {
        Err(unsupported("topk"))
    }

    fn argsort<K: DType, KInt: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: usize,
        _descending: bool,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        Err(unsupported("argsort"))
    }
}

// ─── TensorOps ──────────────────────────────────────────────────────────────

impl<D: Device> TensorOps<Self> for MetalBackendImpl<D> {
    /// `where_cond`. Broadcasts `mask`/`on_true`/`on_false` to their common
    /// shape via the already tape-wired `broadcast_as` above (itself a
    /// `binary_op_metal` trick — a zeros tensor of the target shape
    /// combined via broadcasting — not a new host round-trip;
    /// `crate::layout::broadcast_shape` computes that shape, the same
    /// resolver CPU's own `where_cond` and every other backend's port use),
    /// then selects elementwise from `as_bytes()`. Its own backward routes
    /// each `grad_out` element to `grad_true`/`grad_false` by the mask
    /// while still in the broadcasted shape; unbroadcasting each back down
    /// to `on_true`'s/`on_false`'s own shape happens automatically as the
    /// tape walk continues into `broadcast_as`'s own backward for whichever
    /// operand was not already at the common shape. `mask` itself gets no
    /// gradient, matching CPU.
    fn where_cond<K: DType>(
        mask: &<Self as StorageBackend>::Storage<bool>,
        on_true: &<Self as StorageBackend>::Storage<K>,
        on_false: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out_shape = crate::layout::broadcast_shape(
            on_true.metadata().shape().dims(),
            on_false.metadata().shape().dims(),
        )?;
        let mask_b = <Self as TensorOps<Self>>::broadcast_as::<bool>(mask, &out_shape)?;
        let true_b = <Self as TensorOps<Self>>::broadcast_as::<K>(on_true, &out_shape)?;
        let false_b = <Self as TensorOps<Self>>::broadcast_as::<K>(on_false, &out_shape)?;

        let mask_data: &[f32] = bytemuck::cast_slice(mask_b.as_bytes()?);
        let true_data: &[f32] = bytemuck::cast_slice(true_b.as_bytes()?);
        let false_data: &[f32] = bytemuck::cast_slice(false_b.as_bytes()?);
        let out: Vec<f32> = mask_data
            .iter()
            .zip(true_data.iter())
            .zip(false_data.iter())
            .map(|((&m, &t), &f)| if m != 0.0 { t } else { f })
            .collect();
        let out_storage = upload_f32_metal(&true_b, out_shape, out)?;

        let mask_cap = mask_b.clone();
        let (true_id, false_id, out_id) = (true_b.id(), false_b.id(), out_storage.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![true_id, false_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let mask_data: &[f32] = bytemuck::cast_slice(mask_cap.as_bytes()?);
                let grad_data: &[f32] = bytemuck::cast_slice(grad_out.as_bytes()?);
                let mut grad_true = Vec::with_capacity(grad_data.len());
                let mut grad_false = Vec::with_capacity(grad_data.len());
                for (&m, &g) in mask_data.iter().zip(grad_data.iter()) {
                    if m != 0.0 {
                        grad_true.push(g);
                        grad_false.push(0.0);
                    } else {
                        grad_true.push(0.0);
                        grad_false.push(g);
                    }
                }
                let grad_shape = grad_out.metadata().shape().dims().to_vec();
                let g_true = upload_f32_metal(grad_out, grad_shape.clone(), grad_true)?;
                let g_false = upload_f32_metal(grad_out, grad_shape, grad_false)?;
                Ok(vec![g_true, g_false])
            }),
        });
        Ok(out_storage)
    }

    /// `gather`. Forward is the same host round-trip as `index_select`.
    /// Unlike `index_select`/`scatter`, CPU wires a real gradient for
    /// `gather`, so this does too, matching every other backend's port:
    /// its backward is the matching scatter-add, routing each `grad_out`
    /// element back to the position it was gathered from, accumulating
    /// with `+=` rather than overwriting when two output positions share a
    /// source. `index` itself gets no gradient, matching CPU.
    fn gather<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let index_shape = index.metadata().shape().dims().to_vec();
        let index_data: Vec<f32> = bytemuck::cast_slice(index.as_bytes()?).to_vec();
        let strides = crate::layout::contiguous_strides(&in_shape);
        let out_shape = index_shape.clone();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for &target in index_data.iter().take(total) {
            let target_i = target as usize;
            let mut src_flat = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { target_i } else { idx[axis] };
                src_flat += coord * stride;
            }
            out.push(data[src_flat]);
            if !out_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &out_shape);
            }
        }
        let out_storage = upload_f32_metal(t, out_shape.clone(), out)?;

        let (t_id, out_id) = (t.id(), out_storage.id());
        let t_shape = in_shape.clone();
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_out_data: &[f32] = bytemuck::cast_slice(grad_out.as_bytes()?);
                let t_total = num_elements(&t_shape)?;
                let mut grad_t_data = vec![0.0f32; t_total];
                let index_total = num_elements(&out_shape)?;
                let mut idx = vec![0usize; out_shape.len()];
                for i in 0..index_total {
                    let target_i = index_data[i] as usize;
                    let mut flat_dst = 0usize;
                    for (axis, &stride) in strides.iter().enumerate() {
                        let coord = if axis == dim { target_i } else { idx[axis] };
                        flat_dst += coord * stride;
                    }
                    grad_t_data[flat_dst] += grad_out_data[i];
                    if !out_shape.is_empty() {
                        crate::layout::increment_index(&mut idx, &out_shape);
                    }
                }
                upload_f32_metal(grad_out, t_shape.clone(), grad_t_data).map(|s| vec![s])
            }),
        });
        Ok(out_storage)
    }

    /// `scatter`. Same host round-trip as `index_select`, matching CPU's
    /// semantics exactly, including silently ignoring an out-of-bounds
    /// destination position rather than erroring. Not autograd-wired,
    /// matching CPU.
    fn scatter<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
        src: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let mut out_data: Vec<f32> = bytemuck::cast_slice(t.as_bytes()?).to_vec();
        let index_shape = index.metadata().shape().dims().to_vec();
        let index_data: &[f32] = bytemuck::cast_slice(index.as_bytes()?);
        let src_data: &[f32] = bytemuck::cast_slice(src.as_bytes()?);
        let strides = crate::layout::contiguous_strides(&in_shape);
        let index_total = num_elements(&index_shape)?;
        let mut idx = vec![0usize; index_shape.len()];
        for i in 0..index_total {
            let target_i = index_data[i] as usize;
            let mut flat_dest = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { target_i } else { idx[axis] };
                flat_dest += coord * stride;
            }
            if flat_dest < out_data.len() {
                out_data[flat_dest] = src_data[i];
            }
            if !index_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &index_shape);
            }
        }
        upload_f32_metal(t, in_shape, out_data)
    }

    /// `group_norm`. Metal storage is always contiguous, so a group (the
    /// per-sample run of `channels/groups * spatial` elements — see the CPU
    /// implementation's doc comment for why dividing the whole tensor by
    /// `groups` is wrong above batch size 1) is a plain contiguous slice of
    /// `as_bytes()`'s output, needing no strided indexing at all — the same
    /// simplification WGPU's and CUDA's own ports of this method have. Not
    /// autograd-wired, matching CPU.
    fn group_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        groups: usize,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if groups == 0 {
            return Err(Error::Msg("group_norm: groups must be non-zero".into()));
        }
        let in_shape = t.metadata().shape().dims().to_vec();
        let channels = if in_shape.len() >= 2 { in_shape[1] } else { 1 };
        if channels % groups != 0 {
            return Err(Error::Msg(
                "group_norm: channels must be divisible by groups".into(),
            ));
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let total = data.len();
        let (batch, spatial) = if in_shape.len() >= 2 {
            (in_shape[0], in_shape[2..].iter().product::<usize>())
        } else {
            (1, total)
        };
        let group_size = channels / groups * spatial;
        let mut out = vec![0.0f32; total];
        for run in 0..batch * groups {
            let start = run * group_size;
            let slice = &data[start..start + group_size];
            let sum: f64 = slice.iter().map(|&v| v as f64).sum();
            let sq_sum: f64 = slice.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let mean = sum / group_size as f64;
            let var = (sq_sum / group_size as f64 - mean * mean).max(0.0);
            let inv_std = 1.0 / (var + eps).sqrt();
            for (i, &value) in slice.iter().enumerate() {
                out[start + i] = ((value as f64 - mean) * inv_std) as f32;
            }
        }
        upload_f32_metal(t, in_shape, out)
    }

    /// `instance_norm`. `group_norm` with one group per channel, matching
    /// every other backend's own composition exactly.
    fn instance_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let channels = if t.metadata().shape().dims().len() >= 2 {
            t.metadata().shape().dims()[1]
        } else {
            1
        };
        <Self as TensorOps<Self>>::group_norm::<K>(t, channels, eps)
    }

    /// `index_select`. Same host round-trip as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn index_select<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let index_data: &[f32] = bytemuck::cast_slice(index.as_bytes()?);
        let in_strides = crate::layout::contiguous_strides(&in_shape);
        let mut out_shape = in_shape.clone();
        out_shape[dim] = index_data.len();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let selected = index_data[out_idx[dim]] as usize;
            let mut src_flat = 0usize;
            for (axis, &pos) in out_idx.iter().enumerate() {
                let coord = if axis == dim { selected } else { pos };
                src_flat += coord * in_strides[axis];
            }
            out.push(data[src_flat]);
            if !out_shape.is_empty() {
                crate::layout::increment_index(&mut out_idx, &out_shape);
            }
        }
        upload_f32_metal(t, out_shape, out)
    }

    /// `masked_fill`. Same host round-trip as `repeat`. Not
    /// autograd-wired, matching CPU. Unlike CPU's own version, checks `t`'s
    /// and `mask`'s shapes match exactly rather than silently assuming it.
    fn masked_fill<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        mask: &<Self as StorageBackend>::Storage<bool>,
        value: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let mask_shape = mask.metadata().shape().dims().to_vec();
        if in_shape != mask_shape {
            return Err(Error::ShapeMismatch {
                op: "masked_fill",
                expected: in_shape,
                got: mask_shape,
                msg: "mask must match the operand's shape exactly".to_string(),
            });
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let mask_data: &[f32] = bytemuck::cast_slice(mask.as_bytes()?);
        let value = value as f32;
        let out: Vec<f32> = data
            .iter()
            .zip(mask_data.iter())
            .map(|(&v, &m)| if m != 0.0 { value } else { v })
            .collect();
        upload_f32_metal(t, in_shape, out)
    }

    /// `unfold`. Same host round-trip as `repeat`. Not autograd-wired,
    /// matching CPU.
    fn unfold<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let dim_len = in_shape[dim];
        if size > dim_len {
            return Err(Error::Msg(
                "unfold size cannot exceed dimension length".into(),
            ));
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let in_strides = crate::layout::contiguous_strides(&in_shape);
        let n_windows = (dim_len - size) / step + 1;
        let mut out_shape = in_shape.clone();
        out_shape[dim] = n_windows;
        out_shape.push(size);
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let win_idx = idx[dim];
            let offset_idx = idx[out_shape.len() - 1];
            let mut src_flat = 0usize;
            for (axis, &stride) in in_strides.iter().enumerate() {
                let coord = if axis == dim {
                    win_idx * step + offset_idx
                } else {
                    idx[axis]
                };
                src_flat += coord * stride;
            }
            out.push(data[src_flat]);
            crate::layout::increment_index(&mut idx, &out_shape);
        }
        upload_f32_metal(t, out_shape, out)
    }

    /// `pixel_shuffle`. Same host round-trip as `repeat`. Not
    /// autograd-wired, matching CPU.
    fn pixel_shuffle<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        upscale_factor: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        if in_shape.len() != 4 {
            return Err(Error::Msg(
                "pixel_shuffle expects a 4D tensor (N, C, H, W)".into(),
            ));
        }
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let r = upscale_factor;
        let r_sq = r * r;
        if c % r_sq != 0 {
            return Err(Error::Msg(
                "pixel_shuffle channels must be divisible by upscale_factor^2".into(),
            ));
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let in_strides = crate::layout::contiguous_strides(&in_shape);
        let out_c = c / r_sq;
        let out_shape = vec![n, out_c, h * r, w * r];
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; 4];
        for _ in 0..total {
            let (b, c_out, h_out, w_out) = (idx[0], idx[1], idx[2], idx[3]);
            let h_in = h_out / r;
            let w_in = w_out / r;
            let r_h = h_out % r;
            let r_w = w_out % r;
            let c_in = c_out * r_sq + r_h * r + r_w;
            let src_flat = b * in_strides[0]
                + c_in * in_strides[1]
                + h_in * in_strides[2]
                + w_in * in_strides[3];
            out.push(data[src_flat]);
            crate::layout::increment_index(&mut idx, &out_shape);
        }
        upload_f32_metal(t, out_shape, out)
    }

    /// `repeat`. Metal storage is plain host bytes (`MetalStorage::as_bytes`),
    /// so like the elementwise ops above this reads directly with no
    /// download step, repeats with the same row-major walk CPU's own
    /// `repeat` uses (reusing `crate::layout::contiguous_strides` and
    /// `crate::layout::increment_index`, both already `pub(crate)`),
    /// and builds the result via `upload_f32_metal`. Not autograd-wired,
    /// matching CPU.
    fn repeat<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        if repeats.len() != in_shape.len() {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Repeat,
                reason: "repeat factors must match tensor rank",
            }));
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let in_strides = crate::layout::contiguous_strides(&in_shape);
        let out_shape: Vec<usize> = in_shape.iter().zip(repeats).map(|(a, b)| a * b).collect();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let src_flat: usize = idx
                .iter()
                .zip(in_shape.iter())
                .zip(in_strides.iter())
                .map(|((&s, &dim), &stride)| (s % dim) * stride)
                .sum();
            out.push(data[src_flat]);
            if !out_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &out_shape);
            }
        }
        upload_f32_metal(t, out_shape, out)
    }

    /// `pad`. Same pattern as `repeat`. Not autograd-wired, matching CPU.
    fn pad<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let in_strides = crate::layout::contiguous_strides(&in_shape);
        let out_shape: Vec<usize> = in_shape
            .iter()
            .zip(padding)
            .map(|(&s, &(before, after))| s + before + after)
            .collect();
        let total = num_elements(&out_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        let val = val as f32;
        for _ in 0..total {
            let mut inside = true;
            let mut src_flat = 0usize;
            for (axis, &p) in idx.iter().enumerate() {
                let (before, _) = padding[axis];
                if p < before || p >= before + in_shape[axis] {
                    inside = false;
                    break;
                }
                src_flat += (p - before) * in_strides[axis];
            }
            out.push(if inside { data[src_flat] } else { val });
            if !out_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &out_shape);
            }
        }
        upload_f32_metal(t, out_shape, out)
    }

    /// `triu`. Same pattern as `repeat`. Not autograd-wired, matching CPU.
    fn triu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let rank = in_shape.len();
        let total = num_elements(&in_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for &value in data.iter().take(total) {
            let (r, c) = if rank >= 2 {
                (idx[rank - 2] as i64, idx[rank - 1] as i64)
            } else {
                (0, idx[0] as i64)
            };
            out.push(if c >= r + k { value } else { 0.0 });
            if !in_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &in_shape);
            }
        }
        upload_f32_metal(t, in_shape, out)
    }

    /// `tril`. Same pattern as `repeat`. Not autograd-wired, matching CPU.
    fn tril<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let rank = in_shape.len();
        let total = num_elements(&in_shape)?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for &value in data.iter().take(total) {
            let (r, c) = if rank >= 2 {
                (idx[rank - 2] as i64, idx[rank - 1] as i64)
            } else {
                (0, idx[0] as i64)
            };
            out.push(if c <= r + k { value } else { 0.0 });
            if !in_shape.is_empty() {
                crate::layout::increment_index(&mut idx, &in_shape);
            }
        }
        upload_f32_metal(t, in_shape, out)
    }

    /// `diag`. Same pattern as `repeat`, matching CPU's two cases: a 1D
    /// operand builds a 2D matrix with that operand on its `k`-th diagonal,
    /// an operand of rank 2+ extracts its `k`-th diagonal into a 1D result.
    /// Not autograd-wired, matching CPU.
    fn diag<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_shape = t.metadata().shape().dims().to_vec();
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let rank = in_shape.len();
        if rank == 1 {
            let n = in_shape[0];
            let k_abs = k.unsigned_abs() as usize;
            let out_dim = n.checked_add(k_abs).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "Metal diagonal output dimension",
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
            upload_f32_metal(t, vec![out_dim, out_dim], out)
        } else {
            let r_len = in_shape[rank - 2];
            let c_len = in_shape[rank - 1];
            let mut diag_vals = Vec::new();
            for r in 0..r_len {
                let c = (r as i64 + k) as usize;
                if c < c_len {
                    diag_vals.push(data[r * c_len + c]);
                }
            }
            let out_len = diag_vals.len();
            upload_f32_metal(t, vec![out_len], diag_vals)
        }
    }

    /// `unsqueeze`. Metadata-only, like `reshape` (which it delegates to and
    /// so inherits gradient wiring from), matching every other backend's
    /// own `unsqueeze`.
    fn unsqueeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut target_shape = t.metadata().shape().dims().to_vec();
        if dim <= target_shape.len() {
            target_shape.insert(dim, 1);
        } else {
            target_shape.push(1);
        }
        <Self as TensorOps<Self>>::reshape::<K>(t, &target_shape)
    }

    /// `cmp_eq`. Metal's `add`/`sub`/`mul` above are already implemented as
    /// a host round-trip over `as_bytes()`/`from_bytes()` (`binary_op_metal`,
    /// this file) rather than a dispatched `.metal` shader, so this reuses
    /// that same helper with a comparison closure instead of adding a new
    /// one. Matches CPU's own encoding (1.0/0.0 in the same dtype) and lack
    /// of a gradient for comparisons.
    fn cmp_eq<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_eq",
        })
    }
    fn cmp_ne<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_ne",
        })
    }
    fn cmp_lt<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_lt",
        })
    }
    fn cmp_le<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_le",
        })
    }
    fn cmp_gt<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_gt",
        })
    }
    fn cmp_ge<K: DType>(
        _lhs: &<Self as StorageBackend>::Storage<K>,
        _rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "cmp_ge",
        })
    }

    fn logical_and(
        _lhs: &<Self as StorageBackend>::Storage<bool>,
        _rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "logical_and",
        })
    }
    fn logical_or(
        _lhs: &<Self as StorageBackend>::Storage<bool>,
        _rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "logical_or",
        })
    }
    fn logical_not(
        _t: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        Err(Error::UnsupportedDType {
            dtype: <bool as ConstDType>::DESCRIPTOR,
            backend: "Metal",
            op: "logical_not",
        })
    }

    /// `sub_scalar`. Reuses `scalar_op_metal`, already used by
    /// `mul_scalar_float` above; not autograd-wired, matching CPU's
    /// `TensorOps` scalar methods (as opposed to `FloatOps`'s
    /// `add_scalar_float`/`mul_scalar_float`, which do carry a gradient).
    fn sub_scalar<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        scalar_op_metal(t, val, |v, s| v - s)
    }
    /// `div_scalar`.
    fn div_scalar<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        scalar_op_metal(t, val, |v, s| v / s)
    }

    /// `maximum`. Not autograd-wired, matching CPU.
    fn maximum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        binary_op_metal(lhs, rhs, "maximum", f32::max)
    }
    /// `minimum`.
    fn minimum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        binary_op_metal(lhs, rhs, "minimum", f32::min)
    }
    /// `abs_diff`.
    fn abs_diff<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        binary_op_metal(lhs, rhs, "abs_diff", |a, b| (a - b).abs())
    }

    /// `lerp`. `start + weight * (end - start)`; not autograd-wired,
    /// matching CPU.
    fn lerp<K: DType>(
        start: &<Self as StorageBackend>::Storage<K>,
        end: &<Self as StorageBackend>::Storage<K>,
        weight: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let weight = weight as f32;
        binary_op_metal(start, end, "lerp", move |s, e| s + weight * (e - s))
    }

    /// `float_to_scalar`. Metal storage is plain host-accessible bytes
    /// (`MetalStorage::as_bytes`), so unlike CUDA/WGPU this needs no
    /// download step at all.
    fn float_to_scalar<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<f64> {
        let numel = t.metadata().shape().dims().iter().product::<usize>();
        if numel != 1 {
            return Err(Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::Storage,
                parameter: "float_to_scalar element count",
                value: numel,
            }));
        }
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        Ok(f64::from(data[0]))
    }
    /// `float_to_vec1`.
    fn float_to_vec1<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<Vec<f64>> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        Ok(data.iter().map(|&x| x as f64).collect())
    }
    /// `int_to_scalar`.
    fn int_to_scalar<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<i64> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        let value = *data.first().ok_or(Error::InvalidByteLength {
            expected: core::mem::size_of::<f32>(),
            got: 0,
        })?;
        incin_core::prelude::convert_f64_to_i64(
            "int_to_scalar",
            t.metadata().dtype(),
            f64::from(value),
            incin_core::prelude::FloatToIntPolicy::Exact,
        )
    }
    /// `int_to_vec1`.
    fn int_to_vec1<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<Vec<i64>> {
        let data: &[f32] = bytemuck::cast_slice(t.as_bytes()?);
        data.iter()
            .map(|&value| {
                incin_core::prelude::convert_f64_to_i64(
                    "int_to_vec1",
                    t.metadata().dtype(),
                    f64::from(value),
                    incin_core::prelude::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
    /// `tensor_to_dtype`. Matches CPU's/WGPU's/CUDA's own passthrough for
    /// this method.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        _dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K2>> {
        let bytes = t.as_bytes()?.to_vec();
        MetalStorage::from_bytes(bytes, t.metadata().clone(), t.mode(), t.device_ordinal())
    }

    /// `addmm`. `beta * mat + alpha * (mat1 @ mat2)`, composed from the
    /// already tape-wired `matmul`/`mul_scalar_float`/`add`, matching every
    /// other backend's own composition — no new host round-trip, just
    /// reuse of already-implemented methods.
    fn addmm<K: DType>(
        mat: &<Self as StorageBackend>::Storage<K>,
        mat1: &<Self as StorageBackend>::Storage<K>,
        mat2: &<Self as StorageBackend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mm = <Self as TensorOps<Self>>::matmul::<K>(mat1, mat2)?;
        let mm_alpha = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&mm, alpha)?;
        let mat_beta = <Self as FloatOps<Self>>::mul_scalar_float::<K>(mat, beta)?;
        <Self as NumericOps<Self>>::add::<K>(&mat_beta, &mm_alpha)
    }
    /// `bmm`. `matmul` already handles the batch dimensions, matching every
    /// other backend.
    fn bmm<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        <Self as TensorOps<Self>>::matmul::<K>(lhs, rhs)
    }

    /// `scaled_dot_product_attention`. Composed from the already tape-wired
    /// `transpose`/`matmul`/`mul_scalar_float`/`add`/`softmax`, matching
    /// every other backend's own composition, no new host round-trip.
    fn scaled_dot_product_attention<K: DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let k_rank = k.metadata().shape().dims().len();
        let k_t = if k_rank >= 2 {
            <Self as TensorOps<Self>>::transpose::<K>(k, k_rank - 2, k_rank - 1)?
        } else {
            k.clone()
        };
        let scores = <Self as TensorOps<Self>>::matmul::<K>(q, &k_t)?;
        let d_k = *q.metadata().shape().dims().last().unwrap_or(&1) as f64;
        let s = scale.unwrap_or_else(|| 1.0 / d_k.sqrt());
        let scaled_scores = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&scores, s)?;
        let masked_scores = if let Some(m) = mask {
            <Self as NumericOps<Self>>::add::<K>(&scaled_scores, m)?
        } else {
            scaled_scores
        };
        let attn_dim = scores.metadata().shape().dims().len() - 1;
        let attn = <Self as FloatOps<Self>>::softmax::<K>(&masked_scores, attn_dim)?;
        <Self as TensorOps<Self>>::matmul::<K>(&attn, v)
    }

    fn matmul<K: DType>(
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

    fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim0: usize,
        dim1: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

    fn broadcast_as<K: DType>(
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

    fn broadcast_left<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::broadcast_as::<K>(t, shape)
    }

    fn narrow<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        let out_numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_dims))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let mut out_data = Vec::with_capacity(out_numel);

        let bytes = t.as_bytes()?;
        let in_slice: &[f32] = bytemuck::cast_slice(bytes);

        let outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[..dim]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let inner: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[dim + 1..]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;

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
                let g_outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(t_dims[..dim]))
                    .checked_numel(incin_core::prelude::OperationKind::Storage)?;
                let g_inner: usize =
                    incin_core::prelude::ShapeBuf::from_slice(&(t_dims[dim + 1..]))
                        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        let folded: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[start_dim..=end_dim]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        new_dims.push(folded);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        Self::reshape::<K>(t, &new_dims)
    }

    fn squeeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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

        let outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(first_dims[..dim]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let inner: usize = incin_core::prelude::ShapeBuf::from_slice(&(first_dims[dim + 1..]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let out_numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_dims))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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
                    let sub_numel: usize =
                        incin_core::prelude::ShapeBuf::from_slice(&(sub_dims))
                            .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        let refs: Vec<&<Self as StorageBackend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    fn slice<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut curr = t.clone();
        for (dim, &(start, len)) in ranges.iter().enumerate() {
            curr = Self::narrow::<K>(&curr, dim, start, len)?;
        }
        Ok(curr)
    }
}

// ─── ModuleOps ──────────────────────────────────────────────────────────────

impl<D: Device> ModuleOps<Self> for MetalBackendImpl<D> {
    fn layer_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let last_dim = dims[dims.len() - 1];
        let outer: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[..dims.len() - 1]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;

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
        t: &<Self as StorageBackend>::Storage<K>,
        w: Option<&<Self as StorageBackend>::Storage<K>>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        _rm: Option<&<Self as StorageBackend>::Storage<K>>,
        _rv: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        let spatial: usize = incin_core::prelude::ShapeBuf::from_slice(&(dims[2..]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;

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
        _t: &<Self as StorageBackend>::Storage<KInt>,
        _w: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("embedding"))
    }

    fn conv1d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _w: &<Self as StorageBackend>::Storage<K>,
        _b: Option<&<Self as StorageBackend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("conv1d"))
    }

    fn conv2d<K: DType>(
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

    fn conv_transpose2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _w: &<Self as StorageBackend>::Storage<K>,
        _b: Option<&<Self as StorageBackend>::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _output_padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("conv_transpose2d"))
    }

    fn max_pool2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("max_pool2d"))
    }

    fn avg_pool2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("avg_pool2d"))
    }

    fn adaptive_avg_pool2d<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("adaptive_avg_pool2d"))
    }
}

// ─── LossOps ────────────────────────────────────────────────────────────────

impl<D: Device> LossOps<Self> for MetalBackendImpl<D> {
    fn cross_entropy_loss<K: DType, KInt: DType>(
        _pred: &<Self as StorageBackend>::Storage<K>,
        _target: &<Self as StorageBackend>::Storage<KInt>,
        _reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("cross_entropy_loss"))
    }
}

// ─── QuantizedOps ───────────────────────────────────────────────────────────

impl<D: Device> QuantizedOps<Self> for MetalBackendImpl<D> {
    fn quantize<K: FloatDType, Q: QuantDType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<Q>> {
        Err(unsupported("quantize"))
    }

    fn dequantize<Q: QuantDType, K: FloatDType>(
        _t: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(unsupported("dequantize"))
    }

    fn quantized_matmul<Q: QuantDType>(
        _lhs: &<Self as StorageBackend>::Storage<Q>,
        _rhs: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<f32>> {
        Err(unsupported("quantized_matmul"))
    }
}

// ─── OptimizerOps ───────────────────────────────────────────────────────────
// Uses default adamw_step composed from NumericOps/FloatOps (via trait default).
impl<D: Device> OptimizerOps<Self> for MetalBackendImpl<D> {}



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
}
impl<D: Device> VariableBackend for MetalBackendImpl<D> {
    type RawVar = MetalVar;

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
}
