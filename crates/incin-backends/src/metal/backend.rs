//! Metal compute backend for Incin on Apple Silicon and macOS.

use core::marker::PhantomData;

use incin_core::exec::TensorMeta;
use incin_core::prelude::*;

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

    fn backward<K: DType>(_loss: &Self::Storage<K>) -> Result<Self::Grads> {
        Err(unsupported("backward"))
    }

    fn get_grad<K: DType>(
        _t: &Self::Storage<K>,
        _grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(None)
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
        let meta = TensorMeta::contiguous(
            shape_buf,
            dtype,
            *device,
            MetalStorage::alignment(),
            numel,
        )?;
        MetalStorage::from_bytes(bytes.to_vec(), meta, MetalStorageMode::Shared, device.ordinal())
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
        MetalStorage::zeros(&shape_buf, dtype, MetalStorageMode::Shared, device.ordinal())
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
        let meta = TensorMeta::contiguous(
            shape_buf,
            dtype,
            *device,
            MetalStorage::alignment(),
            n,
        )?;
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
        let meta = TensorMeta::contiguous(
            shape_buf,
            dtype,
            *device,
            MetalStorage::alignment(),
            n,
        )?;
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
        let meta = TensorMeta::contiguous(
            shape_buf,
            dtype,
            *device,
            MetalStorage::alignment(),
            n,
        )?;
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

// ─── NumericOps ─────────────────────────────────────────────────────────────

impl<T: DType, D: Device> NumericOps<Self> for MetalBackendImpl<T, D> {
    fn add<K: DType>(
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("add"))
    }

    fn sub<K: DType>(
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("sub"))
    }

    fn mul<K: DType>(
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("mul"))
    }

    fn div<K: DType>(
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("div"))
    }
}

// ─── FloatOps ───────────────────────────────────────────────────────────────

impl<T: DType, D: Device> FloatOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_float_ops! {
        unary:
            relu, step, elu, gelu, mish, tanh, sigmoid, abs, neg, sqrt, exp, log, swish,
            sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
            atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    fn add_scalar_float<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("add_scalar_float"))
    }

    fn mul_scalar_float<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("mul_scalar_float"))
    }

    fn softmax<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("softmax"))
    }
}

// ─── ReductionOps ───────────────────────────────────────────────────────────

impl<T: DType, D: Device> ReductionOps<Self> for MetalBackendImpl<T, D> {
    crate::unsupported::unsupported_reduction_ops! {
        all: sum_all, mean_all, max_all, min_all, prod_all;
        dim:
            sum_dim, mean_dim, max_dim, min_dim,
            sum_keepdim, mean_keepdim, max_keepdim, min_keepdim,
            prod_dim, cumsum;
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
    ) -> Result<(<Self as Backend>::Storage<K>, <Self as Backend>::Storage<KInt>)> {
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
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("matmul"))
    }

    fn transpose<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim0: usize,
        _dim1: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("transpose"))
    }

    fn reshape<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("reshape"))
    }

    fn broadcast_as<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("broadcast_as"))
    }

    fn broadcast_left<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("broadcast_left"))
    }

    fn narrow<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("narrow"))
    }

    fn flatten<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("flatten"))
    }

    fn squeeze<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("squeeze"))
    }

    fn concat<K: DType>(
        _tensors: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("concat"))
    }

    fn stack<K: DType>(
        _tensors: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("stack"))
    }

    fn slice<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("slice"))
    }
}

// ─── ModuleOps ──────────────────────────────────────────────────────────────

impl<T: DType, D: Device> ModuleOps<Self> for MetalBackendImpl<T, D> {
    fn layer_norm<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _weight: &<Self as Backend>::Storage<K>,
        _bias: Option<&<Self as Backend>::Storage<K>>,
        _eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("layer_norm"))
    }

    fn batch_norm<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _w: Option<&<Self as Backend>::Storage<K>>,
        _b: Option<&<Self as Backend>::Storage<K>>,
        _rm: Option<&<Self as Backend>::Storage<K>>,
        _rv: Option<&<Self as Backend>::Storage<K>>,
        _e: f32,
        _momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(unsupported("batch_norm"))
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
