//! Compatibility `ModuleOps` adapters for the CPU backend.
//!
//! Canonical descriptor executors call the backend-local helpers directly.
//! This adapter remains only while dynamic dispatch migration removes the
//! historical operation-family surface.

use crate::cpu::CpuBackendImpl;
use incin_core::prelude::{DType, Device, Result, StorageBackend};
use incin_core::__backend_compat::legacy::ModuleOps;

impl<D: Device> ModuleOps<Self> for CpuBackendImpl<D> {
    fn layer_norm<K: DType>(t: &<Self as StorageBackend>::Storage<K>, weight: &<Self as StorageBackend>::Storage<K>, bias: Option<&<Self as StorageBackend>::Storage<K>>, eps: f32) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::norm::layer_norm_impl::<D, K>(t, weight, bias, eps)
    }
    fn batch_norm<K: DType>(t: &<Self as StorageBackend>::Storage<K>, w: Option<&<Self as StorageBackend>::Storage<K>>, b: Option<&<Self as StorageBackend>::Storage<K>>, rm: Option<&<Self as StorageBackend>::Storage<K>>, rv: Option<&<Self as StorageBackend>::Storage<K>>, e: f32, momentum: f64) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::norm::batch_norm_impl::<D, K>(t, w, b, rm, rv, e, momentum)
    }
    fn embedding<K: DType, KInt: DType>(t: &<Self as StorageBackend>::Storage<KInt>, w: &<Self as StorageBackend>::Storage<K>) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::embedding::embedding_impl::<D, K, KInt>(t, w)
    }
    fn conv1d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, w: &<Self as StorageBackend>::Storage<K>, b: Option<&<Self as StorageBackend>::Storage<K>>, stride: usize, padding: usize, dilation: usize, groups: usize) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv1d_impl::<D, K>(t, w, b, stride, padding, dilation, groups)
    }
    fn conv2d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, w: &<Self as StorageBackend>::Storage<K>, b: Option<&<Self as StorageBackend>::Storage<K>>, stride: usize, padding: usize, dilation: usize, groups: usize) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv2d_impl::<D, K>(t, w, b, stride, padding, dilation, groups)
    }
    fn conv_transpose2d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, w: &<Self as StorageBackend>::Storage<K>, b: Option<&<Self as StorageBackend>::Storage<K>>, stride: usize, padding: usize, output_padding: usize, dilation: usize, groups: usize) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv_transpose2d_impl::<D, K>(t, w, b, stride, padding, output_padding, dilation, groups)
    }
    fn max_pool2d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize)) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::max_pool2d_impl::<D, K>(t, kernel_size, stride, padding, dilation)
    }
    fn avg_pool2d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::avg_pool2d_impl::<D, K>(t, kernel_size, stride, padding)
    }
    fn adaptive_avg_pool2d<K: DType>(t: &<Self as StorageBackend>::Storage<K>, output_size: (usize, usize)) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::adaptive_avg_pool2d_impl::<D, K>(t, output_size)
    }
}
