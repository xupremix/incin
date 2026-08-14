//! `ModuleOps` for `CpuBackendImpl<D>` — the crate's single `impl ModuleOps`
//! block (Rust disallows splitting one trait impl across multiple files).
//!
//! As of Plan 04-07, ALL NINE `ModuleOps` methods are real — no unsupported-
//! operation stubs remain on `CpuBackendImpl`:
//!   - `layer_norm` — delegates to `ops::norm::layer_norm_impl`
//!   - `batch_norm`  — delegates to `ops::norm::batch_norm_impl`
//!   - `embedding`  — delegates to `ops::embedding::embedding_impl`
//!   - `conv1d`  — delegates to `ops::conv::conv1d_impl`
//!   - `conv2d`  — delegates to `ops::conv::conv2d_impl`
//!   - `conv_transpose2d`  — delegates to `ops::conv::conv_transpose2d_impl`
//!   - `max_pool2d`  — delegates to `ops::pool::max_pool2d_impl`
//!   - `avg_pool2d`  — delegates to `ops::pool::avg_pool2d_impl`
//!   - `adaptive_avg_pool2d`  — delegates to `ops::pool::adaptive_avg_pool2d_impl`
//!
//! This closes out `ModuleOps`'s full trait surface for Phase 4
//! (CPUBACK-08).

use crate::cpu::CpuBackendImpl;
use incin_core::prelude::{DType, Device, Result, StorageBackend};
use incin_core::tensor::backend::ModuleOps;

impl<D: Device> ModuleOps<Self> for CpuBackendImpl<D> {
    /// `layer_norm`.
    fn layer_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::norm::layer_norm_impl::<D, K>(t, weight, bias, eps)
    }

    /// `batch_norm`.
    fn batch_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: Option<&<Self as StorageBackend>::Storage<K>>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        rm: Option<&<Self as StorageBackend>::Storage<K>>,
        rv: Option<&<Self as StorageBackend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::norm::batch_norm_impl::<D, K>(t, w, b, rm, rv, e, momentum)
    }

    /// `embedding`.
    fn embedding<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<KInt>,
        w: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::embedding::embedding_impl::<D, K, KInt>(t, w)
    }

    /// `conv1d`.
    fn conv1d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv1d_impl::<D, K>(t, w, b, stride, padding, dilation, groups)
    }

    /// `conv2d`.
    fn conv2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv2d_impl::<D, K>(t, w, b, stride, padding, dilation, groups)
    }

    /// `conv_transpose2d`.
    fn conv_transpose2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::conv::conv_transpose2d_impl::<D, K>(
            t,
            w,
            b,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )
    }

    /// `max_pool2d`.
    fn max_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::max_pool2d_impl::<D, K>(t, kernel_size, stride, padding, dilation)
    }

    /// `avg_pool2d`.
    fn avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::avg_pool2d_impl::<D, K>(t, kernel_size, stride, padding)
    }

    /// `adaptive_avg_pool2d`.
    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        crate::cpu::ops::pool::adaptive_avg_pool2d_impl::<D, K>(t, output_size)
    }
}
