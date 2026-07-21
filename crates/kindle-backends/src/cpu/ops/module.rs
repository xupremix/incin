//! `ModuleOps` for `CpuBackend<T, D>` — the crate's single `impl ModuleOps`
//! block (Rust disallows splitting one trait impl across multiple files).
//!
//! As of Plan 04-07, ALL NINE `ModuleOps` methods are real — no unsupported-
//! operation stubs remain on `CpuBackend`:
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

use kindle_core::prelude::{Backend, DType, ModuleOps, Result};

use crate::cpu::CpuBackend;

impl<T: DType, D: kindle_core::prelude::Device> ModuleOps<Self> for CpuBackend<T, D> {
    /// Core abstraction for `layer_norm` within the Kindle framework..
    fn layer_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::norm::layer_norm_impl::<T, D, K>(t, weight, bias, eps)
    }

    /// Core abstraction for `batch_norm` within the Kindle framework..
    fn batch_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: Option<&<Self as Backend>::Storage<K>>,
        b: Option<&<Self as Backend>::Storage<K>>,
        rm: Option<&<Self as Backend>::Storage<K>>,
        rv: Option<&<Self as Backend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::norm::batch_norm_impl::<T, D, K>(t, w, b, rm, rv, e, momentum)
    }

    /// Core abstraction for `embedding` within the Kindle framework..
    fn embedding<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<KInt>,
        w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::embedding::embedding_impl::<T, D, K, KInt>(t, w)
    }

    /// Core abstraction for `conv1d` within the Kindle framework..
    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::conv::conv1d_impl::<T, D, K>(t, w, b, stride, padding, dilation, groups)
    }

    /// Core abstraction for `conv2d` within the Kindle framework..
    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::conv::conv2d_impl::<T, D, K>(t, w, b, stride, padding, dilation, groups)
    }

    /// Core abstraction for `conv_transpose2d` within the Kindle framework..
    fn conv_transpose2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::conv::conv_transpose2d_impl::<T, D, K>(
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

    /// Core abstraction for `max_pool2d` within the Kindle framework..
    fn max_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::pool::max_pool2d_impl::<T, D, K>(t, kernel_size, stride, padding, dilation)
    }

    /// Core abstraction for `avg_pool2d` within the Kindle framework..
    fn avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::pool::avg_pool2d_impl::<T, D, K>(t, kernel_size, stride, padding)
    }

    /// Core abstraction for `adaptive_avg_pool2d` within the Kindle framework..
    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cpu::ops::pool::adaptive_avg_pool2d_impl::<T, D, K>(t, output_size)
    }
}
