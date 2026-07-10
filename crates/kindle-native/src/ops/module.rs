//! `ModuleOps` for `NativeBackend<T, D>` — the crate's single `impl ModuleOps`
//! block (Rust disallows splitting one trait impl across multiple files).
//!
//! As of Plan 04-06, eight of the nine `ModuleOps` methods are real:
//!   - `layer_norm` — delegates to `ops::norm::layer_norm_impl`
//!   - `batch_norm`  — delegates to `ops::norm::batch_norm_impl`
//!   - `embedding`  — delegates to `ops::embedding::embedding_impl`
//!   - `conv1d`  — delegates to `ops::conv::conv1d_impl`
//!   - `conv2d`  — delegates to `ops::conv::conv2d_impl`
//!   - `max_pool2d`  — delegates to `ops::pool::max_pool2d_impl`
//!   - `avg_pool2d`  — delegates to `ops::pool::avg_pool2d_impl`
//!   - `adaptive_avg_pool2d`  — delegates to `ops::pool::adaptive_avg_pool2d_impl`
//!
//! The remaining method (`conv_transpose2d`) returns
//! `Error::UnsupportedBackendOperation` and will be replaced by a later plan
//! in this phase.

use kindle_core::err::Error;
use kindle_core::prelude::{Backend, DType, ModuleOps, Result};

use crate::NativeBackend;

impl<T: DType, D: kindle_core::prelude::Device> ModuleOps<Self> for NativeBackend<T, D> {
    fn layer_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::norm::layer_norm_impl::<T, D, K>(t, weight, bias, eps)
    }

    fn batch_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: Option<&<Self as Backend>::Storage<K>>,
        b: Option<&<Self as Backend>::Storage<K>>,
        rm: Option<&<Self as Backend>::Storage<K>>,
        rv: Option<&<Self as Backend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::norm::batch_norm_impl::<T, D, K>(t, w, b, rm, rv, e, momentum)
    }

    fn embedding<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<KInt>,
        w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::embedding::embedding_impl::<T, D, K, KInt>(t, w)
    }

    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::conv::conv1d_impl::<T, D, K>(t, w, b, stride, padding, dilation, groups)
    }

    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::conv::conv2d_impl::<T, D, K>(t, w, b, stride, padding, dilation, groups)
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
        Err(Error::UnsupportedBackendOperation {
            op: "conv_transpose2d",
            backend: "Native",
        })
    }

    fn max_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::pool::max_pool2d_impl::<T, D, K>(t, kernel_size, stride, padding, dilation)
    }

    fn avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::pool::avg_pool2d_impl::<T, D, K>(t, kernel_size, stride, padding)
    }

    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::ops::pool::adaptive_avg_pool2d_impl::<T, D, K>(t, output_size)
    }
}
