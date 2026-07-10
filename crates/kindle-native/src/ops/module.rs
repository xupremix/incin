//! `ModuleOps` for `NativeBackend<T, D>` — the crate's single `impl ModuleOps`
//! block (Rust disallows splitting one trait impl across multiple files).
//!
//! As of Plan 04-03, two of the nine `ModuleOps` methods are real:
//!   - `layer_norm` — delegates to `ops::norm::layer_norm_impl`
//!   - `batch_norm`  — delegates to `ops::norm::batch_norm_impl`
//!
//! The remaining seven methods return `Error::UnsupportedBackendOperation`
//! and will be replaced by later plans in this phase (embedding, conv, pooling).

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
        _t: &<Self as Backend>::Storage<KInt>,
        _w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "embedding",
            backend: "Native",
        })
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
        Err(Error::UnsupportedBackendOperation {
            op: "conv1d",
            backend: "Native",
        })
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
        Err(Error::UnsupportedBackendOperation {
            op: "conv2d",
            backend: "Native",
        })
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
        _t: &<Self as Backend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "max_pool2d",
            backend: "Native",
        })
    }

    fn avg_pool2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "avg_pool2d",
            backend: "Native",
        })
    }

    fn adaptive_avg_pool2d<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "adaptive_avg_pool2d",
            backend: "Native",
        })
    }
}
