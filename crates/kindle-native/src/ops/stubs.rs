//! Trait-completeness stubs for `NativeBackend<T, D>` — `ModuleOps` only.
//!
//! **Plan 05 update:** `LossOps` is now implemented in `ops/loss.rs`.
//! `TensorOps` lives in `ops/shape_ops.rs`, `ReductionOps` in `ops/reduce.rs`.
//! This file retains only `ModuleOps` — all methods return a typed
//! `Error::UnsupportedBackendOperation` since no `ModuleOps` method is
//! reachable by `Linear::forward` + `mse_loss`'s actual call graph (confirmed
//! by direct read of `nn/linear.rs` in RESEARCH.md — `Linear` is implemented
//! as `matmul` + `add`, with no `ModuleOps` calls).

use kindle_core::err::Error;
use kindle_core::prelude::{Backend, DType, ModuleOps, Result};

use crate::NativeBackend;

impl<T: DType, D: kindle_core::prelude::Device> ModuleOps<Self> for NativeBackend<T, D> {
    fn layer_norm<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _weight: &<Self as Backend>::Storage<K>,
        _bias: Option<&<Self as Backend>::Storage<K>>,
        _eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "layer_norm",
            backend: "Native",
        })
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
        Err(Error::UnsupportedBackendOperation {
            op: "batch_norm",
            backend: "Native",
        })
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
