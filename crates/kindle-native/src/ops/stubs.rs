//! Trait-completeness stubs for `NativeBackend<T, D>`.
//!
//! `Backend` requires `CreationOps + NumericOps + TensorOps + FloatOps +
//! ReductionOps + ModuleOps + LossOps` as a single supertrait bound (see
//! `kindle-core/src/tensor/backend.rs`), so the crate cannot compile without
//! *some* implementation of every method on every sub-trait.
//!
//! **Plan 04 update:** `TensorOps` is now implemented in `ops/shape_ops.rs`
//! and `ReductionOps` is now implemented in `ops/reduce.rs`. This file
//! retains only `ModuleOps` and `LossOps` — both remain out of Phase 1 scope
//! except for `mse_loss` which composes from already-tape-tracked primitives.

use kindle_core::err::Error;
use kindle_core::nn::Reduction;
use kindle_core::prelude::{Backend, DType, LossOps, ModuleOps, ReductionOps, Result};

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

impl<T: DType, D: kindle_core::prelude::Device> LossOps<Self> for NativeBackend<T, D> {
    fn mse_loss<K: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Composed from already-tape-tracked primitives (sub, mul,
        // mean_all/sum_all) per the anti-pattern warning — never a fused
        // hand-derived backward formula.
        let diff = <Self as kindle_core::prelude::NumericOps<Self>>::sub::<K>(pred, target)?;
        let sq = <Self as kindle_core::prelude::NumericOps<Self>>::mul::<K>(&diff, &diff)?;
        match reduction {
            Reduction::Mean => <Self as ReductionOps<Self>>::mean_all::<K>(&sq),
            Reduction::Sum => <Self as ReductionOps<Self>>::sum_all::<K>(&sq),
            Reduction::None => Ok(sq),
        }
    }

    fn l1_loss<K: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("l1_loss not implemented for NativeBackend")
    }

    fn bce_with_logits_loss<K: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("bce_with_logits_loss not implemented for NativeBackend")
    }

    fn cross_entropy_loss<K: DType, KInt: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("cross_entropy_loss not implemented for NativeBackend")
    }
}
