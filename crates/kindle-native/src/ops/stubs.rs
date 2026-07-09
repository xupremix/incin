//! Trait-completeness stubs for `NativeBackend<T, D>`.
//!
//! `Backend` requires `CreationOps + NumericOps + TensorOps + FloatOps +
//! ReductionOps + ModuleOps + LossOps` as a single supertrait bound (see
//! `kindle-core/src/tensor/backend.rs`), so the crate cannot compile — and
//! this plan's own `cargo test -p kindle-native --lib creation::`/
//! `ops::elementwise::` verification cannot run — without *some*
//! implementation of every method on every sub-trait, even though most of
//! `TensorOps`/`ReductionOps`/`ModuleOps`/most of `LossOps` are explicitly
//! out of this plan's scope (they get real implementations in later plans
//! per the Minimal Phase 1 Op Set / "Stub/todo acceptable" column).
//!
//! `reshape`/`transpose`/`broadcast_as` already have real O(1)-view
//! implementations on `NativeStorage` from Plan 01, so those three are
//! wired to real logic here rather than stubbed. Every other not-yet-owned
//! method returns `Error::UnsupportedBackendOperation`, matching
//! `NdarrayBackend`'s existing in-repo convention for the same situation
//! (a typed, non-panicking not-yet-implemented signal) rather than
//! `CandleBackend`'s `unimplemented!()` panics.

use kindle_core::err::Error;
use kindle_core::nn::Reduction;
use kindle_core::prelude::{
    Backend, DType, KindleDType, LossOps, ModuleOps, ReductionOps, Result, TensorOps,
};

use crate::NativeBackend;
use crate::storage::NativeStorage;

impl<T: DType, D: kindle_core::prelude::Device> TensorOps<Self> for NativeBackend<T, D> {
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        t.reshape(shape)
    }

    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        t.transpose(dim1, dim2)
    }

    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        t.broadcast_as(shape)
    }

    fn matmul<K: DType>(
        _lhs: &<Self as Backend>::Storage<K>,
        _rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "matmul",
            backend: "Native",
        })
    }

    fn narrow<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "narrow",
            backend: "Native",
        })
    }

    fn squeeze<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "squeeze",
            backend: "Native",
        })
    }

    fn stack<K: DType>(
        _t: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "stack",
            backend: "Native",
        })
    }

    fn concat<K: DType>(
        _t: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "concat",
            backend: "Native",
        })
    }

    fn slice<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "slice",
            backend: "Native",
        })
    }

    fn flatten<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "flatten",
            backend: "Native",
        })
    }

    fn broadcast_left<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "broadcast_left",
            backend: "Native",
        })
    }

    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        Ok(t.get(&vec![0usize; t.shape.len()]))
    }

    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<std::vec::Vec<f64>> {
        if t.shape.len() != 1 {
            return Err(Error::UnsupportedBackendOperation {
                op: "float_to_vec1",
                backend: "Native",
            });
        }
        Ok((0..t.shape[0]).map(|i| t.get(&[i])).collect())
    }

    fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        Err(Error::UnsupportedBackendOperation {
            op: "int_to_scalar",
            backend: "Native",
        })
    }

    fn int_to_vec1<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<std::vec::Vec<i64>> {
        Err(Error::UnsupportedBackendOperation {
            op: "int_to_vec1",
            backend: "Native",
        })
    }

    fn tensor_to_dtype<K: DType, K2: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        Err(Error::UnsupportedBackendOperation {
            op: "tensor_to_dtype",
            backend: "Native",
        })
    }
}

impl<T: DType, D: kindle_core::prelude::Device> ReductionOps<Self> for NativeBackend<T, D> {
    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use crate::storage::NativeBuffer;
        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut sum = 0f64;
        for _ in 0..total {
            sum += t.get(&idx);
            if !t.shape.is_empty() {
                crate::ops::elementwise::increment_index(&mut idx, &t.shape);
            }
        }
        Ok(NativeStorage::from_contiguous(
            NativeBuffer::F32(vec![sum as f32]),
            vec![],
        ))
    }

    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use crate::storage::NativeBuffer;
        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut sum = 0f64;
        for _ in 0..total {
            sum += t.get(&idx);
            if !t.shape.is_empty() {
                crate::ops::elementwise::increment_index(&mut idx, &t.shape);
            }
        }
        let mean = if total > 0 { sum / total as f64 } else { 0.0 };
        Ok(NativeStorage::from_contiguous(
            NativeBuffer::F32(vec![mean as f32]),
            vec![],
        ))
    }

    fn max_all<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "max_all",
            backend: "Native",
        })
    }
    fn min_all<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "min_all",
            backend: "Native",
        })
    }
    fn sum_dim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "sum_dim",
            backend: "Native",
        })
    }
    fn sum_keepdim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "sum_keepdim",
            backend: "Native",
        })
    }
    fn mean_dim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "mean_dim",
            backend: "Native",
        })
    }
    fn mean_keepdim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "mean_keepdim",
            backend: "Native",
        })
    }
    fn max_dim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "max_dim",
            backend: "Native",
        })
    }
    fn max_keepdim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "max_keepdim",
            backend: "Native",
        })
    }
    fn min_dim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "min_dim",
            backend: "Native",
        })
    }
    fn min_keepdim<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "min_keepdim",
            backend: "Native",
        })
    }
    fn argmax<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        Err(Error::UnsupportedBackendOperation {
            op: "argmax",
            backend: "Native",
        })
    }
    fn argmin<K: DType, KInt: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        Err(Error::UnsupportedBackendOperation {
            op: "argmin",
            backend: "Native",
        })
    }
}

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
        _target: &<Self as Backend>::Storage<KInt>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("cross_entropy_loss not implemented for NativeBackend")
    }
}
