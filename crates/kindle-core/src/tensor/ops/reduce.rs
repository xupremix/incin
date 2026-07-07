//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::tensor::ops::*;
use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};
use crate::nn::loss::{Mean, ReductionMode, CrossEntropyReductionShape, MseReductionShape, L1ReductionShape, BceReductionShape, Reduction};

use alloc::vec::Vec;
use alloc::format;

macro_rules! impl_reduction_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(self) -> Result<Tensor<(), B, G>> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::from_parts_unchecked(
                inner,
                (), // Scalar shape field
                self._dtype,
                self._device,
                self._grad,
            ))
        }
    };
}

macro_rules! impl_reduction_dim_op {
    ($method:ident, $backend_method:ident, $trait_bound:ident, $keep_dim:expr) => {
        pub fn $method<const DIM: usize>(&self) -> Result<Tensor<S::Output, B, G>>
        where
            S: DynShape + crate::shapes::$trait_bound<DIM>,
        {
            let inner = B::$backend_method(&self.inner, DIM)?;

            // We just use from_dyn to construct the resulting shape field dynamically,
            // since we know it's a dimensional reduction.
            let mut out_dims = S::dims(&self._shape).into();
            if $keep_dim {
                out_dims[DIM] = 1;
            } else {
                out_dims.remove(DIM);
            }

            Ok(Tensor::from_parts_unchecked(
                inner,
                S::Output::from_dyn(&out_dims).unwrap(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            ))
        }
    };
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    impl_reduction_op!(sum_all, sum_all);
    impl_reduction_op!(mean_all, mean_all);
    impl_reduction_op!(max_all, max_all);
    impl_reduction_op!(min_all, min_all);

    impl_reduction_dim_op!(sum_dim, sum_dim, ReduceDim, false);
    impl_reduction_dim_op!(sum_keepdim, sum_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(mean_dim, mean_dim, ReduceDim, false);
    impl_reduction_dim_op!(mean_keepdim, mean_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(max_dim, max_dim, ReduceDim, false);
    impl_reduction_dim_op!(max_keepdim, max_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(min_dim, min_dim, ReduceDim, false);
    impl_reduction_dim_op!(min_keepdim, min_keepdim, ReduceKeepDim, true);
}

