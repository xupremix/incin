//! Tensor reduction operations (sum, mean, max, min).
//!
//! This module provides methods to reduce tensors across all dimensions (resulting in a scalar,
//! or dimensionless tensor) as well as along specific axes. It supports both static type-level
//! dimensional reductions using `Axis` where the shape statically changes, and dynamic
//! dimensional reductions where the shape becomes `Dyn`.
use crate::prelude::{Backend, DynShape, RequiresGrad, Result, Shape, Tensor};

macro_rules! impl_reduction_op {
    (
        $(#[$meta:meta])*
        $method:ident, $backend_method:ident
    ) => {
        $(#[$meta])*
        pub fn $method(self) -> Result<Tensor<(), B, K, D, G>> {
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
    (
        $(#[$meta:meta])*
        $method:ident, $backend_method:ident, $trait_bound:ident, $keep_dim:expr
    ) => {
        $(#[$meta])*
        pub fn $method<const DIM: usize>(&self) -> Result<Tensor<S::Output, B, K, D, G>>
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

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad> Tensor<S, B, K, D, G>
{
    impl_reduction_op!(
        /// Computes the sum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let s = t.sum_all().unwrap(); // shape is ()
        /// ```
        sum_all, sum_all
    );

    impl_reduction_op!(
        /// Computes the mean of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.mean_all().unwrap(); // shape is ()
        /// ```
        mean_all, mean_all
    );

    impl_reduction_op!(
        /// Computes the maximum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.max_all().unwrap(); // shape is ()
        /// ```
        max_all, max_all
    );

    impl_reduction_op!(
        /// Computes the minimum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.min_all().unwrap(); // shape is ()
        /// ```
        min_all, min_all
    );

    impl_reduction_dim_op!(
        /// Sums the tensor along a specific dimension, removing that dimension from the shape.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let s = t.sum_dim::<0>().unwrap(); // shape is [3]
        /// ```
        sum_dim, sum_dim, ReduceDim, false
    );

    impl_reduction_dim_op!(
        /// Sums the tensor along a specific dimension, keeping it with size 1.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let s = t.sum_keepdim::<0>().unwrap(); // shape is [1, 3]
        /// ```
        sum_keepdim, sum_keepdim, ReduceKeepDim, true
    );

    impl_reduction_dim_op!(
        /// Averages the tensor along a specific dimension, removing that dimension from the shape.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.mean_dim::<0>().unwrap(); // shape is [3]
        /// ```
        mean_dim, mean_dim, ReduceDim, false
    );

    impl_reduction_dim_op!(
        /// Averages the tensor along a specific dimension, keeping it with size 1.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.mean_keepdim::<0>().unwrap(); // shape is [1, 3]
        /// ```
        mean_keepdim, mean_keepdim, ReduceKeepDim, true
    );

    impl_reduction_dim_op!(
        /// Finds the maximum along a specific dimension, removing that dimension from the shape.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.max_dim::<0>().unwrap(); // shape is [3]
        /// ```
        max_dim, max_dim, ReduceDim, false
    );

    impl_reduction_dim_op!(
        /// Finds the maximum along a specific dimension, keeping it with size 1.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.max_keepdim::<0>().unwrap(); // shape is [1, 3]
        /// ```
        max_keepdim, max_keepdim, ReduceKeepDim, true
    );

    impl_reduction_dim_op!(
        /// Finds the minimum along a specific dimension, removing that dimension from the shape.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.min_dim::<0>().unwrap(); // shape is [3]
        /// ```
        min_dim, min_dim, ReduceDim, false
    );

    impl_reduction_dim_op!(
        /// Finds the minimum along a specific dimension, keeping it with size 1.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.min_keepdim::<0>().unwrap(); // shape is [1, 3]
        /// ```
        min_keepdim, min_keepdim, ReduceKeepDim, true
    );

}

impl<S: crate::prelude::Shape + crate::prelude::DynShape, B: crate::prelude::Backend, K: crate::prelude::DType, D: crate::prelude::Device, G: crate::prelude::RequiresGrad> Tensor<S, B, K, D, G> {
    /// Computes the argmax of the tensor.
    /// If `dim` is `None`, the tensor is flattened and the argmax over the entire tensor is returned as a 0D scalar.
    /// If `dim` is `Some(d)`, the argmax is computed along that dimension.
    pub fn argmax(&self, dim: Option<usize>) -> Result<Tensor<crate::prelude::Dyn, B, u32, D, crate::prelude::NoGrad>> {
        let inner = match dim {
            Some(d) => B::argmax::<K, u32>(&self.inner, Some(d))?,
            None => {
                let rank = self.rank();
                let flat = if rank == 0 {
                    self.inner.clone()
                } else {
                    B::flatten::<K>(&self.inner, 0, rank - 1)?
                };
                B::argmax::<K, u32>(&flat, Some(0))?
            }
        };
        let mut out_dims = S::dims(&self._shape).into();
        if let Some(d) = dim {
            out_dims.remove(d);
        } else {
            out_dims = alloc::vec![];
        }
        
        Ok(Tensor::from_parts_unchecked(
            inner,
            crate::prelude::Dyn::from_dyn(&out_dims).unwrap(),
            core::marker::PhantomData,
            self._device.clone(),
            crate::prelude::NoGrad::init(()),
        ))
    }

    /// Computes the argmin of the tensor.
    /// If `dim` is `None`, the tensor is flattened and the argmin over the entire tensor is returned as a 0D scalar.
    /// If `dim` is `Some(d)`, the argmin is computed along that dimension.
    pub fn argmin(&self, dim: Option<usize>) -> Result<Tensor<crate::prelude::Dyn, B, u32, D, crate::prelude::NoGrad>> {
        let inner = match dim {
            Some(d) => B::argmin::<K, u32>(&self.inner, Some(d))?,
            None => {
                let rank = self.rank();
                let flat = if rank == 0 {
                    self.inner.clone()
                } else {
                    B::flatten::<K>(&self.inner, 0, rank - 1)?
                };
                B::argmin::<K, u32>(&flat, Some(0))?
            }
        };
        let mut out_dims = S::dims(&self._shape).into();
        if let Some(d) = dim {
            out_dims.remove(d);
        } else {
            out_dims = alloc::vec![];
        }
        
        Ok(Tensor::from_parts_unchecked(
            inner,
            crate::prelude::Dyn::from_dyn(&out_dims).unwrap(),
            core::marker::PhantomData,
            self._device.clone(),
            crate::prelude::NoGrad::init(()),
        ))
    }
}
