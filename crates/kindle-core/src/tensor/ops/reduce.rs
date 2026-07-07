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
    (
        $(#[$meta:meta])*
        $method:ident, $backend_method:ident, $trait_bound:ident, $keep_dim:expr
    ) => {
        $(#[$meta])*
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
