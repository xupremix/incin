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
        pub fn $method(self) -> Result<Tensor<(), B, K, G>> {
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
        pub fn $method<const DIM: usize>(&self) -> Result<Tensor<S::Output, B, K, G>>
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

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad> Tensor<S, B, K, G> {
    impl_reduction_op!(
        /// Computes the sum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
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
        /// use incin::prelude::*;
        /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
        /// let m = t.min_keepdim::<0>().unwrap(); // shape is [1, 3]
        /// ```
        min_keepdim, min_keepdim, ReduceKeepDim, true
    );

    impl_reduction_op!(
        /// Product of all elements in the tensor, reducing it to a scalar tensor.
        prod_all, prod_all
    );

    impl_reduction_dim_op!(
        /// Product along a specific dimension, removing that dimension.
        prod_dim, prod_dim, ReduceDim, false
    );

    /// Cumulative sum along dimension `DIM`.
    pub fn cumsum<const DIM: usize>(&self) -> Result<Self>
    where
        S: DynShape,
    {
        let inner = B::cumsum::<K>(&self.inner, DIM)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Computes the vector p-norm (`p` norm: 1.0 = L1, 2.0 = L2) over all elements.
    pub fn norm(&self, p: f64) -> Result<Tensor<(), B, K, G>>
    where
        B: crate::tensor::backend::FloatOps<B>,
    {
        if (p - 1.0).abs() < 1e-6 {
            self.abs()?.sum_all()
        } else if (p - 2.0).abs() < 1e-6 {
            let sq = self.mul(self)?;
            sq.sum_all()?.sqrt()
        } else {
            let abs_t = self.abs()?;
            let pow_t = abs_t.powf(p)?;
            let sum_t = pow_t.sum_all()?;
            sum_t.powf(1.0 / p)
        }
    }
}

impl<
    S: crate::prelude::Shape + crate::prelude::DynShape,
    B: crate::prelude::Backend,
    K: crate::prelude::DType,
    G: crate::prelude::RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Computes the argmax of the tensor.
    /// If `dim` is `None`, the tensor is flattened and the argmax over the entire tensor is returned as a 0D scalar.
    /// If `dim` is `Some(d)`, the argmax is computed along that dimension.
    pub fn argmax(
        &self,
        dim: Option<usize>,
    ) -> Result<Tensor<crate::prelude::Dyn, B, u32, crate::prelude::NoGrad>> {
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
    pub fn argmin(
        &self,
        dim: Option<usize>,
    ) -> Result<Tensor<crate::prelude::Dyn, B, u32, crate::prelude::NoGrad>> {
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

    /// Computes the top `k` elements of the tensor along the given dimension.
    /// Returns a tuple of `(values, indices)`.
    pub fn topk(
        &self,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        Tensor<crate::prelude::Dyn, B, K, crate::prelude::NoGrad>,
        Tensor<crate::prelude::Dyn, B, u32, crate::prelude::NoGrad>,
    )> {
        let (values_inner, indices_inner) = B::topk::<K, u32>(&self.inner, k, dim, largest)?;
        let mut out_dims = S::dims(&self._shape).into();
        out_dims[dim] = k;
        let out_shape = crate::prelude::Dyn::from_dyn(&out_dims).unwrap();

        let values = Tensor::from_parts_unchecked(
            values_inner,
            out_shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            crate::prelude::NoGrad::init(()),
        );
        let indices = Tensor::from_parts_unchecked(
            indices_inner,
            out_shape,
            core::marker::PhantomData,
            self._device.clone(),
            crate::prelude::NoGrad::init(()),
        );
        Ok((values, indices))
    }

    /// Sorts the elements of the tensor along the given dimension and returns the sorted indices.
    pub fn argsort(
        &self,
        dim: usize,
        descending: bool,
    ) -> Result<Tensor<S, B, u32, crate::prelude::NoGrad>> {
        let indices_inner = B::argsort::<K, u32>(&self.inner, dim, descending)?;
        Ok(Tensor::from_parts_unchecked(
            indices_inner,
            self._shape.clone(),
            core::marker::PhantomData,
            self._device.clone(),
            crate::prelude::NoGrad::init(()),
        ))
    }
}

// -------------------------------------------------------------------------
// Variance and Standard Deviation Reducers
// -------------------------------------------------------------------------

impl<
    S: crate::prelude::Shape + crate::shapes::DynShape,
    B: crate::prelude::Backend
        + crate::tensor::backend::ReductionOps<B>
        + crate::tensor::backend::FloatOps<B>,
    K: crate::prelude::DType,
    G: crate::prelude::RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Computes the variance over all elements.
    pub fn var_all(&self, unbiased: bool) -> Result<Tensor<(), B, K, G>> {
        let mean = self.clone().mean_all()?;
        let dyn_self = self.clone().into_dyn();
        let dyn_mean = mean.into_dyn();
        let diff = dyn_self.broadcast_sub(&dyn_mean)?;
        let sq_diff = diff.mul(&diff)?;
        let sum_sq = sq_diff.sum_all()?;

        let n = S::numel(&self._shape) as f32;
        let denom = if unbiased {
            if n <= 1.0 { 0.0 } else { n - 1.0 }
        } else {
            n
        };
        let scalar = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        sum_sq.mul_scalar(scalar)
    }

    /// Computes the standard deviation over all elements.
    pub fn std_all(&self, unbiased: bool) -> Result<Tensor<(), B, K, G>> {
        self.var_all(unbiased)?.sqrt()
    }

    /// Computes the variance along a specific dimension, removing that dimension.
    pub fn var_dim<const DIM: usize>(
        &self,
        unbiased: bool,
    ) -> Result<Tensor<<S as crate::shapes::ReduceDim<DIM>>::Output, B, K, G>>
    where
        S: crate::shapes::DynShape
            + crate::shapes::ReduceDim<DIM>
            + crate::shapes::ReduceKeepDim<DIM>,
        <S as crate::shapes::ReduceDim<DIM>>::Output: crate::shapes::DynShape,
        <S as crate::shapes::ReduceKeepDim<DIM>>::Output: crate::shapes::DynShape,
    {
        let mean = self.mean_keepdim::<DIM>()?;
        let dyn_self = self.clone().into_dyn();
        let dyn_mean = mean.into_dyn();
        let diff = dyn_self.broadcast_sub(&dyn_mean)?;
        let sq_diff = diff.mul(&diff)?;
        let sum_sq = sq_diff.sum_dim::<DIM>()?;

        let dims = S::dims(&self._shape);
        let n = dims.as_ref()[DIM] as f32;
        let denom = if unbiased {
            if n <= 1.0 { 0.0 } else { n - 1.0 }
        } else {
            n
        };
        let scalar = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        let res = sum_sq.mul_scalar(scalar)?;
        res.into_shape::<<S as crate::shapes::ReduceDim<DIM>>::Output>()
    }

    /// Computes the standard deviation along a specific dimension, removing that dimension.
    pub fn std_dim<const DIM: usize>(
        &self,
        unbiased: bool,
    ) -> Result<Tensor<<S as crate::shapes::ReduceDim<DIM>>::Output, B, K, G>>
    where
        S: crate::shapes::DynShape
            + crate::shapes::ReduceDim<DIM>
            + crate::shapes::ReduceKeepDim<DIM>,
        <S as crate::shapes::ReduceDim<DIM>>::Output: crate::shapes::DynShape,
        <S as crate::shapes::ReduceKeepDim<DIM>>::Output: crate::shapes::DynShape,
    {
        self.var_dim::<DIM>(unbiased)?.sqrt()
    }

    /// Computes the variance along a specific dimension, keeping it with size 1.
    pub fn var_keepdim<const DIM: usize>(
        &self,
        unbiased: bool,
    ) -> Result<Tensor<<S as crate::shapes::ReduceKeepDim<DIM>>::Output, B, K, G>>
    where
        S: crate::shapes::DynShape + crate::shapes::ReduceKeepDim<DIM>,
        <S as crate::shapes::ReduceKeepDim<DIM>>::Output: crate::shapes::DynShape,
    {
        let mean = self.mean_keepdim::<DIM>()?;
        let dyn_self = self.clone().into_dyn();
        let dyn_mean = mean.into_dyn();
        let diff = dyn_self.broadcast_sub(&dyn_mean)?;
        let sq_diff = diff.mul(&diff)?;
        let sum_sq = sq_diff.sum_keepdim::<DIM>()?;

        let dims = S::dims(&self._shape);
        let n = dims.as_ref()[DIM] as f32;
        let denom = if unbiased {
            if n <= 1.0 { 0.0 } else { n - 1.0 }
        } else {
            n
        };
        let scalar = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        let res = sum_sq.mul_scalar(scalar)?;
        res.into_shape::<<S as crate::shapes::ReduceKeepDim<DIM>>::Output>()
    }

    /// Computes the standard deviation along a specific dimension, keeping it with size 1.
    pub fn std_keepdim<const DIM: usize>(
        &self,
        unbiased: bool,
    ) -> Result<Tensor<<S as crate::shapes::ReduceKeepDim<DIM>>::Output, B, K, G>>
    where
        S: crate::shapes::DynShape + crate::shapes::ReduceKeepDim<DIM>,
        <S as crate::shapes::ReduceKeepDim<DIM>>::Output: crate::shapes::DynShape,
    {
        self.var_keepdim::<DIM>(unbiased)?.sqrt()
    }
}
