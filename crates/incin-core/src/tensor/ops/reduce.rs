//! Tensor reduction operations (sum, mean, max, min).
//!
//! This module provides methods to reduce tensors across all dimensions (resulting in a scalar,
//! or dimensionless tensor) as well as along specific axes. It supports both static type-level
//! dimensional reductions using `Axis` where the shape statically changes, and dynamic
//! dimensional reductions where the shape becomes `Dyn`.
use crate::dist::{Local, Placement};
use crate::err::Result;
use crate::exec::catalog::{
    AxisAttributes, CanonicalOperation, Descriptor, LogicalTensorMeta, Operation, op,
};
use crate::exec::context::ExecutionContext;
use crate::exec::request::TensorHandle;
use crate::exec::{ExecutionDescriptor, GradMode};
use crate::shapes::ShapeBuf;
use crate::shapes::error::OperationKind;
use crate::shapes::idx::StaticCursor;
use crate::shapes::shape_ops::{ReduceAt, ReduceKeepAt};
use crate::shapes::{DynShape, Shape};
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{DType, DTypeId};
use crate::tensor::grad::RequiresGrad;

fn reduction_descriptor<O>(shape: &ShapeBuf, axis: usize) -> Result<Descriptor<O>>
where
    O: CanonicalOperation + Operation<Attributes = AxisAttributes>,
{
    Descriptor::<O>::infer_runtime(
        AxisAttributes { axis },
        alloc::vec![LogicalTensorMeta {
            shape: Some(shape.clone()),
            dtype: None,
            device: None,
        }],
    )
    .map(|validated| validated.into_descriptor())
    .map_err(|error| crate::err::Error::from(crate::exec::CanonicalError::Descriptor(error)))
}

macro_rules! impl_reduction_op {
    (
        $(#[$meta:meta])*
        $method:ident, $operation:ident
    ) => {
        $(#[$meta])*
        pub fn $method(self) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
        where
            B: Execute<op::$operation> + crate::exec::Capabilities,
            <B as Execute<op::$operation>>::Output: Into<B::Storage<K>>,
        {
            let output_shape = crate::shapes::ShapeValue::<crate::shapes::Nil>::try_new(
                crate::shapes::ShapeBuf::scalar(),
            )
            .map_err(crate::err::Error::Shape)?;
            let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
            let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
            let inner = G::grad_mode(&self._grad)
                .restrict(|| {
                    crate::exec::dispatch::execute_shaped::<op::$operation, B, crate::shapes::Nil>(
                        &context,
                        crate::exec::catalog::NoAttributes,
                        &[input],
                        &output_shape,
                    )
                })?
                .into();
            Tensor::from_parts(
                inner,
                output_shape.shape_buf().clone(),
                self._dtype,
                self._device,
                self._grad,
            )
        }
    };
}

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, P: Placement>
    Tensor<S, B, K, G, P>
{
    /// Sums over an arbitrary signed compile-time axis.
    pub fn sum<const AXIS: isize>(&self) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        self.sum_runtime(AXIS)
    }

    /// Sums over a runtime or `axis!` numeric selector.
    pub fn sum_axis<A: crate::shapes::idx::ToAxisIndex>(
        &self,
        axis: A,
    ) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        self.sum_runtime(axis.to_axis_index())
    }

    /// Sums over a compile-time structural axis cursor.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn sum_at<C>(&self) -> Result<Tensor<<S as ReduceAt<C>>::Output, B, K, G, P>>
    where
        C: StaticCursor,
        S: DynShape + ReduceAt<C>,
        <S as ReduceAt<C>>::Output: DynShape,
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(self.shape_buf().len())?
            .into_iter()
            .next()
            .ok_or(crate::err::Error::Shape(
                crate::shapes::error::ShapeError::InvalidAxis {
                    axis: C::INDEX.unsigned_abs(),
                    rank: self.shape_buf().len(),
                },
            ))?;
        let descriptor = <crate::exec::rule::ReduceRule as crate::exec::ShapeRule<(S, C)>>::lower(
            &self.shape_buf_value(),
            crate::exec::catalog::AxisAttributes { axis },
        )
        .map_err(crate::err::Error::Shape)?
        .into_descriptor();
        let output_dims = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::SumDim,
                rank: 0,
            })
        })?;
        let output_shape =
            crate::shapes::ShapeValue::<<S as ReduceAt<C>>::Output>::try_new(output_dims)
                .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::SumDim, B, <S as ReduceAt<C>>::Output>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<<S as ReduceAt<C>>::Output, B, K, G, P>::from_shape_buf_placed(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Sums over an arbitrary signed compile-time axis while retaining it.
    pub fn sum_keepdim<const AXIS: isize>(&self) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.sum_keepdim_runtime(AXIS)
    }

    /// Sums over a runtime or `axis!` numeric selector while retaining it.
    pub fn sum_keepdim_axis<A: crate::shapes::idx::ToAxisIndex>(
        &self,
        axis: A,
    ) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.sum_keepdim_runtime(axis.to_axis_index())
    }

    /// Sums over a compile-time structural axis cursor while retaining it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn sum_keepdim_at<C>(&self) -> Result<Tensor<<S as ReduceKeepAt<C>>::Output, B, K, G, P>>
    where
        C: StaticCursor,
        S: DynShape + ReduceKeepAt<C>,
        <S as ReduceKeepAt<C>>::Output: DynShape,
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(self.shape_buf().len())?
            .into_iter()
            .next()
            .ok_or(crate::err::Error::Shape(
                crate::shapes::error::ShapeError::InvalidAxis {
                    axis: C::INDEX.unsigned_abs(),
                    rank: self.shape_buf().len(),
                },
            ))?;
        let descriptor =
            <crate::exec::rule::ReduceKeepRule as crate::exec::ShapeRule<(S, C)>>::lower(
                &self.shape_buf_value(),
                crate::exec::catalog::AxisAttributes { axis },
            )
            .map_err(crate::err::Error::Shape)?
            .into_descriptor();
        let output_dims = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::SumKeepDim,
                rank: 0,
            })
        })?;
        let output_shape =
            crate::shapes::ShapeValue::<<S as ReduceKeepAt<C>>::Output>::try_new(output_dims)
                .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<
                    op::SumKeepDim,
                    B,
                    <S as ReduceKeepAt<C>>::Output,
                >(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<<S as ReduceKeepAt<C>>::Output, B, K, G, P>::from_shape_buf_placed(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Sums one runtime-selected axis and erases only the unavailable output
    /// position facts.
    pub fn sum_runtime(&self, axis: isize) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[axis])
            .normalize(self.shape_buf().rank())?
            .into_iter()
            .next()
            .expect("one axis selector always yields one axis");
        let descriptor = reduction_descriptor::<op::SumDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction(descriptor)
    }

    /// Sums one runtime-selected axis while retaining it as a length-one axis.
    pub fn sum_keepdim_runtime(&self, axis: isize) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[axis])
            .normalize(self.shape_buf().rank())?
            .into_iter()
            .next()
            .expect("one axis selector always yields one axis");
        let descriptor = reduction_descriptor::<op::SumKeepDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction(descriptor)
    }

    /// Sums the axis identified by a semantic tag.
    ///
    /// Stable Rust cannot currently turn recursive semantic lookup into the
    /// structural `RemoveAt` output type without overlapping implementations.
    /// This honest fallback resolves the selector against the current shape
    /// and returns a runtime-rank shape. Missing and duplicate names remain
    /// typed shape errors.
    pub fn sum_named<Tag>(
        &self,
        selector: crate::shapes::idx::NamedAxisSelector<Tag>,
    ) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        Tag: crate::shapes::AxisTag,
        S: DynShape + crate::shapes::idx::NamedAxisLookup<Tag>,
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = selector.resolve::<S>()?;
        let descriptor = reduction_descriptor::<op::SumDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction(descriptor)
    }

    /// Named-axis sum retaining the selected axis as a runtime length-one
    /// dimension.
    pub fn sum_keepdim_named<Tag>(
        &self,
        selector: crate::shapes::idx::NamedAxisSelector<Tag>,
    ) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        Tag: crate::shapes::AxisTag,
        S: DynShape + crate::shapes::idx::NamedAxisLookup<Tag>,
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        let axis = selector.resolve::<S>()?;
        let descriptor = reduction_descriptor::<op::SumKeepDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction(descriptor)
    }

    fn execute_named_reduction<O>(
        &self,
        descriptor: Descriptor<O>,
    ) -> Result<Tensor<crate::shapes::Dyn, B, K, G, P>>
    where
        O: crate::exec::catalog::CanonicalOperation
            + crate::exec::catalog::Operation<Attributes = crate::exec::catalog::AxisAttributes>,
        B: Execute<O> + crate::exec::Capabilities,
        <B as Execute<O>>::Output: Into<B::Storage<K>>,
    {
        let axis = descriptor.attributes().axis;
        let output_dims = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: O::ID,
                rank: 0,
            })
        })?;
        let output_shape = crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(output_dims)
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<O, B, crate::shapes::Dyn>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<crate::shapes::Dyn, B, K, G, P>::from_shape_buf_placed(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }
}

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G, Local>
{
    impl_reduction_op!(
        /// Computes the sum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust
        /// # extern crate incin_core as incin;
        /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
        /// use incin::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let s = t.sum_all().unwrap(); // shape is ()
        /// ```
        sum_all, SumAll
    );

    impl_reduction_op!(
        /// Computes the mean of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust
        /// # extern crate incin_core as incin;
        /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
        /// use incin::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.mean_all().unwrap(); // shape is ()
        /// ```
        mean_all, MeanAll
    );

    impl_reduction_op!(
        /// Computes the maximum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust
        /// # extern crate incin_core as incin;
        /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
        /// use incin::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.max_all().unwrap(); // shape is ()
        /// ```
        max_all, MaxAll
    );

    impl_reduction_op!(
        /// Computes the minimum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust
        /// # extern crate incin_core as incin;
        /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
        /// use incin::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
        /// let m = t.min_all().unwrap(); // shape is ()
        /// ```
        min_all, MinAll
    );

    impl_reduction_op!(
        /// Product of all elements in the tensor, reducing it to a scalar tensor.
        prod_all, ProdAll
    );

    /// Cumulative sum along a compile-time structural axis cursor.
    pub fn cumsum<C: StaticCursor>(&self) -> Result<Self>
    where
        S: DynShape,
        B: Execute<op::Cumsum> + crate::exec::Capabilities,
        <B as Execute<op::Cumsum>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(self.rank())?
            .into_iter()
            .next()
            .ok_or(crate::err::Error::Shape(
                crate::shapes::error::ShapeError::InvalidAxis {
                    axis: C::INDEX.unsigned_abs(),
                    rank: self.rank(),
                },
            ))?;
        let output_shape = crate::shapes::ShapeValue::<S>::try_new(self.shape_buf().clone())
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::Cumsum, B, S>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Computes the vector p-norm (`p` norm: 1.0 = L1, 2.0 = L2) over all elements.
    pub fn norm(&self, p: f64) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
    where
        G: crate::tensor::grad::GradJoin<G, Output = G>,
        B: Execute<op::Mul>
            + Execute<op::Abs>
            + Execute<op::Sqrt>
            + Execute<op::SumAll>
            + Execute<op::Powf>
            + crate::exec::Capabilities,
        <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Abs>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Sqrt>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Powf>>::Output: Into<B::Storage<K>>,
    {
        if (p - 1.0).abs() < 1e-6 {
            self.abs()?.sum_all()
        } else if (p - 2.0).abs() < 1e-6 {
            let sq = self.mul_exact(self)?;
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
    S: crate::shapes::Shape + crate::shapes::DynShape,
    B: crate::tensor::backend::Backend + Execute<op::Sub> + Execute<op::Mul>,
    K: crate::tensor::dtype::DType,
    G: crate::tensor::grad::RequiresGrad,
> Tensor<S, B, K, G>
where
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
{
    /// Computes argmax along a compile-time numeric axis.
    pub fn argmax<const AXIS: isize>(
        &self,
    ) -> Result<Tensor<crate::shapes::Dyn, B, u32, crate::tensor::grad::NoGrad>>
    where
        B: Execute<op::ArgMax> + crate::exec::Capabilities,
        <B as Execute<op::ArgMax>>::Output: Into<B::Storage<u32>>,
    {
        self.argmax_runtime(Some(AXIS))
    }

    /// Computes argmax over a runtime-selected axis.
    #[doc(hidden)]
    pub fn argmax_runtime(
        &self,
        dim: Option<isize>,
    ) -> Result<Tensor<crate::shapes::Dyn, B, u32, crate::tensor::grad::NoGrad>>
    where
        B: Execute<op::ArgMax> + crate::exec::Capabilities,
        <B as Execute<op::ArgMax>>::Output: Into<B::Storage<u32>>,
    {
        let normalized = dim
            .map(|d| {
                crate::shapes::idx::AxisSelector::new(&[d])
                    .normalize(self.rank())
                    .map(|axes| axes[0])
            })
            .transpose()?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        if let Some(d) = normalized {
            out_dims.remove(d);
        } else {
            out_dims = alloc::vec![];
        }
        let output_shape = crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(
            crate::shapes::ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
        let inner = GradMode::Disabled
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::ArgMax, B, crate::shapes::Dyn>(
                    &context,
                    crate::exec::catalog::IndexReductionAttributes {
                        axis: normalized,
                        dtype: DTypeId::U32.descriptor(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();

        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            u32::init(()),
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )
    }

    /// Computes argmin along a compile-time signed axis.
    pub fn argmin<const AXIS: isize>(
        &self,
    ) -> Result<Tensor<crate::shapes::Dyn, B, u32, crate::tensor::grad::NoGrad>>
    where
        B: Execute<op::ArgMin> + crate::exec::Capabilities,
        <B as Execute<op::ArgMin>>::Output: Into<B::Storage<u32>>,
    {
        self.argmin_runtime(Some(AXIS))
    }

    /// Computes argmin over a runtime-selected axis.
    #[doc(hidden)]
    pub fn argmin_runtime(
        &self,
        dim: Option<isize>,
    ) -> Result<Tensor<crate::shapes::Dyn, B, u32, crate::tensor::grad::NoGrad>>
    where
        B: Execute<op::ArgMin> + crate::exec::Capabilities,
        <B as Execute<op::ArgMin>>::Output: Into<B::Storage<u32>>,
    {
        let normalized = dim
            .map(|d| {
                crate::shapes::idx::AxisSelector::new(&[d])
                    .normalize(self.rank())
                    .map(|axes| axes[0])
            })
            .transpose()?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        if let Some(d) = normalized {
            out_dims.remove(d);
        } else {
            out_dims = alloc::vec![];
        }
        let output_shape = crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(
            crate::shapes::ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
        let inner = GradMode::Disabled
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::ArgMin, B, crate::shapes::Dyn>(
                    &context,
                    crate::exec::catalog::IndexReductionAttributes {
                        axis: normalized,
                        dtype: DTypeId::U32.descriptor(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();

        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            u32::init(()),
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )
    }

    /// Computes the top `k` elements of the tensor along the given dimension.
    /// Returns a tuple of `(values, indices)`.
    #[allow(clippy::type_complexity)]
    pub fn topk(
        &self,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        Tensor<crate::shapes::Dyn, B, K, crate::tensor::grad::NoGrad>,
        Tensor<crate::shapes::Dyn, B, u32, crate::tensor::grad::NoGrad>,
    )>
    where
        B: Execute<op::TopK> + crate::exec::Capabilities,
        <B as Execute<op::TopK>>::Output: Into<(B::Storage<K>, B::Storage<u32>)>,
    {
        let rank = self.rank();
        let dim = crate::shapes::idx::AxisSelector::normalize_unsigned(dim, rank)?;
        let extent = self.shape_buf().as_ref()[dim];
        if k > extent {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::DimensionMismatch {
                    operation: OperationKind::Reduction,
                    axis: crate::shapes::error::Axis::Index(dim),
                    lhs: extent,
                    rhs: k,
                    constraint: crate::shapes::error::DimensionConstraint::AtLeast,
                },
            ));
        }
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims[dim] = k;
        let out_shape = crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(
            crate::shapes::ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
        let (values_inner, indices_inner) = GradMode::Disabled
            .restrict(|| {
                crate::exec::dispatch::execute::<op::TopK, B>(
                    &context,
                    crate::exec::catalog::TopKAttributes {
                        k,
                        axis: dim,
                        largest,
                        index_dtype: DTypeId::U32.descriptor(),
                    },
                    &[input],
                )
            })?
            .into();

        let values = Tensor::from_parts(
            values_inner,
            out_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )?;
        let indices = Tensor::from_parts(
            indices_inner,
            out_shape.shape_buf().clone(),
            core::marker::PhantomData,
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )?;
        Ok((values, indices))
    }

    /// Sorts the elements of the tensor along the given dimension and returns the sorted indices.
    pub fn argsort(
        &self,
        dim: usize,
        descending: bool,
    ) -> Result<Tensor<S, B, u32, crate::tensor::grad::NoGrad>>
    where
        B: Execute<op::Argsort> + crate::exec::Capabilities,
        <B as Execute<op::Argsort>>::Output: Into<B::Storage<u32>>,
    {
        // Disabled rather than this tensor's own mode: the result is `NoGrad`
        // whatever the receiver was, and sec. 1.2.5 makes that a statement
        // about what runs, not only about which APIs the result offers.
        let axis = crate::shapes::idx::AxisSelector::normalize_unsigned(dim, self.rank())?;
        let output_shape = crate::shapes::ShapeValue::<S>::try_new(self.shape_buf().clone())
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
        let indices_inner = GradMode::Disabled
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::Argsort, B, S>(
                    &context,
                    crate::exec::catalog::ArgsortAttributes {
                        axis,
                        descending,
                        index_dtype: DTypeId::U32.descriptor(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            indices_inner,
            output_shape,
            core::marker::PhantomData,
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )
    }
}

// -------------------------------------------------------------------------
// Variance and Standard Deviation Reducers
// -------------------------------------------------------------------------

impl<
    S: crate::shapes::Shape + crate::shapes::DynShape,
    B: crate::tensor::backend::Backend
        + Execute<op::Sub>
        + Execute<op::Mul>
        + Execute<op::Sqrt>
        + Execute<op::MeanAll>
        + Execute<op::SumAll>
        + Execute<op::MulScalar>
        + crate::exec::Capabilities,
    K: crate::tensor::dtype::DType,
    G: crate::tensor::grad::RequiresGrad + crate::tensor::grad::GradJoin<G, Output = G>,
> Tensor<S, B, K, G>
where
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sqrt>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MeanAll>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
{
    /// Computes the variance over all elements.
    pub fn var_all(&self, unbiased: bool) -> Result<Tensor<crate::shapes::Nil, B, K, G>> {
        let mean = self.clone().mean_all()?;
        let dyn_self = self.clone().into_dyn();
        let dyn_mean = mean.into_dyn();
        let diff = dyn_self.broadcast_sub(&dyn_mean)?;
        let sq_diff = diff.mul_exact(&diff)?;
        let sum_sq = sq_diff.sum_all()?;

        let n = self.shape_buf().numel().unwrap_or(0) as f32;
        let denom = if unbiased {
            if n <= 1.0 { 0.0 } else { n - 1.0 }
        } else {
            n
        };
        let scalar = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        sum_sq.mul_scalar(scalar)
    }

    /// Computes the standard deviation over all elements.
    pub fn std_all(&self, unbiased: bool) -> Result<Tensor<crate::shapes::Nil, B, K, G>> {
        self.var_all(unbiased)?.sqrt()
    }
}
