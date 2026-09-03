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
use crate::shapes::Layout;
use crate::shapes::ShapeBuf;
use crate::shapes::StaticCursor;
use crate::shapes::error::OperationKind;
use crate::shapes::shape_ops::{ReduceAt, ReduceKeepAt};
use crate::shapes::{DynShape, RuntimeRankProjection, Shape};
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{DType, DTypeId};
use crate::tensor::grad::RequiresGrad;

/// A user-facing reduction selector with a statically known or runtime axis.
///
/// Static selectors retain the shape algebra's output type. Runtime integer
/// selectors intentionally use `Dyn`; known-rank callers can use the ranked
/// methods below when they need rank preservation independent of the axis
/// position.
pub trait ReduceSelector<S: Shape> {
    /// Shape after removing the selected axis.
    type Drop: Shape;
    /// Shape after retaining the selected axis as extent one.
    type Keep: Shape;

    /// Resolve the selector against the input rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

impl<S, C> ReduceSelector<S> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape + DynShape + ReduceAt<C> + ReduceKeepAt<C>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as ReduceAt<C>>::Output: Shape,
    <S as ReduceKeepAt<C>>::Output: Shape,
{
    type Drop = <S as ReduceAt<C>>::Output;
    type Keep = <S as ReduceKeepAt<C>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Shape(crate::shapes::error::ShapeError::InvalidAxis {
                    axis: C::INDEX.unsigned_abs(),
                    rank,
                })
            })
    }
}

impl<S, C> ReduceSelector<S> for crate::shapes::idx::ReverseAxis<C>
where
    S: Shape
        + DynShape
        + ReduceAt<crate::shapes::idx::FromEnd<C>>
        + ReduceKeepAt<crate::shapes::idx::FromEnd<C>>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as ReduceAt<crate::shapes::idx::FromEnd<C>>>::Output: Shape,
    <S as ReduceKeepAt<crate::shapes::idx::FromEnd<C>>>::Output: Shape,
{
    type Drop = <S as ReduceAt<crate::shapes::idx::FromEnd<C>>>::Output;
    type Keep = <S as ReduceKeepAt<crate::shapes::idx::FromEnd<C>>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[crate::shapes::idx::FromEnd::<C>::INDEX])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Shape(crate::shapes::error::ShapeError::InvalidAxis {
                    axis: crate::shapes::idx::FromEnd::<C>::INDEX.unsigned_abs(),
                    rank,
                })
            })
    }
}

impl<S> ReduceSelector<S> for isize
where
    S: Shape + RuntimeRankProjection,
{
    type Drop = S::Drop;
    type Keep = S::Keep;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Shape(crate::shapes::error::ShapeError::InvalidAxis {
                    axis: self.unsigned_abs(),
                    rank,
                })
            })
    }
}

impl<S, Tag> ReduceSelector<S> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + DynShape + RuntimeRankProjection + crate::shapes::idx::NamedAxisLookup<Tag>,
    Tag: crate::shapes::AxisTag,
{
    type Drop = S::Drop;
    type Keep = S::Keep;

    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

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
        pub fn $method(
            self,
        ) -> Result<crate::shapes::Dense<crate::shapes::Nil, B, K, G, Local>>
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

/// Reductions.
///
/// Generic over the operand's layout. Every result here is a freshly allocated
/// dense buffer, so every signature returns [`Dense`](crate::shapes::Dense) --
/// the layout is *stated*, never carried. That distinction is what `cumsum`
/// makes visible: it preserves the shape, so returning `Self` typechecked and
/// silently handed the operand's claim to a buffer that had nothing to do with
/// it. Pinned by `a_reduction_result_is_dense_even_from_a_strided_operand`,
/// which feeds a `transpose_view` result and would have caught the old
/// signature.
impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, P: Placement, L: Layout>
    Tensor<S, B, K, G, P, L>
{
    /// Sums over a static, named, or runtime axis selector.
    pub fn sum<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Drop, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::SumDim, A::Drop>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Sums over a compile-time structural axis cursor.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn sum_at<C>(&self) -> Result<crate::shapes::Dense<<S as ReduceAt<C>>::Output, B, K, G, P>>
    where
        C: crate::shapes::idx::AxisCursor,
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
        Tensor::<<S as ReduceAt<C>>::Output, B, K, G, P>::from_shape_buf_placed::<
            <S as ReduceAt<C>>::Output,
            crate::shapes::RowMajor<<S as ReduceAt<C>>::Output>,
        >(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Sums over a static, named, or runtime axis selector while retaining it.
    pub fn sum_keepdim<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Keep, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::SumKeepDim, A::Keep>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes the mean over a static, named, or runtime axis selector.
    pub fn mean<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Drop, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MeanDim> + crate::exec::Capabilities,
        <B as Execute<op::MeanDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MeanDim, A::Drop>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes the mean over a static, named, or runtime axis selector and retains it.
    pub fn mean_keepdim<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Keep, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MeanKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::MeanKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MeanKeepDim, A::Keep>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes the maximum over a static, named, or runtime axis selector.
    pub fn max<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Drop, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MaxDim> + crate::exec::Capabilities,
        <B as Execute<op::MaxDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MaxDim, A::Drop>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes the maximum over a static, named, or runtime axis selector and retains it.
    pub fn max_keepdim<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Keep, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MaxKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::MaxKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MaxKeepDim, A::Keep>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes `log(sum(exp(x)))` over an axis without ever forming `exp(x)`.
    ///
    /// The naive spelling is unusable at the magnitudes this is wanted for. A
    /// single entry above roughly 88 sends `exp` to infinity in f32, and a row
    /// that sits far enough below zero sends the whole sum to zero and the
    /// logarithm to negative infinity. Both are ordinary sizes for a router
    /// logit or an unnormalized log-likelihood. Shifting by the axis maximum
    /// first bounds every exponential to `(0, 1]`, so the sum lies between one
    /// and the axis length, and adding the maximum back recovers the answer.
    ///
    /// This is the normalizer [`Self::log_softmax`] subtracts, exposed on its
    /// own because an auxiliary loss usually wants the normalizer rather than
    /// the normalized values.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![3], DefaultBackend>::from_slice(&[300.0, 300.0, 300.0], ()).unwrap();
    /// let total = t.logsumexp(0).unwrap().to_vec1::<f32>().unwrap()[0];
    /// // ln(3 * e^300) = 300 + ln(3). Summing the exponentials directly would
    /// // have overflowed to infinity three times over before the logarithm.
    /// assert!((total - (300.0 + 3.0f32.ln())).abs() < 1e-2);
    /// ```
    pub fn logsumexp<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Drop, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::LogSumExpDim> + crate::exec::Capabilities,
        <B as Execute<op::LogSumExpDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::LogSumExpDim, A::Drop>(axis.resolve(self.shape_buf().rank())?)
    }

    /// [`Self::logsumexp`] over a static, named, or runtime axis selector,
    /// retaining the reduced axis as size one.
    pub fn logsumexp_keepdim<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Keep, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::LogSumExpKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::LogSumExpKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::LogSumExpKeepDim, A::Keep>(
            axis.resolve(self.shape_buf().rank())?,
        )
    }

    /// Computes the minimum over a static, named, or runtime axis selector.
    pub fn min<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Drop, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MinDim> + crate::exec::Capabilities,
        <B as Execute<op::MinDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MinDim, A::Drop>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Computes the minimum over a static, named, or runtime axis selector and retains it.
    pub fn min_keepdim<A>(&self, axis: A) -> Result<crate::shapes::Dense<A::Keep, B, K, G, P>>
    where
        A: ReduceSelector<S>,
        B: Execute<op::MinKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::MinKeepDim>>::Output: Into<B::Storage<K>>,
    {
        self.execute_reduction::<op::MinKeepDim, A::Keep>(axis.resolve(self.shape_buf().rank())?)
    }

    /// Sums over a compile-time structural axis cursor while retaining it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn sum_keepdim_at<C>(
        &self,
    ) -> Result<crate::shapes::Dense<<S as ReduceKeepAt<C>>::Output, B, K, G, P>>
    where
        C: crate::shapes::idx::AxisCursor,
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
        Tensor::<<S as ReduceKeepAt<C>>::Output, B, K, G, P>::from_shape_buf_placed::<
            <S as ReduceKeepAt<C>>::Output,
            crate::shapes::RowMajor<<S as ReduceKeepAt<C>>::Output>,
        >(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    fn execute_reduction<O, Out>(
        &self,
        axis: usize,
    ) -> Result<crate::shapes::Dense<Out, B, K, G, P>>
    where
        Out: crate::shapes::Shape,
        O: crate::exec::catalog::CanonicalOperation
            + crate::exec::catalog::Operation<Attributes = crate::exec::catalog::AxisAttributes>,
        B: Execute<O> + crate::exec::Capabilities,
        <B as Execute<O>>::Output: Into<B::Storage<K>>,
    {
        self.execute_named_reduction_as::<O, Out>(reduction_descriptor::<O>(
            &self.shape_buf_value(),
            axis,
        )?)
    }

    fn execute_named_reduction_as<O, Out>(
        &self,
        descriptor: Descriptor<O>,
    ) -> Result<crate::shapes::Dense<Out, B, K, G, P>>
    where
        Out: crate::shapes::Shape,
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
        let output_shape = crate::shapes::ShapeValue::<Out>::try_new(output_dims)
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<O, B, Out>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<Out, B, K, G, P>::from_shape_buf_placed::<Out, crate::shapes::RowMajor<Out>>(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }
}

impl<
    S: crate::shapes::Shape
        + crate::shapes::DynShape
        + crate::shapes::RemoveOneRank
        + crate::shapes::PreserveRank,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    P: Placement,
    L: Layout,
> Tensor<S, B, K, G, P, L>
where
    <S as crate::shapes::RemoveOneRank>::Output: crate::shapes::Shape,
    <S as crate::shapes::PreserveRank>::Output: crate::shapes::Shape,
{
    /// Reduces one runtime axis while retaining the known rank in the type.
    /// The output dimensions remain runtime values, but the rank changes from
    /// `Ranked<R>` to its `RemoveOneRank::Output`.
    #[doc(hidden)]
    pub fn sum_runtime_ranked(
        &self,
        axis: isize,
    ) -> Result<crate::shapes::Dense<<S as crate::shapes::RemoveOneRank>::Output, B, K, G, P>>
    where
        B: Execute<op::SumDim> + crate::exec::Capabilities,
        <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    {
        let axis =
            crate::shapes::idx::AxisSelector::new(&[axis]).normalize(self.shape_buf().rank())?[0];
        let descriptor = reduction_descriptor::<op::SumDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction_as::<op::SumDim, <S as crate::shapes::RemoveOneRank>::Output>(
            descriptor,
        )
    }

    /// Reduces one runtime axis while preserving the known rank and axis
    /// positions in the type.
    #[doc(hidden)]
    pub fn sum_keepdim_runtime_ranked(
        &self,
        axis: isize,
    ) -> Result<crate::shapes::Dense<<S as crate::shapes::PreserveRank>::Output, B, K, G, P>>
    where
        B: Execute<op::SumKeepDim> + crate::exec::Capabilities,
        <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    {
        let axis =
            crate::shapes::idx::AxisSelector::new(&[axis]).normalize(self.shape_buf().rank())?[0];
        let descriptor = reduction_descriptor::<op::SumKeepDim>(&self.shape_buf_value(), axis)?;
        self.execute_named_reduction_as::<
            op::SumKeepDim,
            <S as crate::shapes::PreserveRank>::Output,
        >(descriptor)
    }
}

/// Whole-tensor reductions.
///
/// Generic over the operand's layout. Each collapses to a scalar, so the
/// result describes a different geometry; the fresh allocation is dense and
/// the macro states so rather than carrying anything through.
impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, L: Layout>
    Tensor<S, B, K, G, Local, L>
{
    impl_reduction_op!(
        /// Computes the sum of all elements in the tensor, reducing it to a scalar tensor.
        ///
        /// # Examples
        /// ```rust
        /// # extern crate incin_core as incin;
        /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
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
        /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
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
        /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
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
        /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
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

    /// Computes a cumulative sum along a static, named, or signed runtime axis.
    pub fn cumsum<A: ReduceSelector<S>>(
        &self,
        axis: A,
    ) -> Result<crate::shapes::Dense<S, B, K, G, Local>>
    where
        S: DynShape,
        B: Execute<op::Cumsum> + crate::exec::Capabilities,
        <B as Execute<op::Cumsum>>::Output: Into<B::Storage<K>>,
    {
        let axis = axis.resolve(self.rank())?;
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
    pub fn norm(&self, p: f64) -> Result<crate::shapes::Dense<crate::shapes::Nil, B, K, G, Local>>
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
    fn index_reduce<O, Out>(
        &self,
        dim: Option<isize>,
    ) -> Result<Tensor<Out, B, u32, crate::tensor::grad::NoGrad>>
    where
        O: CanonicalOperation
            + Operation<Attributes = crate::exec::catalog::IndexReductionAttributes>,
        Out: crate::shapes::Shape + crate::shapes::DynShape,
        B: Execute<O> + crate::exec::Capabilities,
        <B as Execute<O>>::Output: Into<B::Storage<u32>>,
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
        let output_shape = crate::shapes::ShapeValue::<Out>::try_new(
            crate::shapes::ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
        let inner = GradMode::Disabled
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<O, B, Out>(
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

    /// Computes argmax along a static, named, or runtime axis selector.
    pub fn argmax<A>(&self, axis: A) -> Result<Tensor<A::Drop, B, u32, crate::tensor::grad::NoGrad>>
    where
        A: ReduceSelector<S>,
        A::Drop: crate::shapes::DynShape,
        B: Execute<op::ArgMax> + crate::exec::Capabilities,
        <B as Execute<op::ArgMax>>::Output: Into<B::Storage<u32>>,
    {
        self.index_reduce::<op::ArgMax, A::Drop>(Some(axis.resolve(self.rank())? as isize))
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
        self.index_reduce::<op::ArgMax, crate::shapes::Dyn>(dim)
    }

    /// Computes argmin along a static, named, or runtime axis selector.
    pub fn argmin<A>(&self, axis: A) -> Result<Tensor<A::Drop, B, u32, crate::tensor::grad::NoGrad>>
    where
        A: ReduceSelector<S>,
        A::Drop: crate::shapes::DynShape,
        B: Execute<op::ArgMin> + crate::exec::Capabilities,
        <B as Execute<op::ArgMin>>::Output: Into<B::Storage<u32>>,
    {
        self.index_reduce::<op::ArgMin, A::Drop>(Some(axis.resolve(self.rank())? as isize))
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
        self.index_reduce::<op::ArgMin, crate::shapes::Dyn>(dim)
    }

    /// Computes the top `k` elements of the tensor along the given dimension.
    /// Returns a tuple of `(values, indices)`.
    #[allow(clippy::type_complexity)]
    pub fn topk<A>(
        &self,
        k: usize,
        axis: A,
        largest: bool,
    ) -> Result<(
        Tensor<
            <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output,
            B,
            K,
            crate::tensor::grad::NoGrad,
        >,
        Tensor<
            <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output,
            B,
            u32,
            crate::tensor::grad::NoGrad,
        >,
    )>
    where
        A: crate::tensor::ops::manipulation::ReplaceAxisSelector<S>,
        <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output:
            crate::shapes::DynShape,
        B: Execute<op::TopK> + crate::exec::Capabilities,
        <B as Execute<op::TopK>>::Output: Into<(B::Storage<K>, B::Storage<u32>)>,
    {
        let rank = self.rank();
        let dim = axis.resolve(rank)?;
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
        let out_shape = crate::shapes::ShapeValue::<
            <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output,
        >::try_new(crate::shapes::ShapeBuf::from_slice(&out_dims))
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

        let values = Tensor::<
            <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output,
            B,
            K,
            crate::tensor::grad::NoGrad,
        >::from_parts(
            values_inner,
            out_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )?;
        let indices = Tensor::<
            <A as crate::tensor::ops::manipulation::ReplaceAxisSelector<S>>::Output,
            B,
            u32,
            crate::tensor::grad::NoGrad,
        >::from_parts(
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
    pub fn var_all(
        &self,
        unbiased: bool,
    ) -> Result<crate::shapes::Dense<crate::shapes::Nil, B, K, G, Local>> {
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
    pub fn std_all(
        &self,
        unbiased: bool,
    ) -> Result<crate::shapes::Dense<crate::shapes::Nil, B, K, G, Local>> {
        self.var_all(unbiased)?.sqrt()
    }
}
