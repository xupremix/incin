//! Reshaping, squeezing, unsqueezing, flattening, and broadcasting operations.

use crate::backend_authoring::{Backend, Descriptor, Execute};
use crate::dist::placement::Local;
use crate::dist::Placement;
use crate::err::Result;
use crate::exec::catalog::{
    op, AxisAttributes, FlattenAttributes, LogicalTensorMeta, ShapeAttributes,
};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::exec::Capabilities;
use crate::exec::ExecutionDescriptor;
use crate::shapes::error::OperationKind;
use crate::shapes::{Dyn, DynShape, FlattenAt, Shape, ShapeBuf, ShapeSpec, ShapeValue};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::ops::manipulation::selectors::{FlattenSelector, UnsqueezeSelector};
use alloc::vec::Vec;

pub(crate) fn reshape_storage_exact<B, K>(
    storage: &B::Storage<K>,
    shape: &ShapeBuf,
) -> Result<B::Storage<K>>
where
    B: Backend + Execute<op::ReshapeExact>,
    K: DType,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
{
    let target = ShapeValue::<Dyn>::try_new(shape.clone()).map_err(crate::err::Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = ExecutionContext::from_scope(B::default());
    Ok(dispatch::execute_shaped::<op::ReshapeExact, B, Dyn>(
        &context,
        ShapeAttributes {
            shape: shape.as_ref().to_vec(),
        },
        &[input],
        &target,
    )
    .map(Into::into)?)
}

pub(crate) fn squeeze_storage_exact<B, K>(
    storage: &B::Storage<K>,
    logical_dims: &[usize],
    axis: usize,
) -> Result<B::Storage<K>>
where
    B: Backend + Capabilities + Execute<op::SqueezeExact>,
    K: DType,
    <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
{
    crate::shapes::idx::AxisSelector::normalize_unsigned(axis, logical_dims.len())?;
    if logical_dims[axis] != 1 {
        return Err(crate::err::Error::Shape(
            crate::shapes::ShapeError::DimensionMismatch {
                operation: OperationKind::Squeeze,
                axis: crate::shapes::error::Axis::Index(axis),
                lhs: logical_dims[axis],
                rhs: 1,
                constraint: crate::shapes::error::DimensionConstraint::Equal,
            },
        ));
    }
    let mut output_dims = logical_dims.to_vec();
    output_dims.remove(axis);
    let target = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&output_dims))
        .map_err(crate::err::Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = ExecutionContext::from_scope(B::default());
    Ok(dispatch::execute_shaped::<op::SqueezeExact, B, Dyn>(
        &context,
        AxisAttributes { axis },
        &[input],
        &target,
    )
    .map(Into::into)?)
}

impl<
        S: Shape + DynShape,
        B: Backend,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        P: Placement,
    > Tensor<S, B, K, G, P>
{
    /// Reshape this tensor using a [`ShapeSpec`].
    ///
    /// Fully static specifications preserve compile-time element-count
    /// checking. Runtime specifications are checked before dispatch.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape(shape![6]).unwrap();
    /// ```
    pub fn reshape<Spec>(&self, spec: Spec) -> Result<Tensor<Spec::Shape, B, K, G, P>>
    where
        Spec: ShapeSpec + crate::shapes::reshape::ReshapeSpec<S>,
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let new_shape = spec.resolve()?;
        let new_shape_field = new_shape.shape_buf().clone();
        let source_numel = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Reshape,
        )?;
        let target_numel = Spec::Shape::checked_numel(
            new_shape.shape_buf(),
            crate::shapes::error::OperationKind::Reshape,
        )?;
        if source_numel != target_numel {
            return Err(crate::err::Error::ShapeMismatch {
                op: "reshape",
                expected: alloc::vec![source_numel],
                got: alloc::vec![target_numel],
                msg: alloc::format!(
                    "Reshape failed: source numel ({}) != target numel ({})",
                    source_numel,
                    target_numel
                ),
            });
        }

        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ReshapeExact, B, Spec::Shape>(
                    &context,
                    ShapeAttributes {
                        shape: new_shape_field.as_ref().to_vec(),
                    },
                    &[input],
                    &new_shape,
                )
            })?
            .into();
        Tensor::<Spec::Shape, B, K, G, P>::from_shape_value_placed(
            inner,
            new_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Reshape using the legacy explicit `S2::Arg` representation.
    #[doc(hidden)]
    pub fn reshape_typed<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G, P>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::ReshapeShape<S2>,
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let new_shape_field = S2::resolve(args).map_err(crate::err::Error::Shape)?;
        let validated = <crate::exec::ReshapeRule as crate::exec::ShapeRule<(S, S2)>>::lower(
            &(self.shape_buf_value(), new_shape_field.clone()),
            (),
        )
        .map_err(crate::err::Error::Shape)?;
        let new_shape_field =
            validated
                .into_descriptor()
                .output_shape()
                .cloned()
                .ok_or(crate::err::Error::Shape(
                    crate::shapes::ShapeError::TargetShapeRejected {
                        operation: OperationKind::Reshape,
                        rank: S2::RANK.unwrap_or(0),
                    },
                ))?;
        let new_shape =
            ShapeValue::<S2>::try_new(new_shape_field.clone()).map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ReshapeExact, B, S2>(
                    &context,
                    ShapeAttributes {
                        shape: new_shape_field.as_ref().to_vec(),
                    },
                    &[input],
                    &new_shape,
                )
            })?
            .into();
        Tensor::<S2, B, K, G, P>::from_shape_value_placed(
            inner,
            new_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Reshapes with one runtime-inferred extent from `shape![..., infer]`.
    pub fn reshape_infer<S2>(
        &self,
        spec: crate::shapes::InferShape<S2>,
    ) -> Result<Tensor<S2, B, K, G, P>>
    where
        S2: Shape + DynShape,
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let source_numel = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Reshape,
        )?;
        let output_shape = ShapeValue::<S2>::try_new(spec.resolve(source_numel)?)
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ReshapeExact, B, S2>(
                    &context,
                    ShapeAttributes {
                        shape: output_shape.shape_buf().as_ref().to_vec(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<S2, B, K, G, P>::from_shape_value_placed(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Reshapes a tensor based on python-like slicing syntax via the `idx!` macro.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// use incin::advanced::idx;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape_idx::<idx![6]>().unwrap();
    /// ```
    pub fn reshape_idx<T: crate::shapes::idx::ReshapeTarget<S>>(
        &self,
    ) -> Result<Tensor<T::Output, B, K, G, P>>
    where
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let in_shape_vec = self.shape_buf();
        let out_shape_vec = T::calculate_shape(in_shape_vec.as_ref())?;
        let output_shape = ShapeValue::<T::Output>::try_new(ShapeBuf::from_slice(&out_shape_vec))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ReshapeExact, B, T::Output>(
                    &context,
                    ShapeAttributes {
                        shape: out_shape_vec.clone(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<T::Output, B, K, G, P>::from_shape_value_placed(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Removes the selected dimension if its size is 1.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![1, 5], DefaultBackend>::ones(()).unwrap();
    /// let sq = t.try_squeeze(0isize).unwrap(); // shape [5]
    /// ```
    pub fn try_squeeze<A>(self, axis: A) -> Result<Tensor<A::Drop, B, K, G, P>>
    where
        A: crate::tensor::ops::reduce::ReduceSelector<S>,
        A::Drop: DynShape,
        B: Capabilities + Execute<op::SqueezeExact>,
        <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
        let mut shape = self.shape_buf().as_ref().to_vec();
        let extent = *shape.get(dim).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::InvalidAxis {
                axis: dim,
                rank: shape.len(),
            })
        })?;
        if extent != 1 {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::DimensionMismatch {
                    operation: OperationKind::Squeeze,
                    axis: crate::shapes::error::Axis::Index(dim),
                    lhs: extent,
                    rhs: 1,
                    constraint: crate::shapes::error::DimensionConstraint::Equal,
                },
            ));
        }
        let input_shape = self.shape_buf();
        let inner = G::grad_mode(&self._grad)
            .restrict(|| squeeze_storage_exact::<B, K>(&self.inner, input_shape, dim))?;
        shape.remove(dim);
        Tensor::<S, B, K, G, P>::from_shape_buf_placed_checked::<A::Drop>(
            inner,
            ShapeBuf::from_slice(&shape),
            self._dtype,
            self._device,
            self._grad,
            self._placement,
        )
    }

    /// `try_reshape`.
    pub fn try_reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G, P>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::TryReshape<S2>,
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let new_shape_field = S2::resolve(args).map_err(crate::err::Error::Shape)?;

        // Runtime boundaries checking
        let source_numel = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Reshape,
        )?;
        let target_numel = S2::checked_numel(
            &new_shape_field,
            crate::shapes::error::OperationKind::Reshape,
        )?;
        if source_numel != target_numel {
            return Err(crate::err::Error::ShapeMismatch {
                op: "try_reshape",
                expected: alloc::vec![source_numel],
                got: alloc::vec![target_numel],
                msg: alloc::format!(
                    "Reshape failed: source numel ({}) != target numel ({})",
                    source_numel,
                    target_numel
                ),
            });
        }

        let output_shape =
            ShapeValue::<S2>::try_new(new_shape_field.clone()).map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ReshapeExact, B, S2>(
                    &context,
                    ShapeAttributes {
                        shape: new_shape_field.as_ref().to_vec(),
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<S2, B, K, G, P>::from_shape_value_placed(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Broadcast the tensor to the specific shape `S2`.
    pub fn broadcast_to<S2: Shape + DynShape>(
        &self,
        args: S2::Arg,
    ) -> Result<Tensor<S2, B, K, G, P>>
    where
        S: crate::shapes::broadcast::BroadcastShape<S2, Output = S2>,
        B: Execute<
                op::BroadcastAs,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let new_shape_field = S2::resolve(args).map_err(crate::err::Error::Shape)?;
        <<S as crate::shapes::broadcast::BroadcastShape<S2>>::Output as Shape>::STATIC_VALID;
        let descriptor = Descriptor::<op::BroadcastAs>::infer_runtime(
            ShapeAttributes {
                shape: new_shape_field.as_ref().to_vec(),
            },
            alloc::vec![LogicalTensorMeta {
                shape: Some(self.shape_buf_value()),
                dtype: None,
                device: None,
            }],
        )
        .map_err(|error| crate::err::Error::from(crate::exec::CanonicalError::Descriptor(error)))?
        .into_descriptor();
        let new_shape_field = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: OperationKind::Broadcast,
                rank: 0,
            })
        })?;
        let output_shape =
            ShapeValue::<S2>::try_new(new_shape_field.clone()).map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<op::BroadcastAs, B, S2>(
                &context,
                ShapeAttributes {
                    shape: new_shape_field.as_ref().to_vec(),
                },
                &[input],
                &output_shape,
            )
        })?;
        Tensor::<S2, B, K, G, P>::from_shape_value_placed(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
            self._placement.clone(),
        )
    }

    /// Expands the tensor to target shape `S2`.
    pub fn expand<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G, P>>
    where
        S: crate::shapes::broadcast::BroadcastShape<S2, Output = S2>,
        B: Execute<
                op::BroadcastAs,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        self.broadcast_to::<S2>(args)
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    /// Flattens a statically selected inclusive axis range.
    /// Flattens an axis interval selected by compile-time axis selectors.
    ///
    /// The associated output shape is kept in the public signature so the
    /// selector remains visible to type-level callers. Clippy cannot express
    /// that intent without reporting the signature as a complex type.
    #[allow(clippy::type_complexity)]
    pub fn flatten<A, BSel>(
        &self,
        start: A,
        end: BSel,
    ) -> Result<Tensor<<() as FlattenSelector<S, A, BSel>>::Output, B, K, G>>
    where
        (): FlattenSelector<S, A, BSel>,
        <() as FlattenSelector<S, A, BSel>>::Output: Shape + DynShape,
        B: Execute<
                op::FlattenExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let rank = self.shape_buf().rank();
        let (start, end) = <() as FlattenSelector<S, A, BSel>>::resolve(&(start, end), rank)?;
        if start > end {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxisRange {
                    operation: OperationKind::Flatten,
                    start,
                    end,
                    rank,
                },
            ));
        }
        let dims = self.shape_buf().as_ref();
        let product = dims[start..=end]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Flatten,
                expression: "flattened dimension product",
            })?;
        let mut out_dims = Vec::with_capacity(rank - (end - start));
        out_dims.extend_from_slice(&dims[..start]);
        out_dims.push(product);
        out_dims.extend_from_slice(&dims[end + 1..]);
        let output_shape = ShapeValue::<<() as FlattenSelector<S, A, BSel>>::Output>::try_new(
            ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<
                op::FlattenExact,
                B,
                <() as FlattenSelector<S, A, BSel>>::Output,
            >(
                &context,
                FlattenAttributes {
                    start_axis: start,
                    end_axis: end,
                },
                &[input],
                &output_shape,
            )
        })?;
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Advanced structural flatten retained for shape-proof internals.
    #[allow(clippy::type_complexity)]
    pub fn flatten_structural<Start, End>(
        &self,
    ) -> Result<Tensor<<S as FlattenAt<Start, End>>::Output, B, K, G>>
    where
        S: FlattenAt<Start, End>,
        <S as FlattenAt<Start, End>>::Output: Shape + DynShape,
        Start: crate::shapes::idx::StaticCursor,
        End: crate::shapes::idx::StaticCursor,
        B: Execute<
                op::FlattenExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let rank = self.shape_buf().rank();
        let axes =
            crate::shapes::idx::AxisSelector::new(&[Start::INDEX, End::INDEX]).normalize(rank)?;
        if axes[0] > axes[1] {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxisRange {
                    operation: OperationKind::Flatten,
                    start: axes[0],
                    end: axes[1],
                    rank,
                },
            ));
        }
        let dims_buf = self.shape_buf();
        let in_dims = dims_buf.as_ref();
        let product = in_dims[axes[0]..=axes[1]]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Flatten,
                expression: "flattened dimension product",
            })?;
        let mut out_dims = Vec::with_capacity(rank - (axes[1] - axes[0]));
        out_dims.extend_from_slice(&in_dims[..axes[0]]);
        out_dims.push(product);
        out_dims.extend_from_slice(&in_dims[axes[1] + 1..]);
        let output_shape = ShapeValue::<<S as FlattenAt<Start, End>>::Output>::try_new(
            ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<op::FlattenExact, B, <S as FlattenAt<Start, End>>::Output>(
                &context,
                crate::exec::catalog::FlattenAttributes {
                    start_axis: axes[0],
                    end_axis: axes[1],
                },
                &[input],
                &output_shape,
            )
        })?;
        Tensor::<<S as FlattenAt<Start, End>>::Output, B, K, G>::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Runtime flatten range with checked normalization and a dynamic output.
    #[doc(hidden)]
    pub fn flatten_runtime(&self, start: isize, end: isize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<
                op::FlattenExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let rank = self.shape_buf().rank();
        let axes = crate::shapes::idx::AxisSelector::new(&[start, end]).normalize(rank)?;
        let start = axes[0];
        let end = axes[1];
        if start > end || end >= rank {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxisRange {
                    operation: OperationKind::Flatten,
                    start,
                    end,
                    rank,
                },
            ));
        }
        let dims = self.shape_buf().as_ref();
        let product = dims[start..=end]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Flatten,
                expression: "flattened dimension product",
            })?;
        let mut out = alloc::vec::Vec::with_capacity(rank - (end - start));
        out.extend_from_slice(&dims[..start]);
        out.push(product);
        out.extend_from_slice(&dims[end + 1..]);
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<op::FlattenExact, B, Dyn>(
                &context,
                FlattenAttributes {
                    start_axis: start,
                    end_axis: end,
                },
                &[input],
                &output_shape,
            )
        })?;
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Flattens an inclusive signed axis range and returns a dynamic shape.
    #[doc(hidden)]
    pub fn flatten_range(&self, start: isize, end: isize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<
                op::FlattenExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        self.flatten_runtime(start, end)
    }

    /// Inserts a 1-sized dimension at the selected axis.
    pub fn unsqueeze<A>(&self, axis: A) -> Result<Tensor<A::Output, B, K, G>>
    where
        A: UnsqueezeSelector<S>,
        A::Output: DynShape,
        B: Capabilities + Execute<op::UnsqueezeExact>,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        if dim > out_shape.len() {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: dim,
                    rank: out_shape.len() + 1,
                },
            ));
        }
        out_shape.insert(dim, 1);
        let output_shape = ShapeValue::<A::Output>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::UnsqueezeExact, B, A::Output>(
                    &context,
                    AxisAttributes { axis: dim },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<A::Output, B, K, G>::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    pub(crate) fn unsqueeze_dyn(&self, dim: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::UnsqueezeExact>,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        if dim > out_shape.len() {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: dim,
                    rank: out_shape.len() + 1,
                },
            ));
        }
        out_shape.insert(dim, 1);
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::UnsqueezeExact, B, Dyn>(
                    &context,
                    AxisAttributes { axis: dim },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
