//! Concatenation, stacking, repetition, and padding operations.

use crate::backend_authoring::{Backend, Execute};
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::{PadAttributes, RepeatAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::error::OperationKind;
use crate::shapes::{Dyn, DynShape, Shape, ShapeBuf, ShapeValue};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::ops::manipulation::selectors::{ConcatSelector, StackSelector};
use alloc::string::ToString;
use alloc::vec::Vec;

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, L: Layout>
    Tensor<S, B, K, G, Local, L>
{
    /// Dynamically concatenates a slice of tensors along `dim`.
    /// This is fallible at runtime if shapes mismatch or dim is out of bounds.
    pub fn try_concat_slice(
        tensors: &[&Tensor<S, B, K, G, Local, L>],
        dim: usize,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        B: Execute<op::ConcatExact> + Capabilities,
        <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    {
        if tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot concat empty list".to_string(),
            ));
        }
        let rank = tensors[0].shape_buf().rank();
        if let Some(tensor) = tensors
            .iter()
            .find(|tensor| tensor.shape_buf().rank() != rank)
        {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::RankMismatch {
                    operation: OperationKind::Concat,
                    expected: crate::shapes::RankExpectation::Exactly(rank),
                    actual: tensor.shape_buf().rank(),
                },
            ));
        }
        let dim = isize::try_from(dim)
            .ok()
            .and_then(|dim| {
                crate::shapes::idx::AxisSelector::new(&[dim])
                    .normalize(rank)
                    .ok()
                    .map(|axes| axes[0])
            })
            .ok_or(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis { axis: dim, rank },
            ))?;
        let mut out_shape = tensors[0].shape_buf().as_ref().to_vec();
        if let Some((axis, tensor)) = tensors.iter().skip(1).find_map(|tensor| {
            tensor
                .shape_buf()
                .as_ref()
                .iter()
                .enumerate()
                .find(|(axis, extent)| *axis != dim && **extent != out_shape[*axis])
                .map(|(axis, _)| (axis, *tensor))
        }) {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::DimensionMismatch {
                    operation: OperationKind::Concat,
                    axis: crate::shapes::Axis::Index(axis),
                    lhs: out_shape[axis],
                    rhs: tensor.shape_buf().as_ref()[axis],
                    constraint: crate::shapes::DimensionConstraint::Equal,
                },
            ));
        }
        out_shape[dim] = tensors.iter().try_fold(0usize, |total, tensor| {
            total.checked_add(tensor.shape_buf().as_ref()[dim]).ok_or(
                crate::shapes::ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Concat,
                    expression: "concat extent",
                },
            )
        })?;
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let inputs = tensors
            .iter()
            .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
            .collect::<Vec<_>>();
        let context = crate::tensor::grad::execution_context::<B, G>(&tensors[0]._grad);
        let inner = G::grad_mode(&tensors[0]._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ConcatExact, B, Dyn>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis: dim },
                    &inputs,
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        )
    }

    /// Concatenates two tensors along a static, named, or runtime axis.
    ///
    /// Static selectors preserve the exact shape algebra output. Runtime and
    /// named selectors preserve the input rank when the input shape carries
    /// rank information.
    #[allow(clippy::type_complexity)]
    pub fn concat<S2, A>(
        &self,
        other: &Tensor<S2, B, K, G>,
        axis: A,
    ) -> Result<crate::shapes::Dense<<A as ConcatSelector<S, S2>>::Output, B, K, G, Local>>
    where
        S2: Shape,
        A: ConcatSelector<S, S2>,
        <A as ConcatSelector<S, S2>>::Output: Shape,
        B: Execute<op::ConcatExact> + Capabilities,
        <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
        self.concat_resolved::<S2, <A as ConcatSelector<S, S2>>::Output>(other, dim)
    }

    /// Advanced structural concatenation retained for shape-proof internals.
    #[allow(clippy::type_complexity)]
    pub fn concat_structural<S2, Axis>(
        &self,
        other: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output, B, K, G>>
    where
        S2: Shape + DynShape,
        Axis: crate::shapes::idx::StaticCursor,
        S: crate::shapes::concat::ConcatShape<S2, Axis>,
        <S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output: Shape,
        B: Execute<
                op::ConcatExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let dim = crate::shapes::idx::AxisSelector::new(&[Axis::INDEX])
            .normalize(self.shape_buf().rank())?[0];
        let mut out_dims: Vec<usize> = self.shape_buf().as_ref().to_vec();
        let other_dims: Vec<usize> = other.shape_buf().as_ref().to_vec();
        out_dims[dim] = out_dims[dim].checked_add(other_dims[dim]).ok_or(
            crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "concat extent",
            },
        )?;
        let output_shape =
            ShapeValue::<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output>::try_new(
                ShapeBuf::from_slice(&out_dims),
            )
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<
                op::ConcatExact,
                B,
                <S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output,
            >(
                &context,
                crate::exec::catalog::AxisAttributes { axis: dim },
                &inputs,
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

    /// Legacy dynamic concatenation entry point. Prefer [`Self::concat`] with
    /// an `axis!` selector or signed `isize`.
    #[doc(hidden)]
    pub fn try_concat<S2>(
        &self,
        other: &Tensor<S2, B, K, G>,
        dim: usize,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        S2: Shape,
        B: Execute<op::ConcatExact> + Capabilities,
        <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    {
        self.concat_resolved::<S2, Dyn>(other, dim)
    }

    fn concat_resolved<S2, Out>(
        &self,
        other: &Tensor<S2, B, K, G>,
        dim: usize,
    ) -> Result<crate::shapes::Dense<Out, B, K, G, Local>>
    where
        S2: Shape,
        Out: Shape,
        B: Execute<op::ConcatExact> + Capabilities,
        <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    {
        let rank = self.shape_buf().rank();
        let other_rank = other.shape_buf().rank();
        if other_rank != rank {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::RankMismatch {
                    operation: OperationKind::Concat,
                    expected: crate::shapes::RankExpectation::Exactly(rank),
                    actual: other_rank,
                },
            ));
        }
        let dim = isize::try_from(dim)
            .ok()
            .and_then(|dim| {
                crate::shapes::idx::AxisSelector::new(&[dim])
                    .normalize(rank)
                    .ok()
                    .map(|axes| axes[0])
            })
            .ok_or(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis { axis: dim, rank },
            ))?;
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        if let Some((axis, rhs)) = other
            .shape_buf()
            .as_ref()
            .iter()
            .enumerate()
            .find(|(axis, rhs)| *axis != dim && **rhs != out_shape[*axis])
        {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::DimensionMismatch {
                    operation: OperationKind::Concat,
                    axis: crate::shapes::Axis::Index(axis),
                    lhs: out_shape[axis],
                    rhs: *rhs,
                    constraint: crate::shapes::DimensionConstraint::Equal,
                },
            ));
        }
        out_shape[dim] = out_shape[dim]
            .checked_add(other.shape_buf().as_ref()[dim])
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "concat extent",
            })?;
        let output_shape = ShapeValue::<Out>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ConcatExact, B, Out>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis: dim },
                    &inputs,
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

    /// Legacy signed-axis spelling. Prefer [`Self::concat`] with an `isize`.
    #[doc(hidden)]
    pub fn concat_axis<S2>(
        &self,
        other: &Tensor<S2, B, K, G>,
        axis: isize,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        S2: Shape,
        B: Execute<op::ConcatExact> + Capabilities,
        <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[axis]).normalize(self.rank())?[0];
        self.try_concat(other, axis)
    }

    /// Dynamically stacks a slice of tensors along `dim`.
    pub fn try_stack_slice(
        tensors: &[&Tensor<S, B, K, G, Local, L>],
        dim: usize,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        B: Execute<op::StackExact> + Capabilities,
        <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
    {
        if tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot stack empty list".to_string(),
            ));
        }
        let rank = tensors[0].shape_buf().rank();
        let dim = isize::try_from(dim)
            .ok()
            .and_then(|dim| {
                crate::shapes::idx::AxisSelector::new(&[dim])
                    .normalize(rank + 1)
                    .ok()
                    .map(|axes| axes[0])
            })
            .ok_or(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: dim,
                    rank: rank + 1,
                },
            ))?;
        let mut out_shape = tensors[0].shape_buf().as_ref().to_vec();
        out_shape.insert(dim, tensors.len());
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let inputs = tensors
            .iter()
            .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
            .collect::<Vec<_>>();
        let context = crate::tensor::grad::execution_context::<B, G>(&tensors[0]._grad);
        let inner = G::grad_mode(&tensors[0]._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::StackExact, B, Dyn>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis: dim },
                    &inputs,
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        )
    }

    /// Structurally inserts a size-two axis at a cursor position.
    #[doc(hidden)]
    pub fn stack_structural<Axis>(
        &self,
        other: &Tensor<S, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::stack::StackShape<Axis>>::Output, B, K, G>>
    where
        Axis: crate::shapes::idx::StaticCursor,
        S: crate::shapes::stack::StackShape<Axis>,
        <S as crate::shapes::stack::StackShape<Axis>>::Output: Shape,
        B: Execute<
                op::StackExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let dim = crate::shapes::idx::AxisSelector::new(&[Axis::INDEX])
            .normalize(self.shape_buf().rank() + 1)?[0];
        let mut out_dims: Vec<usize> = self.shape_buf().as_ref().to_vec();
        out_dims.insert(dim, 2);
        let output_shape =
            ShapeValue::<<S as crate::shapes::stack::StackShape<Axis>>::Output>::try_new(
                ShapeBuf::from_slice(&out_dims),
            )
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<
                op::StackExact,
                B,
                <S as crate::shapes::stack::StackShape<Axis>>::Output,
            >(
                &context,
                crate::exec::catalog::AxisAttributes { axis: dim },
                &inputs,
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

    /// Stacks `self` with `other` along a static, named, or signed axis selector.
    #[allow(clippy::type_complexity)]
    pub fn stack<A>(
        &self,
        other: &Tensor<S, B, K, G>,
        axis: A,
    ) -> Result<crate::shapes::Dense<<A as StackSelector<S>>::Output, B, K, G, Local>>
    where
        A: StackSelector<S>,
        <A as StackSelector<S>>::Output: Shape + DynShape,
        B: Execute<
                op::StackExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
        let mut out_dims: Vec<usize> = self.shape_buf().as_ref().to_vec();
        out_dims.insert(dim, 2);
        let output_shape =
            ShapeValue::<<A as StackSelector<S>>::Output>::try_new(ShapeBuf::from_slice(&out_dims))
                .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<op::StackExact, B, <A as StackSelector<S>>::Output>(
                &context,
                crate::exec::catalog::AxisAttributes { axis: dim },
                &inputs,
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

    /// Legacy dynamic stacking entry point. Prefer [`Self::stack`] with an
    /// `axis!` selector or signed `isize`.
    #[doc(hidden)]
    pub fn try_stack(&self, other: &Tensor<S, B, K, G>, dim: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<op::StackExact> + Capabilities,
        <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
    {
        let rank = self.shape_buf().rank() + 1;
        let dim = isize::try_from(dim)
            .ok()
            .and_then(|dim| {
                crate::shapes::idx::AxisSelector::new(&[dim])
                    .normalize(rank)
                    .ok()
                    .map(|axes| axes[0])
            })
            .ok_or(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis { axis: dim, rank },
            ))?;
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        out_shape.insert(dim, 2);
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::StackExact, B, Dyn>(
                    &context,
                    crate::exec::catalog::AxisAttributes { axis: dim },
                    &inputs,
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

    /// Repeats tensor data along each dimension according to `repeats`.
    pub fn repeat(&self, repeats: &[usize]) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        B: Capabilities + Execute<op::Repeat>,
        <B as Execute<op::Repeat>>::Output: Into<B::Storage<K>>,
    {
        let rank = self.shape_buf().len();
        if repeats.len() != rank {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::RankMismatch {
                    operation: OperationKind::Repeat,
                    expected: crate::shapes::RankExpectation::Exactly(rank),
                    actual: repeats.len(),
                },
            ));
        }
        let out_shape = self
            .shape_buf()
            .iter()
            .zip(repeats)
            .map(|(&extent, &repeat)| extent.checked_mul(repeat))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Repeat,
                    expression: "repeat output extent",
                })
            })?;
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::Repeat, B, Dyn>(
                    &context,
                    RepeatAttributes {
                        repeats: repeats.to_vec(),
                    },
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

    /// Pads tensor according to `padding` (before, after) pairs per dimension with `val`.
    pub fn pad<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        padding: &[(usize, usize)],
        val: Sc,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        B: Capabilities + Execute<op::Pad>,
        <B as Execute<op::Pad>>::Output: Into<B::Storage<K>>,
    {
        let val_f64 = val.into().to_f64();
        let out_shape = self
            .shape_buf()
            .iter()
            .zip(padding)
            .map(|(&extent, &(before, after))| {
                extent
                    .checked_add(before)
                    .and_then(|value| value.checked_add(after))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Pad,
                    expression: "padded output extent",
                })
            })?;
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::Pad, B, Dyn>(
                    &context,
                    PadAttributes {
                        padding: padding.to_vec(),
                        value: val_f64,
                    },
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

/// `try_stack_tensors`.
pub fn try_stack_tensors<
    S: Shape + DynShape,
    B: Backend + Execute<op::StackExact> + Capabilities,
    K: crate::tensor::dtype::DType,
    G: crate::tensor::grad::RequiresGrad,
    L: Layout,
>(
    tensors: &[&Tensor<S, B, K, G, Local, L>],
    dim: usize,
) -> Result<Tensor<Dyn, B, K, G>>
where
    G::Field: Clone,
    <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
{
    if tensors.is_empty() {
        return Err(crate::err::Error::ShapeMismatch {
            op: "stack_tensors",
            expected: alloc::vec![],
            got: alloc::vec![],
            msg: alloc::string::String::from("Cannot stack empty list of tensors"),
        });
    }
    let rank = tensors[0].shape_buf().rank() + 1;
    let dim = isize::try_from(dim)
        .ok()
        .and_then(|dim| {
            crate::shapes::idx::AxisSelector::new(&[dim])
                .normalize(rank)
                .ok()
                .map(|axes| axes[0])
        })
        .ok_or(crate::err::Error::Shape(
            crate::shapes::ShapeError::InvalidAxis { axis: dim, rank },
        ))?;
    let mut shape = tensors[0].shape_buf().as_ref().to_vec();
    shape.insert(dim, tensors.len());
    let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&shape))
        .map_err(crate::err::Error::Shape)?;
    let inputs = tensors
        .iter()
        .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
        .collect::<Vec<_>>();
    let context = crate::tensor::grad::execution_context::<B, G>(&tensors[0]._grad);
    let inner = G::grad_mode(&tensors[0]._grad)
        .restrict(|| {
            dispatch::execute_shaped::<op::StackExact, B, Dyn>(
                &context,
                crate::exec::catalog::AxisAttributes { axis: dim },
                &inputs,
                &output_shape,
            )
        })?
        .into();
    Tensor::<Dyn, B, K, G>::from_shape_value(
        inner,
        output_shape,
        tensors[0]._dtype.clone(),
        tensors[0]._device.clone(),
        tensors[0]._grad.clone(),
    )
}
