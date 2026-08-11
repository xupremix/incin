//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor
//! without necessarily changing the underlying data. It includes reshaping, transposition,
//! squeezing, flattening, and broadcasting. These operations heavily leverage the
//! compile-time type system to ensure the resulting shapes are strictly valid.
use crate::backend_authoring::{Descriptor, Execute};
use crate::dist::Placement;
use crate::dist::placement::Local;
use crate::exec::Capabilities;
use crate::exec::ExecutionDescriptor;
use crate::exec::catalog::{
    AxisAttributes, DTypeAttributes, DiagonalAttributes, DuplicateIndexRule, EpsilonAttributes,
    FlattenAttributes, GroupNormAttributes, LogicalTensorMeta, NarrowAttributes, NoAttributes,
    PadAttributes, PixelShuffleAttributes, Pool2dAttributes, RepeatAttributes, ScalarAttributes,
    ScatterAttributes, ShapeAttributes, SliceAttributes, TransposeAttributes, UnfoldAttributes, op,
};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::prelude::{
    Backend, DType, Dyn, DynShape, RequiresGrad, Result, Shape, SupportsDType, Tensor, TransferTo,
};
use crate::shapes::error::OperationKind;
use crate::shapes::idx::StaticCursor;
use crate::shapes::shape::shape_buf_from_dims;
use crate::shapes::{FlattenAt, SwapAxes};
use crate::shapes::{ShapeBuf, ShapeValue};
use crate::tensor::backend::{FloatOps, NumericOps, TensorOps};
use crate::tensor::ops::*;

use alloc::string::ToString;
use alloc::vec::Vec;

fn is_valid_scalar_type<E: 'static>() -> bool {
    let tid = core::any::TypeId::of::<E>();
    tid == core::any::TypeId::of::<bool>()
        || tid == core::any::TypeId::of::<u8>()
        || tid == core::any::TypeId::of::<u16>()
        || tid == core::any::TypeId::of::<u32>()
        || tid == core::any::TypeId::of::<u64>()
        || tid == core::any::TypeId::of::<usize>()
        || tid == core::any::TypeId::of::<i8>()
        || tid == core::any::TypeId::of::<i16>()
        || tid == core::any::TypeId::of::<i32>()
        || tid == core::any::TypeId::of::<i64>()
        || tid == core::any::TypeId::of::<isize>()
        || tid == core::any::TypeId::of::<f32>()
        || tid == core::any::TypeId::of::<f64>()
        || tid == core::any::TypeId::of::<half::f16>()
        || tid == core::any::TypeId::of::<half::bf16>()
}

pub(crate) fn reshape_storage_exact<B, K>(
    storage: &B::Storage<K>,
    shape: &ShapeBuf,
) -> Result<B::Storage<K>>
where
    B: Backend + Execute<op::ReshapeExact>,
    K: DType,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
{
    let target = ShapeValue::<Dyn>::try_new(shape.clone()).map_err(crate::prelude::Error::Shape)?;
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

fn narrow_storage_exact<B, K>(
    storage: &B::Storage<K>,
    logical_dims: &[usize],
    axis: usize,
    start: usize,
    length: usize,
) -> Result<B::Storage<K>>
where
    B: Backend + Capabilities + Execute<op::Narrow>,
    K: DType,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
{
    crate::shapes::idx::AxisSelector::normalize_unsigned(axis, logical_dims.len())?;
    let dim_len = logical_dims[axis];
    let end = start.checked_add(length).ok_or(crate::err::Error::Shape(
        crate::shapes::ShapeError::ArithmeticOverflow {
            operation: OperationKind::Narrow,
            expression: "start + length",
        },
    ))?;
    if end > dim_len {
        return Err(crate::err::Error::Shape(
            crate::shapes::ShapeError::InvalidAxisRange {
                operation: OperationKind::Narrow,
                start,
                end,
                rank: dim_len,
            },
        ));
    }
    let mut output_dims = logical_dims.to_vec();
    output_dims[axis] = length;
    let target = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&output_dims))
        .map_err(crate::prelude::Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = ExecutionContext::from_scope(B::default());
    Ok(dispatch::execute_shaped::<op::Narrow, B, Dyn>(
        &context,
        NarrowAttributes {
            axis,
            start,
            length,
        },
        &[input],
        &target,
    )
    .map(Into::into)?)
}

fn squeeze_storage_exact<B, K>(
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
        .map_err(crate::prelude::Error::Shape)?;
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

/// Whether `E` is the exact Rust type the tensor's `dtype` stores.
///
/// The extraction below reads the tensor's bytes through a `*const E`, so
/// agreeing on a byte *width* is not enough: `f32` and `u32` are both four
/// bytes wide, and reading one as the other reinterprets the bit pattern
/// instead of converting it. `1.0f32` extracted as `u32` returned
/// `1065353216` rather than reporting a mismatch, which is a wrong answer
/// with no error attached to it.
///
/// `bool` is deliberately absent. It is not a stored dtype at all; both
/// callers handle it before reaching here, as a per-element truthy test
/// rather than a reinterpret.
///
/// `Q8_0` is also absent, and matches nothing: a block-quantized element has
/// no scalar Rust type to be read as without dequantizing first.
fn scalar_type_matches_dtype<E: 'static>(dtype: crate::tensor::dtype::DTypeDescriptor) -> bool {
    use crate::tensor::dtype::DTypeId;
    let tid = core::any::TypeId::of::<E>();
    match dtype.builtin_id() {
        Some(DTypeId::U8) => tid == core::any::TypeId::of::<u8>(),
        Some(DTypeId::U32) => tid == core::any::TypeId::of::<u32>(),
        Some(DTypeId::I64) => tid == core::any::TypeId::of::<i64>(),
        Some(DTypeId::BF16) => tid == core::any::TypeId::of::<half::bf16>(),
        Some(DTypeId::F16) => tid == core::any::TypeId::of::<half::f16>(),
        Some(DTypeId::F32) => tid == core::any::TypeId::of::<f32>(),
        Some(DTypeId::F64) => tid == core::any::TypeId::of::<f64>(),
        Some(DTypeId::Bool) => tid == core::any::TypeId::of::<bool>(),
        _ => false,
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    /// Slices a tensor dynamically based on a slice of `IndexSpec` configurations.
    /// Returns a dynamically shaped tensor (`Dyn`).
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![3, 3], DefaultBackend>::ones(()).unwrap();
    /// let s = t.slice(&[IndexSpec::All, IndexSpec::Index(0)]).unwrap();
    /// ```
    pub fn slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::Narrow> + Execute<op::SqueezeExact>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        self.dyn_slice(specs)
    }

    /// Ergonomic slicing and indexing API using `IndexArgs`.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let tensor = Tensor::<s![2, 4, 4], DefaultBackend>::ones(())?;
    /// let sliced = tensor.get((0, 1..3, ..))?;
    /// # Ok(()) }
    /// ```
    pub fn get<I: crate::tensor::ops::index::IndexArgs>(
        &self,
        index: I,
    ) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::Narrow> + Execute<op::SqueezeExact>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        self.dyn_slice(&index.into_specs())
    }

    /// Internal alias for `slice`.
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::Narrow> + Execute<op::SqueezeExact>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    {
        let current_dims = self.shape_buf();
        if specs.len() > current_dims.as_ref().len() {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Too many slicing specs ({}) for tensor of rank {}",
                specs.len(),
                current_dims.as_ref().len()
            )));
        }

        // One scope around the whole walk rather than one per axis: a slice
        // over three axes is three backend calls, and the mode they run under
        // is a property of this tensor, not of an iteration.
        let mut logical_dims = current_dims.as_ref().to_vec();
        let inner = self.under_grad_mode(|| -> Result<B::Storage<K>> {
            let mut inner = self.inner.clone();
            let mut dim = 0;
            for spec in specs {
                let dim_len = logical_dims[dim];

                let resolve = |idx: isize| -> Result<usize> {
                    if idx < 0 {
                        let magnitude = idx.unsigned_abs();
                        if magnitude > dim_len {
                            return Err(crate::err::Error::Shape(
                                crate::shapes::ShapeError::InvalidAxis {
                                    axis: magnitude,
                                    rank: dim_len,
                                },
                            ));
                        }
                        Ok(dim_len - magnitude)
                    } else {
                        let index = idx as usize;
                        if index > dim_len {
                            return Err(crate::err::Error::Shape(
                                crate::shapes::ShapeError::InvalidAxis {
                                    axis: index,
                                    rank: dim_len,
                                },
                            ));
                        }
                        Ok(index)
                    }
                };

                match spec {
                    IndexSpec::All => {}
                    IndexSpec::Range(start, end) => {
                        let r_start = resolve(*start)?;
                        let r_end = resolve(*end)?;
                        if r_start > r_end {
                            return Err(crate::err::Error::Shape(
                                crate::shapes::ShapeError::InvalidAxisRange {
                                    operation: OperationKind::Slice,
                                    start: r_start,
                                    end: r_end,
                                    rank: dim_len,
                                },
                            ));
                        }
                        let length = r_end - r_start;
                        inner = narrow_storage_exact::<B, K>(
                            &inner,
                            &logical_dims,
                            dim,
                            r_start,
                            length,
                        )?;
                        logical_dims[dim] = length;
                    }
                    IndexSpec::RangeFrom(start) => {
                        let r_start = resolve(*start)?;
                        let len = dim_len - r_start;
                        inner =
                            narrow_storage_exact::<B, K>(&inner, &logical_dims, dim, r_start, len)?;
                        logical_dims[dim] = len;
                    }
                    IndexSpec::RangeTo(end) => {
                        let r_end = resolve(*end)?;
                        inner = narrow_storage_exact::<B, K>(&inner, &logical_dims, dim, 0, r_end)?;
                        logical_dims[dim] = r_end;
                    }
                    IndexSpec::Index(idx) => {
                        let r_idx = resolve(*idx)?;
                        if r_idx == dim_len {
                            return Err(crate::err::Error::Shape(
                                crate::shapes::ShapeError::InvalidAxis {
                                    axis: r_idx,
                                    rank: dim_len,
                                },
                            ));
                        }
                        inner = narrow_storage_exact::<B, K>(&inner, &logical_dims, dim, r_idx, 1)?;
                        logical_dims[dim] = 1;
                        inner = squeeze_storage_exact::<B, K>(&inner, &logical_dims, dim)?;
                        logical_dims.remove(dim);
                        continue;
                    }
                }
                dim += 1;
            }
            Ok(inner)
        })?;

        Tensor::from_parts(
            inner,
            ShapeBuf::from_slice(&logical_dims),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<
    S: Shape + DynShape,
    B: Backend + Execute<op::MaxPool2d>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
where
    B: Capabilities,
    <B as Execute<op::MaxPool2d>>::Output: Into<B::Storage<K>>,
{
    /// Functional `max_pool2d` operation.
    pub fn max_pool2d<KShape, SShape, Pool, Dilation>(
        &self,
    ) -> Result<
        Tensor<<S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output, B, K, G>,
    >
    where
        KShape: typenum::Unsigned,
        SShape: typenum::Unsigned,
        Pool: typenum::Unsigned,
        Dilation: typenum::Unsigned,
        S: crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>,
        <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output: Shape,
    {
        let shape =
            <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::compute_output_shape(
                &self.shape_buf_value(),
            )?;
        let shape = ShapeValue::<
            <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output,
        >::try_new(shape)
        .map_err(crate::err::Error::Shape)?;
        let inputs = [TensorHandle::from_storage::<B, K, Local>(&self.inner)];
        let context = ExecutionContext::from_scope(B::default());
        let out = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<
                    op::MaxPool2d,
                    B,
                    <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output,
                >(
                    &context,
                    Pool2dAttributes {
                        kernel: [KShape::USIZE; 2],
                        stride: [SShape::USIZE; 2],
                        padding: [Pool::USIZE; 2],
                        dilation: [Dilation::USIZE; 2],
                    },
                    &inputs,
                    &shape,
                )
            })?
            .into();
        Tensor::from_parts(
            out,
            shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

// -------------------------------------------------------------
// Structural Ops (Reshape, Broadcast, Transpose, Flatten)
// -------------------------------------------------------------

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, P: Placement>
    Tensor<S, B, K, G, P>
{
    /// Reshape this tensor into explicitly provided shape `S2`.
    /// This is guaranteed at compile-time to have matching elements.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape::<s![6]>(((), ())).unwrap();
    /// ```
    pub fn reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G, P>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::ReshapeShape<S2>,
        B: Execute<op::ReshapeExact> + Capabilities,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    {
        let new_shape_field = S2::resolve(args).map_err(crate::prelude::Error::Shape)?;
        let new_shape = ShapeValue::<S2>::try_new(new_shape_field.clone())
            .map_err(crate::prelude::Error::Shape)?;

        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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

    /// Reshapes a tensor based on python-like slicing syntax via the `idx!` macro.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
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
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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

    /// Slices a tensor based on python-like slicing syntax via the `idx!` macro.
    pub fn slice_idx<T: crate::shapes::idx::SliceTarget<S>>(
        &self,
    ) -> Result<Tensor<T::Output, B, K, G, P>>
    where
        B: Capabilities + Execute<op::SliceExact>,
        <B as Execute<op::SliceExact>>::Output: Into<B::Storage<K>>,
    {
        let in_shape_vec = self.shape_buf();
        let ranges = T::calculate_bounds(in_shape_vec.as_ref());
        if ranges.len() != in_shape_vec.as_ref().len() {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::RankMismatch {
                    operation: OperationKind::Slice,
                    expected: crate::shapes::error::RankExpectation::Exactly(
                        in_shape_vec.as_ref().len(),
                    ),
                    actual: ranges.len(),
                },
            ));
        }
        for (axis, &(start, end)) in ranges.iter().enumerate() {
            let extent = in_shape_vec.as_ref()[axis];
            if start > end || end > extent {
                return Err(crate::err::Error::Shape(
                    crate::shapes::ShapeError::InvalidAxisRange {
                        operation: OperationKind::Slice,
                        start,
                        end,
                        rank: extent,
                    },
                ));
            }
        }
        let mut out_shape_vec = Vec::new();
        for &(start, end) in &ranges {
            out_shape_vec.push(end - start);
        }

        let output_shape = ShapeValue::<T::Output>::try_new(shape_buf_from_dims::<T::Output>(
            OperationKind::Slice,
            &out_shape_vec,
        )?)
        .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::SliceExact, B, T::Output>(
                    &context,
                    SliceAttributes {
                        ranges: ranges.clone(),
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

    /// Narrows the tensor dynamically, returning a tensor with `Dyn` shape.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![10], DefaultBackend>::ones(()).unwrap();
    /// let n = t.try_narrow(0, 2, 5).unwrap(); // shape [5]
    /// ```
    pub fn try_narrow(self, dim: usize, start: usize, len: usize) -> Result<Tensor<Dyn, B, K, G, P>>
    where
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let mut shape = self.shape_buf().as_ref().to_vec();
        let extent = *shape.get(dim).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::InvalidAxis {
                axis: dim,
                rank: shape.len(),
            })
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Slice,
                expression: "narrow start + length",
            })
        })?;
        if end > extent {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxisRange {
                    operation: OperationKind::Slice,
                    start,
                    end,
                    rank: extent,
                },
            ));
        }
        let input_shape = self.shape_buf();
        let inner = self.under_grad_mode(|| {
            narrow_storage_exact::<B, K>(&self.inner, input_shape, dim, start, len)
        })?;
        shape[dim] = len;
        Tensor::<S, B, K, G, P>::from_shape_buf_placed_checked::<Dyn>(
            inner,
            ShapeBuf::from_slice(&shape),
            self._dtype,
            self._device,
            self._grad,
            self._placement,
        )
    }

    /// Squeezes the tensor dynamically by removing the dimension `dim` if its size is 1.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![1, 5], DefaultBackend>::ones(()).unwrap();
    /// let sq = t.try_squeeze(0).unwrap(); // shape [5]
    /// ```
    pub fn try_squeeze(self, dim: usize) -> Result<Tensor<Dyn, B, K, G, P>>
    where
        B: Capabilities + Execute<op::SqueezeExact>,
        <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    {
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
        let inner =
            self.under_grad_mode(|| squeeze_storage_exact::<B, K>(&self.inner, input_shape, dim))?;
        shape.remove(dim);
        Tensor::<S, B, K, G, P>::from_shape_buf_placed_checked::<Dyn>(
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
        let new_shape_field = S2::resolve(args).map_err(crate::prelude::Error::Shape)?;

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
                expected: alloc::vec![source_numel], // We use numels here
                got: alloc::vec![target_numel],
                msg: alloc::format!(
                    "Reshape failed: source numel ({}) != target numel ({})",
                    source_numel,
                    target_numel
                ),
            });
        }

        let output_shape = ShapeValue::<S2>::try_new(new_shape_field.clone())
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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
        let new_shape_field = S2::resolve(args).map_err(crate::prelude::Error::Shape)?;
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
        .map_err(|error| {
            crate::prelude::Error::from(crate::exec::CanonicalError::Descriptor(error))
        })?
        .into_descriptor();
        let new_shape_field = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::prelude::Error::Shape(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: OperationKind::Broadcast,
                rank: 0,
            })
        })?;
        let output_shape = ShapeValue::<S2>::try_new(new_shape_field.clone())
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self.under_grad_mode(|| {
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
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    /// `to_dtype`.
    pub fn to_dtype<T2: crate::tensor::dtype::DType<Arg = ()>>(&self) -> Result<Tensor<S, B, T2, G>>
    where
        B: Execute<op::ToDType> + Capabilities,
        <B as Execute<op::ToDType>>::Output: Into<B::Storage<T2>>,
    {
        let field = T2::init(());
        let descriptor = T2::descriptor(&field);
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::ToDType, B, S>(
                    &context,
                    DTypeAttributes { dtype: descriptor },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<
    S: Shape + DynShape,
    B: Backend + TensorOps<B> + FloatOps<B> + NumericOps<B>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Extracts a single scalar value from a 0D or 1D tensor.
    /// This will bring the tensor data to the CPU and read the bytes.
    ///
    /// `bool` is handled as a truthy (any-nonzero-byte) conversion rather
    /// than a raw reinterpret, regardless of whether the tensor's actual
    /// dtype element size happens to match `size_of::<bool>()`: `bool` has
    /// only two valid bit patterns (`0x00`/`0x01`), and there is no
    /// `DTypeId::Bool` (ONNX-style boolean tensors are stored as another
    /// dtype, typically `U8`, and read out via this truthy conversion), so
    /// reinterpreting an arbitrary stored byte as `bool` via
    /// `read_unaligned` would be undefined behavior whenever that byte
    /// isn't `0` or `1`.
    pub fn to_scalar<E: Copy + 'static>(&self) -> Result<E> {
        if !is_valid_scalar_type::<E>() {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Invalid target scalar type for tensor extraction: {:?}",
                core::any::type_name::<E>()
            )));
        }

        let bytes = B::to_bytes(&self.inner)?;
        let dtype = self.dtype();

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to scalar. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            if bytes.is_empty() {
                return Err(crate::err::Error::Msg(
                    "cannot convert an empty tensor to a bool scalar".into(),
                ));
            }
            let byte = bytes[0];
            let val = match byte {
                0 => false,
                1 => true,
                other => {
                    return Err(crate::err::Error::Msg(alloc::format!(
                        "Invalid boolean storage byte: expected 0 or 1, found {}",
                        other
                    )));
                }
            };
            // SAFETY: `E` is verified to be exactly `bool` above.
            return Ok(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_size = dtype.encoding().scalar_bytes().unwrap_or(0);
        if bytes.len() != elem_size || elem_size != expected_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to scalar. Tensor dtype {:?} ({} bytes) vs requested type ({} bytes)",
                dtype,
                bytes.len(),
                elem_size
            )));
        }
        // SAFETY: `E` is verified to be a primitive scalar numeric type and bytes.len() == elem_size.
        let val = unsafe { core::ptr::read_unaligned(bytes.as_ptr() as *const E) };
        Ok(val)
    }

    /// Extracts a 1D vector of scalars from this tensor.
    pub fn to_vec1<E: Copy + 'static>(&self) -> Result<alloc::vec::Vec<E>> {
        if !is_valid_scalar_type::<E>() {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Invalid target scalar type for vector extraction: {:?}",
                core::any::type_name::<E>()
            )));
        }

        let bytes = B::to_bytes(&self.inner)?;
        let num_elements = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Storage,
        )?;
        let dtype = self.dtype();

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to vec. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            if bytes.len() != num_elements {
                return Err(crate::err::Error::Msg(alloc::format!(
                    "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                    bytes.len(),
                    num_elements
                )));
            }
            let mut out = alloc::vec::Vec::with_capacity(num_elements);
            for &byte in &bytes {
                let val = match byte {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(crate::err::Error::Msg(alloc::format!(
                            "Invalid boolean storage byte: expected 0 or 1, found {}",
                            other
                        )));
                    }
                };
                // SAFETY: `E` is verified to be exactly `bool` above.
                out.push(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
            }
            return Ok(out);
        }

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to vec. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_elem_size = dtype.encoding().scalar_bytes().unwrap_or(0);
        if elem_size != expected_elem_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Element size mismatch converting to vec: Tensor dtype {:?} element size {} vs requested type size {}",
                dtype,
                expected_elem_size,
                elem_size
            )));
        }
        let expected_bytes = num_elements * elem_size;
        if bytes.len() != expected_bytes {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                bytes.len(),
                expected_bytes
            )));
        }
        let mut out = alloc::vec::Vec::with_capacity(num_elements);
        for chunk in bytes.chunks_exact(elem_size) {
            // SAFETY: `E` is verified to be a primitive scalar type above and chunk is elem_size bytes.
            let val = unsafe { core::ptr::read_unaligned(chunk.as_ptr() as *const E) };
            out.push(val);
        }
        Ok(out)
    }

    /// Transposes two compile-time structural axis cursors.
    pub fn transpose<L, R>(&self) -> Result<Tensor<<S as SwapAxes<L, R>>::Output, B, K, G>>
    where
        L: StaticCursor,
        R: StaticCursor,
        S: SwapAxes<L, R>,
        <S as SwapAxes<L, R>>::Output: Shape + DynShape,
        B: Execute<op::TransposeExact> + Capabilities,
        <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    {
        let axes = crate::shapes::idx::AxisSelector::new(&[L::INDEX, R::INDEX])
            .normalize(self.shape_buf().rank())?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(axes[0], axes[1]);
        let output_shape =
            ShapeValue::<<S as SwapAxes<L, R>>::Output>::try_new(ShapeBuf::from_slice(&out_dims))
                .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::TransposeExact, B, <S as SwapAxes<L, R>>::Output>(
                    &context,
                    TransposeAttributes {
                        first: axes[0],
                        second: axes[1],
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<<S as SwapAxes<L, R>>::Output, B, K, G>::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Runtime-selector transpose. Selectors are normalized before any
    /// backend operation and the result intentionally carries only `Dyn`.
    pub fn transpose_runtime(&self, left: isize, right: isize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<op::TransposeExact> + Capabilities,
        <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    {
        let axes = crate::shapes::idx::AxisSelector::new(&[left, right])
            .normalize(self.shape_buf().rank())?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(axes[0], axes[1]);
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_dims))
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::TransposeExact, B, Dyn>(
                    &context,
                    TransposeAttributes {
                        first: axes[0],
                        second: axes[1],
                    },
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

    /// Flattens a range selected by structural cursors.
    pub fn flatten<Start, End>(
        &self,
    ) -> Result<Tensor<<S as FlattenAt<Start, End>>::Output, B, K, G>>
    where
        S: FlattenAt<Start, End>,
        <S as FlattenAt<Start, End>>::Output: Shape + DynShape,
        Start: StaticCursor,
        End: StaticCursor,
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
        .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self.under_grad_mode(|| {
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
    pub fn flatten_runtime(&self, start: usize, end: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<
                op::FlattenExact,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        let rank = self.shape_buf().rank();
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
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self.under_grad_mode(|| {
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

    /// Dynamically concatenates a slice of tensors along `dim`.
    /// This is fallible at runtime if shapes mismatch or dim is out of bounds.
    pub fn try_concat_slice(
        tensors: &[&Tensor<S, B, K, G>],
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>>
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
        out_shape[dim] = tensors.iter().try_fold(0usize, |total, tensor| {
            total.checked_add(tensor.shape_buf().as_ref()[dim]).ok_or(
                crate::shapes::ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Concat,
                    expression: "concat extent",
                },
            )
        })?;
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = tensors
            .iter()
            .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
            .collect::<Vec<_>>();
        let context = ExecutionContext::from_scope(B::default());
        let inner = tensors[0]
            .under_grad_mode(|| {
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

    /// Structurally concatenates along a cursor axis, preserving the exact
    /// recursive shape output at arbitrary rank.
    pub fn concat<S2, Axis>(
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
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self.under_grad_mode(|| {
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

    /// Dynamically concatenates `self` with `other` along `dim`.
    pub fn try_concat<S2>(
        &self,
        other: &Tensor<S2, B, K, G>,
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>>
    where
        S2: Shape,
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
        out_shape[dim] = out_shape[dim]
            .checked_add(other.shape_buf().as_ref()[dim])
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "concat extent",
            })?;
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Dynamically stacks a slice of tensors along `dim`.
    pub fn try_stack_slice(
        tensors: &[&Tensor<S, B, K, G>],
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>>
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
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = tensors
            .iter()
            .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
            .collect::<Vec<_>>();
        let context = ExecutionContext::from_scope(B::default());
        let inner = tensors[0]
            .under_grad_mode(|| {
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
    pub fn stack<Axis>(
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
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self.under_grad_mode(|| {
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

    /// Dynamically stacks `self` with `other` along `dim`.
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
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&other.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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

    /// Fills elements where `mask` is true with `value`.
    pub fn masked_fill<S2: Shape, G2: RequiresGrad, Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        mask: &Tensor<S2, B, bool, G2>,
        value: Sc,
    ) -> Result<Self>
    where
        S: ShapeEq<S2>,
        B: Execute<op::MaskedFill>,
        <B as Execute<op::MaskedFill>>::Output: Into<B::Storage<K>>,
    {
        let val_f64 = value.into().to_f64();
        self.under_grad_mode(|| {
            execute_masked_fill_descriptor::<S, S2, B, K, G, G2>(self, mask, val_f64)
        })
    }

    /// Gathers values along `dim` specified by `index`.
    pub fn gather<S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<S2, B, K, G>>
    where
        B: Execute<op::Gather> + Capabilities,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Gather, B, S2>(
                    &context,
                    AxisAttributes { axis: dim },
                    &inputs,
                    &index._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            index._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Scatters `src` values along `dim` into `self` using `index`.
    pub fn scatter<
        S2: Shape,
        S3: Shape,
        KInt: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        G3: RequiresGrad,
    >(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
        src: &Tensor<S3, B, K, G3>,
    ) -> Result<Self>
    where
        S2: ShapeEq<S3>,
        B: Execute<op::Scatter> + Capabilities,
        <B as Execute<op::Scatter>>::Output: Into<B::Storage<K>>,
    {
        <S2 as ShapeEq<S3>>::ASSERT_SHAPES_MATCH;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
            TensorHandle::from_storage::<B, K, Local>(&src.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Scatter, B, S>(
                    &context,
                    ScatterAttributes {
                        axis: dim,
                        duplicate_indices: DuplicateIndexRule::LastWriteWins,
                    },
                    &inputs,
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Selects slices along `dim` given 1D `index`.
    pub fn index_select<S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<op::IndexSelect> + Capabilities,
        <B as Execute<op::IndexSelect>>::Output: Into<B::Storage<K>>,
    {
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        if dim >= out_shape.len() {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: dim,
                    rank: out_shape.len(),
                },
            ));
        }
        if index.shape_buf().as_ref().len() != 1 {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::RankMismatch {
                    operation: OperationKind::IndexSelect,
                    expected: crate::shapes::error::RankExpectation::Exactly(1),
                    actual: index.shape_buf().as_ref().len(),
                },
            ));
        }
        out_shape[dim] = index.shape_buf().as_ref()[0];
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::prelude::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::IndexSelect, B, Dyn>(
                    &context,
                    AxisAttributes { axis: dim },
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

    /// Inserts a 1-sized dimension at position `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor<Dyn, B, K, G>>
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
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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

    /// Repeats tensor data along each dimension according to `repeats`.
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor<Dyn, B, K, G>>
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
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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
    ) -> Result<Tensor<Dyn, B, K, G>>
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
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
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

    /// Returns upper triangular part of matrix.
    pub fn triu(&self, k: i64) -> Result<Self>
    where
        B: Execute<op::Triu> + Capabilities,
        <B as Execute<op::Triu>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Triu, B, S>(
                    &context,
                    DiagonalAttributes { offset: k },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Returns lower triangular part of matrix.
    pub fn tril(&self, k: i64) -> Result<Self>
    where
        B: Execute<op::Tril> + Capabilities,
        <B as Execute<op::Tril>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Tril, B, S>(
                    &context,
                    DiagonalAttributes { offset: k },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Extracts or constructs diagonal tensor.
    pub fn diag(&self, k: i64) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Execute<op::Diag> + Capabilities,
        <B as Execute<op::Diag>>::Output: Into<B::Storage<K>>,
    {
        let descriptor = Descriptor::<op::Diag>::infer_runtime(
            DiagonalAttributes { offset: k },
            alloc::vec![LogicalTensorMeta {
                shape: Some(self.shape_buf().clone()),
                dtype: None,
                device: None,
            }],
        )
        .map_err(|error| {
            crate::prelude::Error::from(crate::exec::CanonicalError::Descriptor(error))
        })?
        .into_descriptor();
        let out_shape = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::prelude::Error::Shape(crate::shapes::ShapeError::TargetShapeRejected {
                operation: OperationKind::Diag,
                rank: self.rank(),
            })
        })?;
        let output_shape =
            ShapeValue::<Dyn>::try_new(out_shape).map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Diag, B, Dyn>(
                    &context,
                    DiagonalAttributes { offset: k },
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

    /// Splits tensor into `chunks` equal parts along `dim`.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<alloc::vec::Vec<Tensor<Dyn, B, K, G>>>
    where
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let dim_size = *self.shape_buf().as_ref().get(dim).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::InvalidAxis {
                axis: dim,
                rank: self.shape_buf().rank(),
            })
        })?;
        if chunks == 0 {
            return Err(crate::err::Error::Msg(
                "chunk expects positive number of chunks".into(),
            ));
        }
        let chunk_size = dim_size.div_ceil(chunks);
        let mut out = alloc::vec::Vec::with_capacity(chunks);
        for i in 0..chunks {
            let start = i * chunk_size;
            if start >= dim_size {
                break;
            }
            let len = (dim_size - start).min(chunk_size);
            out.push(self.clone().try_narrow(dim, start, len)?);
        }
        Ok(out)
    }

    /// Splits tensor into sections of size `split_size` along `dim`.
    pub fn split(
        &self,
        split_size: usize,
        dim: usize,
    ) -> Result<alloc::vec::Vec<Tensor<Dyn, B, K, G>>>
    where
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let dim_size = *self.shape_buf().as_ref().get(dim).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::InvalidAxis {
                axis: dim,
                rank: self.shape_buf().rank(),
            })
        })?;
        if split_size == 0 {
            return Err(crate::err::Error::Msg(
                "split expects positive split_size".into(),
            ));
        }
        let chunks = dim_size.div_ceil(split_size);
        let mut out = alloc::vec::Vec::with_capacity(chunks);
        for i in 0..chunks {
            let start = i * split_size;
            let len = (dim_size - start).min(split_size);
            out.push(self.clone().try_narrow(dim, start, len)?);
        }
        Ok(out)
    }

    /// Expands the tensor to target shape `S2`.
    pub fn expand<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>>
    where
        S: crate::shapes::broadcast::BroadcastShape<S2, Output = S2>,
        B: Execute<
                op::BroadcastAs,
                Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
            > + Capabilities,
    {
        self.broadcast_to::<S2>(args)
    }

    /// Extracts sliding window slices along `dim`.
    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::Unfold>,
        <B as Execute<op::Unfold>>::Output: Into<B::Storage<K>>,
    {
        let mut out_shape = self.shape_buf().as_ref().to_vec();
        let extent = *out_shape.get(dim).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::InvalidAxis {
                axis: dim,
                rank: out_shape.len(),
            })
        })?;
        if size == 0 || step == 0 || size > extent {
            return Err(crate::err::Error::Msg(
                "unfold expects positive size and step within the selected extent".into(),
            ));
        }
        let windows = (extent - size) / step + 1;
        out_shape[dim] = windows;
        out_shape.push(size);
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::Unfold, B, Dyn>(
                    &context,
                    UnfoldAttributes {
                        axis: dim,
                        size,
                        step,
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

    /// Rearranges elements in a 4D tensor of shape (N, C, H, W) to (N, C / r^2, H * r, W * r).
    pub fn pixel_shuffle(&self, upscale_factor: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::PixelShuffle>,
        <B as Execute<op::PixelShuffle>>::Output: Into<B::Storage<K>>,
    {
        let dims = self.shape_buf();
        if dims.as_ref().len() != 4 || upscale_factor == 0 {
            return Err(crate::err::Error::Msg(
                "pixel_shuffle expects a rank-4 tensor and a positive upscale factor".into(),
            ));
        }
        let factor = upscale_factor.checked_mul(upscale_factor).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::PixelShuffle,
                expression: "upscale factor squared",
            })
        })?;
        if dims.as_ref()[1] % factor != 0 {
            return Err(crate::err::Error::Msg(
                "pixel_shuffle channels must be divisible by upscale factor squared".into(),
            ));
        }
        let out_shape = vec![
            dims.as_ref()[0],
            dims.as_ref()[1] / factor,
            dims.as_ref()[2]
                .checked_mul(upscale_factor)
                .ok_or_else(|| {
                    crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                        operation: OperationKind::PixelShuffle,
                        expression: "height times upscale factor",
                    })
                })?,
            dims.as_ref()[3]
                .checked_mul(upscale_factor)
                .ok_or_else(|| {
                    crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                        operation: OperationKind::PixelShuffle,
                        expression: "width times upscale factor",
                    })
                })?,
        ];
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::prelude::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::PixelShuffle, B, Dyn>(
                    &context,
                    PixelShuffleAttributes { upscale_factor },
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

    /// Group normalization across `groups`.
    pub fn group_norm(&self, groups: usize, eps: f64) -> Result<Self>
    where
        B: Execute<op::GroupNorm> + Capabilities,
        <B as Execute<op::GroupNorm>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::GroupNorm, B, S>(
                    &context,
                    GroupNormAttributes {
                        groups,
                        epsilon: eps,
                    },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Instance normalization for 4D (N, C, H, W) tensors.
    pub fn instance_norm(&self, eps: f64) -> Result<Self>
    where
        B: Execute<op::InstanceNorm> + Capabilities,
        <B as Execute<op::InstanceNorm>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                dispatch::execute_shaped::<op::InstanceNorm, B, S>(
                    &context,
                    EpsilonAttributes { epsilon: eps },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
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
>(
    tensors: &[&Tensor<S, B, K, G>],
    dim: usize,
) -> Result<Tensor<Dyn, B, K, G>>
where
    G::Field: Clone,
    <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
{
    if tensors.is_empty() {
        return Err(crate::prelude::Error::ShapeMismatch {
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
        .map_err(crate::prelude::Error::Shape)?;
    let inputs = tensors
        .iter()
        .map(|tensor| TensorHandle::from_storage::<B, K, Local>(&tensor.inner))
        .collect::<Vec<_>>();
    let context = ExecutionContext::from_scope(B::default());
    let inner = tensors[0]
        .under_grad_mode(|| {
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

impl<
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    NewD: crate::prelude::Device,
> crate::nn::module::ToDevice<B, NewD> for Tensor<S, B, K, G>
where
    B: Backend + TransferTo<NewD>,
    <B as TransferTo<NewD>>::Output: SupportsDType<K>,
{
    /// The same tensor, rebuilt on backend `NewD`.
    type Output = Tensor<S, <B as TransferTo<NewD>>::Output, K, G>;
    /// Transfers storage to device `arg`, keeping shape/dtype/grad-tracking.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner =
            self.under_grad_mode(|| B::transfer_storage(&self.inner, &self._dtype, &field))?;
        Tensor::from_shape_value(inner, self._shape, self._dtype, field, self._grad)
    }
}

pub(crate) fn execute_where_cond_descriptor<
    S: Shape,
    S2: Shape,
    B: Backend,
    K: DType,
    G1: RequiresGrad,
    G2: RequiresGrad,
>(
    mask: &Tensor<S, B, bool, G1>,
    on_true: &Tensor<S2, B, K, G2>,
    on_false: &Tensor<S2, B, K, G2>,
) -> Result<Tensor<S2, B, K, G2>>
where
    S: ShapeEq<S2>,
    B: Execute<op::WhereCond>,
    <B as Execute<op::WhereCond>>::Output: Into<B::Storage<K>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_mask = TensorHandle::from_storage::<B, bool, Local>(&mask.inner);
    let h_true = TensorHandle::from_storage::<B, K, Local>(&on_true.inner);
    let h_false = TensorHandle::from_storage::<B, K, Local>(&on_false.inner);
    let shape_val = on_true._shape.clone();
    let context = ExecutionContext::from_scope(B::default());
    let storage = dispatch::execute_shaped::<op::WhereCond, B, S2>(
        &context,
        NoAttributes,
        &[h_mask, h_true, h_false],
        &shape_val,
    )
    .map_err(crate::prelude::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        on_true._shape.clone(),
        on_true._dtype.clone(),
        on_true._device.clone(),
        on_true._grad.clone(),
    )
}

pub(crate) fn execute_masked_fill_descriptor<
    S: Shape,
    S2: Shape,
    B: Backend,
    K: DType,
    G1: RequiresGrad,
    G2: RequiresGrad,
>(
    input: &Tensor<S, B, K, G1>,
    mask: &Tensor<S2, B, bool, G2>,
    value: f64,
) -> Result<Tensor<S, B, K, G1>>
where
    S: ShapeEq<S2>,
    B: Execute<op::MaskedFill>,
    <B as Execute<op::MaskedFill>>::Output: Into<B::Storage<K>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_input = TensorHandle::from_storage::<B, K, Local>(&input.inner);
    let h_mask = TensorHandle::from_storage::<B, bool, Local>(&mask.inner);
    let shape_val = input._shape.clone();
    let context = ExecutionContext::from_scope(B::default());
    let storage = dispatch::execute_shaped::<op::MaskedFill, B, S>(
        &context,
        crate::exec::catalog::ScalarAttributes { value },
        &[h_input, h_mask],
        &shape_val,
    )
    .map_err(crate::prelude::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        input._shape.clone(),
        input._dtype.clone(),
        input._device.clone(),
        input._grad.clone(),
    )
}

impl<S: Shape + DynShape, B: Backend + Capabilities + Default, G: RequiresGrad>
    Tensor<S, B, bool, G>
{
    /// Conditional selection: picks elements from `on_true` where `self` is true, and `on_false` elsewhere.
    pub fn where_cond<S2: Shape, K: DType, G2: RequiresGrad>(
        &self,
        on_true: &Tensor<S2, B, K, G2>,
        on_false: &Tensor<S2, B, K, G2>,
    ) -> Result<Tensor<S2, B, K, G2>>
    where
        S: ShapeEq<S2>,
        B: Execute<op::WhereCond>,
        <B as Execute<op::WhereCond>>::Output: Into<B::Storage<K>>,
    {
        self.under_grad_mode(|| {
            execute_where_cond_descriptor::<S, S2, B, K, G, G2>(self, on_true, on_false)
        })
    }
}
