//! Slicing, narrowing, indexing, masking, and gathering operations.

use crate::backend_authoring::{Backend, Descriptor, Execute};
use crate::dist::Placement;
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::ExecutionDescriptor;
use crate::exec::catalog::{
    AxisAttributes, DiagonalAttributes, DuplicateIndexRule, LogicalTensorMeta, NarrowAttributes,
    NoAttributes, ScatterAttributes, SliceAttributes, UnfoldAttributes, op,
};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::shapes::{Dyn, DynShape, Shape, ShapeBuf, ShapeValue};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::ops::index::{IndexArgs, IndexSpec, ShapeEq};
use crate::tensor::ops::manipulation::reshape::squeeze_storage_exact;
use crate::tensor::ops::manipulation::selectors::{AxisSelectorArg, ReplaceAxisSelector};
use alloc::vec::Vec;

pub(crate) fn narrow_storage_exact<B, K>(
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
        .map_err(crate::err::Error::Shape)?;
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

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    /// Slices a tensor dynamically based on a slice of `IndexSpec` configurations.
    /// Returns a dynamically shaped tensor (`Dyn`).
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
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
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// let tensor = Tensor::<s![2, 4, 4], DefaultBackend>::ones(())?;
    /// let sliced = tensor.get(vec![
    ///     IndexSpec::Index(0),
    ///     IndexSpec::Range(1, 3),
    ///     IndexSpec::All,
    /// ])?;
    /// # Ok(()) }
    /// ```
    pub fn get<I: IndexArgs>(&self, index: I) -> Result<Tensor<Dyn, B, K, G>>
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
        let inner = G::grad_mode(&self._grad).restrict(|| -> Result<B::Storage<K>> {
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

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, P: Placement>
    Tensor<S, B, K, G, P>
{
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
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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

    /// Narrows the tensor along a selector, preserving static dimensions when
    /// the selected axis is known and preserving rank for runtime selectors.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![10], DefaultBackend>::ones(()).unwrap();
    /// let n = t.try_narrow(0isize, 2, 5).unwrap(); // shape [5]
    /// ```
    pub fn try_narrow<A>(
        self,
        axis: A,
        start: usize,
        len: usize,
    ) -> Result<Tensor<A::Output, B, K, G, P>>
    where
        A: ReplaceAxisSelector<S>,
        A::Output: DynShape,
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
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
        let inner = G::grad_mode(&self._grad)
            .restrict(|| narrow_storage_exact::<B, K>(&self.inner, input_shape, dim, start, len))?;
        shape[dim] = len;
        Tensor::<S, B, K, G, P>::from_shape_buf_placed_checked::<A::Output>(
            inner,
            ShapeBuf::from_slice(&shape),
            self._dtype,
            self._device,
            self._grad,
            self._placement,
        )
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
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
        G::grad_mode(&self._grad)
            .restrict(|| execute_masked_fill_descriptor::<S, S2, B, K, G, G2>(self, mask, val_f64))
    }

    /// Gathers values along `dim` specified by `index`.
    pub fn gather<A, S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        axis: A,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<S2, B, K, G>>
    where
        A: AxisSelectorArg<S>,
        B: Execute<op::Gather> + Capabilities,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
        A,
        S2: Shape,
        S3: Shape,
        KInt: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        G3: RequiresGrad,
    >(
        &self,
        axis: A,
        index: &Tensor<S2, B, KInt, G2>,
        src: &Tensor<S3, B, K, G3>,
    ) -> Result<Self>
    where
        A: AxisSelectorArg<S>,
        S2: ShapeEq<S3>,
        B: Execute<op::Scatter> + Capabilities,
        <B as Execute<op::Scatter>>::Output: Into<B::Storage<K>>,
    {
        <S2 as ShapeEq<S3>>::ASSERT_SHAPES_MATCH;
        let dim = axis.resolve(self.shape_buf().rank())?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
            TensorHandle::from_storage::<B, K, Local>(&src.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
    pub fn index_select<A, S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        axis: A,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<A::Output, B, K, G>>
    where
        A: ReplaceAxisSelector<S>,
        A::Output: DynShape,
        B: Execute<op::IndexSelect> + Capabilities,
        <B as Execute<op::IndexSelect>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
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
        let output_shape = ShapeValue::<A::Output>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, KInt, Local>(&index.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::IndexSelect, B, A::Output>(
                    &context,
                    AxisAttributes { axis: dim },
                    &inputs,
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

    /// Returns upper triangular part of matrix.
    pub fn triu(&self, k: i64) -> Result<Self>
    where
        B: Execute<op::Triu> + Capabilities,
        <B as Execute<op::Triu>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
        .map_err(|error| crate::err::Error::from(crate::exec::CanonicalError::Descriptor(error)))?
        .into_descriptor();
        let out_shape = descriptor.output_shape().cloned().ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::TargetShapeRejected {
                operation: OperationKind::Diag,
                rank: self.rank(),
            })
        })?;
        let output_shape =
            ShapeValue::<Dyn>::try_new(out_shape).map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
    ///
    /// The explicit output type preserves the selector's shape proof. The
    /// nested generic return is intentionally allowed here because replacing
    /// it with a dynamic tensor would discard that proof.
    #[allow(clippy::type_complexity)]
    pub fn chunk<A>(
        &self,
        chunks: usize,
        axis: A,
    ) -> Result<alloc::vec::Vec<Tensor<A::Output, B, K, G>>>
    where
        A: ReplaceAxisSelector<S> + Copy,
        A::Output: DynShape,
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
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
            out.push(self.clone().try_narrow(axis, start, len)?);
        }
        Ok(out)
    }

    /// Splits tensor into sections of size `split_size` along `dim`.
    ///
    /// The explicit output type preserves the selector's shape proof. The
    /// nested generic return is intentionally allowed here because replacing
    /// it with a dynamic tensor would discard that proof.
    #[allow(clippy::type_complexity)]
    pub fn split<A>(
        &self,
        split_size: usize,
        axis: A,
    ) -> Result<alloc::vec::Vec<Tensor<A::Output, B, K, G>>>
    where
        A: ReplaceAxisSelector<S> + Copy,
        A::Output: DynShape,
        B: Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let dim = axis.resolve(self.shape_buf().rank())?;
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
            out.push(self.clone().try_narrow(axis, start, len)?);
        }
        Ok(out)
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
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
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
    let context = crate::tensor::grad::execution_context::<B, G1>(&input._grad);
    let storage = dispatch::execute_shaped::<op::MaskedFill, B, S>(
        &context,
        crate::exec::catalog::ScalarAttributes { value },
        &[h_input, h_mask],
        &shape_val,
    )
    .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        input._shape.clone(),
        input._dtype.clone(),
        input._device.clone(),
        input._grad.clone(),
    )
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
    let context = crate::tensor::grad::execution_context::<B, G2>(&on_true._grad);
    let storage = dispatch::execute_shaped::<op::WhereCond, B, S2>(
        &context,
        NoAttributes,
        &[h_mask, h_true, h_false],
        &shape_val,
    )
    .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        on_true._shape.clone(),
        on_true._dtype.clone(),
        on_true._device.clone(),
        on_true._grad.clone(),
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
        G2::grad_mode(&on_true._grad).restrict(|| {
            execute_where_cond_descriptor::<S, S2, B, K, G, G2>(self, on_true, on_false)
        })
    }
}
