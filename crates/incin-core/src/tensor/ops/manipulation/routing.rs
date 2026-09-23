//! Routing primitives whose geometry is either data-dependent or tiled by an
//! integer offsets operand: `nonzero` and `grouped_matmul`.
//!
//! Both arrived with #103. `nonzero`'s output extent is not known until the
//! scan finishes, so the public method runs `dispatch::execute` (no expected
//! shape) and reads the result's shape back through the storage backend.
//! `grouped_matmul`'s output is inferable from the lhs and rhs alone, so it
//! takes the ordinary `execute_shaped` path and records a gradient.
//!
//! This file exists rather than `indexing.rs` or `matmul.rs` because both of
//! those are already dirty with concurrent work (#103 asks not to touch them).

use crate::backend_authoring::{Backend, Execute};
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::{NoAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::{Dyn, DynShape, Shape, ShapeBuf, ShapeValue};
use crate::tensor::backend::StorageBackend;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, L: Layout<S>>
    Tensor<S, B, K, G, Local, L>
where
    B: StorageBackend,
{
    /// Row-major coordinates of every non-zero element.
    ///
    /// The result is `[count, rank]` and `i64`: one row per non-zero element,
    /// holding that element's multi-index. The count is not known until the
    /// scan finishes, which is why the shape is [`Dyn`] and why this method
    /// runs [`dispatch::execute`] rather than the shaped form - there is no
    /// expected geometry to pre-assert.
    ///
    /// The result carries no gradient: which positions are non-zero is a
    /// discrete property the tape cannot perturb, the same reason
    /// [`Self::one_hot`] and [`Self::bincount`] record nothing.
    ///
    /// A scalar (rank 0) yields `[count, 0]`: there are zero or one positions
    /// but no coordinates to name them with.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let mask = Cpu.tensor([[1.0f32, 0.0], [0.0, 4.0]]).unwrap();
    /// let coordinates = mask.nonzero().unwrap();
    /// assert_eq!(coordinates.dims().as_ref(), &[2, 2]);
    /// assert_eq!(coordinates.to_vec1::<i64>().unwrap(), vec![0, 0, 1, 1]);
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn nonzero(
        &self,
    ) -> Result<crate::shapes::Dense<Dyn, B, i64, crate::tensor::grad::NoGrad, Local>>
    where
        B: Execute<op::NonZero> + Capabilities + Default,
        <B as Execute<op::NonZero>>::Output: Into<B::Storage<i64>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        // Disabled rather than this tensor's own mode, for the reason
        // `one_hot`'s comment gives: the result is `NoGrad` whatever the
        // receiver was, and the coordinates have no derivative to record.
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let storage = crate::exec::GradMode::Disabled
            .restrict(|| dispatch::execute::<op::NonZero, B>(&context, NoAttributes, &[input]))?
            .into();
        // The shape the kernel produced, not one the caller guessed: the
        // count of non-zero elements is the whole reason this is DataDependent.
        let out_dims = B::shape(&storage);
        crate::shapes::Dense::<Dyn, B, i64, crate::tensor::grad::NoGrad, Local>::from_parts(
            storage,
            out_dims,
            core::marker::PhantomData,
            self._device.clone(),
            crate::tensor::grad::NoGrad::init(()),
        )
    }

    /// Expert-tiled matrix product: `self [T, K]` against stacked
    /// `rhs [E, K, N]`, sliced by `offsets [E+1]`.
    ///
    /// `offsets` is an i64 tile of `[0, T)`: expert `e` owns rows
    /// `offsets[e]..offsets[e + 1]`. An empty span contributes nothing and is
    /// not an error, which is the empty-expert case a router produces when no
    /// token lands on an expert. The result is `[T, N]`.
    ///
    /// The gradient reaches `self` and `rhs` through the usual tape entry;
    /// `offsets` is an integer tile with no cotangent, the same exclusion
    /// `scatter_add` applies to its index operand.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// // Two tokens, two experts, K=2, N=2. Expert 0 owns both rows.
    /// let lhs = Cpu.tensor([[1.0f32, 0.0], [0.0, 1.0]]).unwrap();
    /// let rhs = Cpu.tensor([[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unwrap();
    /// let offsets = Cpu.tensor([0i64, 2, 2]).unwrap();
    /// let out = lhs.grouped_matmul(&rhs, &offsets).unwrap();
    /// assert_eq!(out.dims().as_ref(), &[2, 2]);
    /// // Expert 0's weights apply to both rows; expert 1 owns none.
    /// assert_eq!(
    ///     out.to_vec1::<f32>().unwrap(),
    ///     vec![1.0, 2.0, 3.0, 4.0]
    /// );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn grouped_matmul<
        S2: Shape + DynShape,
        S3: Shape + DynShape,
        G2: RequiresGrad,
        G3: RequiresGrad,
        L2: Layout<S2>,
        L3: Layout<S3>,
    >(
        &self,
        rhs: &Tensor<S2, B, K, G2, Local, L2>,
        offsets: &Tensor<S3, B, i64, G3, Local, L3>,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        B: Execute<op::GroupedMatMul> + Capabilities + Default,
        <B as Execute<op::GroupedMatMul>>::Output: Into<B::Storage<K>>,
    {
        let lhs_dims = self.shape_buf().as_ref();
        if lhs_dims.len() != 2 {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: 0,
                    rank: lhs_dims.len(),
                },
            ));
        }
        let rhs_dims = rhs.shape_buf().as_ref();
        if rhs_dims.len() != 3 {
            return Err(crate::err::Error::Shape(
                crate::shapes::ShapeError::InvalidAxis {
                    axis: 0,
                    rank: rhs_dims.len(),
                },
            ));
        }
        // `[T, N]` from the operands alone; the offsets only partition the rows
        // and never change the product's geometry, which is why this takes the
        // shaped path while `nonzero` does not.
        let out_dims = [lhs_dims[0], rhs_dims[2]];
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_dims))
            .map_err(crate::err::Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&rhs.inner),
            TensorHandle::from_storage::<B, i64, Local>(&offsets.inner),
        ];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::GroupedMatMul, B, Dyn>(
                    &context,
                    NoAttributes,
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
}
