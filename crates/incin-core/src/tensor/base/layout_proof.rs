//! Turning a tensor's runtime layout into a type-level one.
//!
//! A [`Layout`](Layout) parameter is a claim about where a
//! tensor's elements sit, and a claim is only worth having if it cannot be made
//! falsely. There are two honest ways to acquire one: inherit it from an
//! operation whose output layout is determined by its inputs, or *check* it
//! against the metadata the storage actually carries.
//!
//! This module is the second. It exists because most tensors reach a caller
//! through a path that never tracked layout -- construction from raw parts, a
//! dynamically dispatched operation, a backend the frontend cannot see into --
//! and their strides are only discoverable at runtime. Refusing them any proof
//! forever would make the parameter useless; letting them assert one would make
//! it a lie.
//!
//! There is deliberately no `assume_row_major`. An unchecked promotion is the
//! one API that could make every downstream `L: Contiguous` bound meaningless,
//! and the check it would save is a stride comparison over the rank.

use super::Tensor;
use crate::backend_authoring::Backend;
use crate::dist::Placement;
use crate::err::{Error, Result};
use crate::shapes::Layout;
use crate::shapes::{RowMajor, Shape, Unknown};
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use core::marker::PhantomData;

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Tensor<S, B, K, G, P, Unknown> {
    /// Checks the tensor's runtime strides against the dense row-major pattern
    /// and, if they match, returns the same tensor carrying that proof.
    ///
    /// This is the bridge from a runtime fact to a type-level one. The tensor's
    /// buffer, shape and contents are untouched -- only the claim attached to
    /// it changes -- but the claim is earned rather than asserted, so a
    /// downstream `L: Contiguous` bound means what it says.
    ///
    /// Only defined on [`Unknown`], which is the state a tensor is in when
    /// nothing has been established about it. A tensor that already carries a
    /// layout got it from somewhere that knew, and re-deriving it from runtime
    /// metadata would be a step backwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Msg`] when the strides describe anything other than a
    /// dense row-major traversal -- a transposed or narrowed view, a broadcast
    /// with a zero stride, or a tensor whose first element is not at the start
    /// of its buffer.
    ///
    /// # Examples
    ///
    /// ```text
    /// let t: Tensor<s![3, 4], B> = Tensor::zeros(())?;
    /// // `reshape_flat` needs `L: Contiguous`, which `Unknown` cannot satisfy.
    /// let proven = t.into_row_major()?;
    /// let flat = proven.reshape_flat::<s![12]>()?;
    /// ```
    pub fn into_row_major(self) -> Result<Tensor<S, B, K, G, P, RowMajor<S>>> {
        let meta = B::metadata::<K>(&self.inner);
        let dims = meta.shape().as_ref();
        let strides = meta.strides().as_ref();

        if meta.offset_elements() != 0 {
            return Err(Error::Msg(alloc::format!(
                "a row-major layout starts at the beginning of its buffer, but this tensor's first element is at offset {}",
                meta.offset_elements()
            )));
        }
        if dims.len() != strides.len() {
            return Err(Error::Msg(alloc::format!(
                "layout metadata is not congruent: {} extents against {} strides",
                dims.len(),
                strides.len()
            )));
        }

        // Suffix products, innermost first. Compared rather than trusted: this
        // is the whole content of the proof.
        let mut expected = 1usize;
        for axis in (0..dims.len()).rev() {
            if strides[axis] != expected {
                return Err(Error::Msg(alloc::format!(
                    "axis {axis} has stride {} where a dense row-major layout of {dims:?} requires {expected}",
                    strides[axis]
                )));
            }
            expected = expected.checked_mul(dims[axis]).ok_or_else(|| {
                Error::Msg(alloc::format!(
                    "row-major strides for {dims:?} overflow usize"
                ))
            })?;
        }

        Ok(Tensor {
            inner: self.inner,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad,
            _placement: self._placement,
            _layout: PhantomData,
        })
    }
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement, L: Layout>
    Tensor<S, B, K, G, P, L>
{
    /// Reinterprets the tensor's elements under a new shape without copying.
    ///
    /// Only callable when the layout is proven [`Contiguous`]. That bound is
    /// the point: reinterpreting a shape is meaningful exactly when the
    /// elements form one unbroken run, and on a transposed or narrowed view it
    /// silently reads the wrong elements. Every framework this resembles
    /// discovers that at runtime -- here the call does not compile.
    ///
    /// The element count must match, which [`ElementCount`] settles from the
    /// two shape types rather than from their runtime dimensions, so a
    /// mismatched reshape is also a compile error rather than an
    /// `Err(ShapeMismatch)`.
    ///
    /// The result carries [`RowMajor<S2>`], since a contiguous run reinterpreted
    /// under a new shape is dense in that shape by construction.
    ///
    /// [`Contiguous`]: crate::shapes::Contiguous
    /// [`ElementCount`]: crate::shapes::ElementCount
    /// [`RowMajor<S2>`]: crate::shapes::RowMajor
    ///
    /// # Errors
    ///
    /// Returns an error only if the runtime dimensions cannot be resolved for
    /// `S2`; the shape-compatibility question is already settled statically.
    pub fn reshape_view<S2>(self) -> Result<Tensor<S2, B, K, G, P, RowMajor<S2>>>
    where
        L: crate::shapes::Contiguous,
        S2: Shape,
        S: crate::shapes::ElementCount,
        S2: crate::shapes::ElementCount<Count = <S as crate::shapes::ElementCount>::Count>,
    {
        // The target dimensions come from `S2` itself. A view cannot invent
        // them: reinterpreting a buffer means naming the shape you are
        // reinterpreting it *as*, and if that shape has a runtime axis there is
        // nothing here to resolve it against. Those callers want the ordinary
        // `reshape`, which takes the dimensions as an argument.
        let mut dims = alloc::vec::Vec::with_capacity(S2::STATIC_EXTENTS.len());
        for (axis, extent) in S2::STATIC_EXTENTS.iter().enumerate() {
            let Some(extent) = extent else {
                return Err(Error::Msg(alloc::format!(
                    "reshape_view needs every target extent from the type, but axis {axis} of the target shape is only known at runtime; use `reshape` and pass the dimensions"
                )));
            };
            dims.push(*extent);
        }
        if dims.is_empty() && S2::RANK != Some(0) {
            return Err(Error::Msg(
                "reshape_view needs a target shape with a statically known rank".into(),
            ));
        }

        // The element counts were proven equal by the `ElementCount` bound, so
        // this cannot disagree with the source. Asserted anyway: the bound
        // reasons about the shape *types*, and this is the buffer the tensor
        // actually holds.
        let source_numel: usize = B::metadata::<K>(&self.inner)
            .shape()
            .as_ref()
            .iter()
            .product();
        let target_numel: usize = dims.iter().product();
        debug_assert_eq!(
            source_numel, target_numel,
            "ElementCount proved these equal; a difference means the runtime shape and the shape type disagree"
        );

        let reshaped =
            crate::shapes::ShapeValue::<S2>::try_new(crate::shapes::ShapeBuf::from_slice(&dims))?;

        Ok(Tensor {
            inner: self.inner,
            _shape: reshaped,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad,
            _placement: self._placement,
            _layout: PhantomData,
        })
    }
}
