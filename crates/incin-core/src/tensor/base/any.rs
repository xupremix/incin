//! Writing generic code over a tensor without naming all six parameters.
//!
//! `Tensor<S, B, K, G, P, L>` earns each of its parameters, but a caller who
//! wants to be generic over tensors pays for all of them at once. A helper that
//! does nothing but read an element count has to introduce six type parameters
//! and six bounds before it can say anything:
//!
//! ```text
//! fn numel_of<S, B, K, G, P, L>(t: &Tensor<S, B, K, G, P, L>) -> usize
//! where
//!     S: Shape + DynShape,
//!     B: Backend,
//!     K: DType,
//!     G: RequiresGrad,
//!     P: Placement,
//!     L: Layout,
//! { t.numel() }
//! ```
//!
//! [`AnyTensor`] collapses that to one parameter and the bounds that are
//! actually load-bearing:
//!
//! ```text
//! fn numel_of<T: AnyTensor>(t: &T) -> usize
//! where
//!     T::Shape: DynShape,
//! { t.as_tensor().numel() }
//! ```
//!
//! The parameters do not go away -- they are reachable as associated types, so
//! a bound that genuinely needs one still expresses it as `T::Backend:
//! Execute<op::Add>` or `T::Shape: ShapeEq<..>`. What changes is that a helper
//! only names the parts it constrains.
//!
//! This is deliberately not a facade. The trait exposes one method,
//! [`as_tensor`](AnyTensor::as_tensor), rather than mirroring the tensor API;
//! re-declaring every operation here would double the surface and drift from it
//! immediately. Generic code reaches the real API through that one call.

use super::Tensor;
use crate::backend_authoring::Backend;
use crate::dist::Placement;
use crate::shapes::{Layout, Shape};
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;

/// The concrete tensor type behind an [`AnyTensor`].
///
/// Spelling it out is what `AnyTensor` exists to avoid, so it is spelled once
/// here. Useful to callers too: a helper that needs to name the tensor it was
/// handed writes `TensorOf<T>` rather than restating six associated types.
pub type TensorOf<T> = Tensor<
    <T as AnyTensor>::Shape,
    <T as AnyTensor>::Backend,
    <T as AnyTensor>::DType,
    <T as AnyTensor>::Grad,
    <T as AnyTensor>::Placement,
    <T as AnyTensor>::Layout,
>;

/// A tensor, with its parameters reachable as associated types.
///
/// Implemented for every [`Tensor`] and nothing else. The point is to let a
/// generic function take one type parameter instead of six, and [`TensorOf<T>`]
/// spells the reconstruction where a caller needs the concrete type back.
pub trait AnyTensor {
    /// The tensor's shape type.
    type Shape: Shape;
    /// The backend the tensor's storage belongs to.
    type Backend: Backend;
    /// The element type.
    type DType: DType;
    /// Whether the tensor participates in autograd.
    type Grad: RequiresGrad;
    /// Where the tensor lives: local, or sharded across a mesh.
    type Placement: Placement;
    /// What the type settles about where the elements sit.
    type Layout: Layout;

    /// Borrows the concrete tensor.
    ///
    /// The single point of contact with the real API, so this trait never has
    /// to mirror it.
    fn as_tensor(&self) -> &TensorOf<Self>;
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement, L: Layout> AnyTensor
    for Tensor<S, B, K, G, P, L>
{
    type Shape = S;
    type Backend = B;
    type DType = K;
    type Grad = G;
    type Placement = P;
    type Layout = L;

    #[inline]
    fn as_tensor(&self) -> &TensorOf<Self> {
        self
    }
}
