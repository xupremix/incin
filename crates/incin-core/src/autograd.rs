//! Backend-owned gradient handles produced by autograd passes.

use alloc::vec::Vec;

use crate::err::{Error, Result};
use crate::shapes::Shape;
use crate::tensor::backend::{AutogradBackend, Backend, HostInterop};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{DType, PlainDType};
use crate::tensor::grad::RequiresGrad;

/// Encapsulates the backend-specific gradients obtained from a backward pass.
///
/// Autograd owns this wrapper. Optimizers consume it, but do not own the
/// result of graph differentiation.
pub struct Gradients<B: Backend + AutogradBackend>(B::Grads);

impl<B: Backend + AutogradBackend> Gradients<B> {
    pub(crate) fn from_backend(inner: B::Grads) -> Self {
        Self(inner)
    }

    /// Returns the gradient for a tensor when the backward pass produced one.
    pub fn get<S, K, G>(&self, tensor: &Tensor<S, B, K, G>) -> Result<Option<Tensor<S, B, K>>>
    where
        S: Shape,
        K: DType,
        G: RequiresGrad,
    {
        let Some(storage) = B::get_grad(tensor.inner(), &self.0)? else {
            return Ok(None);
        };
        Tensor::from_gradient_storage(tensor, storage).map(Some)
    }

    /// Returns the gradient for a tensor or a structured missing-gradient error.
    pub fn require<S, K, G>(&self, tensor: &Tensor<S, B, K, G>) -> Result<Tensor<S, B, K>>
    where
        S: Shape,
        K: DType,
        G: RequiresGrad,
    {
        self.get(tensor)?.ok_or_else(|| {
            Error::Backward(crate::err::BackwardError::Recipe {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "the backward pass produced no gradient for this tensor",
            })
        })
    }

    /// Returns the gradient values for a tensor of any placement or layout.
    ///
    /// The #150 read path: a parameter obtained from a module keeps whatever
    /// shape and layout the module chose, while [`get`](Self::get) only admits
    /// layout-erased `Dyn` handles. This reads through any `P`/`L` instead,
    /// returning the backend's scalar elements, and fails with the same
    /// structured missing-gradient error as [`require`](Self::require) when
    /// the backward pass produced nothing for the tensor.
    pub fn require_values<S, K, G, P, L>(
        &self,
        tensor: &Tensor<S, B, K, G, P, L>,
    ) -> Result<Vec<K::Elem>>
    where
        S: Shape,
        K: DType + PlainDType,
        G: RequiresGrad,
        P: crate::dist::Placement,
        L: crate::shapes::Layout<S>,
        B: HostInterop,
    {
        let bytes = self.require_bytes(tensor)?;
        let num_elements = tensor.shape_buf().checked_numel(
            crate::shapes::error::OperationKind::Storage,
        )?;
        let elem_size = core::mem::size_of::<K::Elem>();
        let expected_elem_size =
            tensor.dtype().encoding().scalar_bytes().ok_or_else(|| {
                Error::Msg(alloc::format!(
                    "Element extraction requires a scalar encoding. Tensor dtype {:?} is block-packed and has no per-element Rust type",
                    tensor.dtype(),
                ))
            })?;
        if elem_size != expected_elem_size {
            return Err(Error::Msg(alloc::format!(
                "Element size mismatch converting to vec. Tensor dtype {:?} element size {} vs element type {} size {}",
                tensor.dtype(),
                expected_elem_size,
                core::any::type_name::<K::Elem>(),
                elem_size,
            )));
        }
        let expected_bytes = num_elements.checked_mul(elem_size).ok_or(
            crate::shapes::error::ShapeError::ArithmeticOverflow {
                operation: crate::shapes::error::OperationKind::Storage,
                expression: "element count * element size",
            },
        )?;
        if bytes.len() != expected_bytes {
            return Err(Error::Msg(alloc::format!(
                "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                bytes.len(),
                expected_bytes,
            )));
        }
        let mut out = Vec::with_capacity(num_elements);
        for chunk in bytes.chunks_exact(elem_size) {
            // SAFETY: `chunk` is exactly `size_of::<K::Elem>()` bytes, and
            // `K::Elem` is this dtype's own stored element type (pinned by the
            // `PlainDType` bound, not chosen by the caller), mirroring
            // `to_vec_elem`'s round-trip.
            let val = unsafe { core::ptr::read_unaligned(chunk.as_ptr() as *const K::Elem) };
            out.push(val);
        }
        Ok(out)
    }

    /// Returns the raw gradient bytes for a tensor of any dtype, placement,
    /// or layout.
    ///
    /// The fully-generic half of the #150 read path: a typed visitor such as
    /// [`ParameterVisitor`](crate::nn::ParameterVisitor) ranges over
    /// parameters whose dtype is statically unknown, so no element type can
    /// be named and [`require_values`](Self::require_values) cannot apply.
    /// Bytes plus [`Tensor::dtype`] (for width and encoding) are everything
    /// such a caller needs; concrete-dtype callers should prefer
    /// `require_values`.
    pub fn require_bytes<S, K, G, P, L>(
        &self,
        tensor: &Tensor<S, B, K, G, P, L>,
    ) -> Result<Vec<u8>>
    where
        S: Shape,
        K: DType,
        G: RequiresGrad,
        P: crate::dist::Placement,
        L: crate::shapes::Layout<S>,
        B: HostInterop,
    {
        let storage = B::get_grad(tensor.inner(), &self.0)?.ok_or_else(|| {
            Error::Backward(crate::err::BackwardError::Recipe {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "the backward pass produced no gradient for this tensor",
            })
        })?;
        B::to_bytes(&storage)
    }

    /// Borrows the backend-specific gradient container.
    #[must_use]
    pub fn as_backend(&self) -> &B::Grads {
        &self.0
    }

    /// Mutably borrows the backend container for backend-authoring and
    /// distributed gradient synchronization.
    #[must_use]
    pub fn as_backend_mut(&mut self) -> &mut B::Grads {
        &mut self.0
    }

    /// Consumes this handle and returns its backend-specific container.
    #[must_use]
    pub fn into_backend(self) -> B::Grads {
        self.0
    }
}
