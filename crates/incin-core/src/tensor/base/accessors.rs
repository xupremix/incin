use super::Tensor;
use crate::backend_authoring::Backend;
use crate::dist::{Placement, PlacementKind};
use crate::err::Result;
use crate::shapes::{DynShape, Shape, ShapeBuf};
use crate::tensor::device::{Device, DeviceId};
use crate::tensor::dtype::{DType, DTypeId};
use crate::tensor::grad::RequiresGrad;

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Tensor<S, B, K, G, P> {
    #[inline]
    /// Returns a reference to the backend-specific rank-local storage handle.
    pub fn inner(&self) -> &B::Storage<K> {
        &self.inner
    }

    #[inline]
    /// Consumes the tensor and returns its rank-local storage handle.
    pub fn into_inner(self) -> B::Storage<K> {
        self.inner
    }

    #[inline]
    /// Returns the authoritative runtime logical shape buffer.
    pub fn shape_buf(&self) -> &crate::shapes::ShapeBuf {
        self._shape.shape_buf()
    }

    pub(crate) fn shape_buf_value(&self) -> ShapeBuf {
        self._shape.shape_buf().clone()
    }

    #[inline]
    /// Returns a reference to the gradient marker field.
    pub fn grad_field(&self) -> &G::Field {
        &self._grad
    }

    /// Runtime projection of the tensor's placement.
    #[must_use]
    pub fn placement(&self) -> PlacementKind {
        P::to_incin(&self._placement)
    }

    /// Rank whose local storage this tensor owns.
    #[must_use]
    pub fn rank_index(&self) -> usize {
        P::rank(&self._placement)
    }

    /// Shape of the rank-local physical storage.
    ///
    /// [`dims`](Self::dims) reports the global logical shape.
    #[must_use]
    pub fn local_dims(&self) -> alloc::vec::Vec<usize> {
        B::shape(&self.inner).as_ref().to_vec()
    }

    /// Returns the logical descriptor for this tensor's dtype.
    ///
    /// Works for all dtypes including custom third-party non-builtin dtypes.
    #[must_use]
    pub fn dtype(&self) -> crate::tensor::dtype::DTypeDescriptor {
        K::descriptor(&self._dtype)
    }

    /// Returns the `DTypeId` if this tensor's dtype is a built-in Incin dtype,
    /// or `None` for custom third-party dtypes.
    pub fn builtin_dtype_id(&self) -> Option<DTypeId> {
        K::descriptor(&self._dtype).builtin_id()
    }

    /// Alias for [`dtype`](Self::dtype).
    #[must_use]
    #[deprecated(note = "Use `.dtype()` instead")]
    pub fn dtype_descriptor(&self) -> crate::tensor::dtype::DTypeDescriptor {
        self.dtype()
    }

    /// Returns the physical device on which this rank-local storage resides.
    pub fn device(&self) -> Result<DeviceId> {
        B::Device::to_incin(&self._device)
    }

    /// Whether this tensor computes and accumulates gradients.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        G::requires_grad(&self._grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, P: Placement>
    Tensor<S, B, K, G, P>
{
    #[inline]
    /// Returns the number of dimensions (rank) of the tensor.
    pub fn rank(&self) -> usize {
        self._shape.shape_buf().rank()
    }

    #[inline]
    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self._shape.shape_buf().numel().unwrap_or(0)
    }

    #[inline]
    /// Returns the dimensions of the tensor as a slice or container.
    pub fn dims(&self) -> crate::shapes::ShapeBuf {
        self._shape.shape_buf().clone()
    }
}
