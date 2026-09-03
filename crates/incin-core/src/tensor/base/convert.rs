use super::Tensor;
use crate::backend_authoring::{AutogradBackend, Backend, StorageTransfer, SupportsDType};
use crate::dist::Local;
use crate::dist::Placement;
use crate::err::{Error, Result};
use crate::shapes::Layout;
use crate::shapes::{DynShape, Nil, Shape, ShapeValue};
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType};
use crate::tensor::grad::{Grad, NoGrad, RequiresGrad};
use core::marker::PhantomData;

impl<S: Shape, B: Backend + AutogradBackend, K: FloatDType, P: Placement, L: crate::shapes::Layout>
    Tensor<S, B, K, Grad, P, L>
{
    /// Computes a vector-Jacobian product using an explicit output cotangent.
    pub fn backward_with(
        &self,
        seed: &Tensor<S, B, K, NoGrad, P>,
    ) -> Result<crate::autograd::Gradients<B>> {
        if self.shape_buf() != seed.shape_buf() {
            return Err(Error::Backend(crate::err::BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "backward seed shape does not match the output",
            }));
        }
        if self.dtype() != seed.dtype() || self.device()? != seed.device()? {
            return Err(Error::Backend(crate::err::BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "backward seed metadata does not match the output",
            }));
        }
        B::backward_with(&self.inner, seed.inner()).map(crate::autograd::Gradients::from_backend)
    }
}

impl<B: Backend + AutogradBackend, K: FloatDType, P: Placement, L: crate::shapes::Layout>
    Tensor<Nil, B, K, Grad, P, L>
{
    /// Computes the backward pass for a scalar tensor.
    pub fn backward(&self) -> Result<crate::autograd::Gradients<B>> {
        B::backward(&self.inner).map(crate::autograd::Gradients::from_backend)
    }
}

impl<S: Shape, B: Backend, K: DType, P: Placement> Tensor<S, B, K, NoGrad, P> {
    /// Moves this tensor to the specified device as a detached tensor.
    #[allow(clippy::type_complexity)]
    pub fn to_device<D2: Device>(
        &self,
        _device: &D2::Field,
    ) -> Result<Tensor<S, <B as StorageTransfer<D2>>::Output, K, NoGrad, P>>
    where
        B: StorageTransfer<D2>,
        <B as StorageTransfer<D2>>::Output: SupportsDType<K>,
    {
        let new_inner = B::transfer_storage(&self.inner, &self._dtype, _device)?;
        Tensor::<S, <B as StorageTransfer<D2>>::Output, K, NoGrad, P>::from_shape_value_placed(
            new_inner,
            self._shape.clone(),
            self._dtype.clone(),
            _device.clone(),
            PhantomData,
            self._placement.clone(),
        )
    }
}

impl<S: Shape, B: Backend, K: DType, P: Placement> Tensor<S, B, K, Grad, P> {
    /// Moves a tracked tensor to another device and starts a detached tensor.
    ///
    /// A storage transfer does not carry the source backend tape, so retaining
    /// `Grad` here would falsely promise a connected graph.
    #[allow(clippy::type_complexity)]
    pub fn to_device<D2: Device>(
        &self,
        _device: &D2::Field,
    ) -> Result<Tensor<S, <B as StorageTransfer<D2>>::Output, K, NoGrad, P>>
    where
        B: StorageTransfer<D2>,
        <B as StorageTransfer<D2>>::Output: SupportsDType<K>,
    {
        let new_inner = B::transfer_storage(&self.inner, &self._dtype, _device)?;
        Tensor::<S, <B as StorageTransfer<D2>>::Output, K, NoGrad, P>::from_shape_value_placed(
            new_inner,
            self._shape.clone(),
            self._dtype.clone(),
            _device.clone(),
            PhantomData,
            self._placement.clone(),
        )
    }
}

/// Reinterpretations that change the shape type.
///
/// Generic over the operand's layout, and the result states `Dyn`: a layout
/// describes one geometry and cannot be carried to another.
impl<S1: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, L: Layout>
    Tensor<S1, B, K, G, Local, L>
{
    /// Converts this tensor to a new static shape S2.
    pub fn into_shape<S2: Shape + DynShape>(self) -> Result<Tensor<S2, B, K, G>> {
        let dims = self._shape.shape_buf();
        let s2_shape = S2::try_from_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Tensor::from_parts(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    /// Converts this tensor to a dynamically-shaped `Tensor<Dyn>`.
    pub fn into_dyn(self) -> Tensor<crate::shapes::Dyn, B, K, G> {
        let dims = self._shape.shape_buf();
        // `Dyn`'s field *is* the dimension vector, so there is nothing to
        // re-parse and nothing that can fail - the old
        // The old optional raw-dimension conversion asserted that the input
        // was accepted. Building it directly makes that structural rather than
        // assumed, and is the last of the 39 sites `SHP-004` removes.
        let s2_shape = crate::shapes::ShapeBuf::from_slice(dims.as_ref());
        Tensor::from_shape_value_unchecked(
            self.inner,
            ShapeValue::from_validated(s2_shape),
            self._dtype,
            self._device,
            self._grad,
        )
    }

    /// Copies and converts this tensor to a new static shape S2.
    pub fn to_shape<S2: Shape + DynShape>(&self) -> Result<Tensor<S2, B, K, G>> {
        let dims = self._shape.shape_buf();
        let s2_shape = S2::try_from_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Tensor::from_parts(
            self.inner.clone(),
            s2_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<S: Shape, B: Backend, K: FloatDType> Tensor<S, B, K, NoGrad> {
    /// Marks this tensor to require gradient tracking.
    ///
    /// Reverse-mode tracking is available only for floating-point dtypes.
    pub fn require_grad(self) -> Tensor<S, B, K, Grad> {
        Tensor::from_shape_value_unchecked(
            B::fresh_autograd_identity(self.inner),
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, B: Backend, K: DType> Tensor<S, B, K, Grad> {
    /// Detaches this tensor from autodiff tape tracking, returning a NoGrad tensor.
    pub fn detach(self) -> Tensor<S, B, K, NoGrad> {
        Tensor::from_shape_value_unchecked(
            B::fresh_autograd_identity(self.inner),
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}
