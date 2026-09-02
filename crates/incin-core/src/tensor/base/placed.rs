use super::Tensor;
use super::error::validate_gradient_dtype;
use crate::backend_authoring::Backend;
use crate::dist::Placement;
use crate::err::Result;
use crate::shapes::Layout;
use crate::shapes::{Shape, ShapeBuf, ShapeValue};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use core::marker::PhantomData;

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement, L: Layout>
    Tensor<S, B, K, G, P, L>
{
    /// Rebuilds tensor metadata after a checked operation while retaining the
    /// tensor's placement marker and runtime placement field.
    pub(crate) fn from_shape_value_placed<T: Shape>(
        inner: B::Storage<K>,
        shape: ShapeValue<T>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        validate_gradient_dtype::<B, K, G>(&dtype, &grad)?;
        let expected = shape.shape_buf().as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(crate::err::Error::ShapeMismatch {
                op: "from_shape_value_placed",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: placement,
            _layout: PhantomData,
        })
    }

    pub(crate) fn from_shape_buf_placed<T: Shape>(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        T::validate_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Self::from_shape_value_placed(
            inner,
            ShapeValue::from_validated_buf(dims),
            dtype,
            device,
            grad,
            placement,
        )
    }

    /// Rebuilds placed tensor metadata and verifies that the backend returned
    /// the same rank-local shape.  Distributed proof construction uses the
    /// unchecked placement adapter because its local shape is validated
    /// against the proof separately; ordinary operation results use this
    /// boundary so storage and `ShapeValue` cannot diverge.
    pub(crate) fn from_shape_buf_placed_checked<T: Shape>(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        let expected = dims.as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(crate::err::Error::ShapeMismatch {
                op: "from_shape_buf_placed_checked",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Self::from_shape_buf_placed(inner, dims, dtype, device, grad, placement)
    }
}
