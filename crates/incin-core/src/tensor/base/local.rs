use super::Tensor;
use super::error::validate_gradient_dtype;
use crate::backend_authoring::Backend;
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::shapes::{Shape, ShapeBuf, ShapeValue};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{NoGrad, RequiresGrad};
use alloc::string::ToString;
use core::marker::PhantomData;

/// Proof that raw storage and tensor metadata may be joined without repeating
/// their invariant checks inside the constructor.
///
/// This type and both of its variants are private to this module. Other
/// modules must use `try_from_storage`; metadata-only retagging is implemented
/// here, where the old tensor is available as the proof source.
#[derive(Clone, Copy)]
enum ConstructionWitness {
    StorageValidated,
}

/// Local-placement constructors.
///
/// Generic over the layout parameter so a caller that has proven something
/// about its buffer keeps that proof through construction. `Self` carries `L`,
/// so these neither invent nor discard a claim.
impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, L: crate::shapes::Layout>
    Tensor<S, B, K, G, Local, L>
{
    /// Wraps a gradient buffer produced for `source`.
    ///
    /// The gradient's layout is deliberately `Unknown` rather than `source`'s.
    /// A gradient is a fresh allocation the backend made on its own terms, and
    /// carrying the source's claim across would be asserting something about a
    /// buffer this function never inspected. The source is still generic over
    /// `L`, so asking for the gradient of a tensor that has proven something is
    /// allowed -- the proof just does not transfer.
    pub(crate) fn from_gradient_storage(
        source: &Tensor<S, B, K, G, Local, L>,
        inner: B::Storage<K>,
    ) -> Result<Tensor<S, B, K, NoGrad, Local, crate::shapes::Unknown>> {
        Tensor::from_parts(
            inner,
            source.shape_buf().clone(),
            source._dtype.clone(),
            source._device.clone(),
            NoGrad::init(()),
        )
    }

    /// Joins component parts after this module has witnessed their invariants.
    fn from_parts_witnessed(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        _witness: ConstructionWitness,
    ) -> Self {
        Self {
            inner,
            _shape: ShapeValue::from_validated(shape),
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
            _layout: PhantomData,
        }
    }

    /// Reuses a validated logical shape without reconstructing a legacy
    /// representation.  This is the internal path for shape-preserving
    /// operations; `from_parts` remains the checked constructor for caller
    /// supplied shape arguments.
    pub(crate) fn from_shape_value(
        inner: B::Storage<K>,
        shape: ShapeValue<S>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        validate_gradient_dtype::<B, K, G>(&dtype, &grad)?;
        // Compared as slices: this runs on every shape-preserving operation, and
        // allocating both sides to compare them cost two allocations per op on
        // the path that succeeds. The error arm still owns its operands, but it
        // only runs when the operation has already failed.
        let expected = shape.shape_buf();
        let got = B::shape(&inner);
        if expected.as_ref() != got.as_ref() {
            return Err(Error::ShapeMismatch {
                op: "from_shape_value",
                expected: expected.as_ref().to_vec(),
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Ok(Self {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
            _layout: PhantomData,
        })
    }

    pub(crate) fn from_shape_value_unchecked(
        inner: B::Storage<K>,
        shape: ShapeValue<S>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Self {
        Self {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
            _layout: PhantomData,
        }
    }

    pub(crate) fn from_shape_buf(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        S::validate_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Self::from_shape_value(
            inner,
            ShapeValue::from_validated_buf(dims),
            dtype,
            device,
            grad,
        )
    }

    /// Creates a tensor from parts, checking that storage shape matches expected shape.
    pub fn try_from_storage(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        validate_gradient_dtype::<B, K, G>(&dtype, &grad)?;
        S::validate_dims(shape.as_ref()).map_err(crate::err::Error::Shape)?;
        let expected = shape.as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(Error::ShapeMismatch {
                op: "from_parts",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Runtime shape doesn't match expected static/dynamic shape".to_string(),
            });
        }
        let expected_dtype = K::descriptor(&dtype);
        if let Some(got) = B::storage_dtype(&inner)
            && expected_dtype != got
        {
            return Err(Error::DTypeStorageMismatch {
                expected: expected_dtype,
                got,
            });
        }
        let expected_device = B::Device::to_incin(&device)?;
        if let Some(got) = B::storage_device(&inner)
            && expected_device != got
        {
            return Err(Error::DeviceStorageMismatch {
                expected: expected_device,
                got,
            });
        }
        Ok(Self::from_parts_witnessed(
            inner,
            shape,
            dtype,
            device,
            grad,
            ConstructionWitness::StorageValidated,
        ))
    }

    pub(crate) fn from_parts(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        Self::try_from_storage(inner, shape, dtype, device, grad)
    }
}
