use crate::prelude::{
    ArgInto, Backend, ConstDType, DType, DTypeId, Device, DeviceId, DynShape, Error, Grad, NoGrad,
    RequiresGrad, Result, Shape, SupportsDType, TensorArgs, TransferTo,
};
use alloc::string::ToString;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// A marker used as `Shape`, `DType`, `Device`, or their runtime-chosen
/// variant across `Tensor`'s type parameters, deferring that choice from
/// compile time to runtime (e.g. `Tensor<Dyn, B>` has a shape resolved at
/// construction rather than baked into the type).
pub struct Dyn(pub ());

/// The core `Tensor` type representing an n-dimensional array.
///
/// It holds a reference to a backend-specific tensor representation, while statically tracking
/// its `Shape`, `Backend` (which includes `DType` and `Device`), and its `Grad` requirements.
///
/// `Tensor` is the primary workhorse of the Incin framework. By maintaining shape information
/// directly in the type signature, Incin ensures that tensor operations such as matrix multiplication
/// or convolutions are strictly verified at compile time.
///
/// ## Type Parameters
/// * `S`: The [`Shape`] of the tensor. This can be static (e.g., `s![2, 3, 224, 224]`), dynamic (`Dyn`), or partially dynamic.
/// * `B`: The underlying compute [`Backend`]. It defines how the tensor is stored in memory and how mathematical operations are executed.
/// * `G`: Trait marker representing whether the tensor requires gradients ([`Grad`] or [`NoGrad`]). Defaults to `Grad`.
///
/// ## Examples
///
/// Creating and inspecting statically shaped tensors:
/// ```rust,ignore
/// use incin::prelude::*;
/// type Backend = IncinBackend<f32, Cpu>;
///
/// // Compile-time 3D tensor of shape [2, 5, 10]
/// let t = Tensor::<s![2, 5, 10], Backend>::zeros(()).unwrap();
///
/// assert_eq!(t.dims(), vec![2, 5, 10]);
/// ```
///
/// Using dynamically shaped tensors:
/// ```rust,ignore
/// use incin::prelude::*;
/// type Backend = IncinBackend<f32, Cpu>;
///
/// // Shape determined at runtime
/// let dyn_t = Tensor::<Dyn, Backend>::ones(vec![32, 64]).unwrap();
///
/// assert_eq!(dyn_t.dims(), vec![32, 64]);
/// ```
pub struct Tensor<
    S: Shape,
    B: Backend,
    K: DType = <B as Backend>::FloatElem,
    G: RequiresGrad = Grad,
> {
    pub(crate) inner: B::Storage<K>,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: <B::Device as Device>::Field,
    pub(crate) _grad: G::Field,
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad> Clone for Tensor<S, B, K, G> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: self._grad.clone(),
        }
    }
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G> {
    /// Creates a tensor from raw component parts without shape verification.
    pub(crate) fn from_parts_unchecked(
        inner: B::Storage<K>,
        shape: S::Field,
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
        }
    }

    #[inline]
    /// Returns a reference to the backend-specific storage handle.
    pub fn inner(&self) -> &B::Storage<K> {
        &self.inner
    }

    #[inline]
    /// Consumes the Tensor and returns the backend-specific storage handle.
    pub fn into_inner(self) -> B::Storage<K> {
        self.inner
    }

    #[inline]
    /// Returns a reference to the static/dynamic shape field representation.
    pub fn shape_field(&self) -> &S::Field {
        &self._shape
    }

    #[inline]
    /// Returns a reference to the gradient marker field.
    pub fn grad_field(&self) -> &G::Field {
        &self._grad
    }

    /// Creates a tensor from parts, checking that storage shape matches expected shape.
    pub fn try_from_storage(
        inner: B::Storage<K>,
        shape: S::Field,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self>
    where
        S: DynShape,
    {
        let expected = S::dims(&shape).as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got {
            return Err(Error::ShapeMismatch {
                op: "from_parts",
                expected,
                got,
                msg: "Runtime shape doesn't match expected static/dynamic shape".to_string(),
            });
        }
        let expected_dtype = K::to_incin(&dtype);
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
        Ok(Self::from_parts_unchecked(
            inner, shape, dtype, device, grad,
        ))
    }

    pub(crate) fn from_parts(
        inner: B::Storage<K>,
        shape: S::Field,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self>
    where
        S: DynShape,
    {
        Self::try_from_storage(inner, shape, dtype, device, grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G>
where
    (S, K, B::Device, G): TensorArgs<S, K, B::Device, G>,
    B: SupportsDType<K>,
{
    /// Creates a tensor filled with zeros.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with ones.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a slice whose element type fixes its static dtype.
    pub fn from_slice<A>(data: &[K::Elem], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        K: ConstDType,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let byte_len = core::mem::size_of_val(data);
        let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
        let inner = B::from_bytes(bytes, dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a checked native-endian byte payload.
    pub fn from_bytes<A>(bytes: &[u8], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let inner = B::from_bytes(bytes, dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with random values uniform in [0, 1).
    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let inner = B::rand(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with standard normal random values.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let inner = B::randn(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with scalar `val`.
    pub fn full<Sc: Into<crate::tensor::backend::ScalarValue>, A>(val: Sc, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let scalar_f64 = val.into().to_f64();
        let inner = B::full(scalar_f64, dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor starting at `start` with step `step`.
    pub fn arange<Sc: Into<crate::tensor::backend::ScalarValue>, A>(start: Sc, step: Sc, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let st_f64 = step.into().to_f64();
        let inner = B::arange(s_f64, st_f64, dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor with linearly spaced values between `start` and `end`.
    pub fn linspace<Sc: Into<crate::tensor::backend::ScalarValue>, A>(start: Sc, end: Sc, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let e_f64 = end.into().to_f64();
        let inner = B::linspace(s_f64, e_f64, dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Samples a tensor of shape `shape` from a probability distribution `dist`.
    pub fn sample<D: crate::distributions::Distribution<K>, A>(dist: &D, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: SupportsDType<K>,
        B: Backend<FloatElem = K>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        dist.sample::<S, B, G>(_shape, &_device)
    }

    /// Wraps an existing backend storage in a Tensor.
    pub fn from_raw<A>(raw_tensor: B::Storage<K>, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg());
        Self::from_parts(raw_tensor, _shape, _dtype, _device, _grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G> {
    #[inline]
    /// Returns the number of dimensions (rank) of the tensor.
    pub fn rank(&self) -> usize {
        S::rank(&self._shape)
    }

    #[inline]
    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        S::numel(&self._shape)
    }

    #[inline]
    /// Returns the dimensions of the tensor as a slice or container.
    pub fn dims(&self) -> S::Dims {
        S::dims(&self._shape)
    }
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G> {
    #[inline]
    /// Returns the Incin data type variant.
    pub fn dtype(&self) -> DTypeId {
        K::to_incin(&self._dtype)
    }

    /// Returns the device on which this tensor is allocated.
    pub fn device(&self) -> Result<DeviceId> {
        B::Device::to_incin(&self._device)
    }

    #[inline]
    /// Returns true if this tensor computes and accumulates gradients.
    pub fn requires_grad(&self) -> bool {
        G::requires_grad(&self._grad)
    }

    /// Computes the backward pass starting from this tensor, returning the gradients.
    pub fn backward(&self) -> Result<crate::optim::Gradients<B::Grads>> {
        B::backward(&self.inner).map(crate::optim::Gradients)
    }

    /// Moves this tensor to the specified device, returning a new Tensor.
    pub fn to_device<D2: Device>(
        &self,
        _device: &D2::Field,
    ) -> Result<Tensor<S, <B as TransferTo<D2>>::Output, K, G>>
    where
        B: TransferTo<D2>,
        <B as TransferTo<D2>>::Output: SupportsDType<K>,
    {
        let new_inner = B::transfer_storage(&self.inner, &self._dtype, _device)?;
        Ok(Tensor::from_parts_unchecked(
            new_inner,
            self._shape.clone(),
            self._dtype.clone(),
            _device.clone(),
            self._grad.clone(),
        ))
    }
}

impl<S1: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> Tensor<S1, B, K, G> {
    /// Converts this tensor to a new static shape S2.
    pub fn into_shape<S2: Shape + DynShape>(self) -> Result<Tensor<S2, B, K, G>> {
        let dims = S1::dims(&self._shape);
        let s2_shape = S2::from_dyn(dims.as_ref()).ok_or_else(|| {
            crate::err::Error::Msg(alloc::format!(
                "into_shape failed: cannot parse {:?} into {}",
                dims,
                core::any::type_name::<S2>()
            ))
        })?;
        Tensor::from_parts(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    /// Converts this tensor to a dynamically-shaped Tensor<Dyn>.
    pub fn into_dyn(self) -> Tensor<crate::prelude::Dyn, B, K, G> {
        let dims = S1::dims(&self._shape);
        let s2_shape = <crate::prelude::Dyn as Shape>::from_dyn(dims.as_ref()).unwrap();
        Tensor::from_parts_unchecked(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    /// Copies and converts this tensor to a new static shape S2.
    pub fn to_shape<S2: Shape + DynShape>(&self) -> Result<Tensor<S2, B, K, G>> {
        let dims = S1::dims(&self._shape);
        let s2_shape = S2::from_dyn(dims.as_ref()).ok_or_else(|| {
            crate::err::Error::Msg(alloc::format!(
                "to_shape failed: cannot parse {:?} into {}",
                dims,
                core::any::type_name::<S2>()
            ))
        })?;
        Tensor::from_parts(
            self.inner.clone(),
            s2_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<S: Shape, B: Backend, K: DType> Tensor<S, B, K, NoGrad> {
    /// Marks this tensor to require gradient tracking.
    pub fn require_grad(self) -> Tensor<S, B, K, Grad> {
        Tensor::from_parts_unchecked(
            self.inner,
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
        Tensor::from_parts_unchecked(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend, K: DType, G: RequiresGrad>
    core::fmt::Display for Tensor<S, B, K, G>
{
    /// Delegates to the backend's own display formatting of its storage.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", B::format_tensor_display(&self.inner))
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend, K: DType, G: RequiresGrad>
    core::fmt::Debug for Tensor<S, B, K, G>
{
    /// Prints the backend type name, runtime shape, and the backend's own
    /// debug rendering of its storage.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor({}, shape={:?})\n{}",
            core::any::type_name::<B>(),
            B::shape(&self.inner),
            B::format_tensor_debug(&self.inner)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;

    #[test]
    fn test_tensor_creation() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dims(), vec![2, 3]);
    }

    #[test]
    fn test_tensor_ones() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
            Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    // `DummyBackend`'s conv/pool shape math must never panic on a
    /// pathological input, e.g. an input smaller than an (over-dilated)
    /// kernel plus padding — `2*padding + input` underflowing `dilation *
    /// (kernel - 1) + 1` used to panic via unchecked `usize` subtraction in
    /// debug builds (or silently wrap in release).
    fn dummy_backend_conv_pool_shape_math_never_panics_on_tiny_input_large_kernel() {
        use crate::prelude::{Backend, ModuleOps};
        type B = crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>;

        // 1x1x2x2 input, a 5x5 kernel with dilation 3: `dilation*(kernel-1)+1`
        // = 3*4+1 = 13, far larger than `input + 2*padding` = 2 + 0 = 2.
        let input: <B as Backend>::Storage<f32> = alloc::vec![1, 1, 2, 2];
        let weight: <B as Backend>::Storage<f32> = alloc::vec![1, 1, 5, 5];
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&input, &weight, None, 1, 0, 3, 1).unwrap();
        assert_eq!(out.len(), 4);

        let pool_out = <B as crate::prelude::ModuleOps<B>>::max_pool2d::<f32>(
            &input,
            (5, 5),
            (1, 1),
            (0, 0),
            (3, 3),
        )
        .unwrap();
        assert_eq!(pool_out.len(), 4);
    }
}
