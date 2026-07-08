use crate::prelude::{
    ArgInto, Backend, ConstDType, ConstDevice, ConstRequiresGrad, ConstShape, DType, Device,
    DynShape, Error, Grad, KindleDType, KindleDevice, NoGrad, RequiresGrad, Result, Shape,
    TensorArgs,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dyn(pub ());

/// The core `Tensor` type representing an n-dimensional array.
///
/// It holds a reference to a backend-specific tensor representation, while statically tracking
/// its `Shape`, `Backend` (which includes `DType` and `Device`), and its `Grad` requirements.
///
/// `Tensor` is the primary workhorse of the Kindle framework. By maintaining shape information
/// directly in the type signature, Kindle ensures that tensor operations such as matrix multiplication
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
/// use kindle::prelude::*;
/// type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
///
/// // Compile-time 3D tensor of shape [2, 5, 10]
/// let t = Tensor::<s![2, 5, 10], Backend>::zeros(()).unwrap();
///
/// assert_eq!(t.dims(), vec![2, 5, 10]);
/// ```
///
/// Using dynamically shaped tensors:
/// ```rust,ignore
/// use kindle::prelude::*;
/// type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
///
/// // Shape determined at runtime
/// let dyn_t = Tensor::<Dyn, Backend>::ones(vec![32, 64]).unwrap();
///
/// assert_eq!(dyn_t.dims(), vec![32, 64]);
/// ```

pub struct Tensor<S: Shape, B: Backend, G: RequiresGrad = Grad> {
    pub(crate) inner: B::RawTensor,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: <B::DType as DType>::Field,
    pub(crate) _device: <B::Device as Device>::Field,
    pub(crate) _grad: G::Field,
}

impl<S: Shape, B: Backend, G: RequiresGrad> Clone for Tensor<S, B, G> {
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

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    pub fn from_parts_unchecked(
        inner: B::RawTensor,
        shape: S::Field,
        dtype: <B::DType as DType>::Field,
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
    pub fn inner(&self) -> &B::RawTensor {
        &self.inner
    }

    #[inline]
    pub fn into_inner(self) -> B::RawTensor {
        self.inner
    }

    #[inline]
    pub fn shape_field(&self) -> &S::Field {
        &self._shape
    }

    #[inline]
    pub fn grad_field(&self) -> &G::Field {
        &self._grad
    }
}

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G>
where
    (S, B::DType, B::Device, G): TensorArgs<S, B::DType, B::Device, G>,
{
    pub fn from_parts(
        inner: B::RawTensor,
        shape: S::Field,
        dtype: <B::DType as DType>::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
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
        Ok(Self::from_parts_unchecked(
            inner, shape, dtype, device, grad,
        ))
    }

    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn from_slice<A>(data: &[f32], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let bytes =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let inner = B::from_bytes(bytes, dims.as_ref(), KindleDType::F32, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::rand(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::randn(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn from_raw<A>(raw_tensor: B::RawTensor, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) =
            <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        Self::from_parts(raw_tensor, _shape, _dtype, _device, _grad)
    }
}

impl<S: ConstShape + DynShape, B: Backend, G: ConstRequiresGrad> Tensor<S, B, G>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
{
    pub fn static_zeros() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = <B::DType as DType>::init(());
        let _device = <B::Device as Device>::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as ConstDType>::DTYPE;
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn static_ones() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = <B::DType as DType>::init(());
        let _device = <B::Device as Device>::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as ConstDType>::DTYPE;
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }
}

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    #[inline]
    pub fn rank(&self) -> usize {
        S::rank(&self._shape)
    }

    #[inline]
    pub fn numel(&self) -> usize {
        S::numel(&self._shape)
    }

    #[inline]
    pub fn dims(&self) -> S::Dims {
        S::dims(&self._shape)
    }
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    #[inline]
    pub fn dtype(&self) -> KindleDType {
        <B::DType as DType>::to_kindle(&self._dtype)
    }

    pub fn device(&self) -> Result<KindleDevice> {
        <B::Device as Device>::to_kindle(&self._device)
    }

    #[inline]
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
    ) -> Result<Tensor<S, B::BackendWithDevice<D2>, G>> {
        let kindle_device = D2::to_kindle(_device)?;
        let new_inner = B::tensor_to_device(&self.inner, &kindle_device)?;
        Ok(Tensor {
            inner: new_inner,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: _device.clone(),
            _grad: self._grad.clone(),
        })
    }
}

impl<S1: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S1, B, G> {
    pub fn into_shape<S2: Shape + DynShape>(self) -> Result<Tensor<S2, B, G>> {
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

    pub fn into_dyn(self) -> Tensor<crate::prelude::Dyn, B, G> {
        let dims = S1::dims(&self._shape);
        let s2_shape = <crate::prelude::Dyn as Shape>::from_dyn(dims.as_ref()).unwrap();
        Tensor::from_parts_unchecked(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    pub fn to_shape<S2: Shape + DynShape>(&self) -> Result<Tensor<S2, B, G>> {
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

impl<S: Shape, B: Backend> Tensor<S, B, NoGrad> {
    pub fn require_grad(self) -> Tensor<S, B, Grad> {
        Tensor::from_parts_unchecked(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, B: Backend> Tensor<S, B, Grad> {
    pub fn detach(self) -> Tensor<S, B, NoGrad> {
        Tensor::from_parts_unchecked(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{KindleDType, KindleDevice};
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
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> = Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend> core::fmt::Display for Tensor<S, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}",
            B::format_tensor_display(&self.inner)
        )
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend> core::fmt::Debug for Tensor<S, B> {
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
