use crate::prelude::{
    ArgInto, Backend, ConstDType, ConstDevice, ConstRequiresGrad, ConstShape, DType, Device,
    DynShape, Error, Grad, KindleDType, KindleDevice, NoGrad, RequiresGrad, Result, Shape,
    TensorArgs,
};
use alloc::string::ToString;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Auto-generated documentation for Dyn.
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

pub struct Tensor<
    S: Shape,
    B: Backend,
    K: DType = <B as Backend>::FloatElem,
    D: Device = <B as Backend>::Device,
    G: RequiresGrad = Grad,
> {
    pub(crate) inner: B::Storage<K>,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: D::Field,
    pub(crate) _grad: G::Field,
}

impl<S: Shape, B: Backend, K: DType, D: Device, G: RequiresGrad> Clone for Tensor<S, B, K, D, G> {
    /// Auto-generated documentation for clone.
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

impl<S: Shape, B: Backend, K: DType, D: Device, G: RequiresGrad> Tensor<S, B, K, D, G> {
    /// Auto-generated documentation for from_parts_unchecked.
    pub fn from_parts_unchecked(
        inner: B::Storage<K>,
        shape: S::Field,
        dtype: K::Field,
        device: D::Field,
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
    /// Auto-generated documentation for inner.
    pub fn inner(&self) -> &B::Storage<K> {
        &self.inner
    }

    #[inline]
    /// Auto-generated documentation for into_inner.
    pub fn into_inner(self) -> B::Storage<K> {
        self.inner
    }

    #[inline]
    /// Auto-generated documentation for shape_field.
    pub fn shape_field(&self) -> &S::Field {
        &self._shape
    }

    #[inline]
    /// Auto-generated documentation for grad_field.
    pub fn grad_field(&self) -> &G::Field {
        &self._grad
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, D: Device, G: RequiresGrad> Tensor<S, B, K, D, G>
where
    (S, K, D, G): TensorArgs<S, K, D, G>,
{
    /// Auto-generated documentation for from_parts.
    pub fn from_parts(
        inner: B::Storage<K>,
        shape: S::Field,
        dtype: K::Field,
        device: D::Field,
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

    /// Auto-generated documentation for zeros.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for ones.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for from_slice.
    pub fn from_slice<A>(data: &[f32], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <D as Device>::to_kindle(&_device)?;
        let bytes =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let inner = B::from_bytes(bytes, dims.as_ref(), KindleDType::F32, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for rand.
    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::rand(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for randn.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::randn(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for from_raw.
    pub fn from_raw<A>(raw_tensor: B::Storage<K>, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, D, G) as TensorArgs<S, K, D, G>>::Args>,
        K: ConstDType,
        D: ConstDevice,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, D, G)>::construct(args.into_arg());
        Self::from_parts(raw_tensor, _shape, _dtype, _device, _grad)
    }
}

impl<S: ConstShape + DynShape, B: Backend, K: ConstDType, D: ConstDevice, G: ConstRequiresGrad>
    Tensor<S, B, K, D, G>
where
    (S, K, D, G): TensorArgs<S, K, D, G>,
{
    /// Auto-generated documentation for static_zeros.
    pub fn static_zeros() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = <K as DType>::init(());
        let _device = <D as Device>::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Auto-generated documentation for static_ones.
    pub fn static_ones() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = <K as DType>::init(());
        let _device = <D as Device>::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = <D as Device>::to_kindle(&_device)?;
        let dtype = <K as ConstDType>::DTYPE;
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, D: Device, G: RequiresGrad> Tensor<S, B, K, D, G> {
    #[inline]
    /// Auto-generated documentation for rank.
    pub fn rank(&self) -> usize {
        S::rank(&self._shape)
    }

    #[inline]
    /// Auto-generated documentation for numel.
    pub fn numel(&self) -> usize {
        S::numel(&self._shape)
    }

    #[inline]
    /// Auto-generated documentation for dims.
    pub fn dims(&self) -> S::Dims {
        S::dims(&self._shape)
    }
}

impl<S: Shape, B: Backend, K: DType, D: Device, G: RequiresGrad> Tensor<S, B, K, D, G> {
    #[inline]
    /// Auto-generated documentation for dtype.
    pub fn dtype(&self) -> KindleDType {
        K::to_kindle(&self._dtype)
    }

    /// Auto-generated documentation for device.
    pub fn device(&self) -> Result<KindleDevice> {
        D::to_kindle(&self._device)
    }

    #[inline]
    /// Auto-generated documentation for requires_grad.
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
    ) -> Result<Tensor<S, B::BackendWithDevice<D2>, K, D2, G>>
    where
        B::BackendWithDevice<D2>: Backend<Storage<K> = B::Storage<K>>,
    {
        let kindle_device = D2::to_kindle(_device)?;
        let new_inner = B::tensor_to_device(&self.inner, &kindle_device)?;
        Ok(Tensor::from_parts_unchecked(
            new_inner,
            self._shape.clone(),
            self._dtype.clone(),
            _device.clone(),
            self._grad.clone(),
        ))
    }
}

impl<S1: Shape + DynShape, B: Backend, K: DType, D: Device, G: RequiresGrad>
    Tensor<S1, B, K, D, G>
{
    /// Auto-generated documentation for into_shape.
    pub fn into_shape<S2: Shape + DynShape>(self) -> Result<Tensor<S2, B, K, D, G>> {
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

    /// Auto-generated documentation for into_dyn.
    pub fn into_dyn(self) -> Tensor<crate::prelude::Dyn, B, K, D, G> {
        let dims = S1::dims(&self._shape);
        let s2_shape = <crate::prelude::Dyn as Shape>::from_dyn(dims.as_ref()).unwrap();
        Tensor::from_parts_unchecked(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    /// Auto-generated documentation for to_shape.
    pub fn to_shape<S2: Shape + DynShape>(&self) -> Result<Tensor<S2, B, K, D, G>> {
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

impl<S: Shape, B: Backend, K: DType, D: Device> Tensor<S, B, K, D, NoGrad> {
    /// Auto-generated documentation for require_grad.
    pub fn require_grad(self) -> Tensor<S, B, K, D, Grad> {
        Tensor::from_parts_unchecked(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, B: Backend, K: DType, D: Device> Tensor<S, B, K, D, Grad> {
    /// Auto-generated documentation for detach.
    pub fn detach(self) -> Tensor<S, B, K, D, NoGrad> {
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
/// Auto-generated documentation for tests.
mod tests {
    use super::*;

    use alloc::vec;

    #[test]
    /// Auto-generated documentation for test_tensor_creation.
    fn test_tensor_creation() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dims(), vec![2, 3]);
    }

    #[test]
    /// Auto-generated documentation for test_tensor_ones.
    fn test_tensor_ones() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
            Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend, K: DType, D: Device, G: RequiresGrad>
    core::fmt::Display for Tensor<S, B, K, D, G>
{
    /// Auto-generated documentation for fmt.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", B::format_tensor_display(&self.inner))
    }
}

impl<S: crate::prelude::Shape, B: crate::prelude::Backend, K: DType, D: Device, G: RequiresGrad>
    core::fmt::Debug for Tensor<S, B, K, D, G>
{
    /// Auto-generated documentation for fmt.
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
