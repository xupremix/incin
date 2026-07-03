use crate::prelude::{
    ArgInto, Backend, ConstDType, ConstDevice, ConstRequiresGrad, ConstShape, DType, Device,
    DynShape, Grad, NoGrad, RequiresGrad, Result, Shape, TensorArgs, KindleDevice, KindleDType, Error
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dyn(());

#[derive(Debug)]
pub struct Tensor<
    S: Shape,
    B: Backend<S>,
    T: DType = f32,
    #[cfg(feature = "cuda")] D: Device = crate::prelude::Cuda,
    #[cfg(all(not(feature = "cuda"), feature = "metal"))] D: Device = crate::prelude::Metal,
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))] D: Device = crate::prelude::Cpu,
    G: RequiresGrad = Grad,
> {
    pub(crate) inner: B::RawTensor,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: T::Field,
    pub(crate) _device: D::Field,
    pub(crate) _grad: G::Field,
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    pub fn from_parts(
        inner: B::RawTensor,
        shape: S::Field,
        dtype: T::Field,
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
}

impl<S, B: Backend<S>, T, D, G> Tensor<S, B, T, D, G>
where
    S: Shape + DynShape,
    T: DType,
    D: Device,
    G: RequiresGrad,
    (S, T, D, G): TensorArgs<S, T, D, G>,
{
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::rand(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::randn(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    pub fn from_raw<A>(raw_tensor: B::RawTensor, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        Ok(Tensor::<_, B, _, _, _>::from_parts(raw_tensor, _shape, _dtype, _device, _grad))
    }
}

impl<S, B: Backend<S>, T, D, G> Tensor<S, B, T, D, G>
where
    S: ConstShape + DynShape,
    T: ConstDType,
    D: ConstDevice,
    G: ConstRequiresGrad,
{
    pub fn static_zeros() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = T::init(());
        let _device = D::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = D::to_kindle(&_device)?;
        let dtype = T::DTYPE;
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    pub fn static_ones() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = T::init(());
        let _device = D::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = D::to_kindle(&_device)?;
        let dtype = T::DTYPE;
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(inner, _shape, _dtype, _device, _grad))
    }
}

impl<S: Shape + DynShape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
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

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    #[inline]
    pub fn dtype(&self) -> KindleDType {
        T::to_kindle(&self._dtype)
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G>
{
    pub fn device(&self) -> Result<KindleDevice> {
        D::to_kindle(&self._device)
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    #[inline]
    pub fn requires_grad(&self) -> bool {
        G::requires_grad(&self._grad)
    }
}

impl<S1: DynShape, B: Backend<S1>, T: DType, D: Device, G: RequiresGrad> Tensor<S1, B, T, D, G> {
    pub fn into_shape<S2: Shape>(self) -> Result<Tensor<S2, B, T, D, G>> where B: Backend<S2, RawTensor = <B as Backend<S1>>::RawTensor> {
        let current_dims = S1::dims(&self._shape);
        let new_shape = S2::from_dyn(current_dims.as_ref())
            .ok_or_else(|| Error::ShapeMismatch {
                expected: alloc::vec![], // We don't have S2 specific expected shape easily printable
                got: current_dims.as_ref().to_vec() 
            })?;
        
        Ok(Tensor::<S2, B, T, D, G>::from_parts(
            self.inner,
            new_shape,
            self._dtype,
            self._device,
            self._grad,
        ))
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device> Tensor<S, B, T, D, NoGrad> {
    pub fn require_grad(self) -> Tensor<S, B, T, D, Grad> {
        Tensor::<_, B, _, _, _>::from_parts(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device> Tensor<S, B, T, D, Grad> {
    pub fn detach(self) -> Tensor<S, B, T, D, NoGrad> {
        Tensor::<_, B, _, _, _>::from_parts(
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
    use alloc::vec;
    use crate::prelude::{KindleDType, KindleDevice};

    pub struct DummyBackend;
    impl<S: Shape> Backend<S> for DummyBackend {
        type RawTensor = ();
        fn zeros(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn ones(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        
        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        
        fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Ok(()) }
        fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Ok(()) }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
        fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
    }

    #[test]
    fn test_tensor_creation() {
        let t: Tensor<Dyn, DummyBackend> = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dims(), vec![2, 3]);
    }
    
    #[test]
    fn test_tensor_ones() {
        let t: Tensor<Dyn, DummyBackend> = Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }
}
