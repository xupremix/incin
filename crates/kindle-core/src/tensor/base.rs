use crate::prelude::{
    ArgInto, Backend, ConstDType, ConstDevice, ConstRequiresGrad, ConstShape, DType, Device,
    DynShape, Error, Grad, KindleDType, KindleDevice, NoGrad, RequiresGrad, Result, Shape,
    TensorArgs,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dyn(());

#[derive(Debug)]
pub struct Tensor<
    S: Shape,
    B: Backend,
    G: RequiresGrad = Grad,
> {
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
    pub fn from_parts(
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
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::zeros(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::ones(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
    }

    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::rand(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
    }

    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::randn(dims.as_ref(), dtype, &device)?;
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
    }

    pub fn from_raw<A>(raw_tensor: B::RawTensor, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, G) as TensorArgs<S, B::DType, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, B::DType, B::Device, G)>::construct(args.into_arg());
        Ok(Tensor::from_parts(
            raw_tensor, _shape, _dtype, _device, _grad,
        ))
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
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
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
        Ok(Tensor::from_parts(
            inner, _shape, _dtype, _device, _grad,
        ))
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
    pub fn to_device<D2: Device>(&self, _device: &D2::Field) -> Result<Tensor<S, B::BackendWithDevice<D2>, G>> {
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
    pub fn into_shape<S2: Shape>(self) -> Result<Tensor<S2, B, G>> {
        let current_dims = S1::dims(&self._shape);
        let new_shape =
            S2::from_dyn(current_dims.as_ref()).ok_or_else(|| Error::ShapeMismatch {
                expected: alloc::vec![], // We don't have S2 specific expected shape easily printable
                got: current_dims.as_ref().to_vec(),
            })?;

        Ok(Tensor::<S2, B, G>::from_parts(
            self.inner,
            new_shape,
            self._dtype,
            self._device,
            self._grad,
        ))
    }
}

impl<S: Shape, B: Backend> Tensor<S, B, NoGrad> {
    pub fn require_grad(self) -> Tensor<S, B, Grad> {
        Tensor::from_parts(
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
        Tensor::from_parts(
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

    #[derive(Clone)]
    pub struct DummyBackend<T: DType, D: Device>(core::marker::PhantomData<(T, D)>);
    impl<T: DType, D: Device> Backend for DummyBackend<T, D> {

    fn conv1d(
        _t: &Self::RawTensor,
        _w: &Self::RawTensor,
        _b: Option<&Self::RawTensor>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
    ) -> Result<Self::RawTensor> { unimplemented!() }

    fn conv_transpose2d(
        _t: &Self::RawTensor,
        _w: &Self::RawTensor,
        _b: Option<&Self::RawTensor>,
        _stride: usize,
        _padding: usize,
        _output_padding: usize,
        _dilation: usize,
    ) -> Result<Self::RawTensor> { unimplemented!() }

    fn max_pool2d(
        _t: &Self::RawTensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self::RawTensor> { unimplemented!() }

    fn avg_pool2d(
        _t: &Self::RawTensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self::RawTensor> { unimplemented!() }

    fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> { unimplemented!() }

        type Device = D;
        type DType = T;
        type BackendWithDType<NewT: DType> = DummyBackend<NewT, D>;
        type BackendWithDevice<NewD: Device> = DummyBackend<T, NewD>;
        type RawTensor = ();
        type RawVar = ();
        type Grads = ();

        fn shape(_t: &Self::RawTensor) -> alloc::vec::Vec<usize> {
            alloc::vec::Vec::new()
        }

        fn var_as_tensor(_var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn var_from_tensor(_t: &Self::RawTensor) -> Result<Self::RawVar> {
            Ok(())
        }

        fn var_zeros(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_ones(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_rand(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }

        fn tensor_to_device(
            _t: &Self::RawTensor,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn var_to_device(_var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
            Ok(())
        }

        fn var_randn(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }

        fn zeros(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn ones(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn rand(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn randn(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn narrow(
            _t: &Self::RawTensor,
            _dim: usize,
            _start: usize,
            _len: usize,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn conv2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sum_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sum_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn to_dtype(_t: &Self::RawTensor, _dtype: KindleDType) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn broadcast_as(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn broadcast_left(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn transpose(_t: &Self::RawTensor, _dim1: usize, _dim2: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn flatten(_t: &Self::RawTensor, _start: usize, _end: usize) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> {
            Ok(())
        }
        fn step_sgd(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Ok(())
        }
        fn step_adamw(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Ok(())
        }

        fn stack(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn concat(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn layer_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn batch_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _rm: &Self::RawTensor,
            _rv: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
    }

    #[test]
    fn test_tensor_creation() {
        let t: Tensor<Dyn, DummyBackend<f32, crate::prelude::Cpu>> = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dims(), vec![2, 3]);
    }

    #[test]
    fn test_tensor_ones() {
        let t: Tensor<Dyn, DummyBackend<f32, crate::prelude::Cpu>> = Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }
}
