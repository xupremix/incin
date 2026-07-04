use crate::prelude::*;

/// A trainable parameter.
/// Contains the backend-specific dynamic variable (e.g., a `candle_core::Var`)
/// required to compute gradients and step an optimizer.
pub struct Param<S: Shape, B: Backend<S>, T: DType = f32, D: Device = Cpu> {
    pub(crate) inner: B::RawVar,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: T::Field,
    pub(crate) _device: D::Field,
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device> Clone for Param<S, B, T, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
        }
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device> core::fmt::Debug for Param<S, B, T, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Param")
            .field("inner", &"...")
            .finish()
    }
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device> Param<S, B, T, D> {
    /// Extract a functional Tensor from this variable for forward passes
    pub fn as_tensor(&self) -> Result<Tensor<S, B, T, D, Grad>> {
        let inner_tensor = B::var_as_tensor(&self.inner)?;
        Ok(Tensor {
            inner: inner_tensor,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: core::marker::PhantomData,
        })
    }
}

impl<S, B: Backend<S>, T, D> Param<S, B, T, D>
where
    S: Shape + DynShape,
    T: DType,
    D: Device,
    (S, T, D, Grad): TensorArgs<S, T, D, Grad>,
{
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, Grad) as TensorArgs<S, T, D, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, T, D, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::var_zeros(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, Grad) as TensorArgs<S, T, D, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, T, D, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = D::to_kindle(&_device)?;
        let dtype = T::to_kindle(&_dtype);
        let inner = B::var_randn(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Construct a Param directly from a backend's RawVar, typically used when loading checkpoints.
    pub fn from_raw<A>(inner: B::RawVar, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, Grad) as TensorArgs<S, T, D, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, T, D, Grad)>::construct(args.into_arg());
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Moves this parameter to the specified device, returning a new Param.
    pub fn to_device<D2: Device>(self, _device: &D2::Field) -> Result<Param<S, B, T, D2>> {
        let kindle_device = D2::to_kindle(_device)?;
        let new_inner = B::var_to_device(&self.inner, &kindle_device)?;
        Ok(Param {
            inner: new_inner,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: _device.clone(),
        })
    }
}

impl<S: Shape + DynShape, B: Backend<S>, T: DType, D: Device> Parameters<B> for Param<S, B, T, D>
where
    B: Backend<Dyn, RawVar = <B as Backend<S>>::RawVar>,
{
    fn parameters(&self) -> alloc::vec::Vec<<B as Backend<Dyn>>::RawVar> {
        alloc::vec![self.inner.clone()]
    }
}

impl<S1: DynShape, B: Backend<S1>, T: DType, D: Device> Param<S1, B, T, D> {
    pub fn into_shape<S2: Shape>(self) -> Result<Param<S2, B, T, D>>
    where
        B: Backend<S2, RawVar = <B as Backend<S1>>::RawVar>,
    {
        let current_dims = S1::dims(&self._shape);
        let new_shape =
            S2::from_dyn(current_dims.as_ref()).ok_or_else(|| Error::ShapeMismatch {
                expected: alloc::vec![],
                got: current_dims.as_ref().to_vec(),
            })?;

        Ok(Param::<S2, B, T, D> {
            inner: self.inner,
            _shape: new_shape,
            _dtype: self._dtype,
            _device: self._device,
        })
    }
}
