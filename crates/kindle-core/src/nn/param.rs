use crate::prelude::*;

/// A trainable parameter.
/// Contains the backend-specific dynamic variable (e.g., a `candle_core::Var`)
/// required to compute gradients and step an optimizer.
pub struct Param<S: Shape, B: Backend>
{
    pub(crate) inner: <B as Backend>::RawVar,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: <B::DType as DType>::Field,
    pub(crate) _device: <B::Device as Device>::Field,
}

impl<S: Shape, B: Backend> Clone for Param<S, B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
        }
    }
}

impl<S: Shape, B: Backend> core::fmt::Debug for Param<S, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Param").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: Backend> Param<S, B> {
    /// Extract a functional Tensor from this variable for forward passes
    pub fn as_tensor(&self) -> Result<Tensor<S, B, Grad>> {
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

impl<S: Shape + DynShape, B: Backend> Param<S, B>
where
    (S, B::DType, B::Device, Grad): TensorArgs<S, B::DType, B::Device, Grad>,
{
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
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
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::var_randn(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::var_ones(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Construct a Param directly from a backend's RawVar, typically used when loading checkpoints.
    pub fn from_raw<A>(inner: <B as Backend>::RawVar, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Moves this parameter to the specified device, returning a new Param.
    pub fn to_device<D2: Device>(self, _device: &D2::Field) -> Result<Param<S, B::BackendWithDevice<D2>>> {
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

impl<S: Shape + DynShape, B: Backend> Parameters<B> for Param<S, B> {
    fn parameters(&self) -> alloc::vec::Vec<B::RawVar> {
        alloc::vec![self.inner.clone()]
    }
}

impl<S1: DynShape, B: Backend, > Param<S1, B, > {
    pub fn into_shape<S2: Shape>(self) -> Result<Param<S2, B, >>
    where
        B: Backend,
    {
        let current_dims = S1::dims(&self._shape);
        let new_shape =
            S2::from_dyn(current_dims.as_ref()).ok_or_else(|| Error::ShapeMismatch {
                expected: alloc::vec![],
                got: current_dims.as_ref().to_vec(),
            })?;

        Ok(Param::<S2, B, > {
            inner: self.inner,
            _shape: new_shape,
            _dtype: self._dtype,
            _device: self._device,
        })
    }
}

use crate::nn::module::StateDict;
use std::collections::HashMap;

impl<S: Shape + DynShape, B: Backend> StateDict<B> for Param<S, B>
{
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(t) = tensors.get(prefix) {
            // we should replace self.inner or copy values into it.
            // For now, we assume we just replace the RawVar.
            // In a real framework, you might want in-place copy.
            self.inner = B::var_from_tensor(&t.inner)?;
        }
        Ok(())
    }

    fn state_dict(&self, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        if let Ok(t) = self.as_tensor() {
            if let Ok(dyn_t) = t.into_shape::<Dyn>() {
                tensors.insert(prefix.to_string(), dyn_t);
            }
        }
    }
}

/// A non-trainable state buffer (e.g. running_mean in BatchNorm).
pub struct Buffer<S: Shape, B: Backend>
{
    pub(crate) inner: <B as Backend>::RawVar,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: <B::DType as DType>::Field,
    pub(crate) _device: <B::Device as Device>::Field,
}

impl<S: Shape, B: Backend> Clone for Buffer<S, B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
        }
    }
}

impl<S: Shape, B: Backend> core::fmt::Debug for Buffer<S, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: Backend> Buffer<S, B> {
    pub fn as_tensor(&self) -> Result<Tensor<S, B, Grad>> {
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

impl<S: Shape + DynShape, B: Backend> Buffer<S, B>
where
    (S, B::DType, B::Device, Grad): TensorArgs<S, B::DType, B::Device, Grad>,
{
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::var_zeros(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, B::DType, B::Device, Grad) as TensorArgs<S, B::DType, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::DType, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_kindle(&_device)?;
        let dtype = <B::DType as DType>::to_kindle(&_dtype);
        let inner = B::var_ones(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }
}

impl<S: Shape + DynShape, B: Backend> Parameters<B> for Buffer<S, B> {
    fn parameters(&self) -> alloc::vec::Vec<B::RawVar> {
        alloc::vec![]
    }
}

impl<S: Shape + DynShape, B: Backend> StateDict<B> for Buffer<S, B>
{
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(t) = tensors.get(prefix) {
            self.inner = B::var_from_tensor(&t.inner)?;
        }
        Ok(())
    }

    fn state_dict(&self, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        if let Ok(t) = self.as_tensor() {
            if let Ok(dyn_t) = t.into_shape::<Dyn>() {
                tensors.insert(prefix.to_string(), dyn_t);
            }
        }
    }
}
