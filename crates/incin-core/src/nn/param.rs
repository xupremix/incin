use crate::prelude::*;

fn execute_initializer<B>(
    dims: &[usize],
    dtype_field: &<B::FloatElem as DType>::Field,
    device_field: &<B::Device as Device>::Field,
    init: crate::nn::init::Init,
) -> Result<B::RawVar>
where
    B: Backend + SupportsDType<B::FloatElem>,
{
    use crate::nn::init::Init;
    let device = B::Device::to_incin(device_field)?;
    let dtype = B::resolve_dtype(dtype_field, &device)?;
    match init {
        Init::Zeros => B::var_zeros::<B::FloatElem>(dims, dtype, &device),
        Init::Ones => B::var_ones::<B::FloatElem>(dims, dtype, &device),
        Init::Rand => B::var_rand::<B::FloatElem>(dims, dtype, &device),
        Init::Randn => B::var_randn::<B::FloatElem>(dims, dtype, &device),
        Init::Uniform { bound } => {
            let value = B::rand::<B::FloatElem>(dims, dtype, &device)?;
            let value = B::mul_scalar_float(&value, 2.0 * bound)?;
            B::var_from_tensor(&B::add_scalar_float(&value, -bound)?)
        }
        Init::Constant(value) => {
            let ones = B::ones::<B::FloatElem>(dims, dtype, &device)?;
            B::var_from_tensor(&B::mul_scalar_float(&ones, value)?)
        }
        Init::KaimingUniform { fan_in, a } => {
            let std = f64::sqrt(2.0 / ((1.0 + a * a) * fan_in as f64));
            let bound = f64::sqrt(3.0) * std;
            let value = B::rand::<B::FloatElem>(dims, dtype, &device)?;
            let value = B::mul_scalar_float(&value, 2.0 * bound)?;
            B::var_from_tensor(&B::add_scalar_float(&value, -bound)?)
        }
        Init::KaimingNormal { fan_in, a } => {
            let std = f64::sqrt(2.0 / ((1.0 + a * a) * fan_in as f64));
            let value = B::randn::<B::FloatElem>(dims, dtype, &device)?;
            B::var_from_tensor(&B::mul_scalar_float(&value, std)?)
        }
        Init::XavierUniform { fan_in, fan_out } => {
            let bound = f64::sqrt(6.0 / (fan_in as f64 + fan_out as f64));
            let value = B::rand::<B::FloatElem>(dims, dtype, &device)?;
            let value = B::mul_scalar_float(&value, 2.0 * bound)?;
            B::var_from_tensor(&B::add_scalar_float(&value, -bound)?)
        }
        Init::XavierNormal { fan_in, fan_out } => {
            let std = f64::sqrt(2.0 / (fan_in as f64 + fan_out as f64));
            let value = B::randn::<B::FloatElem>(dims, dtype, &device)?;
            B::var_from_tensor(&B::mul_scalar_float(&value, std)?)
        }
    }
}

/// A trainable parameter storing an underlying backend variable that supports gradient computation.
///
/// `Param` is the typed wrapper around a backend's gradient-tracking variable (e.g. `candle_core::Var`).
/// All learnable weights and biases in Incin modules are stored as `Param` values.
///
/// Unlike a regular `Tensor`, a `Param` requires gradient tracking. Gradients are accumulated
/// by the backend during the backward pass and read by the optimizer during a training step.
///
/// ## Examples
/// ```rust,ignore
/// use incin::prelude::*;
///
/// // A 128×256 weight matrix initialized with Kaiming uniform
/// let w: Param<s![128, 256], MyBackend> = Param::zeros(())?;
///
/// // Convert to a tensor for use in a forward pass
/// let w_tensor = w.as_tensor()?;
/// ```
pub struct Param<S: Shape, B: Backend> {
    pub(crate) inner: <B as Backend>::RawVar,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: <B::FloatElem as DType>::Field,
    pub(crate) _device: <B::Device as Device>::Field,
}

impl<S: Shape, B: Backend> Clone for Param<S, B> {
    /// Clones the underlying backend variable and metadata (cheap: most
    /// backends' `RawVar` is a reference-counted handle).
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
    /// Redacted `Debug` output: the backend variable's contents aren't
    /// cheaply inspectable, so only a placeholder is shown for `inner`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Param").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: Backend> Param<S, B> {
    /// Extract a functional Tensor from this variable for forward passes
    pub fn as_tensor(&self) -> Result<Tensor<S, B, B::FloatElem>> {
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

impl<S: Shape + DynShape, B: Backend> Param<S, B> {
    /// Returns the shape dimensions of this parameter.
    pub fn shape_dims(&self) -> Vec<usize> {
        S::dims(&self._shape).as_ref().to_vec()
    }
}

impl<S: Shape, B: Backend, NewD: crate::prelude::Device> crate::nn::module::ToDevice<B, NewD>
    for Param<S, B>
where
    B: TransferTo<NewD>,
    <B as TransferTo<NewD>>::Output: SupportsDType<B::FloatElem>,
{
    /// The same parameter, rebuilt on backend `NewD`.
    type Output = Param<S, <B as TransferTo<NewD>>::Output>;
    /// Transfers the underlying variable to device `arg`.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_var(&self.inner, &self._dtype, &field)?;
        Ok(Param {
            inner,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: field,
        })
    }
}

impl<S: Shape + DynShape, B: Backend> Param<S, B>
where
    (S, B::FloatElem, B::Device, Grad): TensorArgs<S, B::FloatElem, B::Device, Grad>,
{
    /// Allocates storage of the given shape/dtype/device and fills it
    /// according to `init` (e.g. Kaiming/Xavier/zeros/ones/uniform/normal).
    pub fn new_init_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
        init: crate::nn::init::Init,
    ) -> Result<Self>
    where
        B: SupportsDType<B::FloatElem>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let inner = execute_initializer::<B>(dims.as_ref(), &_dtype, &_device, init)?;

        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `new_init_raw`, but accepts any argument type convertible
    /// via `ArgInto` instead of the exact tuple form.
    pub fn new_init<A>(args: A, init: crate::nn::init::Init) -> Result<Self>
    where
        B: SupportsDType<B::FloatElem>,
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::new_init_raw(args.into_arg(), init)
    }

    /// Allocates storage of the given shape/dtype/device, filled with zero.
    pub fn zeros_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_incin(&_device)?;
        let dtype = <B::FloatElem as DType>::to_incin(&_dtype);
        let inner = B::var_zeros::<B::FloatElem>(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `zeros_raw`, but accepts any argument type convertible via
    /// `ArgInto` instead of the exact tuple form.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::zeros_raw(args.into_arg())
    }

    /// Allocates storage of the given shape/dtype/device, filled with
    /// samples from the standard normal distribution `N(0, 1)`.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        let (_shape, _dtype, _device, _) =
            <(S, B::FloatElem, B::Device, Grad)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_incin(&_device)?;
        let dtype = <B::FloatElem as DType>::to_incin(&_dtype);
        let inner = B::var_randn::<B::FloatElem>(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Allocates storage of the given shape/dtype/device, filled with one.
    pub fn ones_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_incin(&_device)?;
        let dtype = <B::FloatElem as DType>::to_incin(&_dtype);
        let inner = B::var_ones::<B::FloatElem>(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `ones_raw`, but accepts any argument type convertible via
    /// `ArgInto` instead of the exact tuple form.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::ones_raw(args.into_arg())
    }

    /// Construct a Param directly from a backend's RawVar, typically used when loading checkpoints.
    pub fn from_raw<A>(inner: <B as Backend>::RawVar, args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        let (_shape, _dtype, _device, _) =
            <(S, B::FloatElem, B::Device, Grad)>::construct(args.into_arg());
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Moves this parameter to the specified device, returning a new Param.
    pub fn to_device<D2: Device>(
        self,
        _device: &D2::Field,
    ) -> Result<Param<S, <B as TransferTo<D2>>::Output>>
    where
        B: TransferTo<D2>,
        <B as TransferTo<D2>>::Output: SupportsDType<B::FloatElem>,
    {
        let new_inner = B::transfer_var(&self.inner, &self._dtype, _device)?;
        Ok(Param {
            inner: new_inner,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: _device.clone(),
        })
    }
}

impl<S: Shape + DynShape, B: Backend> Parameters<B> for Param<S, B> {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        map.insert(prefix.to_string(), self.inner.clone());
    }
}

impl<S1: DynShape, B: Backend> Param<S1, B> {
    /// Reinterprets a dynamically-shaped parameter as a statically-shaped
    /// `S2`, checking at runtime that the actual dimensions match `S2`.
    pub fn into_shape<S2: Shape>(self) -> Result<Param<S2, B>>
    where
        B: Backend,
    {
        let current_dims = S1::dims(&self._shape);
        let new_shape =
            S2::from_dyn(current_dims.as_ref()).ok_or_else(|| Error::ShapeMismatch {
                op: "into_shape",
                expected: alloc::vec![],
                got: current_dims.as_ref().to_vec(),
                msg: alloc::format!(
                    "Cannot convert from {} to new shape.",
                    current_dims.as_ref().len()
                ),
            })?;

        Ok(Param::<S2, B> {
            inner: self.inner,
            _shape: new_shape,
            _dtype: self._dtype,
            _device: self._device,
        })
    }
}

use crate::nn::module::StateDict;
use alloc::collections::BTreeMap;

impl<S: Shape + DynShape, B: Backend> StateDict<B> for Param<S, B> {
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(t) = tensors.get(prefix) {
            // we should replace self.inner or copy values into it.
            // For now, we assume we just replace the RawVar.
            // In a real framework, you might want in-place copy.
            self.inner = B::var_from_tensor(&t.inner)?;
        }
        Ok(())
    }

    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(&self, prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
        if let Ok(t) = self.as_tensor()
            && let Ok(dyn_t) = t.into_shape::<Dyn>()
        {
            tensors.insert(prefix.to_string(), dyn_t);
        }
    }
}

/// A non-trainable state buffer for values that must persist between passes but do not receive gradients.
///
/// `Buffer` has the same internal structure as [`Param`], but [`Parameters::parameters`] returns an
/// empty vector, preventing the optimizer from ever updating it. Use this for statistics like
/// `running_mean` and `running_var` in batch normalization layers.
pub struct Buffer<S: Shape, B: Backend> {
    pub(crate) inner: <B as Backend>::RawVar,
    pub(crate) _shape: S::Field,
    pub(crate) _dtype: <B::FloatElem as DType>::Field,
    pub(crate) _device: <B::Device as Device>::Field,
}

impl<S: Shape, B: Backend> Clone for Buffer<S, B> {
    /// Clones the underlying backend variable and metadata (cheap: most
    /// backends' `RawVar` is a reference-counted handle).
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
    /// Redacted `Debug` output: the backend variable's contents aren't
    /// cheaply inspectable, so only a placeholder is shown for `inner`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: Backend> Buffer<S, B> {
    /// Extract a functional Tensor from this buffer for forward passes.
    pub fn as_tensor(&self) -> Result<Tensor<S, B, B::FloatElem>> {
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

impl<S: Shape + DynShape, B: Backend> Buffer<S, B> {
    /// Returns the shape dimensions of this buffer.
    pub fn shape_dims(&self) -> Vec<usize> {
        S::dims(&self._shape).as_ref().to_vec()
    }
}

impl<S: Shape, B: Backend, NewD: crate::prelude::Device> crate::nn::module::ToDevice<B, NewD>
    for Buffer<S, B>
where
    B: TransferTo<NewD>,
    <B as TransferTo<NewD>>::Output: SupportsDType<B::FloatElem>,
{
    /// The same buffer, rebuilt on backend `NewD`.
    type Output = Buffer<S, <B as TransferTo<NewD>>::Output>;
    /// Transfers the underlying variable to device `arg`.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_var(&self.inner, &self._dtype, &field)?;
        Ok(Buffer {
            inner,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: field,
        })
    }
}

impl<S: Shape + DynShape, B: Backend> Buffer<S, B>
where
    (S, B::FloatElem, B::Device, Grad): TensorArgs<S, B::FloatElem, B::Device, Grad>,
{
    /// Allocates storage of the given shape/dtype/device and fills it
    /// according to `init` (e.g. Kaiming/Xavier/zeros/ones/uniform/normal).
    pub fn new_init_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
        init: crate::nn::init::Init,
    ) -> Result<Self>
    where
        B: SupportsDType<B::FloatElem>,
    {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let inner = execute_initializer::<B>(dims.as_ref(), &_dtype, &_device, init)?;

        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `new_init_raw`, but accepts any argument type convertible
    /// via `ArgInto` instead of the exact tuple form.
    pub fn new_init<A>(args: A, init: crate::nn::init::Init) -> Result<Self>
    where
        B: SupportsDType<B::FloatElem>,
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::new_init_raw(args.into_arg(), init)
    }

    /// Allocates storage of the given shape/dtype/device, filled with zero.
    pub fn zeros_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_incin(&_device)?;
        let dtype = <B::FloatElem as DType>::to_incin(&_dtype);
        let inner = B::var_zeros::<B::FloatElem>(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `zeros_raw`, but accepts any argument type convertible via
    /// `ArgInto` instead of the exact tuple form.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::zeros_raw(args.into_arg())
    }

    /// Allocates storage of the given shape/dtype/device, filled with one.
    pub fn ones_raw(
        args: <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
            S,
            B::FloatElem,
            B::Device,
            Grad,
        >>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, B::FloatElem, B::Device, Grad)>::construct(args);
        let dims: S::Dims = S::dims(&_shape);
        let device = <B::Device as Device>::to_incin(&_device)?;
        let dtype = <B::FloatElem as DType>::to_incin(&_dtype);
        let inner = B::var_ones::<B::FloatElem>(dims.as_ref(), dtype, &device)?;
        Ok(Self {
            inner,
            _shape,
            _dtype,
            _device,
        })
    }

    /// Same as `ones_raw`, but accepts any argument type convertible via
    /// `ArgInto` instead of the exact tuple form.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A:
            ArgInto<
                <(S, B::FloatElem, B::Device, Grad) as TensorArgs<
                    S,
                    B::FloatElem,
                    B::Device,
                    Grad,
                >>::Args,
            >,
    {
        Self::ones_raw(args.into_arg())
    }
}

impl<S: Shape + DynShape, B: Backend> Parameters<B> for Buffer<S, B> {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> StateDict<B> for Buffer<S, B> {
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(t) = tensors.get(prefix) {
            self.inner = B::var_from_tensor(&t.inner)?;
        }
        Ok(())
    }

    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(&self, prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
        if let Ok(t) = self.as_tensor()
            && let Ok(dyn_t) = t.into_shape::<Dyn>()
        {
            tensors.insert(prefix.to_string(), dyn_t);
        }
    }
}
