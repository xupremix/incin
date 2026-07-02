use crate::{
    candle,
    prelude::{
        ArgInto, ConstDType, ConstDevice, ConstRequiresGrad, ConstShape, DType, Device,
        DynShape, Grad, NoGrad, RequiresGrad, Result, Shape, TensorArgs,
    },
};

///
/// Struct which indicates a dynamic parameter of a tensor,
/// that can be:
/// - Shape
/// - DType
/// - Device
/// - RequiresGrad
///
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dyn(());

/// A type-safe tensor wrapping a `candle_core::Tensor`.
///
/// Generic over four axes:
/// - `S`: Shape — determines the rank and dimension sizes
/// - `T`: DType — the element type (f32, f64, u8, etc. or Dyn)
/// - `D`: Device — where the tensor lives (Cpu, Cuda<N>, or Dyn)
/// - `G`: RequiresGrad — whether gradients are tracked
///
/// Each axis can be either:
/// - **Static** (compile-time known): `Const<N>`, `f32`, `Cpu`, `Grad`/`NoGrad`
/// - **Dynamic** (runtime known): `Dyn`, `usize`
///
/// Static parameters are zero-cost (PhantomData) and enable compile-time checks.
/// Dynamic parameters carry runtime values and operations return `Result`.
#[derive(Debug)]
pub struct Tensor<
    S: Shape,
    T: DType = f32,
    #[cfg(feature = "cuda")] D: Device = crate::prelude::Cuda,
    #[cfg(all(not(feature = "cuda"), feature = "metal"))] D: Device = crate::prelude::Metal,
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))] D: Device = crate::prelude::Cpu,
    G: RequiresGrad = Grad,
> {
    inner: candle::Tensor,
    _shape: S::Field,
    _dtype: T::Field,
    _device: D::Field,
    _grad: G::Field,
}

// ============================================================================
// Core construction
// ============================================================================

impl<S: Shape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// Wrap an existing `candle::Tensor` with type-level metadata.
    ///
    /// # Safety contract (not unsafe, but the caller must ensure correctness):
    /// The caller is responsible for ensuring the candle tensor's actual
    /// shape, dtype, and device match the type parameters S, T, D.
    pub(crate) fn from_parts(
        inner: candle::Tensor,
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

    /// Access the underlying `candle::Tensor`.
    #[inline]
    pub fn inner(&self) -> &candle::Tensor {
        &self.inner
    }

    /// Consume self and return the underlying `candle::Tensor`.
    #[inline]
    pub fn into_inner(self) -> candle::Tensor {
        self.inner
    }

    /// Get the shape metadata field.
    #[inline]
    pub fn shape_field(&self) -> &S::Field {
        &self._shape
    }
}

// ============================================================================
// Factory methods for tensors with DynShape (all practical shapes)
// ============================================================================

impl<S, T, D, G> Tensor<S, T, D, G>
where
    S: Shape + DynShape,
    T: DType<DType = candle::DType>,
    D: Device<Device = candle::Device>,
    G: RequiresGrad,
    (S, T, D, G): TensorArgs<S, T, D, G>,
{
    /// Create a tensor filled with zeros.
    ///
    /// ```ignore
    /// // Fully static — no args needed
    /// let t = Tensor::<(Const<2>, Const<3>), f32, Cpu, Grad>::zeros(())?;
    ///
    /// // Dynamic shape
    /// let t = Tensor::<Dyn, f32, Cpu, NoGrad>::zeros([2, 3])?;
    ///
    /// // Dynamic shape + grad
    /// let t = Tensor::<Dyn, f32, Cpu, Dyn>::zeros(([2, 3], true))?;
    /// ```
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims: S::Dims = S::dims(&_shape);
        let device = D::device(&_device)?;
        let dtype = T::dtype(&_dtype);
        let inner = candle::Tensor::zeros(dims.as_ref(), dtype, &device)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    /// Create a tensor filled with ones.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::device(&_device)?;
        let dtype = T::dtype(&_dtype);
        let inner = candle::Tensor::ones(dims.as_ref(), dtype, &device)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    /// Create a tensor with random values drawn from a uniform distribution [0, 1).
    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::device(&_device)?;
        let dtype = T::dtype(&_dtype);
        let inner = candle::Tensor::rand(0f32, 1f32, dims.as_ref(), &device)?.to_dtype(dtype)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    /// Create a tensor with random values from a standard normal distribution.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::device(&_device)?;
        let dtype = T::dtype(&_dtype);
        let inner =
            candle::Tensor::randn(0f32, 1f32, dims.as_ref(), &device)?.to_dtype(dtype)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    /// Wrap an existing candle::Tensor, validating shape metadata.
    pub fn from_candle<A>(candle_tensor: candle::Tensor, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        Ok(Self::from_parts(candle_tensor, _shape, _dtype, _device, _grad))
    }
}

// ============================================================================
// from_slice — only for compile-time-known dtype (ConstDType)
// ============================================================================

impl<S, T, D, G> Tensor<S, T, D, G>
where
    S: Shape + DynShape,
    T: ConstDType<DType = candle::DType>,
    D: Device<Device = candle::Device>,
    G: RequiresGrad,
    T::Elem: candle::WithDType,
    (S, T, D, G): TensorArgs<S, T, D, G>,
{
    /// Create a tensor from a data slice.
    ///
    /// Only available when DType is statically known (not Dyn),
    /// so the element type `T::Elem` is compile-time determined.
    ///
    /// ```ignore
    /// let t = Tensor::<(Const<2>, Const<3>), f32, Cpu, NoGrad>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     (),
    /// )?;
    /// ```
    pub fn from_slice<A>(data: &[T::Elem], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, T, D, G) as TensorArgs<S, T, D, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, T, D, G)>::construct(args.into_arg());
        let dims = S::dims(&_shape);
        let device = D::device(&_device)?;
        let inner = candle::Tensor::new(data, &device)?.reshape(dims.as_ref())?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }
}

// ============================================================================
// Fully-static convenience constructors (no args needed at all)
// ============================================================================

impl<S, T, D, G> Tensor<S, T, D, G>
where
    S: ConstShape + DynShape,
    T: ConstDType<DType = candle::DType>,
    D: ConstDevice<Device = candle::Device>,
    G: ConstRequiresGrad,
{
    /// Create a zeros tensor with fully compile-time-known parameters.
    pub fn static_zeros() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = T::init(());
        let _device = D::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = D::device(&_device)?;
        let dtype = T::DTYPE;
        let inner = candle::Tensor::zeros(dims.as_ref(), dtype, &device)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }

    /// Create a ones tensor with fully compile-time-known parameters.
    pub fn static_ones() -> Result<Self> {
        let _shape = S::Field::default();
        let _dtype = T::init(());
        let _device = D::init(());
        let _grad = G::init(());
        let dims = S::DIMS;
        let device = D::device(&_device)?;
        let dtype = T::DTYPE;
        let inner = candle::Tensor::ones(dims.as_ref(), dtype, &device)?;
        Ok(Self::from_parts(inner, _shape, _dtype, _device, _grad))
    }
}

// ============================================================================
// Accessor methods
// ============================================================================

impl<S: Shape + DynShape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// Runtime rank of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        S::rank(&self._shape)
    }

    /// Runtime number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        S::numel(&self._shape)
    }

    /// Runtime dimension sizes.
    #[inline]
    pub fn dims(&self) -> S::Dims {
        S::dims(&self._shape)
    }
}

impl<S: Shape, T: DType<DType = candle::DType>, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// The candle DType of this tensor.
    #[inline]
    pub fn dtype(&self) -> candle::DType {
        T::dtype(&self._dtype)
    }
}

impl<S: Shape, T: DType, D: Device<Device = candle::Device>, G: RequiresGrad>
    Tensor<S, T, D, G>
{
    /// The candle Device of this tensor.
    pub fn device(&self) -> Result<candle::Device> {
        D::device(&self._device)
    }
}

impl<S: Shape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// Whether this tensor tracks gradients.
    #[inline]
    pub fn requires_grad(&self) -> bool {
        G::requires_grad(&self._grad)
    }
}

// ============================================================================
// to_vec — extract data (only for const dtype)
// ============================================================================

impl<S: Shape, T: ConstDType<DType = candle::DType>, D: Device, G: RequiresGrad>
    Tensor<S, T, D, G>
where
    T::Elem: candle::WithDType,
{
    /// Extract all elements as a flat Vec.
    pub fn to_vec_flat(&self) -> Result<alloc::vec::Vec<T::Elem>> {
        Ok(self.inner.flatten_all()?.to_vec1()?)
    }
}

// ============================================================================
// Grad transitions — change the gradient tracking at the type level
// ============================================================================

impl<S: Shape, T: DType, D: Device> Tensor<S, T, D, NoGrad> {
    /// Enable gradient tracking, changing the type from NoGrad to Grad.
    pub fn require_grad(self) -> Tensor<S, T, D, Grad> {
        Tensor::from_parts(
            self.inner,
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, T: DType, D: Device> Tensor<S, T, D, Grad> {
    /// Detach from the computation graph, changing Grad to NoGrad.
    pub fn detach(self) -> Tensor<S, T, D, NoGrad> {
        Tensor::from_parts(
            self.inner.detach(),
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}
