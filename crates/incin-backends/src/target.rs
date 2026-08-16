//! A device value as the user-facing allocation target.
//!
//! # What problem this solves
//!
//! Constructing a tensor today means naming a backend type and packing shape,
//! dtype, device and grad arguments into one positional tuple whose arity
//! depends on which of those four happen to be static:
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<Cuda>>::zeros(((), Cuda::new(2)))
//! ```
//!
//! The leading `()` is not decoration. A fully static shape's `Shape::Arg` is
//! a tuple of units, `arg_into`'s `NotUnit` marker counts that as an argument
//! the caller supplied, and the device selector therefore has to be shifted
//! into second position. Get it wrong and the diagnostic is an unsatisfied
//! `ArgInto<TensorArgsData<..>>` bound that names none of the four things it
//! is actually talking about.
//!
//! Here the same allocation is:
//!
//! ```text
//! gpu.zeros(shape![2, 3])
//! ```
//!
//! Three things now come from three places that cannot be confused for one
//! another, instead of from one tuple whose arity you had to work out:
//!
//! | What | Comes from | Example |
//! |---|---|---|
//! | backend, device | the target value | `gpu` |
//! | geometry | the shape argument | `shape![batch, 784]` |
//! | dtype | the *data*, or the target's bound float | `[0_i64, 1]` / `gpu` |
//!
//! None of the three can be written in another's position, so there is no
//! order to remember and no `()` placeholder to get wrong.
//!
//! # Why a device value and not a backend, runtime or context
//!
//! - **Backends are zero-sized.** `CpuBackendImpl<D>` is a `PhantomData`
//!   and `IncinBackend<D>` is a *type alias* for
//!   `<Native as EngineOn<D>>::Backend`. Making a caller name one is making them
//!   name nothing.
//! - **There is no runtime state to own.** WGPU's device and queue live in a
//!   process-global `OnceLock`; CUDA's contexts live in a global map that is
//!   deliberately never evicted because releasing the last handle costs a
//!   131 ms re-initialization. A `Runtime` object would claim ownership of
//!   resources it does not hold.
//! - **`ExecutionContext` is policy, not placement.** It pairs a backend value
//!   with an [`ExecutionPolicy`]; that is a real job and a different one.
//!
//! [`ExecutionPolicy`]: incin_core::exec::ExecutionPolicy
//!
//! # Why the target carries a float dtype
//!
//! [`EngineOn<D>`](crate::target::EngineOn) maps *engine × device* to a
//! backend family. Carrying a default float dtype as an associated type on
//! [`TensorTarget`] rather than a generic parameter keeps type inference
//! total: `Cpu` is a target at `f32`, and
//! [`dtype`](DtypeTarget::dtype) produces the view for any other
//! dtype.
//!
//! The float is the dtype of *generated* tensors and layer parameters only.
//! Data tensors take their dtype from the data and are never cast — see
//! [`TargetExt::tensor`].

use core::fmt::Debug;
use core::hash::Hash;

use alloc::vec::Vec;

use incin_core::backend_authoring::{
    Backend, HostInterop, StorageBackend, SupportsDType, VariableBackend,
};
use incin_core::error::{Error, Result};
pub use incin_core::exec::precision;
pub use incin_core::exec::{PrecisionSpec, RuntimePrecisionPolicy};
use incin_core::shapes::dynamic::Dyn;
use incin_core::shapes::{ConstDim, Dim, DimCons, DynShape, Nil, Shape, ShapeBuf, ShapeSpec};
use incin_core::tensor::base::Tensor;
#[cfg(any(feature = "cpu", feature = "external-candle"))]
use incin_core::tensor::device::Cpu;
#[cfg(feature = "metal")]
use incin_core::tensor::device::Metal;
#[cfg(feature = "cuda")]
use incin_core::tensor::device::{Cuda, CudaN};
use incin_core::tensor::device::{Device, DeviceId};
#[cfg(feature = "wgpu")]
use incin_core::tensor::device::{Wgpu, WgpuN};
use incin_core::tensor::dtype::{
    BuiltinDType, ConstDType, DType, DTypeDescriptor, FloatDType, PlainDType,
};
use incin_core::tensor::grad::{Grad, NoGrad, RequiresGrad};

/// A place tensors and parameters can be allocated: a device selector plus the
/// float dtype generated allocations should use.
///
/// Implemented by the device values themselves at their default float
/// (`f32`), and by [`DtypeView`] for every other dtype.
pub trait TensorTarget {
    /// The dtype generated tensors are created at.
    type Dtype: DType;

    /// The dtype stored layer parameters are created at.
    type ParameterDtype: DType;

    /// The device selector this target allocates on.
    type Device: Device;

    /// The backend family owned by this target.
    type Backend: Backend<Device = Self::Device> + VariableBackend;

    /// The selector value the device's own constructor argument needs.
    fn device_arg(&self) -> <Self::Device as Device>::Arg;

    /// Generated dtype field.
    fn dtype_field(&self) -> <Self::Dtype as DType>::Field;

    /// Parameter dtype field.
    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field;

    /// Precision policy.
    fn precision_policy(&self) -> RuntimePrecisionPolicy;
}

/// The backend a target resolves to for element type `K`. Users never write this type.
pub type TargetBackendFor<T> = <T as TensorTarget>::Backend;

/// The default backend a target resolves to for its default generated dtype. Users never write this type.
pub type TargetBackend<T> = <T as TensorTarget>::Backend;

/// A tensor allocated on target `T` from data of element type `K`.
pub type TargetTensor<T, S, K> = Tensor<S, TargetBackend<T>, K, NoGrad>;

/// Rebinding the dtype a target generates.
pub trait DtypeTarget: TensorTarget + Sized + Clone {
    /// Rebinds this target to generate `K` instead of `Self::Dtype`.
    fn dtype<K: ConstDType>(&self) -> Result<DtypeView<Self, K>>
    where
        Self::Backend: SupportsDType<K>,
    {
        let device =
            <Self::Device as Device>::to_incin(&<Self::Device as Device>::init(self.device_arg()))?;
        let field = <K as DType>::init(());
        <Self::Backend as SupportsDType<K>>::resolve_dtype(&field, &device)?;
        Ok(DtypeView::new(self.clone(), field))
    }

    /// Rebinds this target to generate a dynamic runtime descriptor `Dyn`.
    fn dtype_dynamic(&self, descriptor: DTypeDescriptor) -> Result<DtypeView<Self, Dyn>>
    where
        Self::Backend: SupportsDType<Dyn>,
    {
        let device =
            <Self::Device as Device>::to_incin(&<Self::Device as Device>::init(self.device_arg()))?;
        let field = <Dyn as DType>::init(descriptor);
        <Self::Backend as SupportsDType<Dyn>>::resolve_dtype(&field, &device)?;
        Ok(DtypeView::new(self.clone(), field))
    }
}

impl<T: TensorTarget + Sized + Clone> DtypeTarget for T {}

/// A target bound to an explicit dtype `K`, delegating backend selection to the underlying target `T`.
#[derive(Debug, Clone)]
pub struct DtypeView<T, K: DType> {
    target: T,
    field: K::Field,
}

impl<T, K: DType> DtypeView<T, K> {
    pub(crate) const fn new(target: T, field: K::Field) -> Self {
        Self { target, field }
    }

    pub fn target(&self) -> &T {
        &self.target
    }
}

impl<T: TensorTarget, K: DType> TensorTarget for DtypeView<T, K> {
    type Dtype = K;
    type ParameterDtype = T::ParameterDtype;
    type Device = T::Device;
    type Backend = T::Backend;

    fn device_arg(&self) -> <Self::Device as Device>::Arg {
        self.target.device_arg()
    }

    fn dtype_field(&self) -> <K as DType>::Field {
        self.field.clone()
    }

    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        self.target.parameter_dtype_field()
    }

    fn precision_policy(&self) -> RuntimePrecisionPolicy {
        self.target.precision_policy()
    }
}

// ============================================================================
// Shape specification
// ============================================================================

// ============================================================================
// Rust data → tensor
// ============================================================================

/// Ordinary Rust data that carries its own shape and element type.
///
/// Nested arrays already encode both, which is why this exists instead of a
/// literal macro: `[[1.0f32, 2.0], [3.0, 4.0]]` is a `s![2, 2]` tensor of
/// `f32` with no new syntax and no dtype inference rules to learn.
pub trait TensorData {
    /// The element type. Taken from the data, never from the target.
    type Elem: PlainDType<Elem = Self::Elem> + BuiltinDType + bytemuck::Pod;
    /// The static shape the nesting describes.
    type Shape: Shape + DynShape;
    /// Flattens into row-major order.
    fn into_row_major(self) -> Vec<Self::Elem>;
    /// Returns the statically known dimensions represented by the data.
    fn dims() -> Vec<usize>;
}

/// Rank-1 and rank-2 Rust arrays. `typenum`'s `Const<N>`/`ToUInt` bridge turns
/// each array length into the type-level dimension a static shape needs, so
/// `[[1.0f32, 2.0], [3.0, 4.0]]` arrives as `s![2, 2]` without the caller
/// writing a shape at all.
///
/// Higher ranks follow the same pattern and are left out only because each one
/// is another impl; nothing here is rank-specific in principle.
macro_rules! impl_tensor_data {
    ($($elem:ty),* $(,)?) => {
        $(
            impl<const A: usize> TensorData for [$elem; A]
            where
                ConstDim<A>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<ConstDim<A>, Nil>;
                fn into_row_major(self) -> Vec<$elem> {
                    self.to_vec()
                }
                fn dims() -> Vec<usize> { alloc::vec![A] }
            }

            impl<const A: usize, const B: usize> TensorData for [[$elem; B]; A]
            where
                ConstDim<A>: Dim,
                ConstDim<B>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<ConstDim<A>, DimCons<ConstDim<B>, Nil>>;
                fn into_row_major(self) -> Vec<$elem> {
                    // Row-major: the outer array indexes the slowest axis, so
                    // flattening in declaration order is already correct.
                    self.iter().flat_map(|row| row.iter().copied()).collect()
                }
                fn dims() -> Vec<usize> { alloc::vec![A, B] }
            }
        )*
    };
}

impl_tensor_data!(f32, f64, u8, u32, i64);

// ============================================================================
// The allocation surface
// ============================================================================

/// Allocation and construction on a [`TensorTarget`].
///
/// Blanket-implemented, so implementing [`TensorTarget`] is enough to get all
/// of this. Import through the prelude; these are extension methods and only
/// resolve when the trait is in scope.
///
/// Every method here returns a [`NoGrad`] tensor. Data, labels, masks and
/// scratch buffers are not parameters, and making them track gradients by
/// default is the one decision that silently costs memory on every tensor a
/// program builds. Parameters get [`Grad`] from the layer that owns them —
/// see [`TargetExt::parameter`].
pub trait TargetExt: TensorTarget + Sized {
    /// The logical device this target allocates on.
    ///
    /// # Errors
    ///
    /// Propagates a device selector that cannot be resolved.
    fn device_id(&self) -> Result<DeviceId> {
        <Self::Device as Device>::to_incin(&<Self::Device as Device>::init(self.device_arg()))
    }

    /// A tensor holding `data`, keeping the data's own element type.
    ///
    /// The target's float dtype is **not** applied: `cpu.tensor([0_i64, 1])`
    /// is an `i64` tensor on an `f32` target, because silently casting a label
    /// vector to float is a bug that surfaces as bad accuracy rather than as a
    /// type error.
    ///
    /// # Errors
    ///
    /// Propagates backend allocation failure.
    fn tensor<D: TensorData>(&self, data: D) -> Result<TargetTensor<Self, D::Shape, D::Elem>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device> + HostInterop,
    {
        let values = data.into_row_major();
        let dims = D::dims();
        self.allocate_row_major::<D::Shape, D::Elem>(
            &values,
            dims,
            ShapeBuf::from_slice(&D::dims()),
        )
    }

    /// A tensor of `values` laid out row-major into `spec`'s shape.
    ///
    /// The counterpart to [`tensor`](Self::tensor) for data whose length is
    /// only known at runtime.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] when `values.len()` does not equal the
    /// resolved element count.
    fn tensor_from_vec<K: PlainDType<Elem = K> + BuiltinDType + bytemuck::Pod, Sp: ShapeSpec>(
        &self,
        values: Vec<K>,
        spec: Sp,
    ) -> Result<TargetTensor<Self, Sp::Shape, K>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device> + HostInterop,
    {
        let shape_val = spec.resolve()?;
        let dims = shape_val.dims();
        let shape_buf = incin_core::shapes::shape_buf_from_dims::<Sp::Shape>(
            incin_core::shapes::error::OperationKind::Storage,
            &dims,
        )?;
        self.allocate_row_major::<Sp::Shape, K>(&values, dims, shape_buf)
    }

    /// Shared tail of the data constructors: check the element count against
    /// the resolved shape, then hand the bytes to the backend.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] on a length disagreement.
    #[doc(hidden)]
    fn allocate_row_major<
        S: Shape + DynShape,
        K: PlainDType<Elem = K> + BuiltinDType + bytemuck::Pod,
    >(
        &self,
        values: &[K],
        dims: Vec<usize>,
        field: ShapeBuf,
    ) -> Result<TargetTensor<Self, S, K>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device> + HostInterop,
    {
        let expected = incin_core::shapes::ShapeBuf::from_slice(&dims)
            .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
        if expected != values.len() {
            return Err(Error::ShapeMismatch {
                op: "tensor",
                expected: alloc::vec![expected],
                got: alloc::vec![values.len()],
                msg: alloc::string::String::new(),
            });
        }
        let device = self.device_id()?;
        let bytes = bytemuck::cast_slice(values);
        let storage = <TargetBackend<Self> as HostInterop>::from_bytes::<K>(
            bytes,
            &dims,
            K::DTYPE.descriptor(),
            &device,
        )?;
        Tensor::try_from_storage(
            storage,
            field,
            <K as DType>::init(()),
            <Self::Device as Device>::init(self.device_arg()),
            <NoGrad as RequiresGrad>::init(()),
        )
    }

    /// A zero-filled tensor of the target's dtype.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn zeros<Sp: ShapeSpec>(&self, spec: Sp) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Zeros,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.generated_canonical::<incin_core::backend_authoring::op::Zeros, Sp>(spec)
    }

    /// A one-filled tensor of the target's dtype.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn ones<Sp: ShapeSpec>(&self, spec: Sp) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Ones,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.generated_canonical::<incin_core::backend_authoring::op::Ones, Sp>(spec)
    }

    /// A uniform `[0, 1)` tensor of the target's dtype.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn rand<Sp: ShapeSpec>(&self, spec: Sp) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::UniformRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
        Self::Dtype: FloatDType,
    {
        self.generated_canonical::<incin_core::backend_authoring::op::UniformRandom, Sp>(spec)
    }

    /// A standard-normal tensor of the target's dtype.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn randn<Sp: ShapeSpec>(&self, spec: Sp) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::NormalRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
        Self::Dtype: FloatDType,
    {
        self.generated_canonical::<incin_core::backend_authoring::op::NormalRandom, Sp>(spec)
    }

    /// A tensor with every element set to `value`.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn full<Sp: ShapeSpec>(
        &self,
        spec: Sp,
        value: f64,
    ) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Full,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.canonical_creation::<incin_core::backend_authoring::op::Full, Sp>(
            spec,
            |shape, dtype, device| incin_core::backend_authoring::operations::FullAttributes {
                shape,
                dtype,
                device,
                value,
            },
        )
    }

    /// A tensor stepping from `start` by `step`, in row-major order.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn arange<Sp: ShapeSpec>(
        &self,
        spec: Sp,
        start: f64,
        step: f64,
    ) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Arange,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.canonical_creation::<incin_core::backend_authoring::op::Arange, Sp>(
            spec,
            |shape, dtype, device| incin_core::backend_authoring::operations::ArangeAttributes {
                shape,
                dtype,
                device,
                start,
                step,
            },
        )
    }

    /// A tensor of linearly spaced values from `start` to `end`.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    fn linspace<Sp: ShapeSpec>(
        &self,
        spec: Sp,
        start: f64,
        end: f64,
    ) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Linspace,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.canonical_creation::<incin_core::backend_authoring::op::Linspace, Sp>(
            spec,
            |shape, dtype, device| incin_core::backend_authoring::operations::LinspaceAttributes {
                shape,
                dtype,
                device,
                start,
                end,
            },
        )
    }

    /// Wrap a freshly allocated storage handle as a `NoGrad` tensor.
    #[doc(hidden)]
    fn finish<S: Shape>(
        &self,
        storage: <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
        field: ShapeBuf,
    ) -> Result<TargetTensor<Self, S, Self::Dtype>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>,
    {
        Tensor::try_from_storage(
            storage,
            field,
            self.dtype_field(),
            <Self::Device as Device>::init(self.device_arg()),
            <NoGrad as RequiresGrad>::init(()),
        )
    }

    /// Any zero-operand fill, routed through canonical dispatch.
    #[doc(hidden)]
    fn generated_canonical<O, Sp: ShapeSpec>(
        &self,
        spec: Sp,
    ) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        O: incin_core::exec::CanonicalOperation
            + incin_core::exec::Operation<
                Attributes = incin_core::backend_authoring::operations::CreationAttributes,
            >,
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                O,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        self.canonical_creation::<O, Sp>(spec, |shape, dtype, device| {
            incin_core::backend_authoring::operations::CreationAttributes {
                shape,
                dtype,
                device,
            }
        })
    }

    /// Any zero-operand creation, routed through canonical dispatch.
    #[doc(hidden)]
    fn canonical_creation<O, Sp: ShapeSpec>(
        &self,
        spec: Sp,
        build: impl FnOnce(
            Vec<usize>,
            DTypeDescriptor,
            DeviceId,
        ) -> <O as incin_core::exec::Operation>::Attributes,
    ) -> Result<TargetTensor<Self, Sp::Shape, Self::Dtype>>
    where
        O: incin_core::exec::CanonicalOperation,
        O::Attributes: incin_core::exec::AttributeContract,
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                O,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        let shape_val = spec.resolve()?;
        let dims = shape_val.dims();
        let device = self.device_id()?;
        let dtype_field = self.dtype_field();
        let dtype_descriptor = <TargetBackend<Self> as SupportsDType<Self::Dtype>>::resolve_dtype(
            &dtype_field,
            &device,
        )?;
        let context = incin_core::backend_authoring::ExecutionContext::new(
            <TargetBackend<Self> as Default>::default(),
        )
        .with_grad_mode(incin_core::exec::GradMode::Disabled)
        .with_precision_policy(self.precision_policy());
        let storage = incin_core::exec::dispatch::execute_shaped::<O, _, Sp::Shape>(
            &context,
            build(dims.clone(), dtype_descriptor, device),
            &[],
            &shape_val,
        )
        .map_err(Error::from)?;
        let shape_buf = incin_core::shapes::shape_buf_from_dims::<Sp::Shape>(
            incin_core::shapes::error::OperationKind::Storage,
            &dims,
        )?;
        Tensor::try_from_storage(
            storage,
            shape_buf,
            dtype_field,
            <Self::Device as Device>::init(self.device_arg()),
            <NoGrad as RequiresGrad>::init(()),
        )
    }

    /// A non-gradient-tracking state allocation at parameter dtype, for layer buffers/state.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    #[allow(clippy::type_complexity)]
    fn state_tensor<Sp: ShapeSpec>(
        &self,
        spec: Sp,
        fill: GeneratedFill,
    ) -> Result<Tensor<Sp::Shape, TargetBackend<Self>, Self::ParameterDtype, NoGrad>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::ParameterDtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Zeros,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Ones,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::UniformRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::NormalRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::exec::Capabilities
            + Default,
        Self::ParameterDtype: FloatDType,
        Self: Clone,
    {
        let shape_val = spec.resolve()?;
        let dims = shape_val.dims();
        let device = self.device_id()?;
        let param_dtype_field = self.parameter_dtype_field();
        let dtype_id = <TargetBackend<Self> as SupportsDType<Self::ParameterDtype>>::resolve_dtype(
            &param_dtype_field,
            &device,
        )?;
        let context = incin_core::backend_authoring::ExecutionContext::new(
            <TargetBackend<Self> as Default>::default(),
        )
        .with_grad_mode(incin_core::exec::GradMode::Disabled)
        .with_precision_policy(self.precision_policy());

        let storage = match fill {
            GeneratedFill::Zeros => incin_core::exec::dispatch::execute_shaped::<
                incin_core::backend_authoring::op::Zeros,
                _,
                Sp::Shape,
            >(
                &context,
                incin_core::backend_authoring::operations::CreationAttributes {
                    shape: dims.clone(),
                    dtype: dtype_id,
                    device,
                },
                &[],
                &shape_val,
            )?,
            GeneratedFill::Ones => incin_core::exec::dispatch::execute_shaped::<
                incin_core::backend_authoring::op::Ones,
                _,
                Sp::Shape,
            >(
                &context,
                incin_core::backend_authoring::operations::CreationAttributes {
                    shape: dims.clone(),
                    dtype: dtype_id,
                    device,
                },
                &[],
                &shape_val,
            )?,
            GeneratedFill::Uniform => incin_core::exec::dispatch::execute_shaped::<
                incin_core::backend_authoring::op::UniformRandom,
                _,
                Sp::Shape,
            >(
                &context,
                incin_core::backend_authoring::operations::CreationAttributes {
                    shape: dims.clone(),
                    dtype: dtype_id,
                    device,
                },
                &[],
                &shape_val,
            )?,
            GeneratedFill::Normal => incin_core::exec::dispatch::execute_shaped::<
                incin_core::backend_authoring::op::NormalRandom,
                _,
                Sp::Shape,
            >(
                &context,
                incin_core::backend_authoring::operations::CreationAttributes {
                    shape: dims.clone(),
                    dtype: dtype_id,
                    device,
                },
                &[],
                &shape_val,
            )?,
        };
        let shape_buf = incin_core::shapes::shape_buf_from_dims::<Sp::Shape>(
            incin_core::shapes::error::OperationKind::Storage,
            &dims,
        )?;
        Tensor::try_from_storage(
            storage,
            shape_buf,
            param_dtype_field,
            <Self::Device as Device>::init(self.device_arg()),
            <NoGrad as RequiresGrad>::init(()),
        )
    }

    /// A gradient-tracking allocation, for layer parameters.
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure.
    #[allow(clippy::type_complexity)]
    fn parameter<Sp: ShapeSpec>(
        &self,
        spec: Sp,
        fill: GeneratedFill,
    ) -> Result<Tensor<Sp::Shape, TargetBackend<Self>, Self::ParameterDtype, Grad>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::ParameterDtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Zeros,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Ones,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::UniformRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::NormalRandom,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::ParameterDtype>,
            > + incin_core::exec::Capabilities
            + Default,
        Self::ParameterDtype: FloatDType,
        Self: Clone,
    {
        Ok(self.state_tensor(spec, fill)?.require_grad())
    }
}

impl<T: TensorTarget + Sized> TargetExt for T {}

/// Which generated-tensor kernel [`TargetExt::generated_canonical`] should call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedFill {
    /// Every element zero.
    Zeros,
    /// Every element one.
    Ones,
    /// Uniform over `[0, 1)`.
    Uniform,
    /// Standard normal.
    Normal,
}

// ============================================================================
// Device values as targets
// ============================================================================

#[cfg(feature = "cpu")]
macro_rules! impl_unit_arg_target {
    ($($device:ty),* $(,)?) => {
        $(
            impl TensorTarget for $device {
                type Dtype = f32;
                type ParameterDtype = f32;
                type Device = Self;
                type Backend = <Native as EngineOn<Self>>::Backend;
                fn device_arg(&self) {}
                fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
                    <Self::Dtype as DType>::init(())
                }
                fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
                    <Self::ParameterDtype as DType>::init(())
                }
                fn precision_policy(&self) -> RuntimePrecisionPolicy {
                    RuntimePrecisionPolicy::default()
                }
            }
        )*
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
macro_rules! impl_self_arg_target {
    ($($device:ty),* $(,)?) => {
        $(
            impl TensorTarget for $device {
                type Dtype = f32;
                type ParameterDtype = f32;
                type Device = Self;
                type Backend = <Native as EngineOn<Self>>::Backend;
                fn device_arg(&self) -> Self {
                    self.clone()
                }
                fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
                    <Self::Dtype as DType>::init(())
                }
                fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
                    <Self::ParameterDtype as DType>::init(())
                }
                fn precision_policy(&self) -> RuntimePrecisionPolicy {
                    RuntimePrecisionPolicy::default()
                }
            }
        )*
    };
}

#[cfg(feature = "cpu")]
impl_unit_arg_target!(Cpu);

#[cfg(feature = "cuda")]
impl_self_arg_target!(Cuda);

#[cfg(feature = "wgpu")]
impl_self_arg_target!(Wgpu);

#[cfg(feature = "metal")]
impl_self_arg_target!(Metal);

// ============================================================================
// Layer initialization
// ============================================================================
// Engine & Precision abstractions (Target<E, D, P>)
// ============================================================================

/// Trait implemented by type-level execution engines (`Native`, `Candle`, `Dyn`).
pub trait EngineSpec: 'static + Send + Sync + Copy + Debug + Eq + PartialEq + Hash {
    /// Associated state carried by runtime instances of this engine.
    type Field: Clone + Send + Sync + 'static + Debug;
}

/// Execution engine marker for Incin's native backends (CPU, CUDA, WGPU, Metal).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Native;

impl EngineSpec for Native {
    type Field = ();
}

/// Execution engine marker for the Candle backend.
#[cfg(feature = "external-candle")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Candle;

#[cfg(feature = "external-candle")]
impl EngineSpec for Candle {
    type Field = ();
}

/// Runtime engine selection tag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RuntimeEngine {
    Native,
    #[cfg(feature = "external-candle")]
    Candle,
}

impl EngineSpec for Dyn {
    type Field = RuntimeEngine;
}

/// Maps an execution engine `E` and a physical device `D` to a backend family.
pub trait EngineOn<D: Device>: EngineSpec {
    type Backend: Backend<Device = D> + VariableBackend;
}

/// Backend type selected by engine `E` on physical device `D`.
pub type EngineBackend<E, D> = <E as EngineOn<D>>::Backend;

/// Backend type selected by the `Native` engine on physical device `D`.
pub type NativeBackend<D> = <Native as EngineOn<D>>::Backend;

impl EngineOn<Dyn> for Native {
    type Backend = crate::dispatch::DispatchBackend<Dyn>;
}

#[cfg(feature = "cpu")]
impl EngineOn<Cpu> for Native {
    type Backend = crate::cpu::CpuBackendImpl<Cpu>;
}

#[cfg(feature = "cuda")]
impl EngineOn<Cuda> for Native {
    type Backend = crate::cuda::CudaBackendImpl<Cuda>;
}

#[cfg(feature = "cuda")]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<CudaN<O>> for Native {
    type Backend = crate::cuda::CudaBackendImpl<CudaN<O>>;
}

#[cfg(feature = "wgpu")]
impl EngineOn<Wgpu> for Native {
    type Backend = crate::wgpu::WgpuBackendImpl<Wgpu>;
}

#[cfg(feature = "wgpu")]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<WgpuN<O>> for Native {
    type Backend = crate::wgpu::WgpuBackendImpl<WgpuN<O>>;
}

#[cfg(feature = "metal")]
impl EngineOn<Metal> for Native {
    type Backend = crate::metal::MetalBackendImpl<Metal>;
}

#[cfg(feature = "external-candle")]
impl EngineOn<Cpu> for Candle {
    type Backend = crate::external::candle::CandleBackend<Cpu>;
}

#[cfg(all(feature = "external-candle", feature = "cuda"))]
impl EngineOn<Cuda> for Candle {
    type Backend = crate::external::candle::CandleBackend<Cuda>;
}

#[cfg(all(feature = "external-candle", feature = "cuda"))]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<CudaN<O>> for Candle {
    type Backend = crate::external::candle::CandleBackend<CudaN<O>>;
}

/// An engine-aware, physical-device-placed, precision-configured tensor target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target<E, D, P = precision::Default>
where
    E: EngineSpec,
    D: Device,
    P: PrecisionSpec,
{
    pub(crate) engine: E::Field,
    pub(crate) device: D::Arg,
    pub(crate) precision: P::Field,
}

impl<E, D, P> Target<E, D, P>
where
    E: EngineSpec,
    D: Device,
    P: PrecisionSpec,
{
    /// Creates a new target from explicit engine, device argument, and precision policy fields.
    pub fn new(engine: E::Field, device: D::Arg, precision: P::Field) -> Self {
        Self {
            engine,
            device,
            precision,
        }
    }

    /// Rebinds the precision policy of this target, returning a new `Target`.
    pub fn with_precision<P2: PrecisionSpec>(self, policy: P2) -> Target<E, D, P2> {
        Target {
            engine: self.engine,
            device: self.device,
            precision: policy.init_field(),
        }
    }

    /// Rebinds the target to use a dynamic runtime precision policy.
    pub fn with_runtime_precision(self, policy: RuntimePrecisionPolicy) -> Target<E, D, Dyn> {
        Target {
            engine: self.engine,
            device: self.device,
            precision: policy,
        }
    }
}

impl<E, D, P> TensorTarget for Target<E, D, P>
where
    E: EngineOn<D>,
    D: Device,
    P: PrecisionSpec,
{
    type Dtype = P::GeneratedDType;
    type ParameterDtype = P::ParameterDType;
    type Device = D;
    type Backend = E::Backend;

    fn device_arg(&self) -> D::Arg {
        self.device.clone()
    }

    fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
        P::generated_dtype_field(&self.precision)
    }

    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        P::parameter_dtype_field(&self.precision)
    }

    fn precision_policy(&self) -> RuntimePrecisionPolicy {
        P::runtime_policy(&self.precision)
    }
}

impl Native {
    /// Binds the `Native` engine to a physical device target.
    pub fn on<T: TensorTarget>(target: T) -> Target<Native, T::Device, precision::Default> {
        Target {
            engine: (),
            device: target.device_arg(),
            precision: (),
        }
    }
}

#[cfg(feature = "external-candle")]
impl Candle {
    /// Binds the `Candle` engine to a physical device target.
    pub fn on<T: TensorTarget>(target: T) -> Target<Candle, T::Device, precision::Default> {
        Target {
            engine: (),
            device: target.device_arg(),
            precision: (),
        }
    }
}
