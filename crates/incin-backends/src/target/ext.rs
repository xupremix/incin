//! The allocation surface application code calls: `TargetExt`.

use super::*;

/// Allocation and construction on a [`TensorTarget`].
///
/// Blanket-implemented, so implementing [`TensorTarget`] is enough to get all
/// of this. Import through the prelude; these are extension methods and only
/// resolve when the trait is in scope.
///
/// Every method here returns a [`NoGrad`] tensor. Data, labels, masks and
/// scratch buffers are not parameters, and making them track gradients by
/// default is the one decision that silently costs memory on every tensor a
/// program builds. Parameters get [`Grad`] from the layer that owns them -
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 2]);
    /// # Ok::<(), Error>(())
    /// ```
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

    /// A tensor holding `data`, in a layout named by the caller.
    ///
    /// The layout-expressing counterpart to [`tensor`](Self::tensor). Nothing
    /// else differs: the same data, the same shape inference, the same dtype
    /// rule. What changes is that the result *claims* something about its
    /// memory order, and the claim is made by the value that chose the
    /// allocation rather than asserted next to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::shapes::dim::ConstDim;
    /// use incin_core::shapes::{Dense, DimCons, Nil, RowMajor};
    /// use incin_core::tensor::device::Cpu;
    ///
    /// // `incin-backends` has no `incin-macros` dependency, so the shape is
    /// // spelled out here; through the `incin` facade this is `s![2, 2]`.
    /// type S = DimCons<ConstDim<2>, DimCons<ConstDim<2>, Nil>>;
    ///
    /// // The layout is named in the turbofish, and the result carries it --
    /// // this annotation does not compile if the layout came back as `Dyn`.
    /// let x: Dense<S, _> = Cpu.tensor_in::<_, RowMajor<S>>(
    ///     [[1.0_f32, 2.0], [3.0, 4.0]],
    /// )?;
    /// assert_eq!(x.dims().as_ref(), &[2, 2]);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates backend allocation failure, and refuses a layout whose
    /// strides no backend can allocate yet -- see
    /// [`allocate_in`](Self::allocate_in).
    fn tensor_in<D: TensorData, L: incin_core::shapes::FreshLayout<D::Shape>>(
        &self,
        data: D,
    ) -> Result<TargetTensorIn<Self, D::Shape, D::Elem, L>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device> + HostInterop,
    {
        let values = data.into_row_major();
        let dims = D::dims();
        self.allocate_in::<D::Shape, D::Elem, L>(&values, dims, ShapeBuf::from_slice(&D::dims()))
    }

    /// A tensor of `values` laid out row-major into `spec`'s shape.
    ///
    /// The counterpart to [`tensor`](Self::tensor) for data whose length is
    /// only known at runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let values = vec![1.0_f32, 2.0, 3.0, 4.0];
    /// let x = Cpu.tensor_from_vec(values, [2, 2])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 2]);
    /// # Ok::<(), Error>(())
    /// ```
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
        self.allocate_in::<S, K, incin_core::shapes::Dyn>(values, dims, field)
    }

    /// The same, in a named layout.
    ///
    /// The layout-expressing seam under every `*_in` constructor. `L` picks the
    /// strides through [`FreshLayout::strides`], and the result is typed `L` --
    /// so the claim is made by the same value that chose the allocation rather
    /// than asserted alongside it.
    ///
    /// Today every backend uploads host bytes densely, so a layout whose
    /// strides are not the dense ones is **refused** rather than quietly
    /// satisfied with a dense buffer wearing the wrong type. That refusal is
    /// the load-bearing part of the design: a creation API that cannot say no
    /// is a way to mint a proof, which is exactly what
    /// [`FreshDense`](incin_core::shapes::FreshDense) is sealed to prevent.
    ///
    /// The check is on the strides rather than on a list of known layouts, so
    /// a layout added later is handled without touching this function: if it
    /// asks for dense strides it works, and if it does not it is refused until
    /// a backend can honour it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] on a length disagreement, and a
    /// capability error when `L` asks for strides no backend can allocate yet.
    #[doc(hidden)]
    fn allocate_in<
        S: Shape + DynShape,
        K: PlainDType<Elem = K> + BuiltinDType + bytemuck::Pod,
        L: incin_core::shapes::FreshLayout<S>,
    >(
        &self,
        values: &[K],
        dims: Vec<usize>,
        field: ShapeBuf,
    ) -> Result<TargetTensorIn<Self, S, K, L>>
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
        let wanted = <L as incin_core::shapes::FreshLayout<S>>::strides(&dims);
        let dense = incin_core::shapes::dense_strides(&dims);
        if wanted.as_ref() != dense.as_ref() {
            return Err(incin_core::error::BackendError::unsupported(
                <TargetBackend<Self> as StorageBackend>::BACKEND_NAME,
                incin_core::exec::UnsupportedReason::Layout {
                    operation: incin_core::shapes::error::OperationKind::Storage,
                    layout: incin_core::exec::LayoutClass::Strided,
                },
            )
            .into());
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.zeros([2, 3])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 3]);
    /// # Ok::<(), Error>(())
    /// ```
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

    /// A zero-filled tensor of the target's dtype, in a layout named by the
    /// caller.
    ///
    /// The layout-expressing counterpart to [`zeros`](Self::zeros). Unlike the
    /// data constructors it does not upload anything, so the only question is
    /// whether the backend's creation kernel writes the strides `L` wants --
    /// today every one writes dense, so a layout asking for anything else is
    /// refused rather than quietly satisfied.
    ///
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::shapes::{Dyn, RowMajor, ShapeArgs};
    /// use incin_core::tensor::device::Cpu;
    ///
    /// // `ShapeArgs` is what carries a shape through `ShapeSpec`. Spelled with
    /// // `Dyn` here because `incin-backends` has no `incin-macros` dependency;
    /// // through the `incin` facade this reads `ShapeArgs<s![2, 3]>`, and
    /// // `crates/incin-core/tests/target_layout_examples.rs` has the static
    /// // form together with the `reshape_view` that a proof unlocks.
    /// let x = Cpu.zeros_in::<ShapeArgs<Dyn>, RowMajor<Dyn>>(
    ///     ShapeArgs::new(vec![2, 3]),
    /// )?;
    /// assert_eq!(x.dims().as_ref(), &[2, 3]);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates shape resolution and backend allocation failure, and refuses
    /// a layout whose strides the creation path cannot produce.
    fn zeros_in<Sp: ShapeSpec, L: incin_core::shapes::FreshLayout<Sp::Shape>>(
        &self,
        spec: Sp,
    ) -> Result<TargetTensorIn<Self, Sp::Shape, Self::Dtype, L>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>
            + incin_core::backend_authoring::SupportsDType<Self::Dtype>
            + incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::Zeros,
                Output = <TargetBackend<Self> as StorageBackend>::Storage<Self::Dtype>,
            > + incin_core::exec::Capabilities
            + Default,
    {
        let built =
            self.generated_canonical::<incin_core::backend_authoring::op::Zeros, Sp>(spec)?;
        Self::restate_layout::<Sp::Shape, Self::Dtype, L>(built)
    }

    /// Re-types a freshly created tensor in `L`, after checking that `L` asks
    /// for the strides a dense creation kernel actually wrote.
    ///
    /// The counterpart to [`allocate_in`](Self::allocate_in) for the paths that
    /// let the backend allocate rather than uploading bytes. Same rule, same
    /// reason: the type may only claim what the allocation did.
    ///
    /// # Errors
    ///
    /// Refuses a layout whose strides differ from the dense ones.
    #[doc(hidden)]
    fn restate_layout<S: Shape, K: DType, L: incin_core::shapes::FreshLayout<S>>(
        built: Tensor<S, TargetBackend<Self>, K, NoGrad>,
    ) -> Result<TargetTensorIn<Self, S, K, L>>
    where
        TargetBackend<Self>: Backend<Device = Self::Device>,
    {
        let dims = built.shape_buf().as_ref().to_vec();
        let wanted = <L as incin_core::shapes::FreshLayout<S>>::strides(&dims);
        if wanted.as_ref() != incin_core::shapes::dense_strides(&dims).as_ref() {
            return Err(incin_core::error::BackendError::unsupported(
                <TargetBackend<Self> as StorageBackend>::BACKEND_NAME,
                incin_core::exec::UnsupportedReason::Layout {
                    operation: incin_core::shapes::error::OperationKind::Storage,
                    layout: incin_core::exec::LayoutClass::Strided,
                },
            )
            .into());
        }
        // The refusal above is the *capability* answer -- whether the creation
        // path can produce these strides at all -- and this is the check that
        // it did. Both are wanted: the first gives a backend-shaped error for
        // an unsupported request, the second makes the claim conditional on the
        // metadata rather than on the argument above having been right.
        built.into_layout::<L>()
    }

    /// A one-filled tensor of the target's dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.ones([2, 3])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 3]);
    /// # Ok::<(), Error>(())
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.rand([2, 3])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 3]);
    /// # Ok::<(), Error>(())
    /// ```
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
    /// Samples from a standard normal distribution (mean 0, variance 1),
    /// matching the semantics of PyTorch's `torch.randn`.
    ///
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.randn([2, 3])?;
    /// assert_eq!(x.dims().as_ref(), &[2, 3]);
    /// # Ok::<(), Error>(())
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.full([2, 2], 7.0)?;
    /// assert_eq!(x.dims().as_ref(), &[2, 2]);
    /// # Ok::<(), Error>(())
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.arange([4], 1.0, 2.0)?;
    /// assert_eq!(x.dims().as_ref(), &[4]);
    /// # Ok::<(), Error>(())
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use incin_backends::prelude::*;
    /// use incin_core::error::Error;
    /// use incin_core::tensor::device::Cpu;
    ///
    /// let x = Cpu.linspace([3], 0.0, 1.0)?;
    /// assert_eq!(x.dims().as_ref(), &[3]);
    /// # Ok::<(), Error>(())
    /// ```
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
