use crate::backend_authoring::{Descriptor, Execute};
use crate::dist::{Local, Placement, PlacementKind};
use crate::exec::Capabilities;
use crate::exec::catalog::{
    ArangeAttributes, CreationAttributes, FullAttributes, LinspaceAttributes, op,
};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::prelude::{
    ArgInto, Backend, BuiltinDType, ConstDType, DType, DTypeDescriptor, DTypeId, Device, DeviceId,
    DynShape, Error, Grad, NoGrad, RequiresGrad, Result, Shape, ShapeBuf, ShapeValue,
    SupportsDType, TensorArgs, TransferTo,
};
use crate::tensor::dtype::PlainDType;
use alloc::string::ToString;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// A marker used as `Shape`, `DType`, `Device`, or their runtime-chosen
/// variant across `Tensor`'s type parameters, deferring that choice from
/// compile time to runtime (e.g. `Tensor<Dyn, B>` has a shape resolved at
/// construction rather than baked into the type).
pub struct Dyn(());

impl Dyn {
    #[inline]
    pub(crate) const fn marker() -> Self {
        Self(())
    }
}

/// The core `Tensor` type representing an n-dimensional array.
///
/// It holds a reference to a backend-specific tensor representation, while statically tracking
/// its `Shape`, `Backend` (which includes `DType` and `Device`), and its `Grad` requirements.
///
/// `Tensor` is the primary workhorse of the Incin framework. By maintaining shape information
/// directly in the type signature, Incin ensures that tensor operations such as matrix multiplication
/// or convolutions are strictly verified at compile time.
///
/// ## Type Parameters
/// * `S`: The [`Shape`] of the tensor. This can be static (e.g., `s![2, 3, 224, 224]`), dynamic (`Dyn`), or partially dynamic.
/// * `B`: The underlying compute [`Backend`]. It defines how the tensor is stored in memory and how mathematical operations are executed.
/// * `K`: Element [`DType`], which may also be [`Dyn`] and runtime-checked.
/// * `G`: Trait marker representing whether the tensor requires gradients ([`Grad`] or [`NoGrad`]). Defaults to `NoGrad`.
/// * `P`: Logical [`Placement`]. Defaults to [`Local`]; distributed code may
///   select a static placement or [`Dyn`] for runtime placement metadata.
///
/// ## Examples
///
/// Creating and inspecting statically shaped tensors:
/// ```rust
/// # extern crate incin_core as incin;
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
/// // Compile-time 3D tensor of shape [2, 5, 10]
/// let t = Tensor::<s![2, 5, 10], DefaultBackend>::zeros(()).unwrap();
///
/// assert_eq!(t.dims(), [2, 5, 10]);
/// ```
///
/// Using dynamically shaped tensors:
/// ```rust
/// # extern crate incin_core as incin;
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
/// // Shape determined at runtime
/// let dyn_t = Tensor::<Dyn, DefaultBackend>::ones(vec![32, 64]).unwrap();
///
/// assert_eq!(dyn_t.dims(), vec![32, 64]);
/// ```
pub struct Tensor<
    S: Shape,
    B: Backend,
    K: DType = f32,
    G: RequiresGrad = NoGrad,
    P: Placement = Local,
> {
    pub(crate) inner: B::Storage<K>,
    /// Global logical shape. Backend storage carries this rank's local shape.
    pub(crate) _shape: ShapeValue<S>,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: <B::Device as Device>::Field,
    pub(crate) _grad: G::Field,
    pub(crate) _placement: P::Field,
}

/// Proof that raw storage and tensor metadata may be joined without repeating
/// their invariant checks inside the constructor.
///
/// This type and both of its variants are private to this module. Other
/// modules must use `try_from_storage`; metadata-only retagging is implemented
/// here, where the old tensor is available as the proof source.
#[derive(Clone, Copy)]
enum ConstructionWitness {
    StorageValidated,
    MetadataPreserved,
}

/// Failure while joining a distributed proof to one rank's physical storage.
///
/// This remains separate from [`Error`]: placement APIs are preview-gated,
/// while the central error enum must not change its matchable variants when a
/// Cargo feature is toggled.
#[cfg(feature = "distributed")]
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PlacedTensorError {
    /// Reconstructing the validated logical shape at a placement boundary
    /// failed its frontend proof.
    #[error("invalid tensor shape at placement boundary: {0}")]
    Shape(alloc::string::String),
    /// The tensor's static/dynamic global shape disagrees with the proof.
    #[error("tensor global shape {tensor:?} does not match distributed proof {proof:?}")]
    GlobalShape {
        /// Shape represented by canonical `ShapeBuf` dimensions.
        tensor: alloc::vec::Vec<usize>,
        /// Shape carried by the sealed proof.
        proof: alloc::vec::Vec<usize>,
    },
    /// The tensor placement parameter disagrees with the proof's output.
    #[error("tensor placement {tensor:?} does not match distributed proof {proof:?}")]
    OutputPlacement {
        /// Static or runtime tensor placement.
        tensor: PlacementKind,
        /// Placement carried by the sealed proof.
        proof: PlacementKind,
    },
    /// The requested rank has no local result in the proof.
    #[error("rank {rank} is outside a distributed result with {ranks} local values")]
    RankOutOfRange {
        /// Requested rank.
        rank: usize,
        /// Number of rank-local results in the proof.
        ranks: usize,
    },
    /// Physical rank-local storage has the wrong shape.
    #[error("rank {rank} storage shape {storage:?} does not match proof {proof:?}")]
    LocalShape {
        /// Requested rank.
        rank: usize,
        /// Shape reported by backend storage.
        storage: alloc::vec::Vec<usize>,
        /// Expected local shape.
        proof: alloc::vec::Vec<usize>,
    },
    /// Physical storage has the wrong runtime dtype.
    #[error("tensor dtype {expected:?} does not match rank-local storage {got:?}")]
    DType {
        /// Dtype selected statically or through `Dyn`.
        expected: DTypeDescriptor,
        /// Dtype reported by storage.
        got: DTypeDescriptor,
    },
    /// Physical storage is attached to the wrong runtime device.
    #[error("tensor device {expected:?} does not match rank-local storage {got:?}")]
    Device {
        /// Device selected by the tensor field.
        expected: DeviceId,
        /// Device reported by storage.
        got: DeviceId,
    },
    /// A static/runtime device or dtype selection could not be resolved.
    #[error("cannot resolve placed tensor metadata: {message}")]
    MetadataResolution {
        /// Underlying typed resolution failure.
        message: alloc::string::String,
    },
    /// A sealed proof does not describe the tensor's current placement.
    #[error("distributed proof expects input {proof:?}, tensor is {tensor:?}")]
    InputPlacement {
        /// Placement of the tensor being resharded.
        tensor: PlacementKind,
        /// Placement expected by the proof.
        proof: PlacementKind,
    },
    /// Runtime placement transition validation failed.
    #[error(transparent)]
    Distributed(#[from] crate::dist::DistributedError),
    /// A static transition proof and the sealed runtime proof disagree.
    #[error("static transition {expected:?} does not match distributed proof {proof:?}")]
    Transition {
        /// Transition selected by `LegalTransition`.
        expected: crate::dist::PlacementTransition,
        /// Transition carried by the sealed proof.
        proof: crate::dist::PlacementTransition,
    },
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Clone
    for Tensor<S, B, K, G, P>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: self._grad.clone(),
            _placement: self._placement.clone(),
        }
    }
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Tensor<S, B, K, G, P> {
    #[inline]
    /// Returns a reference to the backend-specific rank-local storage handle.
    pub fn inner(&self) -> &B::Storage<K> {
        &self.inner
    }

    #[inline]
    /// Consumes the tensor and returns its rank-local storage handle.
    pub fn into_inner(self) -> B::Storage<K> {
        self.inner
    }

    #[inline]
    /// Returns the authoritative runtime logical shape buffer.
    pub fn shape_buf(&self) -> &crate::shapes::ShapeBuf {
        self._shape.shape_buf()
    }

    /// Rebuilds tensor metadata after a checked operation while retaining the
    /// tensor's placement marker and runtime placement field.
    pub(crate) fn from_shape_value_placed<T: Shape>(
        inner: B::Storage<K>,
        shape: ShapeValue<T>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        let expected = shape.shape_buf().as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(crate::err::Error::ShapeMismatch {
                op: "from_shape_value_placed",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: placement,
        })
    }

    pub(crate) fn from_shape_buf_placed<T: Shape>(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        T::validate_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Self::from_shape_value_placed(
            inner,
            ShapeValue::from_validated_buf(dims),
            dtype,
            device,
            grad,
            placement,
        )
    }

    /// Rebuilds placed tensor metadata and verifies that the backend returned
    /// the same rank-local shape.  Distributed proof construction uses the
    /// unchecked placement adapter because its local shape is validated
    /// against the proof separately; ordinary operation results use this
    /// boundary so storage and `ShapeValue` cannot diverge.
    pub(crate) fn from_shape_buf_placed_checked<T: Shape>(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        placement: P::Field,
    ) -> Result<Tensor<T, B, K, G, P>> {
        let expected = dims.as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(crate::err::Error::ShapeMismatch {
                op: "from_shape_buf_placed_checked",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Self::from_shape_buf_placed(inner, dims, dtype, device, grad, placement)
    }

    pub(crate) fn shape_buf_value(&self) -> ShapeBuf {
        self._shape.shape_buf().clone()
    }

    #[inline]
    /// Returns a reference to the gradient marker field.
    pub fn grad_field(&self) -> &G::Field {
        &self._grad
    }

    /// Runtime projection of the tensor's placement.
    #[must_use]
    pub fn placement(&self) -> PlacementKind {
        P::to_incin(&self._placement)
    }

    /// Rank whose local storage this tensor owns.
    #[must_use]
    pub fn rank_index(&self) -> usize {
        P::rank(&self._placement)
    }

    /// Shape of the rank-local physical storage.
    ///
    /// [`dims`](Self::dims) reports the global logical shape.
    #[must_use]
    pub fn local_dims(&self) -> alloc::vec::Vec<usize> {
        B::shape(&self.inner).as_ref().to_vec()
    }

    /// Returns the logical descriptor for this tensor's dtype.
    ///
    /// Works for all dtypes including custom third-party non-builtin dtypes.
    #[must_use]
    pub fn dtype(&self) -> crate::tensor::dtype::DTypeDescriptor {
        K::descriptor(&self._dtype)
    }

    /// Returns the `DTypeId` if this tensor's dtype is a built-in Incin dtype,
    /// or `None` for custom third-party dtypes.
    #[must_use]
    pub fn builtin_dtype_id(&self) -> Option<DTypeId> {
        K::descriptor(&self._dtype).builtin_id()
    }

    /// Alias for [`dtype`](Self::dtype).
    #[must_use]
    #[deprecated(note = "Use `.dtype()` instead")]
    pub fn dtype_descriptor(&self) -> crate::tensor::dtype::DTypeDescriptor {
        self.dtype()
    }

    /// Returns the physical device on which this rank-local storage resides.
    pub fn device(&self) -> Result<DeviceId> {
        B::Device::to_incin(&self._device)
    }

    /// Whether this tensor computes and accumulates gradients.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        G::requires_grad(&self._grad)
    }

    /// Join one rank's storage to a sealed distributed lowering proof.
    ///
    /// Static `S`, `K`, device, and `P` choices retain their trait-level
    /// guarantees. Any [`Dyn`] choice is checked here against physical storage
    /// and the proof before the tensor can be constructed.
    #[cfg(feature = "distributed")]
    pub fn try_from_distributed_storage<O>(
        inner: B::Storage<K>,
        global_shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        rank: usize,
        proof: &crate::dist::ValidatedDistributed<O>,
    ) -> core::result::Result<Self, PlacedTensorError>
    where
        O: crate::exec::ExecutionDescriptor,
        B: SupportsDType<K>,
    {
        let tensor_global = global_shape.as_ref().to_vec();
        let proof_global = proof.global_shape().dims().to_vec();
        if tensor_global != proof_global {
            return Err(PlacedTensorError::GlobalShape {
                tensor: tensor_global,
                proof: proof_global,
            });
        }

        let Some(placement) = P::try_from_incin(proof.output_placement(), rank) else {
            return Err(PlacedTensorError::OutputPlacement {
                tensor: P::to_incin(&P::Field::default()),
                proof: proof.output_placement(),
            });
        };

        let Some(expected_local) = proof.local_shapes().get(rank) else {
            return Err(PlacedTensorError::RankOutOfRange {
                rank,
                ranks: proof.local_shapes().len(),
            });
        };
        let storage_local = B::shape(&inner);
        if storage_local.as_ref() != expected_local.dims() {
            return Err(PlacedTensorError::LocalShape {
                rank,
                storage: storage_local.as_ref().to_vec(),
                proof: expected_local.dims().to_vec(),
            });
        }

        let expected_device = B::Device::to_incin(&device).map_err(|error| {
            PlacedTensorError::MetadataResolution {
                message: error.to_string(),
            }
        })?;
        let expected_dtype = B::resolve_dtype(&dtype, &expected_device).map_err(|error| {
            PlacedTensorError::MetadataResolution {
                message: error.to_string(),
            }
        })?;
        if let Some(got) = B::storage_dtype(&inner)
            && got != expected_dtype
        {
            return Err(PlacedTensorError::DType {
                expected: expected_dtype,
                got,
            });
        }
        if let Some(got) = B::storage_device(&inner)
            && got != expected_device
        {
            return Err(PlacedTensorError::Device {
                expected: expected_device,
                got,
            });
        }

        Ok(Self {
            inner,
            _shape: ShapeValue::from_validated(global_shape),
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: placement,
        })
    }

    /// Reshard a statically placed tensor through a compile-time legal
    /// transition and a sealed runtime proof.
    #[cfg(feature = "distributed")]
    pub fn try_reshard<To, O>(
        self,
        inner: B::Storage<K>,
        rank: usize,
        proof: &crate::dist::ValidatedDistributed<O>,
    ) -> core::result::Result<Tensor<S, B, K, G, To>, PlacedTensorError>
    where
        O: crate::exec::ExecutionDescriptor,
        P: crate::dist::ConstPlacement + crate::dist::LegalTransition<To>,
        To: crate::dist::ConstPlacement,
        B: SupportsDType<K>,
    {
        validate_reshard_proof(P::PLACEMENT, To::PLACEMENT, proof)?;
        if proof.transition() != P::TRANSITION {
            return Err(PlacedTensorError::Transition {
                expected: P::TRANSITION,
                proof: proof.transition(),
            });
        }
        let shape = self.shape_buf_value();
        Tensor::<S, B, K, G, To>::try_from_distributed_storage(
            inner,
            shape,
            self._dtype,
            self._device,
            self._grad,
            rank,
            proof,
        )
    }
}

#[cfg(feature = "distributed")]
impl<S: Shape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G, Dyn> {
    /// Reshard a runtime-placed tensor through the checked counterpart of
    /// `LegalTransition`.
    pub fn try_reshard_dyn<O>(
        self,
        inner: B::Storage<K>,
        to: PlacementKind,
        rank: usize,
        proof: &crate::dist::ValidatedDistributed<O>,
    ) -> core::result::Result<Self, PlacedTensorError>
    where
        O: crate::exec::ExecutionDescriptor,
        B: SupportsDType<K>,
    {
        let from = self.placement();
        validate_reshard_proof(from, to, proof)?;
        let expected = crate::dist::validate_transition(from, to)?;
        if proof.transition() != expected {
            return Err(PlacedTensorError::Transition {
                expected,
                proof: proof.transition(),
            });
        }
        let shape = self.shape_buf_value();
        Self::try_from_distributed_storage(
            inner,
            shape,
            self._dtype,
            self._device,
            self._grad,
            rank,
            proof,
        )
    }
}

#[cfg(feature = "distributed")]
fn validate_reshard_proof<O>(
    from: PlacementKind,
    to: PlacementKind,
    proof: &crate::dist::ValidatedDistributed<O>,
) -> core::result::Result<(), PlacedTensorError> {
    let Some(&proof_input) = proof.input_placements().as_slice().first() else {
        return Err(crate::dist::DistributedError::NoInputPlacements.into());
    };
    if proof_input != from {
        return Err(PlacedTensorError::InputPlacement {
            tensor: from,
            proof: proof_input,
        });
    }
    if proof
        .input_placements()
        .as_slice()
        .iter()
        .any(|&placement| placement != from)
    {
        let proof_input = proof
            .input_placements()
            .as_slice()
            .iter()
            .copied()
            .find(|&placement| placement != from)
            .unwrap_or(proof_input);
        return Err(PlacedTensorError::InputPlacement {
            tensor: from,
            proof: proof_input,
        });
    }
    if proof.output_placement() != to {
        return Err(PlacedTensorError::OutputPlacement {
            tensor: to,
            proof: proof.output_placement(),
        });
    }
    Ok(())
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G, Local> {
    /// Joins component parts after this module has witnessed their invariants.
    fn from_parts_witnessed(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
        _witness: ConstructionWitness,
    ) -> Self {
        Self {
            inner,
            _shape: ShapeValue::from_validated(shape),
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
        }
    }

    /// Reuses a validated logical shape without reconstructing a legacy
    /// representation.  This is the internal path for shape-preserving
    /// operations; `from_parts` remains the checked constructor for caller
    /// supplied shape arguments.
    pub(crate) fn from_shape_value(
        inner: B::Storage<K>,
        shape: ShapeValue<S>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        let expected = shape.shape_buf().as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(Error::ShapeMismatch {
                op: "from_shape_value",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Backend operation returned storage with an unexpected shape".into(),
            });
        }
        Ok(Self {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
        })
    }

    pub(crate) fn from_shape_value_unchecked(
        inner: B::Storage<K>,
        shape: ShapeValue<S>,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Self {
        Self {
            inner,
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: core::marker::PhantomData,
        }
    }

    pub(crate) fn from_shape_buf(
        inner: B::Storage<K>,
        dims: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        S::validate_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Self::from_shape_value(
            inner,
            ShapeValue::from_validated_buf(dims),
            dtype,
            device,
            grad,
        )
    }

    /// Creates a tensor from parts, checking that storage shape matches expected shape.
    pub fn try_from_storage(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        S::validate_dims(shape.as_ref()).map_err(crate::err::Error::Shape)?;
        let expected = shape.as_ref().to_vec();
        let got = B::shape(&inner);
        if expected != got.as_ref() {
            return Err(Error::ShapeMismatch {
                op: "from_parts",
                expected,
                got: got.as_ref().to_vec(),
                msg: "Runtime shape doesn't match expected static/dynamic shape".to_string(),
            });
        }
        let expected_dtype = K::descriptor(&dtype);
        if let Some(got) = B::storage_dtype(&inner)
            && expected_dtype != got
        {
            return Err(Error::DTypeStorageMismatch {
                expected: expected_dtype,
                got,
            });
        }
        let expected_device = B::Device::to_incin(&device)?;
        if let Some(got) = B::storage_device(&inner)
            && expected_device != got
        {
            return Err(Error::DeviceStorageMismatch {
                expected: expected_device,
                got,
            });
        }
        Ok(Self::from_parts_witnessed(
            inner,
            shape,
            dtype,
            device,
            grad,
            ConstructionWitness::StorageValidated,
        ))
    }

    pub(crate) fn from_parts(
        inner: B::Storage<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
        grad: G::Field,
    ) -> Result<Self> {
        Self::try_from_storage(inner, shape, dtype, device, grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> Tensor<S, B, K, G>
where
    (S, K, B::Device, G): TensorArgs<S, K, B::Device, G>,
    B: SupportsDType<K>,
{
    /// Creates a tensor filled with zeros.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Zeros> + Capabilities,
        <B as Execute<op::Zeros>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Zeros, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with ones.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Ones> + Capabilities,
        <B as Execute<op::Ones>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Ones, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a slice whose element type fixes its static dtype.
    ///
    /// Requires a [`PlainDType`]: dtypes with an actual Rust scalar element per
    /// logical value. Block-quantized dtypes (e.g. `Q8_0`) are rejected at
    /// compile time since they have no plain scalar slice representation.
    pub fn from_slice<A>(data: &[<K as PlainDType>::Elem], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        K: PlainDType + BuiltinDType,
        B: Execute<op::TensorFromData> + Capabilities,
        <B as Execute<op::TensorFromData>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let dims = _shape.clone();
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let byte_len = core::mem::size_of_val(data);
        let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped_with_payload::<op::TensorFromData, B, S>(
            &context,
            crate::exec::catalog::DataAttributes {
                shape: dims.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
            Some(bytes),
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a checked native-endian byte payload.
    pub fn from_bytes<A>(bytes: &[u8], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::TensorFromBytes> + Capabilities,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let dims = _shape.clone();
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped_with_payload::<op::TensorFromBytes, B, S>(
            &context,
            crate::exec::catalog::DataAttributes {
                shape: dims.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
            Some(bytes),
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with random values uniform in [0, 1).
    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::UniformRandom> + Capabilities,
        <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::UniformRandom, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with standard normal random values.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::NormalRandom> + Capabilities,
        <B as Execute<op::NormalRandom>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::NormalRandom, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with scalar `val`.
    pub fn full<Sc: Into<crate::tensor::backend::ScalarValue>, A>(val: Sc, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Full> + Capabilities,
        <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let scalar_f64 = val.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Full, B, S>(
            &context,
            FullAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                value: scalar_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor starting at `start` with step `step`.
    pub fn arange<Sc: Into<crate::tensor::backend::ScalarValue>, A>(
        start: Sc,
        step: Sc,
        args: A,
    ) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Arange> + Capabilities,
        <B as Execute<op::Arange>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let st_f64 = step.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Arange, B, S>(
            &context,
            ArangeAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                start: s_f64,
                step: st_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor with linearly spaced values between `start` and `end`.
    pub fn linspace<Sc: Into<crate::tensor::backend::ScalarValue>, A>(
        start: Sc,
        end: Sc,
        args: A,
    ) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Linspace> + Capabilities,
        <B as Execute<op::Linspace>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let e_f64 = end.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Linspace, B, S>(
            &context,
            LinspaceAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                start: s_f64,
                end: e_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    pub fn sample<D: crate::distributions::Distribution<K>, A>(dist: &D, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: SupportsDType<K> + crate::distributions::DistributionExecutor<D, K>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        dist.sample::<S, B, G>(_shape, &_device)
    }

    /// Wraps an existing backend storage in a Tensor.
    pub fn from_raw<A>(raw_tensor: B::Storage<K>, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        Self::from_parts(raw_tensor, _shape, _dtype, _device, _grad)
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, P: Placement>
    Tensor<S, B, K, G, P>
{
    #[inline]
    /// Returns the number of dimensions (rank) of the tensor.
    pub fn rank(&self) -> usize {
        self._shape.shape_buf().rank()
    }

    #[inline]
    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self._shape.shape_buf().numel().unwrap_or(0)
    }

    #[inline]
    /// Returns the dimensions of the tensor as a slice or container.
    pub fn dims(&self) -> crate::shapes::ShapeBuf {
        self._shape.shape_buf().clone()
    }
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Tensor<S, B, K, G, P> {
    /// Computes the backward pass starting from this tensor, returning the gradients.
    pub fn backward(&self) -> Result<crate::optim::Gradients<B::Grads>> {
        B::backward(&self.inner).map(crate::optim::Gradients::from_backend)
    }

    /// Moves this tensor to the specified device, returning a new Tensor.
    pub fn to_device<D2: Device>(
        &self,
        _device: &D2::Field,
    ) -> Result<Tensor<S, <B as TransferTo<D2>>::Output, K, G>>
    where
        B: TransferTo<D2>,
        <B as TransferTo<D2>>::Output: SupportsDType<K>,
    {
        let new_inner = B::transfer_storage(&self.inner, &self._dtype, _device)?;
        Tensor::from_shape_value(
            new_inner,
            self._shape.clone(),
            self._dtype.clone(),
            _device.clone(),
            self._grad.clone(),
        )
    }
}

impl<S1: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> Tensor<S1, B, K, G> {
    /// Converts this tensor to a new static shape S2.
    pub fn into_shape<S2: Shape + DynShape>(self) -> Result<Tensor<S2, B, K, G>> {
        let dims = self._shape.shape_buf();
        let s2_shape = S2::try_from_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Tensor::from_parts(self.inner, s2_shape, self._dtype, self._device, self._grad)
    }

    /// Converts this tensor to a dynamically-shaped `Tensor<Dyn>`.
    pub fn into_dyn(self) -> Tensor<crate::prelude::Dyn, B, K, G> {
        let dims = self._shape.shape_buf();
        // `Dyn`'s field *is* the dimension vector, so there is nothing to
        // re-parse and nothing that can fail — the old
        // The old optional raw-dimension conversion asserted that the input
        // was accepted. Building it directly makes that structural rather than
        // assumed, and is the last of the 39 sites `SHP-004` removes.
        let s2_shape = crate::shapes::ShapeBuf::from_slice(dims.as_ref());
        Tensor::from_shape_value_unchecked(
            self.inner,
            ShapeValue::from_validated(s2_shape),
            self._dtype,
            self._device,
            self._grad,
        )
    }

    /// Copies and converts this tensor to a new static shape S2.
    pub fn to_shape<S2: Shape + DynShape>(&self) -> Result<Tensor<S2, B, K, G>> {
        let dims = self._shape.shape_buf();
        let s2_shape = S2::try_from_dims(dims.as_ref()).map_err(crate::err::Error::Shape)?;
        Tensor::from_parts(
            self.inner.clone(),
            s2_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<S: Shape, B: Backend, K: DType> Tensor<S, B, K, NoGrad> {
    /// Marks this tensor to require gradient tracking.
    pub fn require_grad(self) -> Tensor<S, B, K, Grad> {
        Tensor::from_shape_value_unchecked(
            B::fresh_autograd_identity(self.inner),
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<S: Shape, B: Backend, K: DType> Tensor<S, B, K, Grad> {
    /// Detaches this tensor from autodiff tape tracking, returning a NoGrad tensor.
    pub fn detach(self) -> Tensor<S, B, K, NoGrad> {
        Tensor::from_shape_value_unchecked(
            B::fresh_autograd_identity(self.inner),
            self._shape,
            self._dtype,
            self._device,
            core::marker::PhantomData,
        )
    }
}

impl<
    S: crate::prelude::Shape,
    B: crate::prelude::Backend + crate::tensor::backend::TensorOps<B>,
    K: DType,
    G: RequiresGrad,
    P: Placement,
> core::fmt::Display for Tensor<S, B, K, G, P>
{
    /// Renders values the way PyTorch's `print(tensor)` does: the backend's
    /// bracketed, right-aligned value grid (`Backend::format_tensor_display`)
    /// wrapped in `tensor(...)`, with nested-bracket rows indented to stay
    /// aligned under the first `[` the way PyTorch's own wrapped output is.
    ///
    /// `dtype=`/`device=`/`requires_grad=` are appended only when they
    /// differ from what a reader would otherwise assume: `f32`
    /// (`DTypeId::default()`), `cpu:0` (`DeviceId::cpu()`), and — not
    /// requiring gradients. That last one is the mirror image of PyTorch's
    /// own rule for the literal reason that `G` defaults to
    /// [`NoGrad`](crate::prelude::NoGrad) here rather than `Grad`: printing
    /// `requires_grad=True` whenever `G::requires_grad` is true still means
    /// "printed exactly when true", while the default tensor remains inert.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let prefix = "tensor(";
        let body = B::format_tensor_display(&self.inner);
        f.write_str(prefix)?;
        for (i, line) in body.lines().enumerate() {
            if i > 0 {
                writeln!(f)?;
                write!(f, "{:1$}", "", prefix.chars().count())?;
            }
            f.write_str(line)?;
        }

        let dtype = self.dtype();
        if dtype != <f32 as ConstDType>::DESCRIPTOR {
            write!(f, ", dtype={}", dtype.name())?;
        }
        if let Ok(device) = <B::Device as Device>::to_incin(&self._device)
            && device != DeviceId::cpu()
        {
            write!(f, ", device={}:{}", device.kind().name(), device.ordinal())?;
        }
        if G::requires_grad(&self._grad) {
            f.write_str(", requires_grad=True")?;
        }
        f.write_str(")")
    }
}

impl<
    S: crate::prelude::Shape,
    B: crate::prelude::Backend + crate::tensor::backend::TensorOps<B>,
    K: DType,
    G: RequiresGrad,
    P: Placement,
> core::fmt::Debug for Tensor<S, B, K, G, P>
{
    /// Prints the backend type name, runtime shape, and the backend's own
    /// debug rendering of its storage.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor({}, global_shape={:?}, local_shape={:?}, placement={:?}, rank={})\n{}",
            core::any::type_name::<B>(),
            self._shape.shape_buf().as_ref(),
            B::shape(&self.inner).as_ref(),
            self.placement(),
            self.rank_index(),
            B::format_tensor_debug(&self.inner)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;

    #[test]
    fn test_tensor_creation() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dims(), vec![2, 3]);
    }

    #[test]
    fn test_tensor_ones() {
        let t: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<crate::prelude::Cpu>> =
            Tensor::ones(vec![4]).unwrap();
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    // `DummyBackend`'s conv/pool shape math must never panic on a
    /// pathological input, e.g. an input smaller than an (over-dilated)
    /// kernel plus padding — `2*padding + input` underflowing `dilation *
    /// (kernel - 1) + 1` used to panic via unchecked `usize` subtraction in
    /// debug builds (or silently wrap in release).
    fn dummy_backend_conv_pool_shape_math_never_panics_on_tiny_input_large_kernel() {
        use crate::backend_authoring::{Backend, ModuleOps};
        type B = crate::tensor::backend::dummy::DummyBackend<crate::prelude::Cpu>;

        // 1x1x2x2 input, a 5x5 kernel with dilation 3: `dilation*(kernel-1)+1`
        // = 3*4+1 = 13, far larger than `input + 2*padding` = 2 + 0 = 2.
        let input: <B as crate::tensor::backend::StorageBackend>::Storage<f32> =
            alloc::vec![1, 1, 2, 2];
        let weight: <B as crate::tensor::backend::StorageBackend>::Storage<f32> =
            alloc::vec![1, 1, 5, 5];
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&input, &weight, None, 1, 0, 3, 1).unwrap();
        assert_eq!(out.len(), 4);

        let pool_out =
            <B as ModuleOps<B>>::max_pool2d::<f32>(&input, (5, 5), (1, 1), (0, 0), (3, 3)).unwrap();
        assert_eq!(pool_out.len(), 4);
    }
}
