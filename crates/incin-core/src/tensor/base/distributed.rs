use super::Tensor;
use crate::backend_authoring::{Backend, SupportsDType};
use crate::dist::{Placement, PlacementKind};
use crate::shapes::{Dyn, Shape, ShapeValue};
use crate::tensor::device::{Device, DeviceId};
use crate::tensor::dtype::{DType, DTypeDescriptor};
use crate::tensor::grad::RequiresGrad;
use alloc::string::ToString;
use core::marker::PhantomData;

/// Failure while joining a distributed proof to one rank's physical storage.
///
/// This remains separate from [`Error`](crate::err::Error): placement APIs
/// are preview-gated, while the central error enum must not change its
/// matchable variants when a Cargo feature is toggled.
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
    /// Gradient tracking was requested for a non-floating dtype.
    #[error(
        "gradient tracking requires a floating dtype, got {dtype:?} on backend {backend} for {op}"
    )]
    GradientDType {
        /// Dtype the placement proof requires.
        dtype: DTypeDescriptor,
        /// Backend name involved in the rejection.
        backend: &'static str,
        /// Operation identity string.
        op: &'static str,
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

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement> Tensor<S, B, K, G, P> {
    /// Join one rank's storage to a sealed distributed lowering proof.
    ///
    /// Static `S`, `K`, device, and `P` choices retain their trait-level
    /// guarantees. Any [`Dyn`] choice is checked here against physical storage
    /// and the proof before the tensor can be constructed.
    pub fn try_from_distributed_storage<O>(
        inner: B::Storage<K>,
        global_shape: crate::shapes::ShapeBuf,
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
        if G::requires_grad(&grad) && !K::descriptor(&dtype).is_float() {
            return Err(PlacedTensorError::GradientDType {
                dtype: K::descriptor(&dtype),
                backend: B::BACKEND_NAME,
                op: "gradient tracking",
            });
        }
        let shape = ShapeValue::<S>::try_new(global_shape.clone())
            .map_err(|error| PlacedTensorError::Shape(error.to_string()))?;
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
            _shape: shape,
            _dtype: dtype,
            _device: device,
            _grad: grad,
            _placement: placement,
            _layout: PhantomData,
        })
    }

    /// Reshard a statically placed tensor through a compile-time legal
    /// transition and a sealed runtime proof.
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
