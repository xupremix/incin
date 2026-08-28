//! Descriptor-to-metadata checks shared by every native executor.
//!
//! A binder's job is to prove that the storage it was handed is the storage the
//! sealed descriptor authorizes. For matmul each backend spells that out for
//! itself, because the checks are entangled with the backend's own transpose
//! handling. The live exact executors perform their metadata checks against
//! the typed descriptor inputs.
//!
//! Device residency and dtype stay with each backend, since those messages name
//! the backend that refused.

use alloc::string::ToString;

use incin_core::error::{BackendError, Error};
use incin_core::exec::UnsupportedReason;
use incin_core::shapes::error::OperationKind;

/// Build an `InvalidInput` error for a descriptor binder.
pub(crate) const fn invalid(operation: OperationKind, reason: &'static str) -> BackendError {
    BackendError::InvalidInput { operation, reason }
}

/// Classify a legacy kernel's failure for the descriptor path.
///
/// The op traits report a gap the backend declared at its own impl site as
/// [`Error::UnsupportedBackendOperation`]. That is a capability answer, not an
/// execution failure: nothing ran and nothing faulted, the backend simply never
/// had a kernel. Reporting it as [`BackendError::Execution`] would tell a caller
/// the device failed, and would split "this backend cannot do that" across two
/// error variants depending on whether the registry or the kernel noticed first.
/// It becomes the same [`BackendError::Unsupported`] a capability query returns.
/// `backend` is the name the refusing backend answers to; see
/// [`StorageBackend::BACKEND_NAME`](incin_core::backend_authoring::StorageBackend::BACKEND_NAME).
/// It is a parameter rather than something derived here because this helper is
/// shared by every executor, and the whole point of the name is that it comes
/// from the backend that actually refused.
pub(crate) fn kernel_error(
    backend: &'static str,
    operation: OperationKind,
    error: Error,
) -> BackendError {
    match error {
        Error::UnsupportedBackendOperation { .. } => {
            BackendError::unsupported(backend, UnsupportedReason::Operation { operation })
        }
        other => BackendError::Execution {
            operation,
            message: other.to_string().into(),
        },
    }
}

macro_rules! impl_data_creation_executors {
    ($backend:ty, $storage:ty) => {
        macro_rules! data_executor {
            ($operation:ident) => {
                impl<D: Device>
                    incin_core::backend_authoring::Execute<
                        incin_core::backend_authoring::op::$operation,
                    > for $backend
                {
                    type Output = $storage;

                    fn execute(
                        &self,
                        request: incin_core::backend_authoring::ExecutionRequest<
                            '_,
                            incin_core::backend_authoring::op::$operation,
                            Self,
                        >,
                    ) -> core::result::Result<$storage, BackendError> {
                        if !request.inputs.is_empty() {
                            return Err(crate::descriptor_bind::invalid(
                                incin_core::shapes::error::OperationKind::$operation,
                                "data creation takes no operand",
                            ));
                        }
                        let attr = request.operation.descriptor().attributes();
                        let bytes = request.payload.ok_or_else(|| {
                            crate::descriptor_bind::invalid(
                                incin_core::shapes::error::OperationKind::$operation,
                                "data creation requires borrowed bytes",
                            )
                        })?;
                        <Self as incin_core::backend_authoring::HostInterop>::from_bytes::<f32>(
                            bytes,
                            &attr.shape,
                            attr.dtype,
                            &attr.device,
                        )
                        .map_err(|err| {
                            crate::descriptor_bind::kernel_error(
                                Self::BACKEND_NAME,
                                incin_core::shapes::error::OperationKind::$operation,
                                err,
                            )
                        })
                    }
                }
            };
        }
        data_executor!(TensorFromData);
        data_executor!(TensorFromBytes);
    };
}

macro_rules! impl_creation_executors {
    ($backend:ty, $storage:ty) => {
        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::Zeros>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::Zeros,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::Zeros,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::zeros::<f32>(&attr.shape, attr.dtype, &attr.device).map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::shapes::error::OperationKind::Zeros,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::Ones>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::Ones,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::Ones,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::ones::<f32>(&attr.shape, attr.dtype, &attr.device).map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::shapes::error::OperationKind::Ones,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::UniformRandom>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::UniformRandom,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::UniformRandom,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::rand::<f32>(&attr.shape, attr.dtype, &attr.device).map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::shapes::error::OperationKind::UniformRandom,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::NormalRandom>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::NormalRandom,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::NormalRandom,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::randn::<f32>(&attr.shape, attr.dtype, &attr.device).map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::shapes::error::OperationKind::NormalRandom,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::Full>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::Full,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::Full,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::full::<f32>(attr.value, &attr.shape, attr.dtype, &attr.device).map_err(
                    |err| {
                        crate::descriptor_bind::kernel_error(
                            Self::BACKEND_NAME,
                            incin_core::shapes::error::OperationKind::Full,
                            err,
                        )
                    },
                )
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::Arange>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::Arange,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::Arange,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::arange::<f32>(attr.start, attr.step, &attr.shape, attr.dtype, &attr.device)
                    .map_err(|err| {
                        crate::descriptor_bind::kernel_error(
                            Self::BACKEND_NAME,
                            incin_core::shapes::error::OperationKind::Arange,
                            err,
                        )
                    })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<incin_core::backend_authoring::op::Linspace>
            for $backend
        {
            type Output = $storage;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::Linspace,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::shapes::error::OperationKind::Linspace,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                Self::linspace::<f32>(attr.start, attr.end, &attr.shape, attr.dtype, &attr.device)
                    .map_err(|err| {
                        crate::descriptor_bind::kernel_error(
                            Self::BACKEND_NAME,
                            incin_core::shapes::error::OperationKind::Linspace,
                            err,
                        )
                    })
            }
        }
    };
}

/// The single operand of a host readback, checked to be readable as one.
///
/// Every backend's `float_to_vec1`/`int_to_vec1`/`to_bytes` reads its buffer as
/// a dense run: the accelerator implementations download the whole allocation
/// and map over it, with no stride walk. Handing one a strided view would
/// therefore return a window of the wrong elements and report success, which is
/// the one failure mode a readback must not have. The layout is checked here,
/// once, for every backend that takes these executors.
///
/// The refusal is a capability answer rather than an invalid input: the operand
/// is well formed, this backend simply has no strided readback path. That is
/// exactly what [`UnsupportedReason::Layout`] states, and it keeps the message
/// aligned with what the capability registry would have said had it been asked
/// about the layout column directly.
///
/// Gated to the accelerator backends because those are its only callers: the
/// CPU reaches the same rows through its own inherent methods, so an ungated
/// definition is dead code in a `--features cpu` build and fails `-D warnings`.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
pub(crate) fn readback_operand<'a, S: core::any::Any>(
    inputs: &'a [incin_core::exec::TensorHandle<'a>],
    operation: OperationKind,
    backend: &'static str,
) -> core::result::Result<&'a S, BackendError> {
    let [handle] = inputs else {
        return Err(invalid(
            operation,
            "a host readback expects exactly one operand",
        ));
    };
    let layout = handle.metadata().layout();
    if layout != incin_core::exec::meta::LayoutClass::Contiguous {
        return Err(BackendError::unsupported(
            backend,
            UnsupportedReason::Layout { operation, layout },
        ));
    }
    handle
        .downcast_ref::<S>()
        .ok_or_else(|| invalid(operation, "operand is not this backend's storage"))
}

/// The four `var_*` allocations, for a backend that already has the plain ones.
///
/// A variable form is the same allocation as its plain sibling plus the promotion
/// to a trainable handle, so these route through the backend's own creation
/// method and then [`VariableBackend::var_from_tensor`] rather than through a
/// second device path. That is why no new kernel source is needed: every backend
/// that can allocate and can hold a variable can already do all four.
///
/// The plain sibling is reached inherently, matching `impl_creation_executors!`
/// above, so a backend takes both macros or neither.
///
/// Gated to the accelerator backends because those are its only callers: the
/// CPU reaches the same rows through its own inherent methods, so an ungated
/// definition is dead code in a `--features cpu` build and fails `-D warnings`.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
macro_rules! impl_variable_creation_executors {
    ($backend:ty, $var:ty) => {
        impl_variable_creation_executors!(@one $backend, $var, VariableZeros, zeros);
        impl_variable_creation_executors!(@one $backend, $var, VariableOnes, ones);
        impl_variable_creation_executors!(@one $backend, $var, VariableUniformRandom, rand);
        impl_variable_creation_executors!(@one $backend, $var, VariableNormalRandom, randn);
    };
    (@one $backend:ty, $var:ty, $operation:ident, $method:ident) => {
        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::$operation,
            > for $backend
        {
            type Output = $var;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::$operation,
                    Self,
                >,
            ) -> core::result::Result<$var, BackendError> {
                const OPERATION: incin_core::shapes::error::OperationKind =
                    incin_core::shapes::error::OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        OPERATION,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                let storage = Self::$method::<f32>(&attr.shape, attr.dtype, &attr.device)
                    .map_err(|err| {
                        crate::descriptor_bind::kernel_error(Self::BACKEND_NAME, OPERATION, err)
                    })?;
                <Self as incin_core::backend_authoring::VariableBackend>::var_from_tensor::<f32>(
                    &storage,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(Self::BACKEND_NAME, OPERATION, err)
                })
            }
        }
    };
}

/// The five host-readback rows, for a backend that already copies to the host.
///
/// `HostReadback` requires `float_to_vec1` and `int_to_vec1` with no default, and
/// `HostInterop` requires `to_bytes`, so a backend reaching this macro has all
/// three. The two scalar rows are the vector rows plus the single-element check,
/// spelled here rather than per backend so the refusal text cannot drift.
///
/// Gated to the accelerator backends because those are its only callers: the
/// CPU reaches the same rows through its own inherent methods, so an ungated
/// definition is dead code in a `--features cpu` build and fails `-D warnings`.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
macro_rules! impl_readback_executors {
    ($backend:ty, $storage:ty) => {
        impl_readback_executors!(@vec $backend, $storage, ToHostFloatVec,
                                 float_to_vec1, f64);
        impl_readback_executors!(@vec $backend, $storage, ToHostIntVec,
                                 int_to_vec1, i64);
        impl_readback_executors!(@scalar $backend, $storage, ToHostFloatScalar,
                                 float_to_vec1, f64, "float_to_scalar");
        impl_readback_executors!(@scalar $backend, $storage, ToHostIntScalar,
                                 int_to_vec1, i64, "int_to_scalar");

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::TensorToBytes,
            > for $backend
        {
            type Output = alloc::vec::Vec<u8>;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::TensorToBytes,
                    Self,
                >,
            ) -> core::result::Result<alloc::vec::Vec<u8>, BackendError> {
                const OPERATION: incin_core::shapes::error::OperationKind =
                    incin_core::shapes::error::OperationKind::TensorToBytes;
                let input = crate::descriptor_bind::readback_operand::<$storage>(
                    request.inputs,
                    OPERATION,
                    Self::BACKEND_NAME,
                )?;
                <Self as incin_core::backend_authoring::HostInterop>::to_bytes::<f32>(input)
                    .map_err(|err| {
                        crate::descriptor_bind::kernel_error(Self::BACKEND_NAME, OPERATION, err)
                    })
            }
        }
    };

    (@vec $backend:ty, $storage:ty, $operation:ident, $method:ident, $element:ty) => {
        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::$operation,
            > for $backend
        {
            type Output = alloc::vec::Vec<$element>;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::$operation,
                    Self,
                >,
            ) -> core::result::Result<alloc::vec::Vec<$element>, BackendError> {
                const OPERATION: incin_core::shapes::error::OperationKind =
                    incin_core::shapes::error::OperationKind::$operation;
                let input = crate::descriptor_bind::readback_operand::<$storage>(
                    request.inputs,
                    OPERATION,
                    Self::BACKEND_NAME,
                )?;
                <Self as incin_core::backend_authoring::HostReadback>::$method::<f32>(input)
                    .map_err(|err| {
                        crate::descriptor_bind::kernel_error(Self::BACKEND_NAME, OPERATION, err)
                    })
            }
        }
    };

    (@scalar $backend:ty, $storage:ty, $operation:ident, $method:ident,
     $element:ty, $name:literal) => {
        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::op::$operation,
            > for $backend
        {
            type Output = $element;
            fn execute(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::op::$operation,
                    Self,
                >,
            ) -> core::result::Result<$element, BackendError> {
                const OPERATION: incin_core::shapes::error::OperationKind =
                    incin_core::shapes::error::OperationKind::$operation;
                let input = crate::descriptor_bind::readback_operand::<$storage>(
                    request.inputs,
                    OPERATION,
                    Self::BACKEND_NAME,
                )?;
                let values =
                    <Self as incin_core::backend_authoring::HostReadback>::$method::<f32>(input)
                        .map_err(|err| {
                            crate::descriptor_bind::kernel_error(Self::BACKEND_NAME, OPERATION, err)
                        })?;
                // The CPU reference reports the same shape mismatch, with the
                // same `op` string and the same `expected`, so a caller that
                // matches on the error does not have to know which device ran.
                let [single] = values.as_slice() else {
                    return Err(crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        OPERATION,
                        incin_core::error::Error::ShapeMismatch {
                            op: $name,
                            expected: alloc::vec![1],
                            got: <Self as incin_core::backend_authoring::StorageBackend>::metadata::<
                                f32,
                            >(input)
                            .shape()
                            .dims()
                            .to_vec(),
                            msg: alloc::string::String::from(concat!(
                                $name,
                                " requires a single-element tensor"
                            )),
                        },
                    ));
                };
                Ok(*single)
            }
        }
    };
}
