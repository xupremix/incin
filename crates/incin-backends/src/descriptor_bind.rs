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

use incin_core::exec::UnsupportedReason;
use incin_core::prelude::{BackendError, Error, OperationKind};

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
                        incin_core::backend_authoring::Descriptor<
                            incin_core::backend_authoring::op::$operation,
                        >,
                    > for $backend
                {
                    type Output = $storage;

                    fn execute_shaped<ShapeTy: Shape>(
                        &self,
                        request: incin_core::backend_authoring::ExecutionRequest<
                            '_,
                            incin_core::backend_authoring::Descriptor<
                                incin_core::backend_authoring::op::$operation,
                            >,
                            Self,
                        >,
                    ) -> core::result::Result<$storage, BackendError> {
                        if !request.inputs.is_empty() {
                            return Err(crate::descriptor_bind::invalid(
                                incin_core::prelude::OperationKind::$operation,
                                "data creation takes no operand",
                            ));
                        }
                        let attr = request.operation.descriptor().attributes();
                        <Self as incin_core::backend_authoring::Backend>::from_bytes::<f32>(
                            &attr.bytes,
                            &attr.shape,
                            attr.dtype,
                            &attr.device,
                        )
                        .map_err(|err| {
                            crate::descriptor_bind::kernel_error(
                                Self::BACKEND_NAME,
                                incin_core::prelude::OperationKind::$operation,
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

#[allow(unused_macros)]
macro_rules! impl_creation_executors {
    ($backend:ty, $storage:ty) => {
        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<incin_core::backend_authoring::op::Zeros>,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::Zeros,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::Zeros,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::zeros::<f32>(
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::Zeros,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<incin_core::backend_authoring::op::Ones>,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::Ones,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::Ones,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::ones::<f32>(
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::Ones,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<
                    incin_core::backend_authoring::op::UniformRandom,
                >,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::UniformRandom,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::UniformRandom,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::rand::<f32>(
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::UniformRandom,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<
                    incin_core::backend_authoring::op::NormalRandom,
                >,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::NormalRandom,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::NormalRandom,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::randn::<f32>(
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::NormalRandom,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<incin_core::backend_authoring::op::Full>,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::Full,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::Full,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::full::<f32>(
                    attr.value,
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::Full,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<
                    incin_core::backend_authoring::op::Arange,
                >,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::Arange,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::Arange,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::arange::<f32>(
                    attr.start,
                    attr.step,
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::Arange,
                        err,
                    )
                })
            }
        }

        impl<D: Device>
            incin_core::backend_authoring::Execute<
                incin_core::backend_authoring::Descriptor<
                    incin_core::backend_authoring::op::Linspace,
                >,
            > for $backend
        {
            type Output = $storage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: incin_core::backend_authoring::ExecutionRequest<
                    '_,
                    incin_core::backend_authoring::Descriptor<
                        incin_core::backend_authoring::op::Linspace,
                    >,
                    Self,
                >,
            ) -> core::result::Result<$storage, BackendError> {
                if !request.inputs.is_empty() {
                    return Err(crate::descriptor_bind::invalid(
                        incin_core::prelude::OperationKind::Linspace,
                        "an allocation takes no operand",
                    ));
                }
                let attr = request.operation.descriptor().attributes();
                <Self as incin_core::backend_authoring::CreationOps<Self>>::linspace::<f32>(
                    attr.start,
                    attr.end,
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        incin_core::prelude::OperationKind::Linspace,
                        err,
                    )
                })
            }
        }
    };
}
