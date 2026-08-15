//! Descriptor execution for the runtime-selected backend.
//!
//! `DispatchBackend` owns no kernels. Its job is to recover the backend that
//! actually holds the operands and hand the *same* sealed descriptor to that
//! backend's executor, so a runtime-selected device gets identical validation
//! to a statically-selected one.
//!
//! There is deliberately no `Capabilities` implementation here. A dispatch
//! backend has no single device, so any answer it gave would either overstate
//! support (the union of every compiled backend) or understate it. The concrete
//! executor this module routes to performs the capability query against its own
//! registry, which is the only device that can answer.

use incin_core::backend_authoring::{Execute, ExecutionRequest, op};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::error::BackendError;
use incin_core::shapes::error::OperationKind;
use incin_core::backend_authoring::StorageBackend;
use incin_core::tensor::device::Device;

use crate::descriptor_bind::invalid;
use crate::dispatch::{DispatchBackend, DispatchStorage};

impl_creation_executors!(DispatchBackend<D>, DispatchStorage);
impl_data_creation_executors!(DispatchBackend<D>, DispatchStorage);

/// Route a one-operand descriptor to the backend that holds its operand.
///
/// Reshape, reduction, and pooling all take a single tensor and differ only in
/// the descriptor type and the operation their diagnostics name. Spelled out,
/// each would be the same three feature-gated arms recovering the same concrete
/// backends, so the three families would be nine near-identical blocks whose
/// only real content is which `DispatchStorage` variant they matched. The
/// descriptor is re-issued unchanged, which is the property that matters: the
/// concrete executor runs the same binder and the same capability query it would
/// have run had the caller named the device statically.
macro_rules! route_single_operand {
    ($descriptor:ty, $kind:expr, $arity:expr, $not_dispatch:expr, $absent:expr) => {
        impl<D: Device> Execute<$descriptor> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, $descriptor, Self>,
            ) -> Result<DispatchStorage, BackendError> {
                let _ = self;
                let [handle] = request.inputs else {
                    return Err(invalid($kind, $arity));
                };
                let storage = handle
                    .downcast_ref::<DispatchStorage>()
                    .ok_or_else(|| invalid($kind, $not_dispatch))?;

                match storage {
                    #[cfg(feature = "cpu")]
                    DispatchStorage::Cpu(input) => {
                        use crate::cpu::CpuBackendImpl;
                        use incin_core::prelude::{Cpu, Dyn, Local};

                        type Concrete = CpuBackendImpl<Cpu>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [TensorHandle::from_storage::<Concrete, Dyn, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cpu)
                    }
                    #[cfg(feature = "wgpu")]
                    DispatchStorage::Wgpu(input) => {
                        use crate::wgpu::WgpuBackendImpl;
                        use incin_core::prelude::{Dyn, Local, WgpuN};

                        type Concrete = WgpuBackendImpl<WgpuN<incin_core::typenum::U0>>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [TensorHandle::from_storage::<Concrete, Dyn, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Wgpu)
                    }
                    #[cfg(feature = "cuda")]
                    DispatchStorage::Cuda(input) => {
                        use crate::cuda::backend::CudaBackendImpl;
                        use incin_core::prelude::{Cuda, Dyn, Local};

                        type Concrete = CudaBackendImpl<Cuda>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [TensorHandle::from_storage::<Concrete, Dyn, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cuda)
                    }
                    #[allow(unreachable_patterns)]
                    _ => Err(invalid($kind, $absent)),
                }
            }
        }
    };
}

route_single_operand!(
    op::ReshapeExact,
    OperationKind::ReshapeExact,
    "reshape expects exactly one tensor input",
    "reshape input is not dispatch storage",
    "reshape input is resident on a backend this build does not include"
);

route_single_operand!(
    op::BroadcastAs,
    OperationKind::BroadcastAs,
    "broadcast_as expects exactly one tensor input",
    "broadcast_as input is not dispatch storage",
    "broadcast_as input is resident on a backend this build does not include"
);

macro_rules! impl_dispatch_binary {
    ($($op:ident),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<DispatchStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let (lhs_s, rhs_s) = (
                    lhs.downcast_ref::<DispatchStorage>().ok_or_else(|| invalid(operation, "input is not dispatch storage"))?,
                    rhs.downcast_ref::<DispatchStorage>().ok_or_else(|| invalid(operation, "input is not dispatch storage"))?,
                );
                match (lhs_s, rhs_s) {
                    #[cfg(feature = "cpu")]
                    (DispatchStorage::Cpu(l), DispatchStorage::Cpu(r)) => {
                        use crate::cpu::CpuBackendImpl;
                        use incin_core::prelude::{Cpu, Dyn, Local};
                        type Concrete = CpuBackendImpl<Cpu>;
                        let context = ExecutionContext::with_policy(Concrete::new(), request.context.policy().clone());
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(l),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(r),
                        ];
                        let inner_req = ExecutionRequest {
                            context: &context,
                            payload: request.payload,
                            operation: request.operation,
                            inputs: &inputs,
                        };
                        Concrete::new()
                            .execute(inner_req)
                            .map(DispatchStorage::Cpu)
                    }
                    #[cfg(feature = "cuda")]
                    (DispatchStorage::Cuda(l), DispatchStorage::Cuda(r)) => {
                        use crate::cuda::CudaBackendImpl;
                        use incin_core::prelude::{Cuda, Dyn, Local};
                        type Concrete = CudaBackendImpl<Cuda>;
                        let context = ExecutionContext::with_policy(Concrete::new(), request.context.policy().clone());
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(l),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(r),
                        ];
                        let inner_req = ExecutionRequest {
                            context: &context,
                            operation: request.operation,
                            inputs: &inputs,
                            payload: request.payload,
                        };
                        Concrete::new()
                            .execute(inner_req)
                            .map(DispatchStorage::Cuda)
                    }
                    #[cfg(feature = "wgpu")]
                    (DispatchStorage::Wgpu(l), DispatchStorage::Wgpu(r)) => {
                        use crate::wgpu::WgpuBackendImpl;
                        use incin_core::prelude::{Dyn, Local, Wgpu};
                        type Concrete = WgpuBackendImpl<Wgpu>;
                        let context = ExecutionContext::with_policy(Concrete::new(), request.context.policy().clone());
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(l),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(r),
                        ];
                        let inner_req = ExecutionRequest {
                            context: &context,
                            operation: request.operation,
                            inputs: &inputs,
                            payload: request.payload,
                        };
                        Concrete::new()
                            .execute(inner_req)
                            .map(DispatchStorage::Wgpu)
                    }
                    _ => Err(BackendError::Execution {
                        operation,
                        message: "mismatched or unsupported devices for dispatch binary operation".into(),
                    }),
                }
            }
        }
    )*};
}

impl_dispatch_binary!(Add, Sub, Mul, Div);

macro_rules! impl_dispatch_binary_with_metal {
    ($($op:ident),* $(,)?) => {
        $(
        impl<D: Device> Execute<op::$op> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<DispatchStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let (lhs, rhs) = (
                    lhs.downcast_ref::<DispatchStorage>().ok_or_else(|| {
                        invalid(operation, "input is not dispatch storage")
                    })?,
                    rhs.downcast_ref::<DispatchStorage>().ok_or_else(|| {
                        invalid(operation, "input is not dispatch storage")
                    })?,
                );
                match (lhs, rhs) {
                    #[cfg(feature = "cpu")]
                    (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                        use crate::cpu::CpuBackendImpl;
                        use incin_core::prelude::{Cpu, Dyn, Local};
                        type Concrete = CpuBackendImpl<Cpu>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(lhs),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(rhs),
                        ];
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cpu)
                    }
                    #[cfg(feature = "wgpu")]
                    (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                        use crate::wgpu::WgpuBackendImpl;
                        use incin_core::prelude::{Dyn, Local, Wgpu};
                        type Concrete = WgpuBackendImpl<Wgpu>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(lhs),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(rhs),
                        ];
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Wgpu)
                    }
                    #[cfg(feature = "cuda")]
                    (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                        use crate::cuda::backend::CudaBackendImpl;
                        use incin_core::prelude::{Cuda, Dyn, Local};
                        type Concrete = CudaBackendImpl<Cuda>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(lhs),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(rhs),
                        ];
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cuda)
                    }
                    #[cfg(feature = "metal")]
                    (DispatchStorage::Metal(lhs), DispatchStorage::Metal(rhs)) => {
                        use crate::metal::MetalBackendImpl;
                        use incin_core::prelude::{Dyn, Local, Metal};
                        type Concrete = MetalBackendImpl<Metal>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(lhs),
                            TensorHandle::from_storage::<Concrete, Dyn, Local>(rhs),
                        ];
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Metal)
                    }
                    _ => Err(invalid(
                        operation,
                        "operation operands must be resident on the same runtime-selected backend",
                    )),
                }
            }
        }
        )*
    };
}

impl_dispatch_binary_with_metal!(MatMulExact);

macro_rules! route_cpu_unary {
    ($op:ident, $kind:expr, $arity:expr, $not_dispatch:expr, $absent:expr) => {
        impl<D: Device> Execute<op::$op> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<DispatchStorage, BackendError> {
                let _ = self;
                let [handle] = request.inputs else {
                    return Err(invalid($kind, $arity));
                };
                let storage = handle
                    .downcast_ref::<DispatchStorage>()
                    .ok_or_else(|| invalid($kind, $not_dispatch))?;
                match storage {
                    #[cfg(feature = "cpu")]
                    DispatchStorage::Cpu(input) => {
                        use crate::cpu::CpuBackendImpl;
                        use incin_core::prelude::{Cpu, Dyn, Local};
                        type Concrete = CpuBackendImpl<Cpu>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let inputs = [TensorHandle::from_storage::<Concrete, Dyn, Local>(input)];
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cpu)
                    }
                    _ => Err(invalid($kind, $absent)),
                }
            }
        }
    };
}

route_cpu_unary!(
    TransposeExact,
    OperationKind::TransposeExact,
    "transpose expects exactly one tensor input",
    "transpose input is not dispatch storage",
    "transpose is only available for CPU dispatch storage"
);

route_cpu_unary!(
    FlattenExact,
    OperationKind::FlattenExact,
    "flatten expects exactly one tensor input",
    "flatten input is not dispatch storage",
    "flatten is only available for CPU dispatch storage"
);

route_cpu_unary!(
    SumAll,
    OperationKind::SumAll,
    "sum_all expects exactly one tensor input",
    "sum_all input is not dispatch storage",
    "sum_all is only available for CPU dispatch storage"
);

route_cpu_unary!(
    MeanAll,
    OperationKind::MeanAll,
    "mean_all expects exactly one tensor input",
    "mean_all input is not dispatch storage",
    "mean_all is only available for CPU dispatch storage"
);

route_cpu_unary!(
    MaxAll,
    OperationKind::MaxAll,
    "max_all expects exactly one tensor input",
    "max_all input is not dispatch storage",
    "max_all is only available for CPU dispatch storage"
);

route_cpu_unary!(
    MinAll,
    OperationKind::MinAll,
    "min_all expects exactly one tensor input",
    "min_all input is not dispatch storage",
    "min_all is only available for CPU dispatch storage"
);

route_cpu_unary!(
    ProdAll,
    OperationKind::ProdAll,
    "prod_all expects exactly one tensor input",
    "prod_all input is not dispatch storage",
    "prod_all is only available for CPU dispatch storage"
);

macro_rules! route_cpu_variadic {
    ($op:ident, $kind:expr, $absent:expr) => {
        impl<D: Device> Execute<op::$op> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<DispatchStorage, BackendError> {
                let _ = self;
                if request.inputs.is_empty() {
                    return Err(invalid($kind, "operation expects at least one operand"));
                }
                let inputs = request
                    .inputs
                    .iter()
                    .map(|input| {
                        input
                            .downcast_ref::<DispatchStorage>()
                            .ok_or_else(|| invalid($kind, "input is not dispatch storage"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match inputs.as_slice() {
                    #[cfg(feature = "cpu")]
                    [DispatchStorage::Cpu(_), rest @ ..]
                        if rest
                            .iter()
                            .all(|input| matches!(input, DispatchStorage::Cpu(_))) =>
                    {
                        use crate::cpu::CpuBackendImpl;
                        use incin_core::prelude::{Cpu, Dyn, Local};
                        type Concrete = CpuBackendImpl<Cpu>;
                        let context = ExecutionContext::with_policy(
                            Concrete::new(),
                            request.context.policy().clone(),
                        );
                        let handles = inputs
                            .iter()
                            .map(|input| match input {
                                DispatchStorage::Cpu(input) => {
                                    Ok(TensorHandle::from_storage::<Concrete, Dyn, Local>(input))
                                }
                                _ => unreachable!(),
                            })
                            .collect::<Result<Vec<_>, BackendError>>()?;
                        Concrete::new()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &handles,
                                context: &context,
                                payload: request.payload,
                            })
                            .map(DispatchStorage::Cpu)
                    }
                    _ => Err(invalid($kind, $absent)),
                }
            }
        }
    };
}

route_cpu_variadic!(
    ConcatExact,
    OperationKind::ConcatExact,
    "concat is only available for CPU dispatch storage"
);
route_cpu_variadic!(
    StackExact,
    OperationKind::StackExact,
    "stack is only available for CPU dispatch storage"
);

impl<D: Device> Execute<op::WhereCond> for DispatchBackend<D> {
    type Output = DispatchStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::WhereCond, Self>,
    ) -> Result<DispatchStorage, BackendError> {
        let operation = OperationKind::WhereCond;
        let [mask_h, true_h, false_h] = request.inputs else {
            return Err(invalid(
                operation,
                "where_cond expects exactly three operands",
            ));
        };
        let (mask_s, true_s, false_s) = (
            mask_h
                .downcast_ref::<DispatchStorage>()
                .ok_or_else(|| invalid(operation, "mask input is not dispatch storage"))?,
            true_h
                .downcast_ref::<DispatchStorage>()
                .ok_or_else(|| invalid(operation, "true input is not dispatch storage"))?,
            false_h
                .downcast_ref::<DispatchStorage>()
                .ok_or_else(|| invalid(operation, "false input is not dispatch storage"))?,
        );
        match (mask_s, true_s, false_s) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(m), DispatchStorage::Cpu(t), DispatchStorage::Cpu(f)) => {
                use crate::cpu::CpuBackendImpl;
                use incin_core::prelude::{Cpu, Dyn, Local};
                type Concrete = CpuBackendImpl<Cpu>;
                let context =
                    ExecutionContext::with_policy(Concrete::new(), request.context.policy());
                let inputs = [
                    TensorHandle::from_storage::<Concrete, bool, Local>(m),
                    TensorHandle::from_storage::<Concrete, Dyn, Local>(t),
                    TensorHandle::from_storage::<Concrete, Dyn, Local>(f),
                ];
                let inner_req = ExecutionRequest {
                    context: &context,
                    payload: request.payload,
                    operation: request.operation,
                    inputs: &inputs,
                };
                Concrete::new().execute(inner_req).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(_), DispatchStorage::Cuda(_), DispatchStorage::Cuda(_)) => {
                Err(BackendError::unsupported(
                    "Cuda",
                    incin_core::exec::UnsupportedReason::Operation { operation },
                ))
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(_), DispatchStorage::Wgpu(_), DispatchStorage::Wgpu(_)) => {
                Err(BackendError::unsupported(
                    "Wgpu",
                    incin_core::exec::UnsupportedReason::Operation { operation },
                ))
            }
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(_), DispatchStorage::Metal(_), DispatchStorage::Metal(_)) => {
                Err(BackendError::unsupported(
                    "Metal",
                    incin_core::exec::UnsupportedReason::Operation { operation },
                ))
            }
            _ => Err(BackendError::Execution {
                operation,
                message: "mismatched or unsupported devices for dispatch where_cond operation"
                    .into(),
            }),
        }
    }
}

impl<D: Device> Execute<op::MaskedFill> for DispatchBackend<D> {
    type Output = DispatchStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaskedFill, Self>,
    ) -> Result<DispatchStorage, BackendError> {
        let operation = OperationKind::MaskedFill;
        let [input_h, mask_h] = request.inputs else {
            return Err(invalid(
                operation,
                "masked_fill expects exactly two operands",
            ));
        };
        let (input_s, mask_s) = (
            input_h
                .downcast_ref::<DispatchStorage>()
                .ok_or_else(|| invalid(operation, "input is not dispatch storage"))?,
            mask_h
                .downcast_ref::<DispatchStorage>()
                .ok_or_else(|| invalid(operation, "mask input is not dispatch storage"))?,
        );
        match (input_s, mask_s) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(i), DispatchStorage::Cpu(m)) => {
                use crate::cpu::CpuBackendImpl;
                use incin_core::prelude::{Cpu, Dyn, Local};
                type Concrete = CpuBackendImpl<Cpu>;
                let context =
                    ExecutionContext::with_policy(Concrete::new(), request.context.policy());
                let inputs = [
                    TensorHandle::from_storage::<Concrete, Dyn, Local>(i),
                    TensorHandle::from_storage::<Concrete, bool, Local>(m),
                ];
                let inner_req = ExecutionRequest {
                    context: &context,
                    operation: request.operation,
                    inputs: &inputs,
                    payload: request.payload,
                };
                Concrete::new().execute(inner_req).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(_), DispatchStorage::Cuda(_)) => Err(BackendError::unsupported(
                "Cuda",
                incin_core::exec::UnsupportedReason::Operation { operation },
            )),
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(_), DispatchStorage::Wgpu(_)) => Err(BackendError::unsupported(
                "Wgpu",
                incin_core::exec::UnsupportedReason::Operation { operation },
            )),
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(_), DispatchStorage::Metal(_)) => {
                Err(BackendError::unsupported(
                    "Metal",
                    incin_core::exec::UnsupportedReason::Operation { operation },
                ))
            }
            _ => Err(BackendError::Execution {
                operation,
                message: "mismatched or unsupported devices for dispatch masked_fill operation"
                    .into(),
            }),
        }
    }
}
