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

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::exec::{
    Conv2dSpec, ExecutionContext, MatMulSpec, Pool2dSpec, ReductionSpec, ReshapeSpec, TensorHandle,
    TensorMeta,
};
use incin_core::prelude::{BackendError, DType, Device, OperationKind, StorageBackend};

use crate::descriptor_bind::invalid;
use crate::dispatch::{DispatchBackend, DispatchStorage};

impl<T: DType, D: Device> StorageBackend for DispatchBackend<T, D> {
    type Storage<K: DType> = DispatchStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<T: DType, D: Device> Execute<MatMulSpec> for DispatchBackend<T, D> {
    type Output = DispatchStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, MatMulSpec, Self>,
    ) -> Result<DispatchStorage, BackendError> {
        let _ = self;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMul,
                "matmul expects exactly two tensor inputs",
            ));
        };
        // Both operands must live on one device before a backend can be chosen.
        // Routing on the first operand alone would silently pick a backend that
        // then rejects the second, reporting a downcast failure in place of the
        // device mismatch that actually occurred.
        let (lhs_storage, rhs_storage) = (
            lhs.downcast_ref::<DispatchStorage>().ok_or_else(|| {
                invalid(
                    OperationKind::MatMul,
                    "matmul input is not dispatch storage",
                )
            })?,
            rhs.downcast_ref::<DispatchStorage>().ok_or_else(|| {
                invalid(
                    OperationKind::MatMul,
                    "matmul input is not dispatch storage",
                )
            })?,
        );

        match (lhs_storage, rhs_storage) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                use crate::cpu::CpuBackendImpl;
                use incin_core::prelude::{Cpu, Local};

                type Concrete = CpuBackendImpl<f32, Cpu>;
                let context = ExecutionContext::new(Concrete::new());
                let inputs = [
                    TensorHandle::from_storage::<Concrete, f32, Local>(lhs),
                    TensorHandle::from_storage::<Concrete, f32, Local>(rhs),
                ];
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                use crate::wgpu::WgpuBackendImpl;
                use incin_core::prelude::{Local, WgpuN};

                type Concrete = WgpuBackendImpl<f32, WgpuN<incin_core::typenum::U0>>;
                let context = ExecutionContext::new(Concrete::new());
                let inputs = [
                    TensorHandle::from_storage::<Concrete, f32, Local>(lhs),
                    TensorHandle::from_storage::<Concrete, f32, Local>(rhs),
                ];
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                use crate::cuda::backend::CudaBackendImpl;
                use incin_core::prelude::{Cuda, Local};

                type Concrete = CudaBackendImpl<f32, Cuda>;
                let context = ExecutionContext::new(Concrete::new());
                let inputs = [
                    TensorHandle::from_storage::<Concrete, f32, Local>(lhs),
                    TensorHandle::from_storage::<Concrete, f32, Local>(rhs),
                ];
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(invalid(
                OperationKind::MatMul,
                "matmul operands must be resident on the same runtime-selected backend",
            )),
        }
    }
}

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
    ($spec:ty, $kind:expr, $arity:expr, $not_dispatch:expr, $absent:expr) => {
        impl<T: DType, D: Device> Execute<$spec> for DispatchBackend<T, D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, $spec, Self>,
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
                        use incin_core::prelude::{Cpu, Local};

                        type Concrete = CpuBackendImpl<f32, Cpu>;
                        let context = ExecutionContext::new(Concrete::new());
                        let inputs = [TensorHandle::from_storage::<Concrete, f32, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                            })
                            .map(DispatchStorage::Cpu)
                    }
                    #[cfg(feature = "wgpu")]
                    DispatchStorage::Wgpu(input) => {
                        use crate::wgpu::WgpuBackendImpl;
                        use incin_core::prelude::{Local, WgpuN};

                        type Concrete = WgpuBackendImpl<f32, WgpuN<incin_core::typenum::U0>>;
                        let context = ExecutionContext::new(Concrete::new());
                        let inputs = [TensorHandle::from_storage::<Concrete, f32, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
                            })
                            .map(DispatchStorage::Wgpu)
                    }
                    #[cfg(feature = "cuda")]
                    DispatchStorage::Cuda(input) => {
                        use crate::cuda::backend::CudaBackendImpl;
                        use incin_core::prelude::{Cuda, Local};

                        type Concrete = CudaBackendImpl<f32, Cuda>;
                        let context = ExecutionContext::new(Concrete::new());
                        let inputs = [TensorHandle::from_storage::<Concrete, f32, Local>(input)];
                        context
                            .backend()
                            .execute(ExecutionRequest {
                                operation: request.operation,
                                inputs: &inputs,
                                context: &context,
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
    ReshapeSpec,
    OperationKind::Reshape,
    "reshape expects exactly one tensor input",
    "reshape input is not dispatch storage",
    "reshape input is resident on a backend this build does not include"
);

route_single_operand!(
    ReductionSpec,
    OperationKind::Reduction,
    "a reduction expects exactly one tensor input",
    "reduction input is not dispatch storage",
    "reduction input is resident on a backend this build does not include"
);

route_single_operand!(
    Pool2dSpec,
    OperationKind::Pool2d,
    "pool2d expects exactly one tensor input",
    "pool2d input is not dispatch storage",
    "pool2d input is resident on a backend this build does not include"
);

impl<T: DType, D: Device> Execute<Conv2dSpec> for DispatchBackend<T, D> {
    type Output = DispatchStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Conv2dSpec, Self>,
    ) -> Result<DispatchStorage, BackendError> {
        let _ = self;
        let (input, weight, bias) = match request.inputs {
            [input, weight] => (input, weight, None),
            [input, weight, bias] => (input, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    OperationKind::Conv2d,
                    "conv2d expects an input and a weight, and optionally a bias",
                ));
            }
        };
        let input = input.downcast_ref::<DispatchStorage>().ok_or_else(|| {
            invalid(
                OperationKind::Conv2d,
                "conv2d input is not dispatch storage",
            )
        })?;
        let weight = weight.downcast_ref::<DispatchStorage>().ok_or_else(|| {
            invalid(
                OperationKind::Conv2d,
                "conv2d input is not dispatch storage",
            )
        })?;
        let bias = bias
            .map(|bias| {
                bias.downcast_ref::<DispatchStorage>().ok_or_else(|| {
                    invalid(
                        OperationKind::Conv2d,
                        "conv2d input is not dispatch storage",
                    )
                })
            })
            .transpose()?;

        // Every operand has to be resident on one device before a backend can
        // be chosen, for the same reason matmul routes on both: picking from the
        // activation alone would report a downcast failure where the real fault
        // is a filter bank on another device.
        match (input, weight, bias) {
            #[cfg(feature = "cpu")]
            (
                DispatchStorage::Cpu(input),
                DispatchStorage::Cpu(weight),
                None | Some(DispatchStorage::Cpu(_)),
            ) => {
                use crate::cpu::CpuBackendImpl;
                use incin_core::prelude::{Cpu, Local};

                type Concrete = CpuBackendImpl<f32, Cpu>;
                let context = ExecutionContext::new(Concrete::new());
                let mut inputs = alloc::vec![
                    TensorHandle::from_storage::<Concrete, f32, Local>(input),
                    TensorHandle::from_storage::<Concrete, f32, Local>(weight),
                ];
                if let Some(DispatchStorage::Cpu(bias)) = bias {
                    inputs.push(TensorHandle::from_storage::<Concrete, f32, Local>(bias));
                }
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (
                DispatchStorage::Wgpu(input),
                DispatchStorage::Wgpu(weight),
                None | Some(DispatchStorage::Wgpu(_)),
            ) => {
                use crate::wgpu::WgpuBackendImpl;
                use incin_core::prelude::{Local, WgpuN};

                type Concrete = WgpuBackendImpl<f32, WgpuN<incin_core::typenum::U0>>;
                let context = ExecutionContext::new(Concrete::new());
                let mut inputs = alloc::vec![
                    TensorHandle::from_storage::<Concrete, f32, Local>(input),
                    TensorHandle::from_storage::<Concrete, f32, Local>(weight),
                ];
                if let Some(DispatchStorage::Wgpu(bias)) = bias {
                    inputs.push(TensorHandle::from_storage::<Concrete, f32, Local>(bias));
                }
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (
                DispatchStorage::Cuda(input),
                DispatchStorage::Cuda(weight),
                None | Some(DispatchStorage::Cuda(_)),
            ) => {
                use crate::cuda::backend::CudaBackendImpl;
                use incin_core::prelude::{Cuda, Local};

                type Concrete = CudaBackendImpl<f32, Cuda>;
                let context = ExecutionContext::new(Concrete::new());
                let mut inputs = alloc::vec![
                    TensorHandle::from_storage::<Concrete, f32, Local>(input),
                    TensorHandle::from_storage::<Concrete, f32, Local>(weight),
                ];
                if let Some(DispatchStorage::Cuda(bias)) = bias {
                    inputs.push(TensorHandle::from_storage::<Concrete, f32, Local>(bias));
                }
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: request.operation,
                        inputs: &inputs,
                        context: &context,
                    })
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(invalid(
                OperationKind::Conv2d,
                "conv2d operands must be resident on the same runtime-selected backend",
            )),
        }
    }
}
