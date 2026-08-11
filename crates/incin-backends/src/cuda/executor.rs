//! Descriptor execution for the CUDA backend.
//!
//! This mirrors the CPU vertical slice from `EXE-007`: the same sealed
//! `Validated<Descriptor<op::MatMulExact>>` binds to CUDA storage through the same
//! `StorageBackend`/`Capabilities`/`Execute` contract, so the descriptor path
//! is not a CPU-only construction.

use incin_core::backend_authoring::{
    Descriptor, Execute, ExecutionRequest, ModuleOps, ReductionOps, StorageBackend, TensorOps, op,
};
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel};
use incin_core::prelude::{BackendError, Device, DeviceKind, OperationKind, Shape};

use super::backend::CudaBackendImpl;
use super::storage::CudaStorage;
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for CudaBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cuda, query)
    }
}

impl_creation_executors!(CudaBackendImpl<D>, CudaStorage);

/// Whether an operand's physical shape is the one the descriptor promised.
///
/// The descriptor states the contracted extents and the broadcast batch; a
/// stride of 0 on a batch axis is the descriptor's own record that the operand
/// is broadcast along it, so that axis is required to be 1 rather than equal.

macro_rules! impl_cuda_canonical {
    ($(($op:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<Descriptor<op::$op>> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$op>, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs.downcast_ref::<CudaStorage>().ok_or_else(|| invalid(operation, "operand is not CUDA storage"))?;
                let rhs = rhs.downcast_ref::<CudaStorage>().ok_or_else(|| invalid(operation, "operand is not CUDA storage"))?;
                crate::cuda::backend::$func(lhs, rhs)
                    .map_err(|error| kernel_error("Cuda", operation, error))
            }
        }
    )*};
}

impl_cuda_canonical![
    (Add, cuda_add_storage),
    (Sub, cuda_sub_storage),
    (Mul, cuda_mul_storage),
    (Div, cuda_div_storage),
];

impl<D: Device> Execute<Descriptor<op::ReshapeExact>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ReshapeExact>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::ReshapeExact,
                "reshape expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::ReshapeExact, "input is not CUDA storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        <Self as TensorOps<Self>>::reshape::<f32>(storage, shape)
            .map_err(|e| kernel_error("Cuda", OperationKind::ReshapeExact, e))
    }
}

impl<D: Device> Execute<Descriptor<op::BroadcastAs>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::BroadcastAs>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::BroadcastAs,
                "broadcast expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::BroadcastAs, "input is not CUDA storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        <Self as TensorOps<Self>>::broadcast_as::<f32>(storage, shape)
            .map_err(|e| kernel_error("Cuda", OperationKind::BroadcastAs, e))
    }
}

impl<D: Device> Execute<Descriptor<op::MatMulExact>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MatMulExact>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMulExact,
                "matmul expects 2 inputs",
            ));
        };
        let lhs = lhs
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "lhs is not CUDA storage"))?;
        let rhs = rhs
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "rhs is not CUDA storage"))?;
        <Self as TensorOps<Self>>::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Cuda", OperationKind::MatMulExact, e))
    }
}

impl<D: Device> Execute<Descriptor<op::Conv2dExact>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Conv2dExact>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [input, weight] = request.inputs else {
            return Err(invalid(
                OperationKind::Conv2dExact,
                "conv2d expects 2 inputs",
            ));
        };
        let input = input
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "input is not CUDA storage"))?;
        let weight = weight
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "weight is not CUDA storage"))?;
        let attrs = request.operation.descriptor().attributes();
        <Self as ModuleOps<Self>>::conv2d::<f32>(
            input,
            weight,
            None,
            attrs.stride[0],
            attrs.padding[0],
            attrs.dilation[0],
            attrs.groups,
        )
        .map_err(|e| kernel_error("Cuda", OperationKind::Conv2dExact, e))
    }
}

impl<D: Device> Execute<Descriptor<op::MaxPool2d>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MaxPool2d>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::MaxPool2d,
                "max_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::MaxPool2d, "input is not CUDA storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::max_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
            pair(attrs.dilation),
        )
        .map_err(|e| kernel_error("Cuda", OperationKind::MaxPool2d, e))
    }
}

impl<D: Device> Execute<Descriptor<op::AvgPool2d>> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::AvgPool2d>, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::AvgPool2d,
                "avg_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::AvgPool2d, "input is not CUDA storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::avg_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
        )
        .map_err(|e| kernel_error("Cuda", OperationKind::AvgPool2d, e))
    }
}

macro_rules! impl_cuda_reduction_all {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<Descriptor<op::$op>> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$op>, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<CudaStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not CUDA storage"))?;
                $func(input).map_err(|e| kernel_error("Cuda", OperationKind::$op, e))
            }
        }
    )*};
}

macro_rules! impl_cuda_reduction_dim {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<Descriptor<op::$op>> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$op>, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<CudaStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not CUDA storage"))?;
                let axis = request.operation.descriptor().attributes().axis;
                $func(input, axis).map_err(|e| kernel_error("Cuda", OperationKind::$op, e))
            }
        }
    )*};
}

impl_cuda_reduction_all![
    (SumAll, |input| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::sum_all::<f32>(input)),
    (MeanAll, |input| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::mean_all::<f32>(input)),
    (MaxAll, |input| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::max_all::<f32>(input)),
    (MinAll, |input| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::min_all::<f32>(input)),
];

impl_cuda_reduction_dim![
    (SumDim, |input, axis| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::sum_dim::<f32>(input, axis)),
    (SumKeepDim, |input, axis| {
        <CudaBackendImpl<D> as ReductionOps<CudaBackendImpl<D>>>::sum_keepdim::<f32>(input, axis)
    }),
    (MeanDim, |input, axis| {
        <CudaBackendImpl<D> as ReductionOps<CudaBackendImpl<D>>>::mean_dim::<f32>(input, axis)
    }),
    (MeanKeepDim, |input, axis| {
        <CudaBackendImpl<D> as ReductionOps<CudaBackendImpl<D>>>::mean_keepdim::<f32>(input, axis)
    }),
    (MaxDim, |input, axis| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::max_dim::<f32>(input, axis)),
    (MaxKeepDim, |input, axis| {
        <CudaBackendImpl<D> as ReductionOps<CudaBackendImpl<D>>>::max_keepdim::<f32>(input, axis)
    }),
    (MinDim, |input, axis| <CudaBackendImpl<D> as ReductionOps<
        CudaBackendImpl<D>,
    >>::min_dim::<f32>(input, axis)),
    (MinKeepDim, |input, axis| {
        <CudaBackendImpl<D> as ReductionOps<CudaBackendImpl<D>>>::min_keepdim::<f32>(input, axis)
    }),
];

macro_rules! assert_every_advertised_cuda_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<Descriptor<O>>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, CudaBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::prelude::Cuda>();
        };
    };
}

crate::capability::cuda_descriptor_operations!(assert_every_advertised_cuda_row_executes,);
