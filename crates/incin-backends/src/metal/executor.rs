//! Descriptor execution for the Metal backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend, op};
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel};
use incin_core::prelude::{BackendError, Device, DeviceKind, OperationKind, Shape};
use incin_core::tensor::backend::{ModuleOps, NumericOps, ReductionOps, TensorOps};

use super::backend::MetalBackendImpl;
use super::storage::MetalStorage;
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for MetalBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Metal, query)
    }
}

impl_creation_executors!(MetalBackendImpl<D>, MetalStorage);
impl_data_creation_executors!(MetalBackendImpl<D>, MetalStorage);

macro_rules! impl_metal_canonical {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
                let rhs = rhs.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
                <Self as NumericOps<Self>>::$method::<f32>(lhs, rhs)
                    .map_err(|error| kernel_error("Metal", operation, error))
            }
        }
    )*};
}

impl_metal_canonical![(Add, add), (Sub, sub), (Mul, mul), (Div, div),];

impl<D: Device> Execute<op::ReshapeExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ReshapeExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::ReshapeExact,
                "reshape expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::ReshapeExact, "input is not Metal storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        <Self as TensorOps<Self>>::reshape::<f32>(storage, shape)
            .map_err(|e| kernel_error("Metal", OperationKind::ReshapeExact, e))
    }
}

impl<D: Device> Execute<op::BroadcastAs> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastAs, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::BroadcastAs,
                "broadcast expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::BroadcastAs, "input is not Metal storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        <Self as TensorOps<Self>>::broadcast_as::<f32>(storage, shape)
            .map_err(|e| kernel_error("Metal", OperationKind::BroadcastAs, e))
    }
}

impl<D: Device> Execute<op::MatMulExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMulExact,
                "matmul expects 2 inputs",
            ));
        };
        let lhs = lhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "lhs is not Metal storage"))?;
        let rhs = rhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "rhs is not Metal storage"))?;
        <Self as TensorOps<Self>>::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Metal", OperationKind::MatMulExact, e))
    }
}

impl<D: Device> Execute<op::Conv2dExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input, weight] = request.inputs else {
            return Err(invalid(
                OperationKind::Conv2dExact,
                "conv2d expects 2 inputs",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "input is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "weight is not Metal storage"))?;
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
        .map_err(|e| kernel_error("Metal", OperationKind::Conv2dExact, e))
    }
}

impl<D: Device> Execute<op::MaxPool2d> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaxPool2d, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::MaxPool2d,
                "max_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MaxPool2d, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::max_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
            pair(attrs.dilation),
        )
        .map_err(|e| kernel_error("Metal", OperationKind::MaxPool2d, e))
    }
}

impl<D: Device> Execute<op::AvgPool2d> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AvgPool2d, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::AvgPool2d,
                "avg_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::AvgPool2d, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::avg_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
        )
        .map_err(|e| kernel_error("Metal", OperationKind::AvgPool2d, e))
    }
}

macro_rules! impl_metal_reduction_all {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not Metal storage"))?;
                $func(input).map_err(|e| kernel_error("Metal", OperationKind::$op, e))
            }
        }
    )*};
}

macro_rules! impl_metal_reduction_dim {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not Metal storage"))?;
                let axis = request.operation.descriptor().attributes().axis;
                $func(input, axis).map_err(|e| kernel_error("Metal", OperationKind::$op, e))
            }
        }
    )*};
}

impl_metal_reduction_all![
    (
        SumAll,
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::sum_all::<f32>
    ),
    (
        MeanAll,
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::mean_all::<f32>
    ),
];

impl_metal_reduction_dim![
    (SumDim, |input, axis| {
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::sum_dim::<f32>(input, axis)
    }),
    (SumKeepDim, |input, axis| {
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::sum_keepdim::<f32>(input, axis)
    }),
    (MeanDim, |input, axis| {
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::mean_dim::<f32>(input, axis)
    }),
    (MeanKeepDim, |input, axis| {
        <MetalBackendImpl<D> as ReductionOps<MetalBackendImpl<D>>>::mean_keepdim::<f32>(input, axis)
    }),
];

macro_rules! assert_every_advertised_metal_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, MetalBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::prelude::Metal>();
        };
    };
}

crate::capability::metal_descriptor_operations!(assert_every_advertised_metal_row_executes,);
