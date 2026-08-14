//! Descriptor execution for the WGPU backend.
//!
//! This mirrors the CPU vertical slice from `EXE-007`: the same sealed
//! `Validated<Descriptor<op::MatMulExact>>` binds to WGPU storage through the same
//! `StorageBackend`/`Capabilities`/`Execute` contract, so the descriptor path
//! is not a CPU-only construction.

use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest, StorageBackend, op};
use incin_core::exec::{CanonicalOperation, Capabilities, CapabilityQuery, SupportLevel};
use incin_core::prelude::{BackendError, Device, DeviceKind, OperationKind};
use incin_core::__backend_compat::legacy::{ModuleOps};

use super::backend::WgpuBackendImpl;
use super::storage::WgpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

fn verify_operand_shape<O: CanonicalOperation>(
    descriptor: &Descriptor<O>,
    index: usize,
    actual: &WgpuStorage,
    operation: OperationKind,
    reason: &'static str,
) -> Result<(), BackendError> {
    if let Some(expected) = descriptor
        .inputs()
        .get(index)
        .and_then(|input| input.shape.as_ref())
        && expected != actual.shape()
    {
        return Err(invalid(operation, reason));
    }
    Ok(())
}

impl<D: Device> Capabilities for WgpuBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Wgpu, query)
    }
}

impl_creation_executors!(WgpuBackendImpl<D>, WgpuStorage);
impl_data_creation_executors!(WgpuBackendImpl<D>, WgpuStorage);

/// Whether an operand's physical shape is the one the descriptor promised.
///
/// The descriptor states the contracted extents and the broadcast batch; a
/// stride of 0 on a batch axis is the descriptor's own record that the operand
/// is broadcast along it, so that axis is required to be 1 rather than equal.
macro_rules! impl_wgpu_canonical {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<WgpuStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs.downcast_ref::<WgpuStorage>().ok_or_else(|| invalid(operation, "operand is not WGPU storage"))?;
                let rhs = rhs.downcast_ref::<WgpuStorage>().ok_or_else(|| invalid(operation, "operand is not WGPU storage"))?;
                Self::$method::<f32>(lhs, rhs)
                    .map_err(|error| kernel_error("Wgpu", operation, error))
            }
        }
    )*};
}

impl_wgpu_canonical![(Add, add), (Sub, sub), (Mul, mul), (Div, div),];

impl<D: Device> Execute<op::ReshapeExact> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ReshapeExact, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::ReshapeExact,
                "reshape expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::ReshapeExact, "input is not WGPU storage"))?;
        verify_operand_shape(
            request.operation.descriptor(),
            0,
            storage,
            OperationKind::ReshapeExact,
            "reshape input metadata does not match the validated descriptor",
        )?;
        let shape = &request.operation.descriptor().attributes().shape;
        Self::reshape::<f32>(storage, shape)
            .map_err(|e| kernel_error("Wgpu", OperationKind::ReshapeExact, e))
    }
}

impl<D: Device> Execute<op::BroadcastAs> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastAs, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::BroadcastAs,
                "broadcast expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::BroadcastAs, "input is not WGPU storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        Self::broadcast_as::<f32>(storage, shape)
            .map_err(|e| kernel_error("Wgpu", OperationKind::BroadcastAs, e))
    }
}

impl<D: Device> Execute<op::MatMulExact> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMulExact,
                "matmul expects 2 inputs",
            ));
        };
        let lhs = lhs
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "lhs is not WGPU storage"))?;
        let rhs = rhs
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "rhs is not WGPU storage"))?;
        verify_operand_shape(
            request.operation.descriptor(),
            0,
            lhs,
            OperationKind::MatMulExact,
            "matmul lhs metadata does not match the validated descriptor",
        )?;
        verify_operand_shape(
            request.operation.descriptor(),
            1,
            rhs,
            OperationKind::MatMulExact,
            "matmul rhs metadata does not match the validated descriptor",
        )?;
        Self::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Wgpu", OperationKind::MatMulExact, e))
    }
}

impl<D: Device> Execute<op::Conv2dExact> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let (input, weight, bias) = match request.inputs {
            [input, weight] => (input, weight, None),
            [input, weight, bias] => (input, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    OperationKind::Conv2dExact,
                    "conv2d expects an activation, a weight and an optional bias",
                ));
            }
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "input is not WGPU storage"))?;
        let weight = weight
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "weight is not WGPU storage"))?;
        verify_operand_shape(
            request.operation.descriptor(),
            0,
            input,
            OperationKind::Conv2dExact,
            "conv2d input metadata does not match the validated descriptor",
        )?;
        verify_operand_shape(
            request.operation.descriptor(),
            1,
            weight,
            OperationKind::Conv2dExact,
            "conv2d weight metadata does not match the validated descriptor",
        )?;
        let bias = bias
            .map(|bias| {
                bias.downcast_ref::<WgpuStorage>()
                    .ok_or_else(|| invalid(OperationKind::Conv2dExact, "bias is not WGPU storage"))
            })
            .transpose()?;
        if let Some(bias) = bias {
            verify_operand_shape(
                request.operation.descriptor(),
                2,
                bias,
                OperationKind::Conv2dExact,
                "conv2d bias metadata does not match the validated descriptor",
            )?;
        }
        let attrs = request.operation.descriptor().attributes();
        <Self as ModuleOps<Self>>::conv2d::<f32>(
            input,
            weight,
            bias,
            attrs.stride[0],
            attrs.padding[0],
            attrs.dilation[0],
            attrs.groups,
        )
        .map_err(|e| kernel_error("Wgpu", OperationKind::Conv2dExact, e))
    }
}

impl<D: Device> Execute<op::MaxPool2d> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaxPool2d, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::MaxPool2d,
                "max_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::MaxPool2d, "input is not WGPU storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::max_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
            pair(attrs.dilation),
        )
        .map_err(|e| kernel_error("Wgpu", OperationKind::MaxPool2d, e))
    }
}

impl<D: Device> Execute<op::AvgPool2d> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AvgPool2d, Self>,
    ) -> Result<WgpuStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::AvgPool2d,
                "avg_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(OperationKind::AvgPool2d, "input is not WGPU storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        <Self as ModuleOps<Self>>::avg_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
        )
        .map_err(|e| kernel_error("Wgpu", OperationKind::AvgPool2d, e))
    }
}

macro_rules! impl_wgpu_reduction_all {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<WgpuStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<WgpuStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not WGPU storage"))?;
                $func(input).map_err(|e| kernel_error("Wgpu", OperationKind::$op, e))
            }
        }
    )*};
}

macro_rules! impl_wgpu_reduction_dim {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<WgpuStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<WgpuStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not WGPU storage"))?;
                let axis = request.operation.descriptor().attributes().axis;
                $func(input, axis).map_err(|e| kernel_error("Wgpu", OperationKind::$op, e))
            }
        }
    )*};
}

impl_wgpu_reduction_all![
    (
        SumAll,
        WgpuBackendImpl::<D>::sum_all::<f32>
    ),
    (
        MeanAll,
        WgpuBackendImpl::<D>::mean_all::<f32>
    ),
    (
        MaxAll,
        WgpuBackendImpl::<D>::max_all::<f32>
    ),
    (
        MinAll,
        WgpuBackendImpl::<D>::min_all::<f32>
    ),
    (
        ProdAll,
        WgpuBackendImpl::<D>::prod_all::<f32>
    ),
];

impl_wgpu_reduction_dim![
    (
        SumDim,
        WgpuBackendImpl::<D>::sum_dim::<f32>
    ),
    (SumKeepDim, |input, axis| {
        WgpuBackendImpl::<D>::sum_keepdim::<f32>(input, axis)
    }),
    (MeanDim, |input, axis| {
        WgpuBackendImpl::<D>::mean_dim::<f32>(input, axis)
    }),
    (MeanKeepDim, |input, axis| {
        WgpuBackendImpl::<D>::mean_keepdim::<f32>(input, axis)
    }),
    (
        MaxDim,
        WgpuBackendImpl::<D>::max_dim::<f32>
    ),
    (MaxKeepDim, |input, axis| {
        WgpuBackendImpl::<D>::max_keepdim::<f32>(input, axis)
    }),
    (
        MinDim,
        WgpuBackendImpl::<D>::min_dim::<f32>
    ),
    (MinKeepDim, |input, axis| {
        WgpuBackendImpl::<D>::min_keepdim::<f32>(input, axis)
    }),
    (ProdDim, |input, axis| {
        WgpuBackendImpl::<D>::prod_dim::<f32>(input, axis)
    }),
];

macro_rules! impl_wgpu_unary_float {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<WgpuStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "unary operation expects 1 input"));
                };
                let input = input.downcast_ref::<WgpuStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not WGPU storage"))?;
                Self::$method::<f32>(input).map_err(|e| kernel_error("Wgpu", OperationKind::$op, e))
            }
        }
    )*};
}

impl_wgpu_unary_float![
    (Relu, relu),
    (Step, step),
    (Mish, mish),
    (Elu, elu),
    (Gelu, gelu),
    (Abs, abs),
    (Exp, exp),
    (Neg, neg),
    (Sqrt, sqrt),
    (Log, log),
    (Tanh, tanh),
    (Sigmoid, sigmoid),
    (Swish, swish),
];

macro_rules! assert_every_advertised_wgpu_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, WgpuBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::prelude::Wgpu>();
        };
    };
}

crate::capability::wgpu_descriptor_operations!(assert_every_advertised_wgpu_row_executes,);
