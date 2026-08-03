//! Canonical descriptor execution for the CPU backend.
//!
//! One `Execute<Descriptor<op::X>>` implementation per exact catalog identity,
//! generated from the same `cpu_descriptor_operations!` declaration that
//! generates `CPU_CAPABILITIES`. Advertising an operation and implementing it
//! are therefore the same edit, and a row that claims support the executor does
//! not provide will not compile.
//!
//! This is the FND-005 replacement for the grouped, attribute-polymorphic
//! `Execute<MatMulSpec>` family: those adapters accept several semantic
//! operations through one descriptor type, so an error or a capability query
//! could not identify which operation was actually refused. Here the identity
//! is the type.

use incin_core::backend_authoring::{Execute, ExecutionRequest, ModuleOps, ReductionOps};
use incin_core::exec::catalog::{Descriptor, op};
use incin_core::exec::{
    Capabilities, CapabilityQuery, MathMode, SupportLevel, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{BackendError, DType, DTypeId, Device, DeviceKind, OperationKind};

use super::CpuBackendImpl;
use super::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

/// Recover CPU storage from a checked handle.
///
/// The handle already carries validated metadata, so the only thing left to
/// establish is that the allocation belongs to this backend. A handle from
/// another device reaching a CPU executor is a dispatch defect, not a user
/// error, but it still fails with a typed reason rather than a panic.
fn operand<'a>(
    handle: &'a TensorHandle<'a>,
    operation: OperationKind,
) -> Result<&'a CpuStorage, BackendError> {
    let storage = handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(operation, "operand is not CPU storage"))?;
    let metadata = storage.metadata();
    if metadata.device().kind() != DeviceKind::Cpu {
        return Err(invalid(operation, "operand is not on a CPU device"));
    }
    Ok(storage)
}

/// Re-check the exact capability row from inside the executor.
///
/// `dispatch::execute` already queried it, but an executor must not depend on
/// having been reached through that path: a backend that only refuses when its
/// caller remembers to ask is a backend whose capability output is advisory.
fn admitted<T: DType, D: Device>(
    backend: &CpuBackendImpl<T, D>,
    operation: OperationKind,
    storage: &CpuStorage,
) -> Result<(), BackendError> {
    let metadata = storage.metadata();
    let query = CapabilityQuery {
        operation,
        dtype: metadata.dtype(),
        layout: metadata.layout(),
        rank: metadata.shape().rank(),
        training: true,
        math_mode: MathMode::Precise,
    };
    match backend.support(&query) {
        SupportLevel::Unsupported(reason) => Err(reason.into()),
        _ => Ok(()),
    }
}

/// Binary pointwise operations over broadcast operands.
macro_rules! pointwise_binary_executors {
    ($(($operation:ident, $kernel:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = operand(lhs, operation)?;
                let rhs = operand(rhs, operation)?;
                admitted(self, operation, lhs)?;
                admitted(self, operation, rhs)?;
                crate::cpu::ops::elementwise::$kernel(lhs, rhs)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

pointwise_binary_executors![
    (Add, add_storage),
    (Sub, sub_storage),
    (Mul, mul_storage),
    (Div, div_storage),
];

/// Reshape to the descriptor's declared shape.
impl<T: DType, D: Device> Execute<Descriptor<op::ReshapeExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ReshapeExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ReshapeExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "reshape expects exactly one operand"));
        };
        let input = operand(input, operation)?;
        admitted(self, operation, input)?;
        let shape = &request.operation.descriptor().attributes().shape;
        crate::cpu::ops::shape_ops::reshape_storage(input, shape)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Broadcast to the descriptor's declared shape.
impl<T: DType, D: Device> Execute<Descriptor<op::BroadcastAs>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::BroadcastAs>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BroadcastAs;
        let [input] = request.inputs else {
            return Err(invalid(
                operation,
                "broadcast_as expects exactly one operand",
            ));
        };
        let input = operand(input, operation)?;
        admitted(self, operation, input)?;
        let shape = &request.operation.descriptor().attributes().shape;
        crate::cpu::ops::shape_ops::broadcast_as_storage(input, shape)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Matrix multiplication over the last two axes, batched over the rest.
impl<T: DType, D: Device> Execute<Descriptor<op::MatMulExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MatMulExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MatMulExact;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "matmul expects exactly two operands"));
        };
        let lhs = operand(lhs, operation)?;
        let rhs = operand(rhs, operation)?;
        for storage in [lhs, rhs] {
            if storage.metadata().dtype() != DTypeId::F32 {
                return Err(UnsupportedReason::DType {
                    operation,
                    dtype: storage.metadata().dtype(),
                }
                .into());
            }
            admitted(self, operation, storage)?;
        }
        crate::cpu::ops::shape_ops::matmul_storage(lhs, rhs)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Bind the single operand a reduction consumes.
fn reduction_operand<'a, T: DType, D: Device>(
    backend: &CpuBackendImpl<T, D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
) -> Result<&'a CpuStorage, BackendError> {
    let [input] = inputs else {
        return Err(invalid(
            operation,
            "a reduction expects exactly one operand",
        ));
    };
    let input = operand(input, operation)?;
    admitted(backend, operation, input)?;
    Ok(input)
}

// The reduction bodies still live on `ReductionOps`. Reaching them from here is
// the migration's temporary compatibility adapter: it is private to this
// module, it is the only remaining call into the legacy family from the
// canonical path, and it is deleted when the reduction kernels move down here
// the way the pointwise and view kernels already have. It is deliberately not
// a source for anything new.

/// Whole-tensor reductions, which take no attributes.
macro_rules! reduce_all_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                <Self as ReductionOps<Self>>::$method::<T>(input)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

reduce_all_executors![
    (SumAll, sum_all),
    (MeanAll, mean_all),
    (MaxAll, max_all),
    (MinAll, min_all),
    (ProdAll, prod_all),
];

/// Single-axis reductions, which read the axis from their typed attributes.
macro_rules! reduce_axis_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                let axis = request.operation.descriptor().attributes().axis;
                <Self as ReductionOps<Self>>::$method::<T>(input, axis)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

reduce_axis_executors![
    (SumDim, sum_dim),
    (MeanDim, mean_dim),
    (MaxDim, max_dim),
    (MinDim, min_dim),
    (ProdDim, prod_dim),
    (SumKeepDim, sum_keepdim),
    (MeanKeepDim, mean_keepdim),
    (MaxKeepDim, max_keepdim),
    (MinKeepDim, min_keepdim),
];

/// Collapse a per-axis window to the single extent the routed CPU kernel takes.
///
/// The descriptor is more expressive than the kernel behind it: it carries one
/// extent per spatial axis, while `ModuleOps::conv2d` takes one for both. An
/// anisotropic window is therefore a real gap, and it is reported as one rather
/// than silently using the first axis for both.
fn isotropic(
    operation: OperationKind,
    [first, second]: [usize; 2],
    reason: &'static str,
) -> Result<usize, BackendError> {
    if first == second {
        Ok(first)
    } else {
        Err(invalid(operation, reason))
    }
}

/// Two-dimensional convolution with an optional bias.
impl<T: DType, D: Device> Execute<Descriptor<op::Conv2dExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Conv2dExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Conv2dExact;
        let attributes = request.operation.descriptor().attributes();
        // The bias operand's presence is part of the descriptor, so a mismatch
        // between what the attributes declare and what the caller passed is
        // caught by validation before this point. Destructuring here only
        // recovers the storage.
        let (activation, weight, bias) = match request.inputs {
            [activation, weight] => (activation, weight, None),
            [activation, weight, bias] => (activation, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    operation,
                    "conv2d expects an activation, a weight and an optional bias",
                ));
            }
        };
        let activation = operand(activation, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, activation)?;

        let stride = isotropic(
            operation,
            attributes.stride,
            "conv2d strides differ per axis; the routed kernel takes one stride for both",
        )?;
        let padding = isotropic(
            operation,
            attributes.padding,
            "conv2d paddings differ per axis; the routed kernel takes one padding for both",
        )?;
        let dilation = isotropic(
            operation,
            attributes.dilation,
            "conv2d dilations differ per axis; the routed kernel takes one dilation for both",
        )?;

        <Self as ModuleOps<Self>>::conv2d::<T>(
            activation,
            weight,
            bias,
            stride,
            padding,
            dilation,
            attributes.groups,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Two-dimensional maximum pooling.
impl<T: DType, D: Device> Execute<Descriptor<op::MaxPool2d>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MaxPool2d>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MaxPool2d;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        let pair = |[height, width]: [usize; 2]| (height, width);

        <Self as ModuleOps<Self>>::max_pool2d::<T>(
            input,
            pair(attributes.kernel),
            pair(attributes.stride),
            pair(attributes.padding),
            pair(attributes.dilation),
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Two-dimensional average pooling, which has no dilated form.
impl<T: DType, D: Device> Execute<Descriptor<op::AvgPool2d>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::AvgPool2d>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::AvgPool2d;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        let pair = |[height, width]: [usize; 2]| (height, width);

        <Self as ModuleOps<Self>>::avg_pool2d::<T>(
            input,
            pair(attributes.kernel),
            pair(attributes.stride),
            pair(attributes.padding),
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Prove, at compile time, that every identity `CPU_CAPABILITIES` advertises
/// has an executor above.
///
/// This is the property the module doc claims, made mechanical. The same
/// declaration that generates the capability rows generates these bounds, so
/// adding a row without an implementation is a compile error rather than a
/// support claim discovered at runtime by whoever believed it.
macro_rules! assert_every_advertised_row_executes {
    (
        ;
        pointwise = [$($pointwise:ident),* $(,)?],
        broadcast = [$($broadcast:ident),* $(,)?],
        reshape = [$($reshape:ident),* $(,)?],
        reduction = [$($reduction:ident),* $(,)?],
        spatial = [$($spatial:ident),* $(,)?],
        matmul = [$($matmul:ident),* $(,)?]
    ) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<Descriptor<O>>,
            {
            }

            const fn assert_all<T: DType, D: Device>() {
                $(executes::<op::$pointwise, CpuBackendImpl<T, D>>();)*
                $(executes::<op::$broadcast, CpuBackendImpl<T, D>>();)*
                $(executes::<op::$reshape, CpuBackendImpl<T, D>>();)*
                $(executes::<op::$reduction, CpuBackendImpl<T, D>>();)*
                $(executes::<op::$spatial, CpuBackendImpl<T, D>>();)*
                $(executes::<op::$matmul, CpuBackendImpl<T, D>>();)*
            }

            assert_all::<f32, incin_core::prelude::Cpu>();
        };
    };
}

crate::capability::cpu_descriptor_operations!(assert_every_advertised_row_executes,);
