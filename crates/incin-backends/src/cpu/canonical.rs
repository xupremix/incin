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

use incin_core::backend_authoring::{
    Execute, ExecutionRequest, FloatOps, ModuleOps, ReductionOps, TensorOps,
};
use incin_core::exec::catalog::{Descriptor, DuplicateIndexRule, op};
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

/// Running sum along the axis its attributes name.
///
/// This is the one operation in the index-reduction neighbourhood that is
/// dtype-honest: the CPU kernel accumulates in `f64` and converts back through
/// the operand's own buffer, so the result carries the operand's dtype.
impl<T: DType, D: Device> Execute<Descriptor<op::Cumsum>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Cumsum>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Cumsum;
        let input = reduction_operand(self, request.inputs, operation)?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as ReductionOps<Self>>::cumsum::<T>(input, axis)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Reject an index dtype the CPU kernel does not actually produce.
///
/// The index-returning reductions take an index dtype as a type parameter and
/// then ignore it: `argmax` and `argmin` always build an `i64` buffer, while
/// `argsort` and `topk` always build a `u32` one. Forwarding a descriptor that
/// asked for anything else would return storage labelled with a dtype it does
/// not hold, which is the failure mode the descriptor contract exists to make
/// impossible. The backend is not even self-consistent about which integer it
/// picks, so the check names the produced dtype rather than a shared constant.
fn produced_index_dtype(
    operation: OperationKind,
    declared: DTypeId,
    produced: DTypeId,
) -> Result<(), BackendError> {
    if declared == produced {
        return Ok(());
    }
    Err(BackendError::Unsupported {
        reason: UnsupportedReason::DType {
            operation,
            dtype: declared,
        },
    })
}

/// Index of the extremum, either flattened or along one axis.
macro_rules! index_reduction_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                let attributes = request.operation.descriptor().attributes();
                produced_index_dtype(operation, attributes.dtype, DTypeId::I64)?;
                <Self as ReductionOps<Self>>::$method::<T, T>(input, attributes.axis)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

index_reduction_executors![(ArgMax, argmax), (ArgMin, argmin)];

/// Indices that would sort the operand along one axis.
impl<T: DType, D: Device> Execute<Descriptor<op::Argsort>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Argsort>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Argsort;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        produced_index_dtype(operation, attributes.index_dtype, DTypeId::U32)?;
        <Self as ReductionOps<Self>>::argsort::<T, T>(input, attributes.axis, attributes.descending)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// The `k` extreme elements along one axis, as a value and an index tensor.
///
/// This is the first migrated identity whose output is not a single storage
/// handle. The catalog already describes it as two: the value tensor carries
/// the operand dtype and the index tensor carries the declared index dtype, so
/// returning a pair is what the descriptor was already promising.
///
/// Its capability row is f32 only, which is narrower than the operand dtypes
/// the kernel accepts. That is deliberate. The kernel builds its value buffer
/// as `f32` whatever the operand held, so for any other dtype it returns
/// storage labelled with a dtype it does not hold. Advertising f32 alone makes
/// the canonical path refuse what the legacy path silently mislabels.
impl<T: DType, D: Device> Execute<Descriptor<op::TopK>> for CpuBackendImpl<T, D> {
    type Output = (CpuStorage, CpuStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::TopK>, Self>,
    ) -> Result<(CpuStorage, CpuStorage), BackendError> {
        let operation = OperationKind::TopK;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        produced_index_dtype(operation, attributes.index_dtype, DTypeId::U32)?;
        <Self as ReductionOps<Self>>::topk::<T, T>(
            input,
            attributes.k,
            attributes.axis,
            attributes.largest,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

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

/// Refuse any operand the routed kernel would silently narrow.
///
/// `admitted` runs one capability query, so on a multi-operand operation it
/// speaks for the operand it was handed and no other. That is enough where the
/// operands share a dtype by construction, and not enough here: the convolution
/// and pooling kernels below build their result buffer as f32 regardless of what
/// the operand held, so an f64 weight would be accepted, narrowed, and returned
/// labelled f64. Measured against the real kernels rather than inferred from
/// their signatures, which are generic over the element type and suggest
/// otherwise.
///
/// The `Option` operands are the descriptor's optional ones. Taking them here
/// rather than at each call site keeps a bias from being the operand nobody
/// remembered to check.
fn f32_only(
    operation: OperationKind,
    operands: &[Option<&CpuStorage>],
) -> Result<(), BackendError> {
    for storage in operands.iter().flatten() {
        let dtype = storage.metadata().dtype();
        if dtype != DTypeId::F32 {
            return Err(UnsupportedReason::DType { operation, dtype }.into());
        }
    }
    Ok(())
}

/// Narrow a descriptor epsilon to the width the routed kernel accepts.
///
/// `LayerNormAttributes` and `BatchNormAttributes` carry an `f64`, while
/// `ModuleOps::layer_norm` and `ModuleOps::batch_norm` take an `f32`. Almost
/// every epsilon survives that, but one below `f32::MIN_POSITIVE` flushes to
/// zero and turns a guarded division into an unguarded one, which shows up as a
/// non-finite activation far from here. Refused rather than rounded.
fn narrowed_epsilon(operation: OperationKind, epsilon: f64) -> Result<f32, BackendError> {
    let narrowed = epsilon as f32;
    if narrowed.is_finite() && narrowed > 0.0 {
        Ok(narrowed)
    } else {
        Err(invalid(
            operation,
            "epsilon is not representable as a positive finite f32, and narrowing it would \
             remove the guard it exists to provide",
        ))
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

/// One-dimensional convolution with an optional bias.
impl<T: DType, D: Device> Execute<Descriptor<op::Conv1dExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Conv1dExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Conv1dExact;
        let attributes = request.operation.descriptor().attributes();
        let (activation, weight, bias) = match request.inputs {
            [activation, weight] => (activation, weight, None),
            [activation, weight, bias] => (activation, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    operation,
                    "conv1d expects an activation, a weight and an optional bias",
                ));
            }
        };
        let activation = operand(activation, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, activation)?;
        f32_only(operation, &[Some(activation), Some(weight), bias])?;

        // `Conv1dAttributes` already carries one extent per field, so unlike the
        // two-dimensional forms there is nothing to collapse.
        <Self as ModuleOps<Self>>::conv1d::<T>(
            activation,
            weight,
            bias,
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.groups,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Two-dimensional transposed convolution with an optional bias.
impl<T: DType, D: Device> Execute<Descriptor<op::ConvTranspose2d>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ConvTranspose2d>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ConvTranspose2d;
        let attributes = request.operation.descriptor().attributes();
        let (activation, weight, bias) = match request.inputs {
            [activation, weight] => (activation, weight, None),
            [activation, weight, bias] => (activation, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    operation,
                    "conv_transpose2d expects an activation, a weight and an optional bias",
                ));
            }
        };
        let activation = operand(activation, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, activation)?;
        f32_only(operation, &[Some(activation), Some(weight), bias])?;

        // The same descriptor-to-kernel width gap the forward convolution has,
        // with one more field: the transposed form also carries an output
        // padding per axis and the kernel takes one for both.
        let stride = isotropic(
            operation,
            attributes.stride,
            "conv_transpose2d strides differ per axis; the routed kernel takes one stride for both",
        )?;
        let padding = isotropic(
            operation,
            attributes.padding,
            "conv_transpose2d paddings differ per axis; the routed kernel takes one padding for \
             both",
        )?;
        let output_padding = isotropic(
            operation,
            attributes.output_padding,
            "conv_transpose2d output paddings differ per axis; the routed kernel takes one for \
             both",
        )?;
        let dilation = isotropic(
            operation,
            attributes.dilation,
            "conv_transpose2d dilations differ per axis; the routed kernel takes one dilation for \
             both",
        )?;

        <Self as ModuleOps<Self>>::conv_transpose2d::<T>(
            activation,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            attributes.groups,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Average pooling to a requested output extent rather than a requested window.
impl<T: DType, D: Device> Execute<Descriptor<op::AdaptiveAvgPool2dExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::AdaptiveAvgPool2dExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::AdaptiveAvgPool2dExact;
        let input = reduction_operand(self, request.inputs, operation)?;
        f32_only(operation, &[Some(input)])?;
        let [height, width] = request.operation.descriptor().attributes().output;
        <Self as ModuleOps<Self>>::adaptive_avg_pool2d::<T>(input, (height, width))
            .map_err(|error| kernel_error(operation, error))
    }
}

// The float family. Its kernel bodies still live on `FloatOps`, reached through
// the same private compatibility adapter the reductions use: these are the
// operations whose bodies move down next, and the macros below are shaped so
// that moving one is a change to a single row.

/// Unary elementwise float operations, which take no attributes.
macro_rules! unary_float_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                <Self as FloatOps<Self>>::$method::<T>(input)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

unary_float_executors![
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
    (Sign, sign),
    (Floor, floor),
    (Ceil, ceil),
    (Round, round),
    (Log2, log2),
    (Log10, log10),
    (Sin, sin),
    (Cos, cos),
    (Tan, tan),
    (Asin, asin),
    (Acos, acos),
    (Atan, atan),
    (Sinh, sinh),
    (Cosh, cosh),
    (Asinh, asinh),
    (Acosh, acosh),
    (Atanh, atanh),
    (Erf, erf),
    (Rsqrt, rsqrt),
    (Trunc, trunc),
    (Frac, frac),
];

/// Unary float operations parametrised by one scalar attribute.
macro_rules! scalar_float_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                let value = request.operation.descriptor().attributes().value;
                <Self as FloatOps<Self>>::$method::<T>(input, value)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

scalar_float_executors![
    (AddScalar, add_scalar_float),
    (MulScalar, mul_scalar_float),
    (Powf, powf),
];

/// Binary elementwise float operations over broadcast operands.
macro_rules! binary_float_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
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
                <Self as FloatOps<Self>>::$method::<T>(lhs, rhs)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

binary_float_executors![(Atan2, atan2), (Fmod, fmod), (Remainder, remainder),];

/// Elementwise clamp, whose two bounds are a single typed attribute set.
impl<T: DType, D: Device> Execute<Descriptor<op::Clamp>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Clamp>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Clamp;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as FloatOps<Self>>::clamp::<T>(input, attributes.min, attributes.max)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Softmax along the axis its attributes name.
impl<T: DType, D: Device> Execute<Descriptor<op::Softmax>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Softmax>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Softmax;
        let input = reduction_operand(self, request.inputs, operation)?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as FloatOps<Self>>::softmax::<T>(input, axis)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Normalize over the trailing axes the attributes name, then scale and shift.
impl<T: DType, D: Device> Execute<Descriptor<op::LayerNorm>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::LayerNorm>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::LayerNorm;
        let attributes = request.operation.descriptor().attributes();
        let (input, weight, bias) = match request.inputs {
            [input, weight] => (input, weight, None),
            [input, weight, bias] => (input, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    operation,
                    "layer norm expects an input, a weight and an optional bias",
                ));
            }
        };
        let input = operand(input, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, input)?;
        f32_only(operation, &[Some(input), Some(weight), bias])?;
        let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;

        // `normalized_shape` is not passed on: the descriptor has already
        // checked that it is the operand's trailing suffix and that the weight
        // and bias match it, and the kernel derives the same split from the
        // weight's own shape.
        <Self as ModuleOps<Self>>::layer_norm::<T>(input, weight, bias, epsilon)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Normalize each channel by its running statistics, then scale and shift.
///
/// Inference only, and refused rather than approximated otherwise. See the
/// refusal below.
impl<T: DType, D: Device> Execute<Descriptor<op::BatchNorm>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::BatchNorm>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BatchNorm;
        let attributes = request.operation.descriptor().attributes();
        // The CPU kernel never computes batch statistics and never updates the
        // running ones; its momentum parameter is bound to `_momentum`. A
        // training request routed to it would come back with the inference
        // result and nothing to distinguish it from a correct one, and the
        // running statistics the caller expects to have been updated would be
        // unchanged. That is the one failure this executor exists to prevent,
        // so it is a refusal and not a note in the documentation.
        if attributes.training {
            return Err(invalid(
                operation,
                "the CPU batch norm kernel evaluates inference mode only: it computes no batch \
                 statistics and updates no running statistics, so a training request cannot be \
                 answered correctly",
            ));
        }
        let Some((input, optional)) = request.inputs.split_first() else {
            return Err(invalid(
                operation,
                "batch norm expects at least the input operand",
            ));
        };
        // The optional operands are positional, in the order the presence flags
        // are declared: weight, bias, running mean, running variance. Only the
        // present ones occupy a slot.
        let mut remaining = optional.iter();
        let mut next = |present: bool| present.then(|| remaining.next()).flatten();
        let weight = next(attributes.has_weight);
        let bias = next(attributes.has_bias);
        let running_mean = next(attributes.has_running_mean);
        let running_variance = next(attributes.has_running_variance);
        if remaining.next().is_some() {
            return Err(invalid(
                operation,
                "batch norm was given more operands than its presence flags account for",
            ));
        }
        // Descriptor validation already pairs these with `training`, but the
        // kernel substitutes a zero mean and a unit variance for an absent one
        // instead of failing, so an executor that trusted the caller here would
        // return a plausible wrong answer rather than an error.
        let (Some(running_mean), Some(running_variance)) = (running_mean, running_variance) else {
            return Err(invalid(
                operation,
                "inference batch norm needs a running mean and a running variance; without them \
                 the kernel substitutes a zero mean and a unit variance",
            ));
        };

        let input = operand(input, operation)?;
        let weight = weight.map(|value| operand(value, operation)).transpose()?;
        let bias = bias.map(|value| operand(value, operation)).transpose()?;
        let running_mean = operand(running_mean, operation)?;
        let running_variance = operand(running_variance, operation)?;
        admitted(self, operation, input)?;
        f32_only(
            operation,
            &[
                Some(input),
                weight,
                bias,
                Some(running_mean),
                Some(running_variance),
            ],
        )?;
        let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;

        <Self as ModuleOps<Self>>::batch_norm::<T>(
            input,
            weight,
            bias,
            Some(running_mean),
            Some(running_variance),
            epsilon,
            attributes.momentum,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Bind the two operands a binary tensor operation consumes.
fn binary_operands<'a, T: DType, D: Device>(
    backend: &CpuBackendImpl<T, D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
) -> Result<(&'a CpuStorage, &'a CpuStorage), BackendError> {
    let [lhs, rhs] = inputs else {
        return Err(invalid(operation, "operation expects exactly two operands"));
    };
    let lhs = operand(lhs, operation)?;
    let rhs = operand(rhs, operation)?;
    admitted(backend, operation, lhs)?;
    admitted(backend, operation, rhs)?;
    Ok((lhs, rhs))
}

/// Binary elementwise tensor operations that take no attributes.
///
/// Comparisons and logical connectives are here rather than with the float
/// family because their semantic profile preserves the operand dtype instead of
/// producing a boolean one, and because they carry no gradient.
macro_rules! binary_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (lhs, rhs) = binary_operands(self, request.inputs, operation)?;
                <Self as TensorOps<Self>>::$method::<T>(lhs, rhs)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

binary_tensor_executors![
    (Maximum, maximum),
    (Minimum, minimum),
    (AbsDiff, abs_diff),
    (CmpEq, cmp_eq),
    (CmpNe, cmp_ne),
    (CmpLt, cmp_lt),
    (CmpLe, cmp_le),
    (CmpGt, cmp_gt),
    (CmpGe, cmp_ge),
    (LogicalAnd, logical_and),
    (LogicalOr, logical_or),
];

/// Batched matrix multiplication, whose operand rank contract differs from the
/// plain `matmul` row and so does not share its registration.
impl<T: DType, D: Device> Execute<Descriptor<op::BatchedMatMul>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::BatchedMatMul>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BatchedMatMul;
        let (lhs, rhs) = binary_operands(self, request.inputs, operation)?;
        <Self as TensorOps<Self>>::bmm::<T>(lhs, rhs)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Unary tensor operations parametrised by one scalar attribute.
macro_rules! scalar_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                let value = request.operation.descriptor().attributes().value;
                <Self as TensorOps<Self>>::$method::<T>(input, value)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

scalar_tensor_executors![(SubScalar, sub_scalar), (DivScalar, div_scalar)];

/// Triangular and diagonal views, parametrised by a signed diagonal offset.
macro_rules! diagonal_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation)?;
                let offset = request.operation.descriptor().attributes().offset;
                <Self as TensorOps<Self>>::$method::<T>(input, offset)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

diagonal_tensor_executors![(Triu, triu), (Tril, tril), (Diag, diag)];

/// Rank-changing views parametrised by a single axis.
macro_rules! axis_tensor_executors {
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
                <Self as TensorOps<Self>>::$method::<T>(input, axis)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

axis_tensor_executors![(SqueezeExact, squeeze), (UnsqueezeExact, unsqueeze)];

/// Elementwise logical negation.
impl<T: DType, D: Device> Execute<Descriptor<op::LogicalNot>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::LogicalNot>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::LogicalNot;
        let input = reduction_operand(self, request.inputs, operation)?;
        <Self as TensorOps<Self>>::logical_not::<T>(input)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Swap the two axes the descriptor names.
impl<T: DType, D: Device> Execute<Descriptor<op::TransposeExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::TransposeExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::TransposeExact;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::transpose::<T>(input, attributes.first, attributes.second)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Take a contiguous run along one axis.
impl<T: DType, D: Device> Execute<Descriptor<op::Narrow>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Narrow>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Narrow;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::narrow::<T>(
            input,
            attributes.axis,
            attributes.start,
            attributes.length,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Collapse an inclusive axis range into one axis.
impl<T: DType, D: Device> Execute<Descriptor<op::FlattenExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::FlattenExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::FlattenExact;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::flatten::<T>(input, attributes.start_axis, attributes.end_axis)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Elementwise selection between two operands under a mask.
///
/// The operand order is the one the catalog's legacy source names -
/// `TensorOps::where_cond(mask, on_true, on_false)` - so a caller that reads
/// the catalog row gets the same meaning from either path.
impl<T: DType, D: Device> Execute<Descriptor<op::WhereCond>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::WhereCond>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::WhereCond;
        let [mask, on_true, on_false] = request.inputs else {
            return Err(invalid(
                operation,
                "where_cond expects exactly three operands",
            ));
        };
        let mask = operand(mask, operation)?;
        let on_true = operand(on_true, operation)?;
        let on_false = operand(on_false, operation)?;
        for storage in [mask, on_true, on_false] {
            admitted(self, operation, storage)?;
        }
        <Self as TensorOps<Self>>::where_cond::<T, T>(mask, on_true, on_false)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Overwrite the masked positions with the declared scalar.
impl<T: DType, D: Device> Execute<Descriptor<op::MaskedFill>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MaskedFill>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MaskedFill;
        let (input, mask) = binary_operands(self, request.inputs, operation)?;
        let value = request.operation.descriptor().attributes().value;
        <Self as TensorOps<Self>>::masked_fill::<T, T>(input, mask, value)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Interpolate between two operands at the declared weight.
impl<T: DType, D: Device> Execute<Descriptor<op::Lerp>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Lerp>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Lerp;
        let (start, end) = binary_operands(self, request.inputs, operation)?;
        let weight = request.operation.descriptor().attributes().weight;
        <Self as TensorOps<Self>>::lerp::<T>(start, end, weight)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Bind a variable-length operand list, checking each one.
///
/// `concat` and `stack` take one or more operands, so their arity contract is
/// a lower bound rather than a fixed count. An empty list is still a defect.
fn variadic_operands<'a, T: DType, D: Device>(
    backend: &CpuBackendImpl<T, D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
) -> Result<Vec<&'a CpuStorage>, BackendError> {
    if inputs.is_empty() {
        return Err(invalid(operation, "operation expects at least one operand"));
    }
    inputs
        .iter()
        .map(|handle| {
            let storage = operand(handle, operation)?;
            admitted(backend, operation, storage)?;
            Ok(storage)
        })
        .collect()
}

/// Bind the three operands an indexing or fused operation consumes.
fn ternary_operands<'a, T: DType, D: Device>(
    backend: &CpuBackendImpl<T, D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
) -> Result<[&'a CpuStorage; 3], BackendError> {
    let [first, second, third] = inputs else {
        return Err(invalid(
            operation,
            "operation expects exactly three operands",
        ));
    };
    let bound = [
        operand(first, operation)?,
        operand(second, operation)?,
        operand(third, operation)?,
    ];
    for storage in bound {
        admitted(backend, operation, storage)?;
    }
    Ok(bound)
}

/// Join operands along an existing axis.
impl<T: DType, D: Device> Execute<Descriptor<op::ConcatExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ConcatExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ConcatExact;
        let operands = variadic_operands(self, request.inputs, operation)?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as TensorOps<Self>>::concat::<T>(&operands, axis)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Join operands along a new axis.
impl<T: DType, D: Device> Execute<Descriptor<op::StackExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::StackExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::StackExact;
        let operands = variadic_operands(self, request.inputs, operation)?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as TensorOps<Self>>::stack::<T>(&operands, axis)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Take a half-open window per axis.
impl<T: DType, D: Device> Execute<Descriptor<op::SliceExact>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::SliceExact>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::SliceExact;
        let input = reduction_operand(self, request.inputs, operation)?;
        let ranges = &request.operation.descriptor().attributes().ranges;
        <Self as TensorOps<Self>>::slice::<T>(input, ranges)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Indexing operations that read one axis and one index operand.
macro_rules! indexing_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<T: DType, D: Device> Execute<Descriptor<op::$operation>> for CpuBackendImpl<T, D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, Descriptor<op::$operation>, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (input, index) = binary_operands(self, request.inputs, operation)?;
                let axis = request.operation.descriptor().attributes().axis;
                <Self as TensorOps<Self>>::$method::<T, T>(input, axis, index)
                    .map_err(|error| kernel_error(operation, error))
            }
        }
    )*};
}

indexing_executors![(Gather, gather), (IndexSelect, index_select)];

/// Write `src` into the operand at the indexed positions.
///
/// The descriptor can ask for duplicate indices to be rejected. The CPU kernel
/// has no duplicate detection: it writes in index order, so the last write
/// wins. Answering a `Reject` request with that behaviour would report success
/// for a contract the backend did not honour, so it is refused instead.
impl<T: DType, D: Device> Execute<Descriptor<op::Scatter>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Scatter>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Scatter;
        let [input, index, source] = ternary_operands(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        if attributes.duplicate_indices == DuplicateIndexRule::Reject {
            return Err(invalid(
                operation,
                "this backend applies last-write-wins and cannot reject duplicate indices",
            ));
        }
        <Self as TensorOps<Self>>::scatter::<T, T>(input, attributes.axis, index, source)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Tile the operand per axis.
impl<T: DType, D: Device> Execute<Descriptor<op::Repeat>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Repeat>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Repeat;
        let input = reduction_operand(self, request.inputs, operation)?;
        let repeats = &request.operation.descriptor().attributes().repeats;
        <Self as TensorOps<Self>>::repeat::<T>(input, repeats)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Extend each axis with the declared constant.
impl<T: DType, D: Device> Execute<Descriptor<op::Pad>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Pad>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Pad;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::pad::<T>(input, &attributes.padding, attributes.value)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Extract sliding windows along one axis.
impl<T: DType, D: Device> Execute<Descriptor<op::Unfold>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Unfold>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Unfold;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::unfold::<T>(
            input,
            attributes.axis,
            attributes.size,
            attributes.step,
        )
        .map_err(|error| kernel_error(operation, error))
    }
}

/// Redistribute channel depth into spatial extent.
impl<T: DType, D: Device> Execute<Descriptor<op::PixelShuffle>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::PixelShuffle>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::PixelShuffle;
        let input = reduction_operand(self, request.inputs, operation)?;
        let factor = request.operation.descriptor().attributes().upscale_factor;
        <Self as TensorOps<Self>>::pixel_shuffle::<T>(input, factor)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Normalize within channel groups.
impl<T: DType, D: Device> Execute<Descriptor<op::GroupNorm>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::GroupNorm>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::GroupNorm;
        let input = reduction_operand(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::group_norm::<T>(input, attributes.groups, attributes.epsilon)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Normalize each channel independently.
impl<T: DType, D: Device> Execute<Descriptor<op::InstanceNorm>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::InstanceNorm>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::InstanceNorm;
        let input = reduction_operand(self, request.inputs, operation)?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        <Self as TensorOps<Self>>::instance_norm::<T>(input, epsilon)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Extend the operand on the left to the declared target shape.
///
/// The descriptor and the legacy method disagree about what the shape argument
/// means: the descriptor's `ShapeAttributes` is the whole target shape, and
/// validates the operand against it, while `TensorOps::broadcast_left` takes
/// only the extents to prepend. Passing the descriptor's shape straight
/// through would prepend the target to the operand and produce a tensor of
/// twice the intended rank, so the prefix is derived here.
impl<T: DType, D: Device> Execute<Descriptor<op::BroadcastLeft>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::BroadcastLeft>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BroadcastLeft;
        let input = reduction_operand(self, request.inputs, operation)?;
        let target = &request.operation.descriptor().attributes().shape;
        let rank = input.metadata().shape().rank();
        let Some(prefix) = target.len().checked_sub(rank) else {
            return Err(invalid(
                operation,
                "the declared target shape has fewer axes than the operand",
            ));
        };
        <Self as TensorOps<Self>>::broadcast_left::<T>(input, &target[..prefix])
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Fused `beta * mat + alpha * (mat1 @ mat2)`.
impl<T: DType, D: Device> Execute<Descriptor<op::Addmm>> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Addmm>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Addmm;
        let [mat, lhs, rhs] = ternary_operands(self, request.inputs, operation)?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::addmm::<T>(mat, lhs, rhs, attributes.beta, attributes.alpha)
            .map_err(|error| kernel_error(operation, error))
    }
}

/// Scaled dot-product attention, with the mask as an optional fourth operand.
///
/// The attribute set says whether a mask is present, so the operand count and
/// the declared contract have to agree before anything runs; a descriptor that
/// declares a mask and supplies three operands is a defect, not a request to
/// attend without one.
impl<T: DType, D: Device> Execute<Descriptor<op::ScaledDotProductAttention>>
    for CpuBackendImpl<T, D>
{
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ScaledDotProductAttention>, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ScaledDotProductAttention;
        let attributes = request.operation.descriptor().attributes();
        let (operands, mask) = match request.inputs {
            [query, key, value] if !attributes.has_mask => ([query, key, value], None),
            [query, key, value, mask] if attributes.has_mask => {
                ([query, key, value], Some(operand(mask, operation)?))
            }
            _ => {
                return Err(invalid(
                    operation,
                    "operand count does not match the declared mask",
                ));
            }
        };
        let mut bound = Vec::with_capacity(3);
        for handle in operands {
            let storage = operand(handle, operation)?;
            admitted(self, operation, storage)?;
            bound.push(storage);
        }
        if let Some(mask) = mask {
            admitted(self, operation, mask)?;
        }
        <Self as TensorOps<Self>>::scaled_dot_product_attention::<T>(
            bound[0],
            bound[1],
            bound[2],
            mask,
            attributes.scale,
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
///
/// The group names are matched generically. Which capability shape an identity
/// was declared under is the registry's business; the only thing this proof
/// asserts is that every advertised identity, in every group, has an executor.
/// Naming the groups here would mean a second place to update whenever the
/// declaration grows one, and a group silently omitted from that list would
/// disable the proof for its members without failing anything.
macro_rules! assert_every_advertised_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<Descriptor<O>>,
            {
            }

            const fn assert_all<T: DType, D: Device>() {
                $($(executes::<op::$operation, CpuBackendImpl<T, D>>();)*)*
            }

            assert_all::<f32, incin_core::prelude::Cpu>();
        };
    };
}

crate::capability::cpu_descriptor_operations!(assert_every_advertised_row_executes,);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::CpuBuffer;
    use incin_core::exec::catalog::{
        AxisAttributes, BatchNormAttributes, Conv1dAttributes, Conv2dAttributes,
        LayerNormAttributes, NoAttributes, ShapeAttributes,
    };
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch};
    use incin_core::prelude::{Cpu, Local};

    type TestBackend = CpuBackendImpl<f32, Cpu>;

    fn storage(values: &[f32], shape: &[usize]) -> CpuStorage {
        CpuStorage::try_from_contiguous(CpuBuffer::F32(values.to_vec()), shape.to_vec())
            .expect("test storage must be well formed")
    }

    fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
        TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
    }

    fn context() -> ExecutionContext<TestBackend> {
        ExecutionContext::new(TestBackend::new())
    }

    /// Step size and tolerance, and why these values.
    ///
    /// Every function checked below is a polynomial of degree at most two in
    /// its inputs, so a central difference has no truncation error and the only
    /// error is f32 cancellation, of order `machine_epsilon * |f| / (eps *
    /// |gradient|)`. That term *shrinks* as the step grows, which is why the
    /// step is `1e-2` rather than the more usual `1e-4`: at `1e-4` the same
    /// gradients came out ~1% off purely from cancellation, and loosening the
    /// tolerance to absorb that would have hidden real errors of the same size.
    ///
    /// What this proves is bounded. `gradcheck` ignores any element whose
    /// absolute difference is below `1e-3`, so this catches a gradient that is
    /// structurally wrong - missing, misrouted, or wrongly scaled - and does
    /// not resolve differences finer than that ceiling. The exact agreement
    /// between the canonical and legacy paths is asserted separately, by
    /// `canonical_and_legacy_gradients_are_identical`.
    const GRADIENT_STEP: f64 = 1e-2;
    const GRADIENT_TOLERANCE: f64 = 1e-3;

    /// A gradient that flows through the canonical path must match a finite
    /// difference of the same path.
    ///
    /// This is the property that makes the migration safe to depend on: the
    /// descriptor executors reuse the legacy kernels' tape entries, and a
    /// reuse that lost one would still produce the right forward value.
    #[test]
    fn canonical_pointwise_gradients_match_finite_differences() {
        let context = context();
        let lhs = storage(&[0.5, 1.5, -2.0, 3.0], &[4]);
        let rhs = storage(&[2.0, -1.0, 0.5, 1.25], &[4]);

        let error = gradcheck(
            |inputs| {
                let product = dispatch::execute::<op::Mul, _>(
                    &context,
                    NoAttributes,
                    &[handle(&inputs[0]), handle(&inputs[1])],
                )
                .expect("mul executes");
                dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
                    .expect("sum_all executes")
            },
            &[lhs, rhs],
            GRADIENT_STEP,
        );
        assert!(
            error < GRADIENT_TOLERANCE,
            "canonical mul gradient error {error} exceeds {GRADIENT_TOLERANCE}"
        );
    }

    /// The same check across a view operation, whose backward rule is the
    /// inverse view rather than an arithmetic rule.
    #[test]
    fn canonical_view_gradients_match_finite_differences() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let error = gradcheck(
            |inputs| {
                let reshaped = dispatch::execute::<op::ReshapeExact, _>(
                    &context,
                    ShapeAttributes { shape: vec![3, 2] },
                    &[handle(&inputs[0])],
                )
                .expect("reshape executes");
                let scaled = dispatch::execute::<op::Mul, _>(
                    &context,
                    NoAttributes,
                    &[handle(&reshaped), handle(&reshaped)],
                )
                .expect("mul executes");
                dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &[handle(&scaled)])
                    .expect("mean_all executes")
            },
            &[input],
            GRADIENT_STEP,
        );
        assert!(
            error < GRADIENT_TOLERANCE,
            "canonical reshape gradient error {error} exceeds {GRADIENT_TOLERANCE}"
        );
    }

    /// A single-axis reduction's gradient, which must scatter back over the
    /// reduced axis rather than over the whole tensor.
    #[test]
    fn canonical_axis_reduction_gradients_match_finite_differences() {
        let context = context();
        let input = storage(&[0.5, 1.5, -2.0, 3.0, 0.25, -0.75], &[2, 3]);

        let error = gradcheck(
            |inputs| {
                let reduced = dispatch::execute::<op::SumDim, _>(
                    &context,
                    AxisAttributes { axis: 1 },
                    &[handle(&inputs[0])],
                )
                .expect("sum_dim executes");
                let squared = dispatch::execute::<op::Mul, _>(
                    &context,
                    NoAttributes,
                    &[handle(&reduced), handle(&reduced)],
                )
                .expect("mul executes");
                dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&squared)])
                    .expect("sum_all executes")
            },
            &[input],
            GRADIENT_STEP,
        );
        assert!(
            error < GRADIENT_TOLERANCE,
            "canonical sum_dim gradient error {error} exceeds {GRADIENT_TOLERANCE}"
        );
    }

    /// Matrix multiplication, whose gradient routes each operand through a
    /// different transposed product.
    #[test]
    fn canonical_matmul_gradients_match_finite_differences() {
        let context = context();
        let lhs = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = storage(&[0.5, -1.0, 2.0, 0.25, -0.5, 1.5], &[3, 2]);

        let error = gradcheck(
            |inputs| {
                let product = dispatch::execute::<op::MatMulExact, _>(
                    &context,
                    NoAttributes,
                    &[handle(&inputs[0]), handle(&inputs[1])],
                )
                .expect("matmul executes");
                dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
                    .expect("sum_all executes")
            },
            &[lhs, rhs],
            GRADIENT_STEP,
        );
        assert!(
            error < GRADIENT_TOLERANCE,
            "canonical matmul gradient error {error} exceeds {GRADIENT_TOLERANCE}"
        );
    }

    /// The canonical and legacy paths must produce the *same* gradient, not
    /// merely two gradients that each survive a finite-difference check.
    ///
    /// This is the assertion that makes the migration a migration. Because
    /// both paths run the same kernel body and push the same tape entry, the
    /// agreement is exact rather than approximate, so it is compared exactly:
    /// a tolerance here would let a genuine divergence through.
    #[test]
    fn canonical_and_legacy_gradients_are_identical() {
        use crate::cpu::tape;
        use incin_core::backend_authoring::{NumericOps, ReductionOps};

        let context = context();
        let lhs = storage(&[0.5, 1.5, -2.0, 3.0], &[4]);
        let rhs = storage(&[2.0, -1.0, 0.5, 1.25], &[4]);

        let canonical_scalar = {
            let product = dispatch::execute::<op::Mul, _>(
                &context,
                NoAttributes,
                &[handle(&lhs), handle(&rhs)],
            )
            .expect("mul executes");
            dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
                .expect("sum_all executes")
        };
        let canonical = tape::backward(&canonical_scalar).expect("backward succeeds");
        let canonical_lhs = canonical
            .get(lhs.id)
            .expect("lhs receives a gradient")
            .clone();
        let canonical_rhs = canonical
            .get(rhs.id)
            .expect("rhs receives a gradient")
            .clone();

        let legacy_scalar = {
            let product = <TestBackend as NumericOps<TestBackend>>::mul::<f32>(&lhs, &rhs).unwrap();
            <TestBackend as ReductionOps<TestBackend>>::sum_all::<f32>(&product).unwrap()
        };
        let legacy = tape::backward(&legacy_scalar).expect("backward succeeds");
        let legacy_lhs = legacy.get(lhs.id).expect("lhs receives a gradient");
        let legacy_rhs = legacy.get(rhs.id).expect("rhs receives a gradient");

        for (index, (canonical, legacy)) in
            [(&canonical_lhs, legacy_lhs), (&canonical_rhs, legacy_rhs)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                canonical.shape.to_vec(),
                legacy.shape.to_vec(),
                "operand {index} gradient shape diverged"
            );
            for flat in 0..canonical.shape.iter().product::<usize>() {
                let mut multi = vec![0usize; canonical.shape.len()];
                let mut remaining = flat;
                for axis in (0..canonical.shape.len()).rev() {
                    multi[axis] = remaining % canonical.shape[axis];
                    remaining /= canonical.shape[axis];
                }
                assert_eq!(
                    canonical.get(&multi),
                    legacy.get(&multi),
                    "operand {index} gradient diverged at {multi:?}"
                );
            }
        }
    }

    fn batch_norm_attributes(training: bool, epsilon: f64) -> BatchNormAttributes {
        BatchNormAttributes {
            epsilon,
            momentum: 0.1,
            training,
            has_weight: false,
            has_bias: false,
            has_running_mean: !training,
            has_running_variance: !training,
        }
    }

    /// A training batch norm must be refused, not answered in inference mode.
    ///
    /// This is the failure the executor was written to prevent. The kernel
    /// behind it binds its momentum to `_momentum` and computes no batch
    /// statistics, so the wrong answer here is not an error or a NaN but a
    /// correctly shaped tensor of finite values that silently used the running
    /// statistics, with the caller's running statistics left unchanged.
    #[test]
    fn a_training_batch_norm_is_refused_rather_than_evaluated_in_inference_mode() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let error = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(true, 1e-5),
            &[handle(&input)],
        )
        .expect_err("a training batch norm has no correct answer on this backend");
        let message = format!("{error}");
        assert!(
            message.contains("inference mode only"),
            "the refusal must name the reason, not just fail: {message}"
        );
    }

    /// The same call in inference mode, with running statistics, succeeds.
    ///
    /// Without this the test above would pass equally well if the executor
    /// refused everything.
    #[test]
    fn an_inference_batch_norm_with_running_statistics_executes() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
        let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

        let output = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(false, 1e-5),
            &[
                handle(&input),
                handle(&running_mean),
                handle(&running_variance),
            ],
        )
        .expect("an inference batch norm with running statistics executes");
        assert_eq!(output.shape.to_vec(), vec![2, 3]);
    }

    /// An epsilon that survives descriptor validation but not the narrowing to
    /// the kernel's `f32` is refused.
    ///
    /// `validate_epsilon` accepts any finite non-negative `f64`, so `1e-300` is
    /// a legal descriptor. It narrows to exactly zero, which would leave the
    /// division in the normalization unguarded, so the executor is the layer
    /// that has to catch it.
    #[test]
    fn an_epsilon_that_flushes_to_zero_in_f32_is_refused() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
        let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

        let error = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(false, 1e-300),
            &[
                handle(&input),
                handle(&running_mean),
                handle(&running_variance),
            ],
        )
        .expect_err("an epsilon that narrows to zero is not the epsilon that was asked for");
        let message = format!("{error}");
        assert!(
            message.contains("positive finite f32"),
            "the refusal must name the reason, not just fail: {message}"
        );
    }

    /// Layer norm reaches its kernel through the canonical path.
    #[test]
    fn canonical_layer_norm_executes_and_normalizes_each_row() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weight = storage(&[1.0, 1.0, 1.0], &[3]);

        let output = dispatch::execute::<op::LayerNorm, _>(
            &context,
            LayerNormAttributes {
                normalized_shape: vec![3],
                epsilon: 1e-5,
                has_bias: false,
            },
            &[handle(&input), handle(&weight)],
        )
        .expect("layer norm executes");
        assert_eq!(output.shape.to_vec(), vec![2, 3]);
        // Each row is [x-1, x, x+1] with unit spacing, so normalizing it leaves
        // a zero mean and the same ordering. Checking the mean rather than the
        // exact values keeps this a routing test rather than a second copy of
        // the kernel's own numerical tests in `ops::norm`.
        for row in 0..2 {
            let mean: f64 = (0..3).map(|column| output.get(&[row, column])).sum::<f64>() / 3.0;
            assert!(
                mean.abs() < 1e-5,
                "row {row} was not normalized: mean {mean}"
            );
        }
    }

    /// A convolution with a bias dispatches.
    ///
    /// The bias is rank one and the activation is rank four, and
    /// `dispatch::execute` applies the operation's single capability row to
    /// every operand in turn. A row whose minimum rank was set from the
    /// activation alone therefore refuses its own bias, which is how this test
    /// found `conv2d` advertising `3..=4` while being undispatchable with one.
    /// It belongs here rather than beside the kernel tests because the property
    /// it protects is the capability row's, not the kernel's.
    #[test]
    fn a_convolution_with_a_bias_is_not_refused_by_its_own_rank_bound() {
        let context = context();
        let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let weight = storage(&[1.0], &[1, 1, 1, 1]);
        let bias = storage(&[0.5], &[1]);

        let output = dispatch::execute::<op::Conv2dExact, _>(
            &context,
            Conv2dAttributes {
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
                groups: 1,
                has_bias: true,
            },
            &[handle(&activation), handle(&weight), handle(&bias)],
        )
        .expect("a biased conv2d executes");
        assert_eq!(output.shape.to_vec(), vec![1, 1, 2, 2]);
    }

    /// One-dimensional convolution reaches its kernel through the canonical path.
    #[test]
    fn canonical_conv1d_executes_at_the_ranks_its_row_advertises() {
        let context = context();
        // [N, C, L] with a single input and output channel, so the result is a
        // plain sliding sum over pairs.
        let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let weight = storage(&[1.0, 1.0], &[1, 1, 2]);

        let output = dispatch::execute::<op::Conv1dExact, _>(
            &context,
            Conv1dAttributes {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                has_bias: false,
            },
            &[handle(&activation), handle(&weight)],
        )
        .expect("conv1d executes");
        assert_eq!(output.shape.to_vec(), vec![1, 1, 3]);
        for (index, expected) in [3.0, 5.0, 7.0].into_iter().enumerate() {
            assert_eq!(output.get(&[0, 0, index]), expected);
        }
    }

    /// An operand the kernel would narrow to f32 is refused before it runs.
    ///
    /// The capability row for the spatial group is f32-only, but `admitted` is
    /// asked about the activation alone, so without the explicit check a
    /// double-precision weight would be accepted here, narrowed inside the
    /// kernel, and returned in an f32 buffer.
    #[test]
    fn a_non_f32_convolution_operand_is_refused() {
        let context = context();
        let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let weight = CpuStorage::try_from_contiguous(CpuBuffer::F64(vec![1.0, 1.0]), vec![1, 1, 2])
            .expect("test storage must be well formed");

        let error = dispatch::execute::<op::Conv1dExact, _>(
            &context,
            Conv1dAttributes {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                has_bias: false,
            },
            &[handle(&activation), handle(&weight)],
        )
        .expect_err("an f64 weight is not a dtype this kernel honours");
        let message = format!("{error}");
        assert!(
            message.to_lowercase().contains("dtype") || message.contains("F64"),
            "the refusal must name the dtype: {message}"
        );
    }
}
