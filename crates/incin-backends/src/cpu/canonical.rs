//! Canonical descriptor execution for the CPU backend.
//!
//! One `Execute<op::X>` implementation per exact catalog identity,
//! generated from the same `cpu_descriptor_operations!` declaration that
//! generates `CPU_CAPABILITIES`. Advertising an operation and implementing it
//! are therefore the same edit, and a row that claims support the executor does
//! not provide will not compile.
//!
//! This is the FND-005 replacement for the grouped, attribute-polymorphic
//! `Execute<op::MatMulExact>` family: those adapters accept several semantic
//! operations through one descriptor type, so an error or a capability query
//! could not identify which operation was actually refused. Here the identity
//! is the type.

use incin_core::backend_authoring::{Backend, Execute, ExecutionRequest, HostInterop, StorageBackend};
use incin_core::exec::catalog::{
    AxisVarianceAttributes, Descriptor, DuplicateIndexRule, LossReduction, VarianceAttributes, op,
};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, MathMode, SupportLevel, TensorHandle,
    UnsupportedReason,
};
use incin_core::prelude::{
    BackendError, ConstDType, Cpu, DTypeId, Device, DeviceKind, OperationKind, Q8_0, Reduction,
};
use incin_core::__backend_compat::legacy::{QuantizedOps, TensorOps};
use crate::legacy::LossOps;

use super::CpuBackendImpl;
use super::ops::conv::{conv_transpose2d_impl, conv1d_impl};
use super::ops::embedding::embedding_impl;
use super::ops::elementwise::{
    canonical_abs, canonical_acos, canonical_acosh, canonical_add_scalar, canonical_asin,
    canonical_asinh, canonical_atan, canonical_atan2, canonical_atanh, canonical_clamp,
    canonical_cosh, canonical_elu, canonical_erf, canonical_exp, canonical_fmod, canonical_frac,
    canonical_gelu, canonical_log, canonical_mish, canonical_mul_scalar, canonical_neg,
    canonical_powf, canonical_relu, canonical_remainder, canonical_rsqrt, canonical_sigmoid,
    canonical_sinh, canonical_softmax, canonical_sqrt, canonical_step, canonical_swish,
    canonical_tan, canonical_tanh, canonical_trunc, canonical_unary,
};
use super::ops::norm::{batch_norm_impl, layer_norm_impl};
use super::ops::pool::{adaptive_avg_pool2d_impl, avg_pool2d_impl, max_pool2d_impl};
use super::ops::shape_ops::{
    broadcast_left_storage, diag_storage, div_scalar_storage, flatten_storage, float_to_scalar_storage,
    float_to_vec1_storage, group_norm_storage, instance_norm_storage, int_to_scalar_storage,
    int_to_vec1_storage, masked_fill_storage, narrow_storage, squeeze_storage, sub_scalar_storage,
    transpose_storage,
    tril_storage, triu_storage, unsqueeze_storage,
};
use super::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for CpuBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cpu, query)
    }
}

/// The name the CPU executors answer to when they refuse work.
///
/// Read off the `StorageBackend` impl rather than spelled out again, so renaming
/// the backend cannot leave this file naming one that no longer exists. The type
/// parameters are arbitrary: `CpuBackendImpl` reports the same name for every
/// instantiation, and the free helpers below have no `Self` to ask.
const CPU_NAME: &str = <CpuBackendImpl<Cpu> as StorageBackend>::BACKEND_NAME;

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

/// Whether the request is in module training mode.
///
/// Capability admission uses the same policy bit as dispatch. Gradient
/// recording is a separate execution permission and must not change which
/// training-phase kernel row is selected.
fn training_mode<B: StorageBackend>(context: &ExecutionContext<B>) -> bool {
    context.training()
}

/// Re-check the exact capability row from inside the executor.
///
/// `dispatch::execute` already queried it, but an executor must not depend on
/// having been reached through that path: a backend that only refuses when its
/// caller remembers to ask is a backend whose capability output is advisory.
///
/// `training` is the caller's, not this function's to assume. It used to be
/// hardcoded to `true`, which was invisible while every migrated row supported
/// training and became a refusal of every legal call the moment one did not:
/// the quantization rows carry `training = false`, because their kernels push
/// no tape entry and a training row would promise a gradient that never
/// arrives.
fn admitted<D: Device>(
    backend: &CpuBackendImpl<D>,
    operation: OperationKind,
    storage: &CpuStorage,
    training: bool,
) -> Result<(), BackendError> {
    let metadata = storage.metadata();
    let query = CapabilityQuery {
        operation: incin_core::exec::OperationIdentity::Builtin(operation),
        dtype: metadata.dtype(),
        layout: metadata.layout(),
        rank: metadata.shape().rank(),
        training,
        math_mode: MathMode::Precise,
    };
    match backend.support(&query) {
        SupportLevel::Unsupported(reason) => Err(BackendError::unsupported(CPU_NAME, reason)),
        _ => Ok(()),
    }
}

/// Binary pointwise operations over broadcast operands.
///
/// These take the output shape from the descriptor rather than re-deriving it
/// from the operands. See [`resolved_output_shape`] for why that is sound and
/// what it saves.
macro_rules! pointwise_binary_executors {
    ($(($operation:ident, $kernel:ident, $kernel_with_shape:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = operand(lhs, operation)?;
                let rhs = operand(rhs, operation)?;
                admitted(self, operation, lhs, training_mode(request.context))?;
                admitted(self, operation, rhs, training_mode(request.context))?;
                match resolved_output_shape(request.operation) {
                    Some(out_shape) => {
                        // The whole point of taking the descriptor's answer is
                        // not to compute this one. Doing it anyway under
                        // `debug_assert` means every test run checks that
                        // `infer_outputs` and `broadcast_shape` agree, and a
                        // release build pays nothing for the guarantee.
                        debug_assert_eq!(
                            crate::layout::broadcast_shape(&lhs.shape, &rhs.shape).ok().as_deref(),
                            Some(out_shape),
                            "the descriptor's inferred output shape must be the operands' broadcast",
                        );
                        crate::cpu::ops::elementwise::$kernel_with_shape(lhs, rhs, out_shape)
                    }
                    None => crate::cpu::ops::elementwise::$kernel(lhs, rhs),
                }
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

/// The output shape the descriptor already resolved, if it has one.
///
/// `dispatch::execute_shaped` runs `infer_outputs` and seals the result
/// in a [`Validated`] before any backend is reached, so by the time an executor
/// runs, the broadcast has been computed *and* validated. Re-deriving it with
/// `broadcast_shape` repeats a fallible right-aligned loop and a heap
/// allocation to reach an answer the request is already carrying.
///
/// This is the first place a CPU executor reads the descriptor instead of
/// re-deriving from raw storage. What makes the value trustworthy is that the
/// request is [`Validated`] — not the proof level, which says when the geometry
/// became known rather than whether it is right. The callers cross-check it
/// against a re-derivation under `debug_assert`, so the trust is verified on
/// every test run and free in release.
///
/// `None` is returned when inference produced no single output shape. The
/// caller falls back to deriving it, so this is an optimization that declines
/// rather than a correctness fork.
///
/// [`Validated`]: incin_core::exec::Validated
fn resolved_output_shape<O: incin_core::exec::CanonicalOperation>(
    operation: &incin_core::exec::Validated<Descriptor<O>>,
) -> Option<&[usize]> {
    let [single] = operation.descriptor().outputs() else {
        return None;
    };
    single.shape.as_deref()
}

pointwise_binary_executors![
    (Add, add_storage, add_storage_with_shape),
    (Sub, sub_storage, sub_storage_with_shape),
    (Mul, mul_storage, mul_storage_with_shape),
    (Div, div_storage, div_storage_with_shape),
];

/// Allocation, which has no operand to read anything from.
///
/// Every other executor here recovers its shape, dtype and device from the
/// storage it was handed. These have none, so all three come from the
/// attributes, and the descriptor has already checked that the shape is
/// well formed and the dtype and device are declared rather than guessed.
///
/// They are also the reason `dispatch::execute` grew a capability query keyed
/// on the descriptor's output: its per-operand loop runs zero times here, so
/// without that these would have been the only path to a backend that skipped
/// the registry entirely.
///
/// The element count is resolved through `ShapeTy` rather than the attributes'
/// runtime shape: `numel_for` reads `ShapeTy::STATIC_NUMEL` when the caller's
/// shape type is fully static, so `zeros::<s![2, 3]>()` never runs the
/// checked-multiplication loop over `attributes.shape` at all — the count was
/// already a compile-time constant. This is the one place in the crate the
/// shape type parameter `execute_shaped` carries actually changes what code
/// runs, rather than only what gets asserted against.
macro_rules! allocating_executors {
    ($(($operation:ident, $method:ident $(, $argument:ident)*)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "an allocation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let total = crate::cpu::stride::numel_for_evidence(
                    &attributes.shape,
                    request.operation.shape_evidence().static_numel(),
                )
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                crate::cpu::creation::$method(
                    total,
                    $(attributes.$argument,)*
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

// The scalar parameters come first in every one of these signatures, which is
// why the macro takes them as a prefix rather than trying to place them.
allocating_executors![
    (Zeros, zeros_with_total),
    (Ones, ones_with_total),
    (UniformRandom, rand_with_total),
    (NormalRandom, randn_with_total),
    (Full, full_with_total, value),
    (Arange, arange_with_total, start, step),
    (Linspace, linspace_with_total, start, end),
];

macro_rules! data_executors {
    ($($operation:ident),* $(,)?) => {$ (
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "data creation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let bytes = request
                    .payload
                    .ok_or_else(|| invalid(operation, "data creation requires borrowed bytes"))?;
                <Self as HostInterop>::from_bytes::<f32>(
                    bytes,
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

data_executors![TensorFromData, TensorFromBytes];

/// The same four allocations, returning a trainable variable.
///
/// A separate macro rather than a fourth column on the one above, because the
/// output type is what differs and that is exactly the thing the associated
/// `Output` on `Execute` exists to let vary. Reporting a `CpuVar` as if it were
/// storage would need a conversion the caller then has to undo.
///
/// Their capability rows sit in the same two groups as the storage forms, and
/// carry `training = false` for the same reason: allocating a variable records
/// nothing, whatever is done to it afterwards.
macro_rules! variable_allocating_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = super::var::CpuVar;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<super::var::CpuVar, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "an allocation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let total = crate::cpu::stride::numel_for_evidence(
                    &attributes.shape,
                    request.operation.shape_evidence().static_numel(),
                )
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                crate::cpu::creation::$method(
                    total,
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

variable_allocating_executors![
    (VariableZeros, var_zeros_with_total),
    (VariableOnes, var_ones_with_total),
    (VariableUniformRandom, var_rand_with_total),
    (VariableNormalRandom, var_randn_with_total),
];

/// Reading a value back to the host.
///
/// These are the only migrated operations that do not produce storage, and the
/// reason `Execute` names its output as an associated type rather than fixing
/// it: a scalar readback returns an `f64`, a vector one returns a `Vec<f64>`,
/// and neither is a tensor. Wrapping them in storage to fit a fixed output type
/// would create an allocation whose only purpose is to be immediately unwrapped.
///
/// Each is a synchronisation point on a device backend even though it is free
/// here, which is why they are a group of their own rather than tensor
/// operations that happen to return something small.
macro_rules! readback_executors {
    ($(($operation:ident, $method:ident, $output:ty)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = $output;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<$output, BackendError> {
                let operation = OperationKind::$operation;
                let training = training_mode(request.context);
                let input = reduction_operand(self, request.inputs, operation, training)?;
                $method(input)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

readback_executors![
    (ToHostFloatScalar, float_to_scalar_storage, f64),
    (ToHostFloatVec, float_to_vec1_storage, Vec<f64>),
    (ToHostIntScalar, int_to_scalar_storage, i64),
    (ToHostIntVec, int_to_vec1_storage, Vec<i64>),
];

/// The raw bytes behind an allocation.
///
/// On `Backend` rather than on any operation family, which is why it is not in
/// the macro above: the byte view is a property of an allocation itself and
/// exists before any operation over it does.
impl<D: Device> Execute<op::TensorToBytes> for CpuBackendImpl<D> {
    type Output = Vec<u8>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TensorToBytes, Self>,
    ) -> Result<Vec<u8>, BackendError> {
        let operation = OperationKind::TensorToBytes;
        let training = training_mode(request.context);
        let input = reduction_operand(self, request.inputs, operation, training)?;
        <Self as HostInterop>::to_bytes::<f32>(input)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Reshape to the descriptor's declared shape.
impl<D: Device> Execute<op::ReshapeExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ReshapeExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ReshapeExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "reshape expects exactly one operand"));
        };
        let input = operand(input, operation)?;
        admitted(self, operation, input, training_mode(request.context))?;
        let shape = &request.operation.descriptor().attributes().shape;
        crate::cpu::ops::shape_ops::reshape_storage(input, shape)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Broadcast to the descriptor's declared shape.
impl<D: Device> Execute<op::BroadcastAs> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastAs, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BroadcastAs;
        let [input] = request.inputs else {
            return Err(invalid(
                operation,
                "broadcast_as expects exactly one operand",
            ));
        };
        let input = operand(input, operation)?;
        admitted(self, operation, input, training_mode(request.context))?;
        let shape = &request.operation.descriptor().attributes().shape;
        crate::cpu::ops::shape_ops::broadcast_as_storage(input, shape)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Matrix multiplication over the last two axes, batched over the rest.
impl<D: Device> Execute<op::MatMulExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MatMulExact;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "matmul expects exactly two operands"));
        };
        let lhs = operand(lhs, operation)?;
        let rhs = operand(rhs, operation)?;
        for storage in [lhs, rhs] {
            if storage.metadata().dtype() != DTypeId::F32.descriptor() {
                return Err(BackendError::unsupported(
                    CPU_NAME,
                    UnsupportedReason::DType {
                        operation,
                        dtype: storage.metadata().dtype(),
                    },
                ));
            }
            admitted(self, operation, storage, training_mode(request.context))?;
        }
        crate::cpu::ops::shape_ops::matmul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Bind the single operand a reduction consumes.
fn reduction_operand<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
) -> Result<&'a CpuStorage, BackendError> {
    let [input] = inputs else {
        return Err(invalid(
            operation,
            "a reduction expects exactly one operand",
        ));
    };
    let input = operand(input, operation)?;
    admitted(backend, operation, input, training)?;
    Ok(input)
}

/// Whole-tensor reductions, which take no attributes.
macro_rules! reduce_all_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                crate::cpu::ops::reduce::$method(input)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
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
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let axis = request.operation.descriptor().attributes().axis;
                crate::cpu::ops::reduce::$method(input, axis)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
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
impl<D: Device> Execute<op::Cumsum> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Cumsum, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Cumsum;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let axis = request.operation.descriptor().attributes().axis;
        crate::cpu::ops::reduce::cumsum(input, axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Call an index-returning reduction with the index dtype the descriptor asked
/// for, rather than with whichever one the kernel used to hardcode.
///
/// The index dtype is a type parameter and the descriptor names it at runtime,
/// so the two are bridged by a match. `u8`, `u32` and `i64` are the integer
/// dtypes the backend has; anything else is refused with the operation
/// attached, which is more use than the kernel's untyped refusal.
///
/// This used to be a `produced_index_dtype` check that compared the requested
/// dtype against a constant the kernel was known to build, because the kernels
/// declared the type parameter and then ignored it. They honour it now, so the
/// canonical path forwards the request instead of narrowing it away.
macro_rules! dispatch_index_dtype {
    ($operation:expr, $dtype:expr, |$index:ident| $body:expr) => {{
        let operation = $operation;
        let desc: incin_core::prelude::DTypeDescriptor = $dtype;
        match desc.builtin_id() {
            Some(DTypeId::U8) => {
                type $index = u8;
                $body
            }
            Some(DTypeId::U32) => {
                type $index = u32;
                $body
            }
            Some(DTypeId::I64) => {
                type $index = i64;
                $body
            }
            _ => Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType {
                    operation,
                    dtype: desc,
                },
            )),
        }
    }};
}

/// Index of the extremum, either flattened or along one axis.
macro_rules! index_reduction_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let attributes = request.operation.descriptor().attributes();
                dispatch_index_dtype!(operation, attributes.dtype, |KIndex| {
                    crate::cpu::ops::reduce::$method::<KIndex>(input, attributes.axis)
                        .map_err(|error| kernel_error(CPU_NAME, operation, error))
                })
            }
        }
    )*};
}

index_reduction_executors![(ArgMax, argmax), (ArgMin, argmin)];

/// Indices that would sort the operand along one axis.
impl<D: Device> Execute<op::Argsort> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Argsort, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Argsort;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        dispatch_index_dtype!(operation, attributes.index_dtype, |KIndex| {
            crate::cpu::ops::reduce::argsort::<KIndex>(
                input,
                attributes.axis,
                attributes.descending,
            )
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
        })
    }
}

/// The `k` extreme elements along one axis, as a value and an index tensor.
///
/// This is the first migrated identity whose output is not a single storage
/// handle. The catalog already describes it as two: the value tensor carries
/// the operand dtype and the index tensor carries the declared index dtype, so
/// returning a pair is what the descriptor was already promising.
///
/// The value buffer used to be built as `f32` whatever the operand held, so
/// the row was narrowed to f32 alone to stop the canonical path routing to a
/// mislabel. The kernel converts through the operand's own buffer now, so the
/// row no longer has to be narrower than the kernel.
impl<D: Device> Execute<op::TopK> for CpuBackendImpl<D> {
    type Output = (CpuStorage, CpuStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TopK, Self>,
    ) -> Result<(CpuStorage, CpuStorage), BackendError> {
        let operation = OperationKind::TopK;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        dispatch_index_dtype!(operation, attributes.index_dtype, |KIndex| {
            crate::cpu::ops::reduce::topk::<KIndex>(
                input,
                attributes.k,
                attributes.axis,
                attributes.largest,
            )
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
        })
    }
}

/// Collapse a per-axis window to the single extent the routed CPU kernel takes.
///
/// The descriptor is more expressive than the kernel behind it: it carries one
/// extent per spatial axis, while the historical module-family contract takes one for both. An
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
        if dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
    }
    Ok(())
}

/// Narrow a descriptor epsilon to the width the routed kernel accepts.
///
/// `LayerNormAttributes` and `BatchNormAttributes` carry an `f64`, while
/// The historical module-family normalization methods take an `f32`. Almost
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
impl<D: Device> Execute<op::Conv2dExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
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
        admitted(self, operation, activation, training_mode(request.context))?;

        // The descriptor's per-axis window is forwarded whole. It used to be
        // collapsed to a single extent, and an anisotropic one refused,
        // because the historical module-family contract states one extent for both axes. The
        // kernel behind that signature never needed them equal, so the pair
        // goes straight to it rather than through the narrower spelling.
        crate::cpu::ops::conv::conv2d_windowed_impl::<D, f32>(
            activation,
            weight,
            bias,
            crate::cpu::ops::conv::Window2d {
                stride: attributes.stride,
                padding: attributes.padding,
                dilation: attributes.dilation,
            },
            attributes.groups,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Two-dimensional maximum pooling.
impl<D: Device> Execute<op::MaxPool2d> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaxPool2d, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MaxPool2d;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        let pair = |[height, width]: [usize; 2]| (height, width);

        max_pool2d_impl::<D, f32>(
            input,
            pair(attributes.kernel),
            pair(attributes.stride),
            pair(attributes.padding),
            pair(attributes.dilation),
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Two-dimensional average pooling, which has no dilated form.
impl<D: Device> Execute<op::AvgPool2d> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AvgPool2d, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::AvgPool2d;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        let pair = |[height, width]: [usize; 2]| (height, width);

        avg_pool2d_impl::<D, f32>(
            input,
            pair(attributes.kernel),
            pair(attributes.stride),
            pair(attributes.padding),
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// One-dimensional convolution with an optional bias.
impl<D: Device> Execute<op::Conv1dExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv1dExact, Self>,
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
        admitted(self, operation, activation, training_mode(request.context))?;
        f32_only(operation, &[Some(activation), Some(weight), bias])?;

        // `Conv1dAttributes` already carries one extent per field, so unlike the
        // two-dimensional forms there is nothing to collapse.
        conv1d_impl::<D, f32>(
            activation,
            weight,
            bias,
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.groups,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Two-dimensional transposed convolution with an optional bias.
impl<D: Device> Execute<op::ConvTranspose2d> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ConvTranspose2d, Self>,
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
        admitted(self, operation, activation, training_mode(request.context))?;
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

        conv_transpose2d_impl::<D, f32>(
            activation,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            attributes.groups,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Average pooling to a requested output extent rather than a requested window.
impl<D: Device> Execute<op::AdaptiveAvgPool2dExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AdaptiveAvgPool2dExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::AdaptiveAvgPool2dExact;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        f32_only(operation, &[Some(input)])?;
        let [height, width] = request.operation.descriptor().attributes().output;
        adaptive_avg_pool2d_impl::<D, f32>(input, (height, width))
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Gather rows of a weight table addressed by an integer index tensor.
///
/// `embedding`'s two operands admit different dtypes by construction: the
/// index operand is one of the integer dtypes, and the weight operand is f32
/// only, because `embedding_impl` always reads and writes f32 regardless of
/// the operand's declared dtype. `INDEX_AND_F32_DTYPES` (see `capability.rs`)
/// states the union of the two, the loosest set one row can honestly claim,
/// so both operands pass the registry re-check below; `f32_only` then
/// enforces the weight's real, tighter constraint directly, the same way the
/// convolution executors enforce theirs for a bias the row's dtype set does
/// not cover either.
impl<D: Device> Execute<op::EmbeddingExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::EmbeddingExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::EmbeddingExact;
        let [indices, weight] = request.inputs else {
            return Err(invalid(
                operation,
                "embedding expects an index tensor and a weight table",
            ));
        };
        let indices = operand(indices, operation)?;
        let weight = operand(weight, operation)?;
        admitted(self, operation, indices, training_mode(request.context))?;
        admitted(self, operation, weight, training_mode(request.context))?;
        f32_only(operation, &[Some(weight)])?;
        embedding_impl::<D, f32, i64>(indices, weight)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

// Canonical elementwise executors use concrete CPU helpers; no operation-family
// trait is needed on this execution path.

macro_rules! canonical_unary_executors {
    ($(($operation:ident, $helper:ident)),* $(,)?) => {
        $(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(
                    self,
                    request.inputs,
                    operation,
                    training_mode(request.context),
                )?;
                $helper(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
        )*
    };
}

impl<D: Device> Execute<op::Relu> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Relu, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Relu;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_relu(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

canonical_unary_executors![(Swish, canonical_swish),];

canonical_unary_executors![
    (Gelu, canonical_gelu),
    (Tan, canonical_tan),
    (Asin, canonical_asin),
    (Acos, canonical_acos),
    (Atan, canonical_atan),
    (Sinh, canonical_sinh),
    (Cosh, canonical_cosh),
    (Asinh, canonical_asinh),
    (Acosh, canonical_acosh),
    (Atanh, canonical_atanh),
    (Erf, canonical_erf),
    (Rsqrt, canonical_rsqrt),
];

impl<D: Device> Execute<op::Tanh> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Tanh, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Tanh;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_tanh(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Sigmoid> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Sigmoid, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Sigmoid;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_sigmoid(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Log> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Log, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Log;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_log(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

macro_rules! direct_unary_no_grad_executors {
    ($(($operation:ident, $kernel:path)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(
                    self,
                    request.inputs,
                    operation,
                    training_mode(request.context),
                )?;
                $kernel(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

direct_unary_no_grad_executors![(Trunc, canonical_trunc), (Frac, canonical_frac),];

impl<D: Device> Execute<op::Elu> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Elu, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Elu;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_elu(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Mish> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Mish, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Mish;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_mish(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Sqrt> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Sqrt, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Sqrt;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_sqrt(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Abs> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Abs, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Abs;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_abs(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Exp> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Exp, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Exp;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_exp(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Step> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Step, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Step;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_step(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Neg> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Neg, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Neg;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        canonical_neg(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

macro_rules! direct_unary_float_executors {
    ($(($operation:ident, $kernel:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(
                    self,
                    request.inputs,
                    operation,
                    training_mode(request.context),
                )?;
                canonical_unary(crate::cpu::ops::elementwise_kernel::UnaryOp::$kernel, input)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

direct_unary_float_executors![
    (Sign, Sign),
    (Floor, Floor),
    (Ceil, Ceil),
    (Round, Round),
    (Log2, Log2),
    (Log10, Log10),
    (Sin, Sin),
    (Cos, Cos),
];

/// Unary float operations parametrised by one scalar attribute.
macro_rules! scalar_float_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let value = request.operation.descriptor().attributes().value;
                $method(input, value)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

scalar_float_executors![
    (AddScalar, canonical_add_scalar),
    (MulScalar, canonical_mul_scalar),
    (Powf, canonical_powf),
];

/// Binary elementwise float operations over broadcast operands.
macro_rules! binary_float_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = operand(lhs, operation)?;
                let rhs = operand(rhs, operation)?;
                admitted(self, operation, lhs, training_mode(request.context))?;
                admitted(self, operation, rhs, training_mode(request.context))?;
                $method(lhs, rhs)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

binary_float_executors![(Fmod, canonical_fmod), (Remainder, canonical_remainder),];

impl<D: Device> Execute<op::Atan2> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Atan2, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Atan2;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "operation expects exactly two operands"));
        };
        let lhs = operand(lhs, operation)?;
        let rhs = operand(rhs, operation)?;
        admitted(self, operation, lhs, training_mode(request.context))?;
        admitted(self, operation, rhs, training_mode(request.context))?;
        canonical_atan2(lhs, rhs).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Elementwise clamp, whose two bounds are a single typed attribute set.
impl<D: Device> Execute<op::Clamp> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Clamp, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Clamp;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        canonical_clamp(input, attributes.min, attributes.max)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Softmax along the axis its attributes name.
impl<D: Device> Execute<op::Softmax> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Softmax, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Softmax;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let axis = request.operation.descriptor().attributes().axis;
        canonical_softmax::<D>(input, axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Normalize over the trailing axes the attributes name, then scale and shift.
impl<D: Device> Execute<op::LayerNorm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LayerNorm, Self>,
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
        admitted(self, operation, input, training_mode(request.context))?;
        f32_only(operation, &[Some(input), Some(weight), bias])?;
        let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;

        // `normalized_shape` is not passed on: the descriptor has already
        // checked that it is the operand's trailing suffix and that the weight
        // and bias match it, and the kernel derives the same split from the
        // weight's own shape.
        layer_norm_impl::<D, f32>(input, weight, bias, epsilon)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Normalize each channel by its running statistics, then scale and shift.
///
/// Inference only, and refused rather than approximated otherwise. See the
/// refusal below.
impl<D: Device> Execute<op::BatchNorm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchNorm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BatchNorm;
        let attributes = request.operation.descriptor().attributes();
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
        // Training mode normalizes by the batch's own statistics and takes no
        // running ones. The two modes are separate kernels rather than a flag,
        // because the inference one substitutes a zero mean and a unit
        // variance for an absent operand instead of failing: an executor that
        // routed a training request there would get a plausible wrong answer
        // rather than an error, which is what this used to refuse outright.
        if attributes.training {
            let input = operand(input, operation)?;
            let weight = weight.map(|value| operand(value, operation)).transpose()?;
            let bias = bias.map(|value| operand(value, operation)).transpose()?;
            admitted(self, operation, input, training_mode(request.context))?;
            f32_only(operation, &[Some(input), weight, bias])?;
            let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;
            return crate::cpu::ops::norm::batch_norm_training_impl::<D, f32>(
                input, weight, bias, epsilon,
            )
            .map_err(|error| kernel_error(CPU_NAME, operation, error));
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
        admitted(self, operation, input, training_mode(request.context))?;
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

        batch_norm_impl::<D, f32>(
            input,
            weight,
            bias,
            Some(running_mean),
            Some(running_variance),
            epsilon,
            attributes.momentum,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Bind the two operands a binary tensor operation consumes.
fn binary_operands<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
) -> Result<(&'a CpuStorage, &'a CpuStorage), BackendError> {
    let [lhs, rhs] = inputs else {
        return Err(invalid(operation, "operation expects exactly two operands"));
    };
    let lhs = operand(lhs, operation)?;
    let rhs = operand(rhs, operation)?;
    admitted(backend, operation, lhs, training)?;
    admitted(backend, operation, rhs, training)?;
    Ok((lhs, rhs))
}

/// Binary elementwise tensor operations that take no attributes.
///
/// Comparisons and logical connectives are here rather than with the float
/// family because their semantic profile preserves the operand dtype instead of
/// producing a boolean one, and because they carry no gradient.
macro_rules! numeric_binary_tensor_executors {
    ($(($operation:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (lhs, rhs) = binary_operands(self, request.inputs, operation, training_mode(request.context))?;
                let out_shape = resolved_output_shape(request.operation)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape).unwrap_or_default());
                crate::cpu::ops::elementwise::elementwise_binary(lhs, rhs, &out_shape, $func)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

numeric_binary_tensor_executors![
    (Maximum, |a, b| a.max(b)),
    (Minimum, |a, b| a.min(b)),
    (AbsDiff, |a, b| (a - b).abs()),
];

macro_rules! cmp_tensor_executors {
    ($(($operation:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (lhs, rhs) = binary_operands(self, request.inputs, operation, training_mode(request.context))?;
                if lhs.meta.dtype() != rhs.meta.dtype() {
                    return Err(invalid(operation, "comparison operands must have matching dtypes"));
                }
                crate::cpu::ops::shape_ops::elementwise_cmp(lhs, rhs, $func)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

cmp_tensor_executors![
    (CmpEq, |a, b| a == b),
    (CmpNe, |a, b| a != b),
    (CmpLt, |a, b| a < b),
    (CmpLe, |a, b| a <= b),
    (CmpGt, |a, b| a > b),
    (CmpGe, |a, b| a >= b),
];

macro_rules! logical_binary_tensor_executors {
    ($(($operation:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (lhs, rhs) = binary_operands(self, request.inputs, operation, training_mode(request.context))?;
                if lhs.meta.dtype() != <bool as ConstDType>::DESCRIPTOR || rhs.meta.dtype() != <bool as ConstDType>::DESCRIPTOR {
                    return Err(invalid(operation, "logical operation operands must be bool"));
                }
                crate::cpu::ops::shape_ops::elementwise_cmp(lhs, rhs, $func)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

logical_binary_tensor_executors![
    (LogicalAnd, |a, b| a != 0.0 && b != 0.0),
    (LogicalOr, |a, b| a != 0.0 || b != 0.0),
];

/// Batched matrix multiplication, whose operand rank contract differs from the
/// plain `matmul` row and so does not share its registration.
impl<D: Device> Execute<op::BatchedMatMul> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchedMatMul, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BatchedMatMul;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        crate::cpu::ops::shape_ops::matmul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Unary tensor operations parametrised by one scalar attribute.
macro_rules! scalar_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let value = request.operation.descriptor().attributes().value;
                $method(input, value)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

scalar_tensor_executors![
    (SubScalar, sub_scalar_storage),
    (DivScalar, div_scalar_storage)
];

/// Triangular and diagonal views, parametrised by a signed diagonal offset.
macro_rules! diagonal_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let offset = request.operation.descriptor().attributes().offset;
                $method(input, offset)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

diagonal_tensor_executors![
    (Triu, triu_storage),
    (Tril, tril_storage),
    (Diag, diag_storage)
];

/// Rank-changing views parametrised by a single axis.
macro_rules! axis_tensor_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let axis = request.operation.descriptor().attributes().axis;
                $method(input, axis)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

axis_tensor_executors![
    (SqueezeExact, squeeze_storage),
    (UnsqueezeExact, unsqueeze_storage)
];

/// Elementwise logical negation.
impl<D: Device> Execute<op::LogicalNot> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LogicalNot, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::LogicalNot;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        if input.meta.dtype() != <bool as ConstDType>::DESCRIPTOR {
            return Err(invalid(operation, "logical_not operand must be bool"));
        }
        crate::cpu::ops::shape_ops::elementwise_cmp(input, input, |a, _| a == 0.0)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Swap the two axes the descriptor names.
impl<D: Device> Execute<op::TransposeExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TransposeExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::TransposeExact;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        transpose_storage(input, attributes.first, attributes.second)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Take a contiguous run along one axis.
impl<D: Device> Execute<op::Narrow> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Narrow, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Narrow;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        narrow_storage(input, attributes.axis, attributes.start, attributes.length)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Collapse an inclusive axis range into one axis.
impl<D: Device> Execute<op::FlattenExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::FlattenExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::FlattenExact;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        flatten_storage(input, attributes.start_axis, attributes.end_axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Elementwise selection between two operands under a mask.
///
/// The operand order is the one the catalog's legacy source names -
/// `TensorOps::where_cond(mask, on_true, on_false)` - so a caller that reads
/// the catalog row gets the same meaning from either path.
impl<D: Device> Execute<op::WhereCond> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::WhereCond, Self>,
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
            admitted(self, operation, storage, training_mode(request.context))?;
        }
        match on_true.dtype().builtin_id() {
            Some(DTypeId::F32) => {
                <Self as TensorOps<Self>>::where_cond::<f32>(mask, on_true, on_false)
            }
            Some(DTypeId::F64) => {
                <Self as TensorOps<Self>>::where_cond::<f64>(mask, on_true, on_false)
            }
            Some(DTypeId::I64) => {
                <Self as TensorOps<Self>>::where_cond::<i64>(mask, on_true, on_false)
            }
            Some(DTypeId::U8) => {
                <Self as TensorOps<Self>>::where_cond::<u8>(mask, on_true, on_false)
            }
            Some(DTypeId::U32) => {
                <Self as TensorOps<Self>>::where_cond::<u32>(mask, on_true, on_false)
            }
            Some(DTypeId::Bool) => {
                <Self as TensorOps<Self>>::where_cond::<bool>(mask, on_true, on_false)
            }
            _ => return Err(invalid(operation, "unsupported value dtype for where_cond")),
        }
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Overwrite the masked positions with the declared scalar.
impl<D: Device> Execute<op::MaskedFill> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaskedFill, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MaskedFill;
        let (input, mask) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let value = request.operation.descriptor().attributes().value;
        masked_fill_storage(input, mask, value)
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Interpolate between two operands at the declared weight.
impl<D: Device> Execute<op::Lerp> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Lerp, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Lerp;
        let (start, end) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let weight = request.operation.descriptor().attributes().weight;
        <Self as TensorOps<Self>>::lerp::<f32>(start, end, weight)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Bind a variable-length operand list, checking each one.
///
/// `concat` and `stack` take one or more operands, so their arity contract is
/// a lower bound rather than a fixed count. An empty list is still a defect.
fn variadic_operands<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
) -> Result<Vec<&'a CpuStorage>, BackendError> {
    if inputs.is_empty() {
        return Err(invalid(operation, "operation expects at least one operand"));
    }
    inputs
        .iter()
        .map(|handle| {
            let storage = operand(handle, operation)?;
            admitted(backend, operation, storage, training)?;
            Ok(storage)
        })
        .collect()
}

/// Bind the three operands an indexing or fused operation consumes.
fn ternary_operands<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
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
        admitted(backend, operation, storage, training)?;
    }
    Ok(bound)
}

/// Join operands along an existing axis.
impl<D: Device> Execute<op::ConcatExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ConcatExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ConcatExact;
        let operands = variadic_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as TensorOps<Self>>::concat::<f32>(&operands, axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Join operands along a new axis.
impl<D: Device> Execute<op::StackExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StackExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::StackExact;
        let operands = variadic_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let axis = request.operation.descriptor().attributes().axis;
        <Self as TensorOps<Self>>::stack::<f32>(&operands, axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Take a half-open window per axis.
impl<D: Device> Execute<op::SliceExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::SliceExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::SliceExact;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let ranges = &request.operation.descriptor().attributes().ranges;
        <Self as TensorOps<Self>>::slice::<f32>(input, ranges)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Indexing operations that read one axis and one index operand.
macro_rules! indexing_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (input, index) = binary_operands(self, request.inputs, operation, training_mode(request.context))?;
                let axis = request.operation.descriptor().attributes().axis;
                <Self as TensorOps<Self>>::$method::<f32, i64>(input, axis, index)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
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
impl<D: Device> Execute<op::Scatter> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Scatter, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Scatter;
        let [input, index, source] = ternary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        if attributes.duplicate_indices == DuplicateIndexRule::Reject {
            return Err(invalid(
                operation,
                "this backend applies last-write-wins and cannot reject duplicate indices",
            ));
        }
        <Self as TensorOps<Self>>::scatter::<f32, i64>(input, attributes.axis, index, source)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Tile the operand per axis.
impl<D: Device> Execute<op::Repeat> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Repeat, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Repeat;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let repeats = &request.operation.descriptor().attributes().repeats;
        <Self as TensorOps<Self>>::repeat::<f32>(input, repeats)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Extend each axis with the declared constant.
impl<D: Device> Execute<op::Pad> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Pad, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Pad;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::pad::<f32>(input, &attributes.padding, attributes.value)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Extract sliding windows along one axis.
impl<D: Device> Execute<op::Unfold> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Unfold, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Unfold;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::unfold::<f32>(
            input,
            attributes.axis,
            attributes.size,
            attributes.step,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Redistribute channel depth into spatial extent.
impl<D: Device> Execute<op::PixelShuffle> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::PixelShuffle, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::PixelShuffle;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let factor = request.operation.descriptor().attributes().upscale_factor;
        <Self as TensorOps<Self>>::pixel_shuffle::<f32>(input, factor)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Normalize within channel groups.
impl<D: Device> Execute<op::GroupNorm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::GroupNorm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::GroupNorm;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        group_norm_storage(input, attributes.groups, attributes.epsilon)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Zero a random share of the operand and scale the rest to compensate.
///
/// The three cases are separate on purpose. Outside training, and at a
/// probability of zero, dropout is the identity and must not consume a random
/// draw: doing so would make an inference pass perturb the generator and change
/// every later training step. At a probability of one it is a multiply by zero,
/// which the general path cannot express because its compensating scale is
/// `1 / (1 - p)`.
///
/// The mask is drawn fresh on every call and is deliberately not reproducible
/// from the descriptor. That makes this the one migrated operation whose result
/// two identical invocations disagree about, which is a property of dropout and
/// not a defect here, and the reason its test asserts the distribution rather
/// than the values.
impl<D: Device> Execute<op::Dropout> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dropout, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Dropout;
        let training = training_mode(request.context);
        let input = reduction_operand(self, request.inputs, operation, training)?;
        let attributes = request.operation.descriptor().attributes();
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        if !attributes.training || attributes.probability <= 0.0 {
            return Ok(input.clone());
        }
        if attributes.probability >= 1.0 {
            return canonical_mul_scalar(input, 0.0).map_err(wrap);
        }

        let metadata = input.metadata();
        let total = crate::cpu::stride::checked_numel(input.shape.as_ref()).map_err(wrap)?;
        let draw = super::creation::rand_with_total(
            total,
            input.shape.as_ref(),
            metadata.dtype(),
            &metadata.device(),
        )
        .map_err(wrap)?;
        // `step` is one above zero and zero at or below it, so shifting the
        // uniform draw down by the probability turns it into the mask directly
        // and keeps exactly the share of elements the attribute asked for.
        let shifted = canonical_add_scalar(&draw, -attributes.probability).map_err(wrap)?;
        let mask = canonical_step(&shifted).map_err(wrap)?;
        let kept = crate::cpu::ops::elementwise::mul_storage(input, &mask).map_err(wrap)?;
        canonical_mul_scalar(&kept, 1.0 / (1.0 - attributes.probability)).map_err(wrap)
    }
}

/// An affine layer: the operand against a transposed weight, plus a bias.
///
/// The transpose is the whole reason this is not just a matmul. `Linear` stores
/// its weight as `[out, in]`, which is the layout every checkpoint format uses,
/// and the product needs `[in, out]`. Doing it here rather than asking the
/// caller to pre-transpose keeps the descriptor's operand order the same as the
/// module's field order.
impl<D: Device> Execute<op::Linear> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Linear, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Linear;
        let (input, weight, bias) = match request.inputs {
            [input, weight] => (input, weight, None),
            [input, weight, bias] => (input, weight, Some(bias)),
            _ => {
                return Err(invalid(
                    operation,
                    "linear expects an input, a weight and an optional bias",
                ));
            }
        };
        let training = training_mode(request.context);
        let input = operand(input, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, input, training)?;
        admitted(self, operation, weight, training)?;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        let transposed = transpose_storage(weight, 0, 1).map_err(wrap)?;
        let product =
            crate::cpu::ops::shape_ops::matmul_storage(input, &transposed).map_err(wrap)?;
        match bias {
            None => Ok(product),
            Some(bias) => {
                admitted(self, operation, bias, training)?;
                // The add broadcasts, which is what lets a `[out]` bias meet a
                // `[.., out]` product without the caller reshaping it.
                crate::cpu::ops::elementwise::add_storage(&product, bias).map_err(wrap)
            }
        }
    }
}

/// Scale by the root mean square over the trailing axis, then by a weight.
///
/// No mean is subtracted, which is the whole difference from `layer_norm` and
/// the reason this cannot be routed to it.
impl<D: Device> Execute<op::RmsNorm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::RmsNorm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::RmsNorm;
        let training = training_mode(request.context);
        let (input, weight) = binary_operands(self, request.inputs, operation, training)?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        // The trailing axis, and the keep-dim reduction over it so the divisor
        // broadcasts back against the operand.
        let axis = input.shape.len().saturating_sub(1);
        let squared = crate::cpu::ops::elementwise::mul_storage(input, input).map_err(wrap)?;
        let mean = crate::cpu::ops::reduce::mean_keepdim(&squared, axis).map_err(wrap)?;
        let guarded = canonical_add_scalar(&mean, epsilon).map_err(wrap)?;
        let scale = canonical_sqrt(&guarded).map_err(wrap)?;
        let normalized = crate::cpu::ops::elementwise::div_storage(input, &scale).map_err(wrap)?;
        crate::cpu::ops::elementwise::mul_storage(&normalized, weight).map_err(wrap)
    }
}

/// Normalize each channel independently.
impl<D: Device> Execute<op::InstanceNorm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::InstanceNorm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::InstanceNorm;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        instance_norm_storage(input, epsilon)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
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
impl<D: Device> Execute<op::BroadcastLeft> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastLeft, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BroadcastLeft;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let target = &request.operation.descriptor().attributes().shape;
        let rank = input.metadata().shape().rank();
        let Some(prefix) = target.len().checked_sub(rank) else {
            return Err(invalid(
                operation,
                "the declared target shape has fewer axes than the operand",
            ));
        };
        broadcast_left_storage(input, &target[..prefix])
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Fused `beta * mat + alpha * (mat1 @ mat2)`.
impl<D: Device> Execute<op::Addmm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Addmm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Addmm;
        let [mat, lhs, rhs] = ternary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        <Self as TensorOps<Self>>::addmm::<f32>(mat, lhs, rhs, attributes.beta, attributes.alpha)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Scaled dot-product attention, with the mask as an optional fourth operand.
///
/// The attribute set says whether a mask is present, so the operand count and
/// the declared contract have to agree before anything runs; a descriptor that
/// declares a mask and supplies three operands is a defect, not a request to
/// attend without one.
impl<D: Device> Execute<op::ScaledDotProductAttention> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ScaledDotProductAttention, Self>,
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
            admitted(self, operation, storage, training_mode(request.context))?;
            bound.push(storage);
        }
        if let Some(mask) = mask {
            admitted(self, operation, mask, training_mode(request.context))?;
        }
        <Self as TensorOps<Self>>::scaled_dot_product_attention::<f32>(
            bound[0],
            bound[1],
            bound[2],
            mask,
            attributes.scale,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Reinterpret an operand's values under a different dtype.
///
/// The target dtype is an attribute rather than an operand, so the capability
/// row constrains only what this reads. What it may be asked to write is
/// constrained here: the CPU kernel refuses a quantized target, and a refusal
/// that names the operation is more use than the kernel's untyped one.
impl<D: Device> Execute<op::ToDType> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ToDType, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ToDType;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype == DTypeId::Q8_0.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        // Both type parameters are phantom here: CPU storage carries its dtype
        // in the buffer variant, and the kernel switches on the runtime value.
        <Self as TensorOps<Self>>::tensor_to_dtype::<f32, i64>(input, dtype)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Compress float storage into blocks.
///
/// The kernel's supported pair is fixed by its type parameters rather than by
/// its operand, so the concrete types are named here instead of forwarding `T`.
/// A `CpuBackendImpl<D>` executing this still compresses f32 storage into
/// `Q8_0`, because the storage carries its own dtype and `T` never described it.
impl<D: Device> Execute<op::Quantize> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Quantize, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Quantize;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        // The attribute names the representation to compress into, and `Q8_0`
        // is the only one this backend has. Refusing here rather than letting
        // the kernel refuse keeps the reason attached to the operation.
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::Q8_0.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        <Self as QuantizedOps<Self>>::quantize::<f32, Q8_0>(input)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Expand blocks back into float storage.
///
/// Lossy, and the inverse of `quantize` only up to the quantization error. That
/// is a property of the representation rather than something this layer can
/// correct, so it is stated and not compensated for.
impl<D: Device> Execute<op::Dequantize> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dequantize, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Dequantize;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        <Self as QuantizedOps<Self>>::dequantize::<Q8_0, f32>(input)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// Multiply two compressed operands without expanding either one first.
///
/// The row for it reads the quantized dtype set while the value it produces is
/// float. That asymmetry is legal because a capability row constrains operands,
/// and it is the clearest example in the registry of why the row and the output
/// rule are separate claims.
impl<D: Device> Execute<op::QuantizedMatMul> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::QuantizedMatMul, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::QuantizedMatMul;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        <Self as QuantizedOps<Self>>::quantized_matmul::<Q8_0>(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// The scalar inner product of two operands of equal shape.
///
/// Composed rather than routed to a BLAS dot, because that is what the frontend
/// does and the two must agree while both exist. `mul` broadcasts, but the
/// descriptor has already required the operands to match, so no broadcast can
/// occur here.
impl<D: Device> Execute<op::Dot> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dot, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Dot;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let product = crate::cpu::ops::elementwise::mul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
        crate::cpu::ops::reduce::sum_all(&product)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

/// The outer product of two vectors, as a matrix.
///
/// Each operand grows an axis on the side the other one occupies, and the
/// broadcast multiply fills the grid that leaves.
impl<D: Device> Execute<op::Outer> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Outer, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Outer;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);
        let column = unsqueeze_storage(lhs, 1).map_err(wrap)?;
        let row = unsqueeze_storage(rhs, 0).map_err(wrap)?;
        crate::cpu::ops::elementwise::mul_storage(&column, &row).map_err(wrap)
    }
}

/// Divide an axis into consecutive pieces.
///
/// `chunk` names how many pieces it wants and `split` names how long each one
/// should be; both answer with as many narrows as that implies, and both leave
/// a shorter final piece when the axis does not divide evenly. The two differ
/// only in how they derive the piece length from the axis, which is why the
/// walk itself is written once.
///
/// The output is a `Vec`, which the execution contract carries because `Execute`
/// names its output as an associated type rather than fixing it to one storage.
fn consecutive_pieces<D: Device>(
    backend: &CpuBackendImpl<D>,
    input: &CpuStorage,
    axis: usize,
    piece: usize,
    operation: OperationKind,
) -> Result<Vec<CpuStorage>, BackendError> {
    let Some(&extent) = input.shape.get(axis) else {
        return Err(invalid(
            operation,
            "the split axis is outside the operand rank",
        ));
    };
    if piece == 0 {
        return Err(invalid(
            operation,
            "a piece of length zero would never advance",
        ));
    }
    let _ = backend;
    let mut pieces = Vec::with_capacity(extent.div_ceil(piece));
    let mut start = 0;
    while start < extent {
        let length = (extent - start).min(piece);
        pieces.push(
            narrow_storage(input, axis, start, length)
                .map_err(|error| kernel_error(CPU_NAME, operation, error))?,
        );
        start += length;
    }
    Ok(pieces)
}

impl<D: Device> Execute<op::Chunk> for CpuBackendImpl<D> {
    type Output = Vec<CpuStorage>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Chunk, Self>,
    ) -> Result<Vec<CpuStorage>, BackendError> {
        let operation = OperationKind::Chunk;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        let Some(&extent) = input.shape.get(attributes.axis) else {
            return Err(invalid(
                operation,
                "the chunk axis is outside the operand rank",
            ));
        };
        if attributes.chunks == 0 {
            return Err(invalid(
                operation,
                "a chunk count of zero divides into nothing",
            ));
        }
        // Rounding up is what makes a request for more chunks than the axis can
        // supply produce fewer pieces than asked for rather than empty ones.
        let piece = extent.div_ceil(attributes.chunks);
        consecutive_pieces(self, input, attributes.axis, piece, operation)
    }
}

impl<D: Device> Execute<op::Split> for CpuBackendImpl<D> {
    type Output = Vec<CpuStorage>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Split, Self>,
    ) -> Result<Vec<CpuStorage>, BackendError> {
        let operation = OperationKind::Split;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        consecutive_pieces(
            self,
            input,
            attributes.axis,
            attributes.split_size,
            operation,
        )
    }
}

/// The reciprocal of the divisor a variance uses over `count` samples.
///
/// Returned as the factor rather than the divisor so the caller multiplies
/// instead of dividing, which keeps the degenerate case expressible: an
/// unbiased variance over one sample has no defined value, and the frontend has
/// always answered zero there rather than a NaN. Reproducing that exactly
/// matters more than improving on it, because the two paths must agree while
/// both exist.
fn variance_scale(count: usize, unbiased: bool) -> f64 {
    let count = count as f64;
    let divisor = if unbiased {
        if count <= 1.0 { 0.0 } else { count - 1.0 }
    } else {
        count
    };
    if divisor > 0.0 { 1.0 / divisor } else { 0.0 }
}

/// The sum of squared deviations from `mean`, and the scaling that turns it
/// into a variance.
///
/// `mean` is broadcast back against the operand, which is why it is passed in
/// rather than computed here: the all-reduced form wants a scalar mean and the
/// axis forms want a keep-dim mean, and only the caller knows which.
fn squared_deviations(
    input: &CpuStorage,
    mean: &CpuStorage,
    operation: OperationKind,
) -> Result<CpuStorage, BackendError> {
    let deviation = crate::cpu::ops::elementwise::sub_storage(input, mean)
        .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
    crate::cpu::ops::elementwise::mul_storage(&deviation, &deviation)
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
}

/// Variance and standard deviation, over everything or along one axis.
///
/// Seven identities that are one composition each: subtract the mean, square,
/// reduce, and scale. None of them has a kernel of its own on any backend, and
/// the primitives they rewrite into are already migrated, so the executor is
/// where the composition belongs rather than the frontend, which is the only
/// place it exists today.
///
/// Every step pushes its own tape entry, so the gradient is correct by
/// composition and no backward rule is written here.
macro_rules! variance_executors {
    ($(($operation:ident, $mean:ident, $reduce:ident, $finish:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let attributes = request.operation.descriptor().attributes();
                let (mean, count) = <Self as VarianceAxis<D>>::$mean(input, attributes)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let squared = squared_deviations(input, &mean, operation)?;
                let summed = <Self as VarianceAxis<D>>::$reduce(&squared, attributes)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let scaled = canonical_mul_scalar(
                    &summed,
                    variance_scale(count, attributes.unbiased),
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let finish: fn(&CpuStorage) -> incin_core::prelude::Result<CpuStorage> = $finish;
                finish(&scaled).map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

/// The half of a variance that differs between the all-reduced and axis forms.
///
/// A private helper trait rather than four more macro parameters, because the
/// two forms differ in the *type* of their attributes as well as in which
/// reduction they call, and a macro that papered over that would stop the
/// compiler from checking either.
trait VarianceAxis<D: Device> {
    fn mean_over_all(
        input: &CpuStorage,
        attributes: &VarianceAttributes,
    ) -> incin_core::prelude::Result<(CpuStorage, usize)>;
    fn sum_over_all(
        input: &CpuStorage,
        attributes: &VarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage>;
    fn mean_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<(CpuStorage, usize)>;
    fn sum_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage>;
    fn sum_along_axis_keeping_it(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage>;
}

impl<D: Device> VarianceAxis<D> for CpuBackendImpl<D> {
    fn mean_over_all(
        input: &CpuStorage,
        _: &VarianceAttributes,
    ) -> incin_core::prelude::Result<(CpuStorage, usize)> {
        let count = input.shape.iter().product::<usize>();
        Ok((crate::cpu::ops::reduce::mean_all(input)?, count))
    }

    fn sum_over_all(
        input: &CpuStorage,
        _: &VarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_all(input)
    }

    /// The mean keeps the axis so it broadcasts back against the operand, and
    /// the count is that axis' extent rather than the whole element count.
    fn mean_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<(CpuStorage, usize)> {
        let count = input.shape.get(attributes.axis).copied().unwrap_or(0);
        Ok((
            crate::cpu::ops::reduce::mean_keepdim(input, attributes.axis)?,
            count,
        ))
    }

    fn sum_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_dim(input, attributes.axis)
    }

    fn sum_along_axis_keeping_it(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::prelude::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_keepdim(input, attributes.axis)
    }
}

fn identity(storage: &CpuStorage) -> incin_core::prelude::Result<CpuStorage> {
    Ok(storage.clone())
}

fn square_root<D: Device>(storage: &CpuStorage) -> incin_core::prelude::Result<CpuStorage> {
    let _ = core::marker::PhantomData::<D>;
    canonical_sqrt(storage)
}

variance_executors![
    (VarianceAll, mean_over_all, sum_over_all, identity),
    (VarianceDim, mean_along_axis, sum_along_axis, identity),
    (
        VarianceKeepDim,
        mean_along_axis,
        sum_along_axis_keeping_it,
        identity
    ),
    (StdAll, mean_over_all, sum_over_all, square_root::<D>),
    (StdDim, mean_along_axis, sum_along_axis, square_root::<D>),
    (
        StdKeepDim,
        mean_along_axis,
        sum_along_axis_keeping_it,
        square_root::<D>
    ),
];

/// The p-norm of every element, as a scalar.
///
/// The two common orders are special-cased the way the frontend special-cases
/// them, and for the same reason: `p = 2` through the general path would raise
/// each element to the power two and the sum to the power one half, which is
/// two transcendental calls where a multiply and a square root do exactly the
/// same thing more accurately.
impl<D: Device> Execute<op::Norm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Norm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Norm;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let order = request.operation.descriptor().attributes().order;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        if (order - 1.0).abs() < NORM_ORDER_TOLERANCE {
            let magnitude = canonical_abs(input).map_err(wrap)?;
            return crate::cpu::ops::reduce::sum_all(&magnitude).map_err(wrap);
        }
        if (order - 2.0).abs() < NORM_ORDER_TOLERANCE {
            let squared = crate::cpu::ops::elementwise::mul_storage(input, input).map_err(wrap)?;
            let summed = crate::cpu::ops::reduce::sum_all(&squared).map_err(wrap)?;
            return canonical_sqrt(&summed).map_err(wrap);
        }
        let magnitude = canonical_abs(input).map_err(wrap)?;
        let raised = canonical_powf(&magnitude, order).map_err(wrap)?;
        let summed = crate::cpu::ops::reduce::sum_all(&raised).map_err(wrap)?;
        canonical_powf(&summed, 1.0 / order).map_err(wrap)
    }
}

/// How close an order has to be to one or two to take the exact path.
///
/// Copied from `Tensor::norm` rather than chosen here. The two paths must agree
/// on which order they take while both exist, and a tolerance that differed
/// would make them disagree only for orders in the gap between them, which is
/// the hardest kind of divergence to find.
const NORM_ORDER_TOLERANCE: f64 = 1e-6;

/// The losses `LossOps` supplies as composed defaults.
///
/// Each takes a prediction and a target of the same shape and reduces the
/// elementwise result according to its attributes. They are grouped here rather
/// than written out because the only thing that differs between them is the
/// method name; the operand binding and the reduction mapping are identical,
/// and three copies of that would be three places for it to drift.
macro_rules! loss_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let (prediction, target) = binary_operands(self, request.inputs, operation, training_mode(request.context))?;
                let reduction = loss_reduction(
                    request.operation.descriptor().attributes().reduction,
                );
                <Self as LossOps<Self>>::$method::<f32>(prediction, target, reduction)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

/// Translate the descriptor's reduction mode into the loss trait's.
///
/// Two enums for one concept, because the catalog's vocabulary is deliberately
/// independent of `nn`. The translation is total and exhaustive, so adding a
/// mode to either enum is a compile error here rather than a silent default.
fn loss_reduction(reduction: LossReduction) -> Reduction {
    match reduction {
        LossReduction::None => Reduction::None,
        LossReduction::Mean => Reduction::Mean,
        LossReduction::Sum => Reduction::Sum,
    }
}

loss_executors![
    (MseLoss, mse_loss),
    (L1Loss, l1_loss),
    (BceWithLogitsLoss, bce_with_logits_loss),
];

/// Negative log likelihood over logits addressed by integer class targets.
///
/// Not a member of `loss_executors!` above, for the reason it is not a member
/// of the `composed_reduction` capability group either: those take a
/// prediction and a target of the *same* shape and dtype, and this takes
/// `[batch, classes]` f32 logits against `[batch]` integer class indices.
/// `LossOps::cross_entropy_loss` is correspondingly the one loss with two
/// dtype parameters rather than one, so the shared macro's
/// `$method::<f32>(prediction, target, reduction)` call could not name it.
///
/// The dtype split is enforced the same way `embedding`'s is: the row carries
/// `INDEX_AND_F32_DTYPES` (the union both operands fall inside, which is all
/// one row can state), the descriptor's per-operand contract has already
/// refused a non-integer target or a swapped pair, and `f32_only` enforces
/// the logits' real constraint here. The target is deliberately *not* passed
/// to `f32_only`: an integer class index is exactly what it should be.
impl<D: Device> Execute<op::CrossEntropyLoss> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::CrossEntropyLoss, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::CrossEntropyLoss;
        let (logits, target) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        f32_only(operation, &[Some(logits)])?;
        let reduction = loss_reduction(request.operation.descriptor().attributes().reduction);
        <Self as LossOps<Self>>::cross_entropy_loss::<f32, i64>(logits, target, reduction)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
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
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, CpuBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::prelude::Cpu>();
        };
    };
}

crate::capability::cpu_descriptor_operations!(assert_every_advertised_row_executes,);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::CpuBuffer;
    use incin_core::exec::GradMode;
    use incin_core::exec::catalog::{
        ArangeAttributes, AxisAttributes, BatchNormAttributes, ChunkAttributes, Conv1dAttributes,
        Conv2dAttributes, CreationAttributes, DTypeAttributes, DropoutAttributes,
        EpsilonAttributes, LayerNormAttributes, LinearAttributes, LinspaceAttributes,
        LossAttributes, NoAttributes, NormAttributes, QuantizationAttributes, ShapeAttributes,
        SplitAttributes,
    };
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch};
    use incin_core::prelude::{Cpu, DeviceId, Local};

    type TestBackend = CpuBackendImpl<Cpu>;

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

    /// A context the quantization rows will answer.
    ///
    /// Those rows carry `training = false`, because neither block kernel pushes
    /// a tape entry and a training row would promise a gradient that never
    /// arrives. `dispatch::execute` reads the grad mode off the context's own
    /// policy rather than off an ambient gradient scope, so the mode has to be
    /// set on the context.
    fn inference_context() -> ExecutionContext<TestBackend> {
        ExecutionContext::new(TestBackend::new()).with_grad_mode(GradMode::Disabled)
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
        use incin_core::__backend_compat::legacy::{NumericOps, ReductionOps};

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

    /// A training batch norm normalizes by the batch's own statistics.
    ///
    /// It used to be refused, because the only kernel behind it bound its
    /// momentum to `_momentum` and computed no batch statistics, so a training
    /// request came back as a correctly shaped tensor of finite values that
    /// had silently used the running ones. Refusing was right while that was
    /// the only option; a separate training kernel is better than either.
    ///
    /// The statistics are per channel, so for `[2, 3]` each of the three
    /// columns normalizes over its own two rows. A two-element population
    /// normalizes to exactly -1 and +1 whatever the two values are, which is
    /// what makes the expected result independent of the input here.
    #[test]
    fn a_training_batch_norm_normalizes_by_the_batch_statistics() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let normalized = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(true, 1e-5),
            &[handle(&input)],
        )
        .expect("a training batch norm computes batch statistics");

        assert_eq!(normalized.shape.to_vec(), vec![2, 3]);
        for row in 0..2 {
            for column in 0..3 {
                let value = normalized.get(&[row, column]);
                let expected = if row == 0 { -1.0 } else { 1.0 };
                assert!(
                    (value - expected).abs() < 1e-3,
                    "[{row}, {column}] was {value}, expected {expected}"
                );
            }
        }
    }

    /// Training and inference are different answers, not the same one reached
    /// twice. Without this, a training mode that quietly fell through to the
    /// inference kernel would pass the test above whenever the running
    /// statistics happened to match the batch ones.
    #[test]
    fn training_and_inference_batch_norm_disagree() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
        let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

        let training = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(true, 1e-5),
            &[handle(&input)],
        )
        .unwrap();
        let inference = dispatch::execute::<op::BatchNorm, _>(
            &context,
            batch_norm_attributes(false, 1e-5),
            &[
                handle(&input),
                handle(&running_mean),
                handle(&running_variance),
            ],
        )
        .unwrap();

        assert!(
            (training.get(&[0, 0]) - inference.get(&[0, 0])).abs() > 1e-3,
            "training {} and inference {} agree",
            training.get(&[0, 0]),
            inference.get(&[0, 0])
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

    /// Each loss reduces to the value its own formula defines.
    ///
    /// Hand-computed rather than compared against the trait method, because
    /// comparing an executor to the function it calls proves only that the call
    /// happened. These check that the descriptor's reduction mode reached the
    /// kernel and that the right kernel was picked: mse and l1 differ on this
    /// input, so a swapped pair of arms in the macro would show up here.
    #[test]
    fn each_canonical_loss_computes_its_own_formula() {
        let context = context();
        let prediction = storage(&[1.0, 3.0], &[2]);
        let target = storage(&[0.0, 0.0], &[2]);

        // Squared error: (1 + 9) / 2 = 5. Absolute error: (1 + 3) / 2 = 2.
        for (expected, run) in [
            (
                5.0,
                dispatch::execute::<op::MseLoss, _>(
                    &context,
                    LossAttributes {
                        reduction: LossReduction::Mean,
                    },
                    &[handle(&prediction), handle(&target)],
                ),
            ),
            (
                2.0,
                dispatch::execute::<op::L1Loss, _>(
                    &context,
                    LossAttributes {
                        reduction: LossReduction::Mean,
                    },
                    &[handle(&prediction), handle(&target)],
                ),
            ),
        ] {
            let output = run.expect("the loss executes");
            assert!(
                output.shape.is_empty(),
                "a mean reduction produces a scalar"
            );
            assert!(
                (output.get(&[]) - expected).abs() < 1e-6,
                "expected {expected}, got {}",
                output.get(&[])
            );
        }
    }

    /// The reduction mode is carried, not defaulted.
    ///
    /// `LossReduction::None` is the mode that changes the result's shape, so it
    /// is the one that fails visibly if the attribute were dropped and the
    /// trait's own default used instead.
    #[test]
    fn a_loss_with_no_reduction_keeps_the_elementwise_shape() {
        let context = context();
        let prediction = storage(&[1.0, 3.0], &[2]);
        let target = storage(&[0.0, 0.0], &[2]);

        let output = dispatch::execute::<op::MseLoss, _>(
            &context,
            LossAttributes {
                reduction: LossReduction::None,
            },
            &[handle(&prediction), handle(&target)],
        )
        .expect("an unreduced loss executes");
        assert_eq!(output.shape.to_vec(), vec![2]);
        assert_eq!(output.get(&[0]), 1.0);
        assert_eq!(output.get(&[1]), 9.0);
    }

    /// An allocation is capability-checked even though it has no operand.
    ///
    /// This is the assertion the whole zero-operand branch in
    /// `dispatch::execute` exists for. `rand` is advertised for the float
    /// dtypes only, and a request for an integer one has to be refused by the
    /// registry rather than reaching the kernel, which would happily produce
    /// something. Before that branch existed the per-operand loop ran zero
    /// times here and nothing was checked at all.
    #[test]
    fn an_allocation_is_refused_by_a_row_it_does_not_match() {
        let context = inference_context();

        let error = dispatch::execute::<op::UniformRandom, _>(
            &context,
            CreationAttributes {
                shape: vec![2, 2],
                dtype: DTypeId::I64.descriptor(),
                device: DeviceId::cpu(),
            },
            &[],
        )
        .expect_err("a uniform draw over an integer dtype is not advertised");
        let message = format!("{error}");
        assert!(
            message.to_lowercase().contains("dtype") || message.contains("I64"),
            "the refusal must name the dtype the row does not carry: {message}"
        );

        // The same call over a dtype the row does carry still runs, so the test
        // above is about the dtype rather than about allocation being broken.
        dispatch::execute::<op::UniformRandom, _>(
            &context,
            CreationAttributes {
                shape: vec![2, 2],
                dtype: DTypeId::F32.descriptor(),
                device: DeviceId::cpu(),
            },
            &[],
        )
        .expect("a uniform draw over f32 is advertised");
    }

    /// The ranged fills produce the values their attributes describe.
    ///
    /// `arange` and `linspace` differ only in whether the second parameter is a
    /// step or an endpoint, which makes them the pair most easily confused in a
    /// macro that passes both positionally.
    #[test]
    fn the_ranged_fills_read_their_parameters_in_the_right_order() {
        let context = inference_context();
        let shape = vec![4];
        let device = DeviceId::cpu();

        let stepped = dispatch::execute::<op::Arange, _>(
            &context,
            ArangeAttributes {
                shape: shape.clone(),
                dtype: DTypeId::F32.descriptor(),
                device,
                start: 10.0,
                step: 2.0,
            },
            &[],
        )
        .expect("arange executes");
        for (index, expected) in [10.0, 12.0, 14.0, 16.0].into_iter().enumerate() {
            assert_eq!(stepped.get(&[index]), expected);
        }

        let spaced = dispatch::execute::<op::Linspace, _>(
            &context,
            LinspaceAttributes {
                shape,
                dtype: DTypeId::F32.descriptor(),
                device,
                start: 0.0,
                end: 3.0,
            },
            &[],
        )
        .expect("linspace executes");
        for (index, expected) in [0.0, 1.0, 2.0, 3.0].into_iter().enumerate() {
            assert!(
                (spaced.get(&[index]) - expected).abs() < 1e-6,
                "element {index}: expected {expected}, got {}",
                spaced.get(&[index])
            );
        }
    }

    /// An allocation refuses an operand rather than ignoring it.
    #[test]
    fn an_allocation_given_an_operand_is_refused() {
        let context = inference_context();
        let stray = storage(&[1.0], &[1]);

        let error = dispatch::execute::<op::Zeros, _>(
            &context,
            CreationAttributes {
                shape: vec![2],
                dtype: DTypeId::F32.descriptor(),
                device: DeviceId::cpu(),
            },
            &[handle(&stray)],
        )
        .expect_err("zeros reads nothing, so an operand is a malformed request");
        assert!(format!("{error}").contains("zeros"));
    }

    /// Dropout keeps roughly the share it promises and scales what survives.
    ///
    /// The result is random, so the assertions are about the distribution and
    /// about the two exact cases. Every surviving element must be the input
    /// scaled by `1 / (1 - p)` and every other must be zero: that is checkable
    /// exactly even though which elements survive is not, and it is what
    /// catches a missing or misapplied compensating scale.
    #[test]
    fn dropout_zeroes_some_elements_and_scales_the_rest_by_the_keep_reciprocal() {
        let context = context();
        let input = storage(&[1.0; 4096], &[4096]);

        let output = dispatch::execute::<op::Dropout, _>(
            &context,
            DropoutAttributes {
                probability: 0.5,
                training: true,
            },
            &[handle(&input)],
        )
        .expect("dropout executes");
        assert_eq!(output.shape.to_vec(), vec![4096]);

        let mut kept = 0;
        for index in 0..4096 {
            let value = output.get(&[index]);
            if value == 0.0 {
                continue;
            }
            kept += 1;
            assert!(
                (value - 2.0).abs() < 1e-6,
                "a surviving element was {value}, not the input scaled by 1 / (1 - p)"
            );
        }
        // Four standard deviations of a fair coin over 4096 draws is about 128,
        // so this fails on a genuinely wrong rate and effectively never on an
        // unlucky draw.
        assert!(
            (2048 - 128..=2048 + 128).contains(&kept),
            "kept {kept} of 4096 elements, which is not a half-probability drop"
        );
    }

    /// Dropout outside training is the identity, exactly.
    ///
    /// Not approximately, and not a scaled copy: an inference pass that
    /// perturbed the values would move every evaluation metric, and one that
    /// merely consumed a random draw would shift every later training step
    /// without changing anything visible here.
    #[test]
    fn dropout_outside_training_returns_the_operand_unchanged() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0], &[4]);

        for (probability, training) in [(0.5, false), (0.0, true)] {
            let output = dispatch::execute::<op::Dropout, _>(
                &context,
                DropoutAttributes {
                    probability,
                    training,
                },
                &[handle(&input)],
            )
            .expect("dropout executes");
            for index in 0..4 {
                assert_eq!(
                    output.get(&[index]),
                    input.get(&[index]),
                    "p={probability} training={training} changed element {index}"
                );
            }
        }
    }

    /// A linear layer transposes its weight before the product.
    ///
    /// The weight is stored as `[out, in]`, so a rectangular one is the only
    /// shape that catches a missing transpose: with a square weight the product
    /// would still have the right shape and only the values would be wrong,
    /// which is the failure this asserts against.
    #[test]
    fn a_linear_layer_transposes_its_weight_and_adds_its_bias() {
        let context = context();
        // One sample of three features, into two outputs.
        let input = storage(&[1.0, 2.0, 3.0], &[1, 3]);
        let weight = storage(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let bias = storage(&[10.0, 20.0], &[2]);

        let plain = dispatch::execute::<op::Linear, _>(
            &context,
            LinearAttributes { has_bias: false },
            &[handle(&input), handle(&weight)],
        )
        .expect("linear executes");
        assert_eq!(plain.shape.to_vec(), vec![1, 2]);
        // Row zero of the weight selects feature zero, row one selects feature
        // one.
        assert_eq!(plain.get(&[0, 0]), 1.0);
        assert_eq!(plain.get(&[0, 1]), 2.0);

        let biased = dispatch::execute::<op::Linear, _>(
            &context,
            LinearAttributes { has_bias: true },
            &[handle(&input), handle(&weight), handle(&bias)],
        )
        .expect("a biased linear executes");
        assert_eq!(biased.get(&[0, 0]), 11.0);
        assert_eq!(biased.get(&[0, 1]), 22.0);
    }

    /// RMS norm divides by the root mean square and subtracts no mean.
    ///
    /// The input is deliberately not centred: its mean is two, so an
    /// implementation that subtracted one would give a visibly different answer
    /// rather than a slightly different one. Root mean square of [1, 2, 3] is
    /// sqrt(14/3), so the first element lands at 1/sqrt(14/3).
    #[test]
    fn rms_norm_scales_by_the_root_mean_square_without_centring() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0], &[1, 3]);
        let weight = storage(&[1.0, 1.0, 1.0], &[3]);

        let output = dispatch::execute::<op::RmsNorm, _>(
            &context,
            EpsilonAttributes { epsilon: 0.0 },
            &[handle(&input), handle(&weight)],
        )
        .expect("rms_norm executes");
        assert_eq!(output.shape.to_vec(), vec![1, 3]);

        let root_mean_square = (14.0f64 / 3.0).sqrt();
        for (index, original) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            let expected: f64 = original / root_mean_square;
            let actual = output.get(&[0, index]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "element {index}: expected {expected}, got {actual}"
            );
        }
    }

    /// Compression and expansion round-trip within the error the block format
    /// allows.
    ///
    /// The tolerance is derived, not tuned: a Q8_0 block scales by the largest
    /// magnitude it holds divided by 127, so every element lands within half of
    /// that scale of its original value. Asserting a tighter bound would be
    /// asserting something about this particular input rather than about the
    /// format.
    #[test]
    fn quantization_round_trips_within_the_block_format_error() {
        let context = inference_context();
        let values: Vec<f32> = (0..32).map(|index| index as f32 - 8.0).collect();
        let input = storage(&values, &[32]);

        let blocks = dispatch::execute::<op::Quantize, _>(
            &context,
            QuantizationAttributes {
                dtype: DTypeId::Q8_0.descriptor(),
            },
            &[handle(&input)],
        )
        .expect("quantize executes");
        let restored = dispatch::execute::<op::Dequantize, _>(
            &context,
            QuantizationAttributes {
                dtype: DTypeId::F32.descriptor(),
            },
            &[handle(&blocks)],
        )
        .expect("dequantize executes");
        assert_eq!(blocks.dtype, DTypeId::Q8_0.descriptor());
        assert_eq!(restored.dtype, DTypeId::F32.descriptor());
        assert_eq!(restored.shape.to_vec(), vec![32]);

        let largest = values
            .iter()
            .fold(0.0f32, |seen, &value| seen.max(value.abs()));
        let tolerance = f64::from(largest / 127.0 / 2.0) + 1e-6;
        for (index, &original) in values.iter().enumerate() {
            let difference = (restored.get(&[index]) - f64::from(original)).abs();
            assert!(
                difference <= tolerance,
                "element {index} moved by {difference}, more than the {tolerance} the \
                 block scale allows"
            );
        }
    }

    /// A compression target the backend does not have is refused by name.
    ///
    /// The attribute names a representation and the row constrains the operand,
    /// so nothing before the executor is in a position to compare the two.
    #[test]
    fn a_compression_into_an_unsupported_representation_is_refused() {
        let context = inference_context();
        let input = storage(&[0.0; 32], &[32]);

        let error = dispatch::execute::<op::Quantize, _>(
            &context,
            QuantizationAttributes {
                dtype: DTypeId::F16.descriptor(),
            },
            &[handle(&input)],
        )
        .expect_err("f16 is not a quantized representation this backend produces");
        let message = format!("{error}");
        assert!(
            message.contains("quantize"),
            "the refusal must name the operation: {message}"
        );
    }

    /// A dot product contracts, an outer product expands.
    ///
    /// The two are checked together because they are the pair most easily
    /// swapped: both take two vectors and multiply them, and only the shape of
    /// the answer says which one ran.
    #[test]
    fn the_dot_and_outer_products_contract_and_expand() {
        let context = context();
        let lhs = storage(&[1.0, 2.0], &[2]);
        let rhs = storage(&[3.0, 4.0], &[2]);

        let inner =
            dispatch::execute::<op::Dot, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
                .expect("dot executes");
        assert!(inner.shape.is_empty(), "a dot product is a scalar");
        assert_eq!(inner.get(&[]), 11.0);

        let grid = dispatch::execute::<op::Outer, _>(
            &context,
            NoAttributes,
            &[handle(&lhs), handle(&rhs)],
        )
        .expect("outer executes");
        assert_eq!(grid.shape.to_vec(), vec![2, 2]);
        for (row, left) in [1.0, 2.0].into_iter().enumerate() {
            for (column, right) in [3.0, 4.0].into_iter().enumerate() {
                assert_eq!(grid.get(&[row, column]), left * right);
            }
        }
    }

    /// An axis that does not divide evenly leaves a shorter final piece.
    ///
    /// This is the behaviour the frontend has, and the case a naive
    /// implementation gets wrong by emitting a full-length final piece that
    /// reads past the axis or an empty one that reads nothing. Five elements
    /// into two chunks is three then two, and into pieces of two is two, two,
    /// one.
    #[test]
    fn an_uneven_axis_leaves_a_shorter_final_piece() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);

        let chunks = dispatch::execute::<op::Chunk, _>(
            &context,
            ChunkAttributes { chunks: 2, axis: 0 },
            &[handle(&input)],
        )
        .expect("chunk executes");
        assert_eq!(
            chunks
                .iter()
                .map(|piece| piece.shape.to_vec())
                .collect::<Vec<_>>(),
            vec![vec![3], vec![2]]
        );
        assert_eq!(chunks[1].get(&[0]), 4.0);

        let pieces = dispatch::execute::<op::Split, _>(
            &context,
            SplitAttributes {
                split_size: 2,
                axis: 0,
            },
            &[handle(&input)],
        )
        .expect("split executes");
        assert_eq!(
            pieces
                .iter()
                .map(|piece| piece.shape.to_vec())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![2], vec![1]]
        );
        assert_eq!(pieces[2].get(&[0]), 5.0);
    }

    /// Asking for more chunks than the axis can supply yields fewer, not empty
    /// ones.
    ///
    /// Rounding the piece length up is what produces that, and it is the only
    /// place the two split operations differ, so it is asserted rather than
    /// left to follow from the arithmetic.
    #[test]
    fn chunking_beyond_the_axis_extent_produces_fewer_pieces_not_empty_ones() {
        let context = context();
        let input = storage(&[1.0, 2.0], &[2]);

        let chunks = dispatch::execute::<op::Chunk, _>(
            &context,
            ChunkAttributes { chunks: 5, axis: 0 },
            &[handle(&input)],
        )
        .expect("chunk executes");
        assert_eq!(chunks.len(), 2, "a two-wide axis has at most two pieces");
        assert!(chunks.iter().all(|piece| piece.shape.to_vec() == vec![1]));
    }

    /// The variance family computes the value its estimator defines.
    ///
    /// Hand-computed against [1, 2, 3, 4]: the mean is 2.5, the squared
    /// deviations sum to 5, so the biased variance is 1.25 and the unbiased one
    /// is 5/3. Checking both settings is the point: `unbiased` is the only
    /// attribute these carry, and an executor that ignored it would still pass
    /// a test that only ever asked for one.
    #[test]
    fn the_variance_estimators_differ_by_their_correction() {
        let context = context();
        let input = storage(&[1.0, 2.0, 3.0, 4.0], &[4]);

        for (unbiased, expected) in [(false, 1.25), (true, 5.0 / 3.0)] {
            let output = dispatch::execute::<op::VarianceAll, _>(
                &context,
                VarianceAttributes { unbiased },
                &[handle(&input)],
            )
            .expect("var_all executes");
            assert!(
                (output.get(&[]) - expected).abs() < 1e-6,
                "unbiased={unbiased}: expected {expected}, got {}",
                output.get(&[])
            );
        }
    }

    /// The standard deviation is the square root of the variance, and the axis
    /// forms differ from each other only in whether the axis survives.
    #[test]
    fn the_axis_variance_forms_reduce_the_axis_they_name() {
        let context = context();
        // Two rows of [1, 2, 3]: each has a biased variance of 2/3.
        let input = storage(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let attributes = AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        };
        let expected = 2.0 / 3.0;

        let reduced = dispatch::execute::<op::VarianceDim, _>(
            &context,
            attributes.clone(),
            &[handle(&input)],
        )
        .expect("var_dim executes");
        assert_eq!(reduced.shape.to_vec(), vec![2]);
        assert!((reduced.get(&[0]) - expected).abs() < 1e-6);

        let kept = dispatch::execute::<op::VarianceKeepDim, _>(
            &context,
            attributes.clone(),
            &[handle(&input)],
        )
        .expect("var_keepdim executes");
        assert_eq!(kept.shape.to_vec(), vec![2, 1]);
        assert!((kept.get(&[0, 0]) - expected).abs() < 1e-6);

        let deviation = dispatch::execute::<op::StdDim, _>(&context, attributes, &[handle(&input)])
            .expect("std_dim executes");
        assert_eq!(deviation.shape.to_vec(), vec![2]);
        assert!((deviation.get(&[0]) - expected.sqrt()).abs() < 1e-6);
    }

    /// Each norm order takes its own path and they agree where they meet.
    ///
    /// The executor special-cases orders one and two. Two is the interesting
    /// one: the fast path multiplies and takes a square root while the general
    /// path raises to a power twice, so a mistake in either shows up as a
    /// disagreement at exactly this order.
    #[test]
    fn the_norm_orders_agree_where_the_fast_paths_meet_the_general_one() {
        let context = context();
        let input = storage(&[3.0, -4.0], &[2]);

        let l1 = dispatch::execute::<op::Norm, _>(
            &context,
            NormAttributes { order: 1.0 },
            &[handle(&input)],
        )
        .expect("the l1 norm executes");
        assert!((l1.get(&[]) - 7.0).abs() < 1e-6, "got {}", l1.get(&[]));

        let l2 = dispatch::execute::<op::Norm, _>(
            &context,
            NormAttributes { order: 2.0 },
            &[handle(&input)],
        )
        .expect("the l2 norm executes");
        assert!((l2.get(&[]) - 5.0).abs() < 1e-6, "got {}", l2.get(&[]));

        // Just outside the tolerance, so this takes the general path and must
        // still land next to the exact answer.
        let near = dispatch::execute::<op::Norm, _>(
            &context,
            NormAttributes { order: 2.001 },
            &[handle(&input)],
        )
        .expect("a general-order norm executes");
        assert!(
            (near.get(&[]) - 5.0).abs() < 1e-2,
            "the general path diverged from the fast one: {}",
            near.get(&[])
        );
    }

    /// A conversion to a quantized dtype is refused by the executor.
    ///
    /// The capability row constrains what `to_dtype` reads, and the target is
    /// an attribute rather than an operand, so nothing before the executor is
    /// in a position to check it. The kernel does refuse, but with an untyped
    /// error that does not name the operation.
    #[test]
    fn a_conversion_to_a_quantized_dtype_is_refused_by_name() {
        let context = context();
        let input = storage(&[1.0, 2.0], &[2]);

        let error = dispatch::execute::<op::ToDType, _>(
            &context,
            DTypeAttributes {
                dtype: DTypeId::Q8_0.descriptor(),
            },
            &[handle(&input)],
        )
        .expect_err("the CPU conversion kernel has no quantized target");
        let message = format!("{error}");
        assert!(
            message.contains("to_dtype"),
            "the refusal must name the operation: {message}"
        );
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
