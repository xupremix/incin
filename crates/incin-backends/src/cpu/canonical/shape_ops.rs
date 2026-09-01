//! Shape manipulation, slicing, concatenation, and layout view executors for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::error::BackendError;
use incin_core::exec::catalog::{DuplicateIndexRule, op};
use incin_core::exec::{TensorHandle, UnsupportedReason};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DTypeId;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{admitted, operand, reduction_operand, training_mode};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::ops::shape_ops::{
    broadcast_left_storage, concat_storage, diag_storage, flatten_storage, gather_storage,
    index_select_storage, lerp_storage, masked_fill_storage, narrow_storage, pad_storage,
    pixel_shuffle_storage, repeat_storage, scatter_add_storage, scatter_storage, slice_storage,
    squeeze_storage, stack_storage, tensor_to_dtype_storage, transpose_storage, tril_storage,
    triu_storage, unfold_storage, unsqueeze_storage, where_storage,
};
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

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

fn variadic_operands<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
) -> Result<alloc::vec::Vec<&'a CpuStorage>, BackendError> {
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
        where_storage(mask, on_true, on_false)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        lerp_storage(start, end, weight).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        concat_storage(&operands, axis).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        stack_storage(&operands, axis).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        slice_storage(input, ranges).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
                $method(input, axis, index)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

indexing_executors![
    (Gather, gather_storage),
    (IndexSelect, index_select_storage)
];

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
        scatter_storage(input, attributes.axis, index, source)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::ScatterAdd> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ScatterAdd, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::ScatterAdd;
        let [input, index, source] = ternary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        // The mirror of the guard on `Scatter` above. There the backend refuses
        // the one rule it does not implement; here it refuses every rule but
        // the one it does, because summing is the operation rather than a mode
        // of it, and a caller who asked for last-write-wins on this descriptor
        // has asked for `scatter` and should be told so rather than quietly
        // handed a different answer.
        if attributes.duplicate_indices != DuplicateIndexRule::Accumulate {
            return Err(invalid(
                operation,
                "scatter_add accumulates duplicate indices and implements no other rule; \
                 use scatter for last-write-wins",
            ));
        }
        scatter_add_storage(input, attributes.axis, index, source)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        repeat_storage(input, repeats).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        pad_storage(input, &attributes.padding, attributes.value)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        unfold_storage(input, attributes.axis, attributes.size, attributes.step)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        pixel_shuffle_storage(input, factor)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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

fn consecutive_pieces<D: Device>(
    backend: &CpuBackendImpl<D>,
    input: &CpuStorage,
    axis: usize,
    piece: usize,
    operation: OperationKind,
) -> Result<alloc::vec::Vec<CpuStorage>, BackendError> {
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
    let mut pieces = alloc::vec::Vec::with_capacity(extent.div_ceil(piece));
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
    type Output = alloc::vec::Vec<CpuStorage>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Chunk, Self>,
    ) -> Result<alloc::vec::Vec<CpuStorage>, BackendError> {
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
        let piece = extent.div_ceil(attributes.chunks);
        consecutive_pieces(self, input, attributes.axis, piece, operation)
    }
}

impl<D: Device> Execute<op::Split> for CpuBackendImpl<D> {
    type Output = alloc::vec::Vec<CpuStorage>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Split, Self>,
    ) -> Result<alloc::vec::Vec<CpuStorage>, BackendError> {
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
        tensor_to_dtype_storage(input, dtype)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}
