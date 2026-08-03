//! Descriptor execution for the Metal backend.

use incin_core::backend_authoring::{
    Execute, ExecutionRequest, ReductionOps, StorageBackend, TensorOps,
};
use incin_core::exec::{
    Capabilities, CapabilityQuery, MatMulSpec, MathMode, ReduceOp, ReductionSpec, ReshapeSpec,
    SupportLevel, TensorMeta,
};
use incin_core::prelude::{BackendError, DType, DTypeId, Device, DeviceKind, OperationKind};

use super::backend::MetalBackendImpl;
use super::storage::MetalStorage;
use crate::descriptor_bind::{
    check_reduction_operand, invalid, kernel_error, reduce_axis_run, reduction_run,
};

impl<T: DType, D: Device> StorageBackend for MetalBackendImpl<T, D> {
    type Storage<K: DType> = MetalStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<T: DType, D: Device> Capabilities for MetalBackendImpl<T, D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Metal, query)
    }
}

fn supports_operand(
    shape: &[usize],
    expected_rank: u8,
    rows: usize,
    columns: usize,
    transposed: bool,
    batch: &[usize],
    batch_strides: &[usize],
) -> bool {
    if shape.len() != usize::from(expected_rank)
        || shape.len() < 2
        || batch_strides.len() != batch.len()
    {
        return false;
    }

    let matrix = &shape[shape.len() - 2..];
    let expected_matrix = if transposed {
        [columns, rows]
    } else {
        [rows, columns]
    };
    if matrix != expected_matrix {
        return false;
    }

    let operand_batch = &shape[..shape.len() - 2];
    if operand_batch.len() > batch.len() {
        return false;
    }
    let offset = batch.len() - operand_batch.len();
    batch.iter().enumerate().all(|(axis, &output_dim)| {
        let input_dim = axis
            .checked_sub(offset)
            .map_or(1, |input_axis| operand_batch[input_axis]);
        if batch_strides[axis] == 0 {
            input_dim == 1
        } else {
            input_dim == output_dim
        }
    })
}

fn bind_matmul<'a, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, MatMulSpec, MetalBackendImpl<T, D>>,
) -> Result<(&'a MetalStorage, &'a MetalStorage), BackendError> {
    let [lhs_handle, rhs_handle] = request.inputs else {
        return Err(invalid(
            OperationKind::MatMul,
            "matmul expects exactly two tensor inputs",
        ));
    };
    let lhs = lhs_handle
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not Metal storage"))?;
    let rhs = rhs_handle
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not Metal storage"))?;
    let spec = request.operation.descriptor();

    for metadata in [lhs.metadata(), rhs.metadata()] {
        if metadata.device().kind() != DeviceKind::Metal {
            return Err(invalid(
                OperationKind::MatMul,
                "matmul inputs must use Metal device",
            ));
        }
        if metadata.dtype() != DTypeId::F32 {
            return Err(incin_core::exec::UnsupportedReason::DType {
                operation: OperationKind::MatMul,
                dtype: metadata.dtype(),
            }
            .into());
        }
        let query = CapabilityQuery {
            operation: OperationKind::MatMul,
            dtype: metadata.dtype(),
            layout: metadata.layout(),
            rank: metadata.shape().rank(),
            training: true,
            math_mode: MathMode::Precise,
        };
        if let SupportLevel::Unsupported(reason) = request.context.backend().support(&query) {
            return Err(reason.into());
        }
    }

    if !supports_operand(
        lhs.shape(),
        spec.lhs_rank,
        spec.m,
        spec.k,
        spec.transpose_lhs,
        spec.batch.dims(),
        spec.lhs_batch_strides.strides(),
    ) {
        return Err(invalid(
            OperationKind::MatMul,
            "matmul lhs metadata does not match the validated descriptor",
        ));
    }
    if !supports_operand(
        rhs.shape(),
        spec.rhs_rank,
        spec.k,
        spec.n,
        spec.transpose_rhs,
        spec.batch.dims(),
        spec.rhs_batch_strides.strides(),
    ) {
        return Err(invalid(
            OperationKind::MatMul,
            "matmul rhs metadata does not match the validated descriptor",
        ));
    }

    Ok((lhs, rhs))
}

fn bind_reshape<'a, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, ReshapeSpec, MetalBackendImpl<T, D>>,
) -> Result<&'a MetalStorage, BackendError> {
    let [handle] = request.inputs else {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape expects exactly one tensor input",
        ));
    };
    let input = handle
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| invalid(OperationKind::Reshape, "reshape input is not Metal storage"))?;
    let spec = request.operation.descriptor();
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Metal {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape input must use Metal device",
        ));
    }
    let query = CapabilityQuery {
        operation: OperationKind::Reshape,
        dtype: metadata.dtype(),
        layout: metadata.layout(),
        rank: metadata.shape().rank(),
        training: false,
        math_mode: MathMode::Precise,
    };
    if let SupportLevel::Unsupported(reason) = request.context.backend().support(&query) {
        return Err(reason.into());
    }

    if metadata.shape().dims() != spec.input.dims() {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape input metadata does not match the validated descriptor",
        ));
    }

    Ok(input)
}

impl<T: DType, D: Device> Execute<ReshapeSpec> for MetalBackendImpl<T, D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReshapeSpec, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let input = bind_reshape(&request)?;

        let output = <Self as TensorOps<Self>>::reshape::<f32>(input, spec.output.dims())
            .map_err(|error| kernel_error(OperationKind::Reshape, error))?;

        if output.shape() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reshape,
                message: "Metal reshape output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<T: DType, D: Device> Execute<MatMulSpec> for MetalBackendImpl<T, D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, MatMulSpec, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let (lhs, rhs) = bind_matmul(&request)?;

        let execution_error = |error: incin_core::prelude::Error| BackendError::Execution {
            operation: OperationKind::MatMul,
            message: error.to_string(),
        };
        let lhs_transposed = if spec.transpose_lhs {
            let rank = lhs.shape().len();
            Some(
                <Self as TensorOps<Self>>::transpose::<f32>(lhs, rank - 2, rank - 1)
                    .map_err(execution_error)?,
            )
        } else {
            None
        };
        let rhs_transposed = if spec.transpose_rhs {
            let rank = rhs.shape().len();
            Some(
                <Self as TensorOps<Self>>::transpose::<f32>(rhs, rank - 2, rank - 1)
                    .map_err(execution_error)?,
            )
        } else {
            None
        };
        let lhs = lhs_transposed.as_ref().unwrap_or(lhs);
        let rhs = rhs_transposed.as_ref().unwrap_or(rhs);

        let output = <Self as TensorOps<Self>>::matmul::<f32>(lhs, rhs).map_err(execution_error)?;

        if output.shape() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::MatMul,
                message: "Metal matmul output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

fn bind_single_operand<'a, O, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, O, MetalBackendImpl<T, D>>,
    operation: OperationKind,
    wrong_arity: &'static str,
    wrong_device: &'static str,
) -> Result<&'a MetalStorage, BackendError>
where
    O: incin_core::exec::OperationSpec,
{
    let [handle] = request.inputs else {
        return Err(invalid(operation, wrong_arity));
    };
    let input = handle
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Metal {
        return Err(invalid(operation, wrong_device));
    }
    let query = CapabilityQuery {
        operation,
        dtype: metadata.dtype(),
        layout: metadata.layout(),
        rank: metadata.shape().rank(),
        training: false,
        math_mode: MathMode::Precise,
    };
    if let SupportLevel::Unsupported(reason) = request.context.backend().support(&query) {
        return Err(reason.into());
    }

    Ok(input)
}

impl<T: DType, D: Device> Execute<ReductionSpec> for MetalBackendImpl<T, D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReductionSpec, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let run = reduction_run(spec)?;
        let input = bind_single_operand(
            &request,
            OperationKind::Reduction,
            "a reduction expects exactly one tensor input",
            "reduction input must use Metal device",
        )?;
        check_reduction_operand(spec, run, input.metadata())?;

        let output = match run {
            Some((start, end)) if end - start > 1 => {
                reduce_axis_run::<Self, T>(spec, input, input.metadata().shape().dims(), start, end)
            }
            _ => {
                let axis = run.map_or(0, |(start, _)| start);
                match (spec.op, spec.keep_dims) {
                    (ReduceOp::Sum, false) => {
                        <Self as ReductionOps<Self>>::sum_dim::<f32>(input, axis)
                    }
                    (ReduceOp::Sum, true) => {
                        <Self as ReductionOps<Self>>::sum_keepdim::<f32>(input, axis)
                    }
                    (ReduceOp::Mean, false) => {
                        <Self as ReductionOps<Self>>::mean_dim::<f32>(input, axis)
                    }
                    (ReduceOp::Mean, true) => {
                        <Self as ReductionOps<Self>>::mean_keepdim::<f32>(input, axis)
                    }
                    (ReduceOp::Max, false) => {
                        <Self as ReductionOps<Self>>::max_dim::<f32>(input, axis)
                    }
                    (ReduceOp::Max, true) => {
                        <Self as ReductionOps<Self>>::max_keepdim::<f32>(input, axis)
                    }
                    (ReduceOp::Min, false) => {
                        <Self as ReductionOps<Self>>::min_dim::<f32>(input, axis)
                    }
                    (ReduceOp::Min, true) => {
                        <Self as ReductionOps<Self>>::min_keepdim::<f32>(input, axis)
                    }
                    (ReduceOp::Prod, false) => {
                        <Self as ReductionOps<Self>>::prod_dim::<f32>(input, axis)
                    }
                    (ReduceOp::Prod, true) => <Self as ReductionOps<Self>>::prod_dim::<f32>(
                        input, axis,
                    )
                    .and_then(|dropped| {
                        <Self as TensorOps<Self>>::reshape::<f32>(&dropped, spec.output.dims())
                    }),
                }
            }
        }
        .map_err(|error| kernel_error(OperationKind::Reduction, error))?;

        if output.shape() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reduction,
                message: "Metal reduction output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}
