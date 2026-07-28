//! Descriptor execution for the CUDA backend.
//!
//! This mirrors the CPU vertical slice from `EXE-007`: the same sealed
//! `Validated<MatMulSpec>` binds to CUDA storage through the same
//! `StorageBackend`/`Capabilities`/`Execute` contract, so the descriptor path
//! is not a CPU-only construction.

use incin_core::exec::{
    Capabilities, CapabilityQuery, Conv2dSpec, MatMulSpec, MathMode, Pool2dSpec, PoolOp, ReduceOp,
    ReductionSpec, ReshapeSpec, SupportLevel, TensorMeta,
};
use incin_core::prelude::{
    BackendError, DType, DTypeId, Device, DeviceKind, Execute, ExecutionRequest, ModuleOps,
    OperationKind, ReductionOps, StorageBackend, TensorOps,
};

use super::backend::CudaBackendImpl;
use super::storage::CudaStorage;
use crate::descriptor_bind::{
    check_conv2d_operands, check_pool2d_operand, check_reduction_operand, conv2d_window, invalid,
    kernel_error, pool2d_window, reduction_axis,
};

impl<T: DType, D: Device> StorageBackend for CudaBackendImpl<T, D> {
    type Storage<K: DType> = CudaStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<T: DType, D: Device> Capabilities for CudaBackendImpl<T, D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cuda, query)
    }
}

/// Whether an operand's physical shape is the one the descriptor promised.
///
/// The descriptor states the contracted extents and the broadcast batch; a
/// stride of 0 on a batch axis is the descriptor's own record that the operand
/// is broadcast along it, so that axis is required to be 1 rather than equal.
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
    request: &'a ExecutionRequest<'_, MatMulSpec, CudaBackendImpl<T, D>>,
) -> Result<(&'a CudaStorage, &'a CudaStorage), BackendError> {
    let [lhs_handle, rhs_handle] = request.inputs else {
        return Err(invalid(
            OperationKind::MatMul,
            "matmul expects exactly two tensor inputs",
        ));
    };
    let lhs = lhs_handle
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not CUDA storage"))?;
    let rhs = rhs_handle
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not CUDA storage"))?;
    let spec = request.operation.descriptor();

    for metadata in [lhs.metadata(), rhs.metadata()] {
        if metadata.device().kind() != DeviceKind::Cuda || metadata.device().ordinal() != 0 {
            return Err(invalid(
                OperationKind::MatMul,
                "matmul inputs must use CUDA device ordinal 0",
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

/// Bind the single reshape operand and check it against the sealed descriptor.
///
/// See the CPU binder for why the capability query asks `training: false`: a
/// `Validated<ReshapeSpec>` proves shapes, not a gradient obligation. Unlike
/// matmul there is no hand-written dtype check either — CUDA registers
/// `Reshape` for every dtype its storage can hold, and the registry is the one
/// place that fact is written down.
fn bind_reshape<'a, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, ReshapeSpec, CudaBackendImpl<T, D>>,
) -> Result<&'a CudaStorage, BackendError> {
    let [handle] = request.inputs else {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape expects exactly one tensor input",
        ));
    };
    let input = handle
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| invalid(OperationKind::Reshape, "reshape input is not CUDA storage"))?;
    let spec = request.operation.descriptor();
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Cuda || metadata.device().ordinal() != 0 {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape input must use CUDA device ordinal 0",
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

impl<T: DType, D: Device> Execute<ReshapeSpec> for CudaBackendImpl<T, D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReshapeSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let input = bind_reshape(&request)?;

        let output = <Self as TensorOps<Self>>::reshape::<T>(input, spec.output.dims())
            .map_err(|error| kernel_error(OperationKind::Reshape, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reshape,
                message: "CUDA reshape output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<T: DType, D: Device> Execute<MatMulSpec> for CudaBackendImpl<T, D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, MatMulSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let (lhs, rhs) = bind_matmul(&request)?;

        // The CUDA GEMM consumes row-major operands. A transposing descriptor
        // is materialized first rather than folded into the shader, so the
        // descriptor path and the legacy path run the identical kernel.
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

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::MatMul,
                message: "CUDA matmul output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

/// Bind the two or three convolution operands against the sealed descriptor.
///
/// A bias is optional, so the operand count carries meaning: two inputs is a
/// bias-free convolution and three is a biased one. Any other count is a
/// malformed request rather than a defaultable one.
fn bind_conv2d<'a, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, Conv2dSpec, CudaBackendImpl<T, D>>,
) -> Result<(&'a CudaStorage, &'a CudaStorage, Option<&'a CudaStorage>), BackendError> {
    let (input_handle, weight_handle, bias_handle) = match request.inputs {
        [input, weight] => (input, weight, None),
        [input, weight, bias] => (input, weight, Some(bias)),
        _ => {
            return Err(invalid(
                OperationKind::Conv2d,
                "conv2d expects an input and a weight, and optionally a bias",
            ));
        }
    };
    let downcast = |handle: &'a incin_core::exec::TensorHandle<'_>| {
        handle
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2d, "conv2d input is not CUDA storage"))
    };
    let input = downcast(input_handle)?;
    let weight = downcast(weight_handle)?;
    let bias = bias_handle.map(downcast).transpose()?;
    let spec = request.operation.descriptor();

    for storage in [Some(input), Some(weight), bias].into_iter().flatten() {
        let metadata = storage.metadata();
        if metadata.device().kind() != DeviceKind::Cuda || metadata.device().ordinal() != 0 {
            return Err(invalid(
                OperationKind::Conv2d,
                "conv2d inputs must use CUDA device ordinal 0",
            ));
        }
        if metadata.dtype() != DTypeId::F32 {
            return Err(incin_core::exec::UnsupportedReason::DType {
                operation: OperationKind::Conv2d,
                dtype: metadata.dtype(),
            }
            .into());
        }
        let query = CapabilityQuery {
            operation: OperationKind::Conv2d,
            dtype: metadata.dtype(),
            layout: metadata.layout(),
            // The registry's `Conv2d` rank window covers the activation, not the
            // rank-1 bias, so every operand is queried at the descriptor's own
            // rank rather than its individual one.
            rank: spec.output.rank(),
            training: false,
            math_mode: MathMode::Precise,
        };
        if let SupportLevel::Unsupported(reason) = request.context.backend().support(&query) {
            return Err(reason.into());
        }
    }

    check_conv2d_operands(
        spec,
        input.metadata(),
        weight.metadata(),
        bias.map(CudaStorage::metadata),
    )?;

    Ok((input, weight, bias))
}

impl<T: DType, D: Device> Execute<Conv2dSpec> for CudaBackendImpl<T, D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Conv2dSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let window = conv2d_window(spec)?;
        let (input, weight, bias) = bind_conv2d(&request)?;

        let output = <Self as ModuleOps<Self>>::conv2d::<f32>(
            input,
            weight,
            bias,
            window.stride,
            window.padding,
            window.dilation,
            window.groups,
        )
        .map_err(|error| kernel_error(OperationKind::Conv2d, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Conv2d,
                message: "CUDA conv2d output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

/// Bind the one operand of a single-input descriptor and clear it through the
/// registry.
///
/// The CPU binder of the same name explains the shape of this; the differences
/// here are the storage type and the device the diagnostics name.
fn bind_single_operand<'a, O, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, O, CudaBackendImpl<T, D>>,
    operation: OperationKind,
    wrong_arity: &'static str,
    wrong_device: &'static str,
) -> Result<&'a CudaStorage, BackendError>
where
    O: incin_core::exec::OperationSpec,
{
    let [handle] = request.inputs else {
        return Err(invalid(operation, wrong_arity));
    };
    let input = handle
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| invalid(operation, "input is not CUDA storage"))?;
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Cuda {
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

impl<T: DType, D: Device> Execute<ReductionSpec> for CudaBackendImpl<T, D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReductionSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let axis = reduction_axis(spec)?;
        let input = bind_single_operand(
            &request,
            OperationKind::Reduction,
            "a reduction expects exactly one tensor input",
            "reduction input must use a CUDA device",
        )?;
        check_reduction_operand(spec, axis, input.metadata())?;

        // Like WGPU, CUDA declares `prod_dim` unsupported at its impl site, so
        // `ReduceOp::Prod` is answered as unsupported rather than as a fault.
        let output = match (spec.op, spec.keep_dims) {
            (ReduceOp::Sum, false) => <Self as ReductionOps<Self>>::sum_dim::<T>(input, axis),
            (ReduceOp::Sum, true) => <Self as ReductionOps<Self>>::sum_keepdim::<T>(input, axis),
            (ReduceOp::Mean, false) => <Self as ReductionOps<Self>>::mean_dim::<T>(input, axis),
            (ReduceOp::Mean, true) => <Self as ReductionOps<Self>>::mean_keepdim::<T>(input, axis),
            (ReduceOp::Max, false) => <Self as ReductionOps<Self>>::max_dim::<T>(input, axis),
            (ReduceOp::Max, true) => <Self as ReductionOps<Self>>::max_keepdim::<T>(input, axis),
            (ReduceOp::Min, false) => <Self as ReductionOps<Self>>::min_dim::<T>(input, axis),
            (ReduceOp::Min, true) => <Self as ReductionOps<Self>>::min_keepdim::<T>(input, axis),
            (ReduceOp::Prod, false) => <Self as ReductionOps<Self>>::prod_dim::<T>(input, axis),
            (ReduceOp::Prod, true) => <Self as ReductionOps<Self>>::prod_dim::<T>(input, axis)
                .and_then(|dropped| {
                    <Self as TensorOps<Self>>::reshape::<T>(&dropped, spec.output.dims())
                }),
        }
        .map_err(|error| kernel_error(OperationKind::Reduction, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reduction,
                message: "CUDA reduction output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<T: DType, D: Device> Execute<Pool2dSpec> for CudaBackendImpl<T, D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Pool2dSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let window = pool2d_window(spec)?;
        let input = bind_single_operand(
            &request,
            OperationKind::Pool2d,
            "pool2d expects exactly one tensor input",
            "pool2d input must use a CUDA device",
        )?;
        check_pool2d_operand(spec, input.metadata())?;

        let output = match spec.op {
            PoolOp::Max => <Self as ModuleOps<Self>>::max_pool2d::<T>(
                input,
                window.kernel,
                window.stride,
                window.padding,
                window.dilation,
            ),
            PoolOp::Average => <Self as ModuleOps<Self>>::avg_pool2d::<T>(
                input,
                window.kernel,
                window.stride,
                window.padding,
            ),
        }
        .map_err(|error| kernel_error(OperationKind::Pool2d, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Pool2d,
                message: "CUDA pool2d output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}
