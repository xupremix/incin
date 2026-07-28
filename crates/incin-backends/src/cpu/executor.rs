//! Descriptor execution for the native CPU vertical slice.

use incin_core::exec::{
    Capabilities, CapabilityQuery, Conv2dSpec, MatMulSpec, MathMode, Pool2dSpec, PoolOp, ReduceOp,
    ReductionSpec, ReshapeSpec, SupportLevel, TensorMeta,
};
use incin_core::prelude::{
    BackendError, DType, DTypeId, Device, DeviceKind, Execute, ExecutionRequest, ModuleOps,
    OperationKind, ReductionOps, StorageBackend, TensorOps,
};

use super::CpuBackendImpl;
use super::storage::CpuStorage;
use crate::descriptor_bind::{
    check_conv2d_operands, check_pool2d_operand, check_reduction_operand, conv2d_window, invalid,
    kernel_error, pool2d_window, reduction_axis,
};

impl<T: DType, D: Device> StorageBackend for CpuBackendImpl<T, D> {
    type Storage<K: DType> = CpuStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}
impl<T: DType, D: Device> Capabilities for CpuBackendImpl<T, D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cpu, query)
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
    request: &'a ExecutionRequest<'_, MatMulSpec, CpuBackendImpl<T, D>>,
) -> Result<(&'a CpuStorage, &'a CpuStorage), BackendError> {
    let [lhs_handle, rhs_handle] = request.inputs else {
        return Err(invalid(
            OperationKind::MatMul,
            "matmul expects exactly two tensor inputs",
        ));
    };
    let lhs = lhs_handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not CPU storage"))?;
    let rhs = rhs_handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not CPU storage"))?;
    let spec = request.operation.descriptor();

    for metadata in [lhs.metadata(), rhs.metadata()] {
        if metadata.device().kind() != DeviceKind::Cpu || metadata.device().ordinal() != 0 {
            return Err(invalid(
                OperationKind::MatMul,
                "matmul inputs must use CPU device ordinal 0",
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

impl<T: DType, D: Device> Execute<MatMulSpec> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, MatMulSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let (lhs, rhs) = bind_matmul(&request)?;

        let lhs_transposed = if spec.transpose_lhs {
            let rank = lhs.shape().len();
            Some(
                lhs.transpose(rank - 2, rank - 1)
                    .map_err(|error| BackendError::Execution {
                        operation: OperationKind::MatMul,
                        message: error.to_string(),
                    })?,
            )
        } else {
            None
        };
        let rhs_transposed = if spec.transpose_rhs {
            let rank = rhs.shape().len();
            Some(
                rhs.transpose(rank - 2, rank - 1)
                    .map_err(|error| BackendError::Execution {
                        operation: OperationKind::MatMul,
                        message: error.to_string(),
                    })?,
            )
        } else {
            None
        };
        let lhs = lhs_transposed.as_ref().unwrap_or(lhs);
        let rhs = rhs_transposed.as_ref().unwrap_or(rhs);

        let output = if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
            super::ops::matmul::matmul_impl(lhs, rhs)
        } else {
            super::ops::matmul::batched_matmul_impl(lhs, rhs)
        }
        .map_err(|error| kernel_error(OperationKind::MatMul, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::MatMul,
                message: "CPU matmul output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

/// Bind the single reshape operand and check it against the sealed descriptor.
///
/// The capability query asks `training: false` on purpose. A
/// `Validated<ReshapeSpec>` is a shape proof, not a request for a gradient, and
/// nothing in the request obliges the backend to differentiate the result.
/// Asking for trainability anyway would refuse a `u8` or `u32` reshape that the
/// backend performs exactly right, because no backend registers integer dtypes
/// as trainable.
fn bind_reshape<'a, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, ReshapeSpec, CpuBackendImpl<T, D>>,
) -> Result<&'a CpuStorage, BackendError> {
    let [handle] = request.inputs else {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape expects exactly one tensor input",
        ));
    };
    let input = handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(OperationKind::Reshape, "reshape input is not CPU storage"))?;
    let spec = request.operation.descriptor();
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Cpu || metadata.device().ordinal() != 0 {
        return Err(invalid(
            OperationKind::Reshape,
            "reshape input must use CPU device ordinal 0",
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

impl<T: DType, D: Device> Execute<ReshapeSpec> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReshapeSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let input = bind_reshape(&request)?;

        // The legacy entry point, not `CpuStorage::reshape` directly: it is what
        // records the tape entry, so a descriptor-executed reshape is the same
        // graph node the typed frontend produces.
        let output = <Self as TensorOps<Self>>::reshape::<T>(input, spec.output.dims())
            .map_err(|error| kernel_error(OperationKind::Reshape, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reshape,
                message: "CPU reshape output disagrees with the validated descriptor".into(),
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
    request: &'a ExecutionRequest<'_, Conv2dSpec, CpuBackendImpl<T, D>>,
) -> Result<(&'a CpuStorage, &'a CpuStorage, Option<&'a CpuStorage>), BackendError> {
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
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2d, "conv2d input is not CPU storage"))
    };
    let input = downcast(input_handle)?;
    let weight = downcast(weight_handle)?;
    let bias = bias_handle.map(downcast).transpose()?;
    let spec = request.operation.descriptor();

    for storage in [Some(input), Some(weight), bias].into_iter().flatten() {
        let metadata = storage.metadata();
        if metadata.device().kind() != DeviceKind::Cpu || metadata.device().ordinal() != 0 {
            return Err(invalid(
                OperationKind::Conv2d,
                "conv2d inputs must use CPU device ordinal 0",
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
        bias.map(CpuStorage::metadata),
    )?;

    Ok((input, weight, bias))
}

impl<T: DType, D: Device> Execute<Conv2dSpec> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Conv2dSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let window = conv2d_window(spec)?;
        let (input, weight, bias) = bind_conv2d(&request)?;

        let output = <Self as ModuleOps<Self>>::conv2d::<T>(
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
                message: "CPU conv2d output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

/// Bind the one operand of a single-input descriptor and clear it through the
/// registry.
///
/// Reduction and pooling differ only in the operation their diagnostics name, so
/// they share this. Reshape keeps a binder of its own because it also compares
/// the operand against an input shape the descriptor stores, which neither of
/// these two records.
///
/// `training: false` for the same reason [`bind_reshape`] gives: a validated
/// descriptor is a shape proof, and nothing in the request obliges the backend
/// to differentiate what it produces.
fn bind_single_operand<'a, O, T: DType, D: Device>(
    request: &'a ExecutionRequest<'_, O, CpuBackendImpl<T, D>>,
    operation: OperationKind,
    wrong_arity: &'static str,
    wrong_device: &'static str,
) -> Result<&'a CpuStorage, BackendError>
where
    O: incin_core::exec::OperationSpec,
{
    let [handle] = request.inputs else {
        return Err(invalid(operation, wrong_arity));
    };
    let input = handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(operation, "input is not CPU storage"))?;
    let metadata = input.metadata();

    if metadata.device().kind() != DeviceKind::Cpu || metadata.device().ordinal() != 0 {
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

impl<T: DType, D: Device> Execute<ReductionSpec> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

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
            "reduction input must use CPU device ordinal 0",
        )?;
        check_reduction_operand(spec, axis, input.metadata())?;

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
            // `ReductionOps` has no `prod_keepdim`. Composing it is exact rather
            // than approximate: keeping the axis reinserts a length-1 dimension
            // and moves no element, which is what the descriptor's own output
            // shape already says, so the reshape below is the whole difference.
            (ReduceOp::Prod, true) => <Self as ReductionOps<Self>>::prod_dim::<T>(input, axis)
                .and_then(|dropped| {
                    <Self as TensorOps<Self>>::reshape::<T>(&dropped, spec.output.dims())
                }),
        }
        .map_err(|error| kernel_error(OperationKind::Reduction, error))?;

        if output.shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reduction,
                message: "CPU reduction output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<T: DType, D: Device> Execute<Pool2dSpec> for CpuBackendImpl<T, D> {
    type Output = CpuStorage;

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
            "pool2d input must use CPU device ordinal 0",
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
            // The dilation is absent from the call rather than dropped:
            // `pool2d_window` has already refused any descriptor that carries a
            // non-trivial one for this operator.
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
                message: "CPU pool2d output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use incin_core::exec::{ExecutionContext, MatMulRule, ShapeRule, TensorHandle, Validated};
    use incin_core::prelude::{Cpu, DeviceId, Dyn, Local, Shape};

    use super::*;
    use crate::cpu::storage::CpuBuffer;

    type TestBackend = CpuBackendImpl<f32, Cpu>;

    fn field<S: Shape>(dims: &[usize]) -> S::Field {
        S::from_dyn(dims).expect("test dimensions must match the shape type")
    }

    fn descriptor() -> Validated<MatMulSpec> {
        <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(
            &(field::<Dyn>(&[2, 3]), field::<Dyn>(&[3, 2])),
            (),
        )
        .unwrap()
    }

    #[test]
    fn binder_rejects_corrupted_non_cpu_storage_metadata() {
        let mut lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0; 6]), vec![2, 3]);
        lhs.meta.device = DeviceId::cuda(0);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0; 6]), vec![3, 2]);
        let inputs = [
            TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs),
            TensorHandle::from_storage::<TestBackend, f32, Local>(&rhs),
        ];
        let context = ExecutionContext::new(TestBackend::new());
        let validated = descriptor();
        let error = context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &inputs,
                context: &context,
            })
            .unwrap_err();

        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::MatMul,
                reason: "matmul inputs must use CPU device ordinal 0"
            }
        ));
    }
}
