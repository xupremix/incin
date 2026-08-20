//! Descriptor execution for the CUDA backend.
//!
//! This mirrors the CPU vertical slice from `EXE-007`: the same sealed
//! `Validated<Descriptor<op::MatMulExact>>` binds to CUDA storage through the same
//! `StorageBackend`/`Capabilities`/`Execute` contract, so the descriptor path
//! is not a CPU-only construction.

use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend, op};
use incin_core::error::BackendError;
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{Device, DeviceKind};

use super::backend::CudaBackendImpl;
use super::storage::CudaStorage;
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for CudaBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cuda, query)
    }
}

impl_creation_executors!(CudaBackendImpl<D>, CudaStorage);
impl_data_creation_executors!(CudaBackendImpl<D>, CudaStorage);

/// Whether an operand's physical shape is the one the descriptor promised.
///
/// The descriptor states the contracted extents and the broadcast batch; a
/// stride of 0 on a batch axis is the descriptor's own record that the operand
/// is broadcast along it, so that axis is required to be 1 rather than equal.
macro_rules! impl_cuda_canonical {
    ($(($op:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
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

macro_rules! impl_cuda_canonical_unary {
    ($(($op:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly one operand"));
                };
                let input = input.downcast_ref::<CudaStorage>().ok_or_else(|| invalid(operation, "operand is not CUDA storage"))?;
                crate::cuda::backend::$func(input)
                    .map_err(|error| kernel_error("Cuda", operation, error))
            }
        }
    )*};
}

impl_cuda_canonical_unary![
    (Relu, cuda_relu_storage),
    (Exp, cuda_exp_storage),
    (Sqrt, cuda_sqrt_storage),
    (Log, cuda_log_storage),
    (Tanh, cuda_tanh_storage),
    (Sigmoid, cuda_sigmoid_storage),
];

impl<D: Device> Execute<op::ReshapeExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ReshapeExact, Self>,
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
        Self::reshape::<f32>(storage, shape)
            .map_err(|e| kernel_error("Cuda", OperationKind::ReshapeExact, e))
    }
}

impl<D: Device> Execute<op::BroadcastAs> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastAs, Self>,
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
        Self::broadcast_as::<f32>(storage, shape)
            .map_err(|e| kernel_error("Cuda", OperationKind::BroadcastAs, e))
    }
}

impl<D: Device> Execute<op::MatMulExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
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
        Self::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Cuda", OperationKind::MatMulExact, e))
    }
}

impl<D: Device> Execute<op::Conv2dExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
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
        Self::conv2d::<f32>(
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

impl<D: Device> Execute<op::MaxPool2d> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaxPool2d, Self>,
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
        Self::max_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
            pair(attrs.dilation),
        )
        .map_err(|e| kernel_error("Cuda", OperationKind::MaxPool2d, e))
    }
}

impl<D: Device> Execute<op::AvgPool2d> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AvgPool2d, Self>,
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
        Self::avg_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
        )
        .map_err(|e| kernel_error("Cuda", OperationKind::AvgPool2d, e))
    }
}

/// Narrow a descriptor's `f64` epsilon to the `f32` the kernels below compute
/// in, refusing rather than silently admitting a value that would stop
/// guarding against division by a near-zero variance.
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

fn downcast<'a>(
    operand: &'a incin_core::exec::request::TensorHandle<'a>,
    operation: OperationKind,
    name: &'static str,
) -> Result<&'a CudaStorage, BackendError> {
    operand
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| invalid(operation, name))
}

/// Fused Welford kernel, normalizing over the trailing axes the weight's own
/// shape names. Forward only: no backward has been written for it yet, so the
/// capability row this answers to keeps `training` at `false` rather than
/// promise a gradient that would never arrive.
impl<D: Device> Execute<op::LayerNorm> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LayerNorm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::LayerNorm;
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
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        let bias = bias
            .map(|bias| downcast(bias, operation, "bias is not CUDA storage"))
            .transpose()?;
        let epsilon = narrowed_epsilon(
            operation,
            request.operation.descriptor().attributes().epsilon,
        )?;
        crate::cuda::ops::norm::launch_layer_norm(input, weight, bias, epsilon)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// Per-channel normalization by running statistics. Inference only, and
/// refused rather than approximated for a training-mode request: the kernel
/// this calls only reads precomputed `running_mean`/`running_variance`, it
/// does not reduce the batch's own statistics the way CPU's second kernel
/// does, so admitting `attributes.training` here would return a plausible
/// wrong answer instead of an error.
impl<D: Device> Execute<op::BatchNorm> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchNorm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::BatchNorm;
        let attributes = request.operation.descriptor().attributes();
        if attributes.training {
            return Err(invalid(
                operation,
                "CUDA batch norm has no on-the-fly batch statistics kernel yet; only \
                 inference mode, driven by a running mean and variance, is implemented",
            ));
        }
        let Some((input, optional)) = request.inputs.split_first() else {
            return Err(invalid(
                operation,
                "batch norm expects at least the input operand",
            ));
        };
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
        let (Some(running_mean), Some(running_variance)) = (running_mean, running_variance) else {
            return Err(invalid(
                operation,
                "inference batch norm needs a running mean and a running variance; without \
                 them the kernel substitutes a zero mean and a unit variance",
            ));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let weight = weight
            .map(|value| downcast(value, operation, "weight is not CUDA storage"))
            .transpose()?;
        let bias = bias
            .map(|value| downcast(value, operation, "bias is not CUDA storage"))
            .transpose()?;
        let running_mean = downcast(running_mean, operation, "running mean is not CUDA storage")?;
        let running_variance = downcast(
            running_variance,
            operation,
            "running variance is not CUDA storage",
        )?;
        let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;
        crate::cuda::ops::norm::launch_batch_norm(
            input,
            weight,
            bias,
            Some(running_mean),
            Some(running_variance),
            epsilon,
        )
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// `exp(x - max(x)) / sum(exp(x - max(x)))` along the descriptor's axis,
/// composed entirely from already tape-tracked primitives. `max_keepdim` is
/// not itself tape-tracked, which is exactly right here: softmax is invariant
/// to a constant shift, so the true gradient through the stabilizing max is
/// zero, and an untracked leaf gives that for free instead of needing a
/// hand-written zero.
impl<D: Device> Execute<op::Softmax> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Softmax, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Softmax;
        let [input] = request.inputs else {
            return Err(invalid(operation, "softmax expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let axis = request.operation.descriptor().attributes().axis;
        crate::cuda::backend::cuda_softmax::<D>(input, axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// `x / sqrt(mean(x^2, axis=-1) + eps) * weight`, composed the same way
/// CPU's kernel is: every step below already pushes a correct tape entry, so
/// the composite's backward is the tape replay, not new hand-derived math.
impl<D: Device> Execute<op::RmsNorm> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::RmsNorm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::RmsNorm;
        let [input, weight] = request.inputs else {
            return Err(invalid(operation, "rms norm expects an input and a weight"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        let axis = input.shape.len().saturating_sub(1);
        (|| {
            let squared = crate::cuda::backend::cuda_mul_storage(input, input)?;
            let mean = CudaBackendImpl::<D>::mean_keepdim::<f32>(&squared, axis)?;
            let guarded = CudaBackendImpl::<D>::add_scalar_float::<f32>(&mean, epsilon)?;
            let scale = crate::cuda::backend::cuda_sqrt_storage(&guarded)?;
            let normalized = crate::cuda::backend::cuda_div_storage(input, &scale)?;
            crate::cuda::backend::cuda_mul_storage(&normalized, weight)
        })()
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

macro_rules! impl_cuda_axis_view {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly one operand"));
                };
                let input = downcast(input, operation, "operand is not CUDA storage")?;
                let axis = request.operation.descriptor().attributes().axis;
                CudaBackendImpl::<D>::$method::<f32>(input, axis)
                    .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_axis_view![(SqueezeExact, squeeze), (UnsqueezeExact, unsqueeze)];

impl<D: Device> Execute<op::TransposeExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TransposeExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::TransposeExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "transpose expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::transpose::<f32>(input, attrs.first, attrs.second)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Narrow> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Narrow, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Narrow;
        let [input] = request.inputs else {
            return Err(invalid(operation, "narrow expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::narrow::<f32>(input, attrs.axis, attrs.start, attrs.length)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::FlattenExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::FlattenExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::FlattenExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "flatten expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::flatten::<f32>(input, attrs.start_axis, attrs.end_axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// `attributes.shape` is the whole target shape, not just the prefix
/// `broadcast_left` prepends - the same split CPU's own executor computes
/// before calling `broadcast_left_storage`.
impl<D: Device> Execute<op::BroadcastLeft> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastLeft, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::BroadcastLeft;
        let [input] = request.inputs else {
            return Err(invalid(operation, "broadcast_left expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let target = &request.operation.descriptor().attributes().shape;
        let rank = input.shape.len();
        let Some(prefix) = target.len().checked_sub(rank) else {
            return Err(invalid(
                operation,
                "the declared target shape has fewer axes than the operand",
            ));
        };
        CudaBackendImpl::<D>::broadcast_left::<f32>(input, &target[..prefix])
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::SliceExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::SliceExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::SliceExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "slice expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let ranges = &request.operation.descriptor().attributes().ranges;
        CudaBackendImpl::<D>::slice::<f32>(input, ranges)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::ConcatExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ConcatExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::ConcatExact;
        if request.inputs.is_empty() {
            return Err(invalid(operation, "operation expects at least one operand"));
        }
        let operands = request
            .inputs
            .iter()
            .map(|handle| downcast(handle, operation, "operand is not CUDA storage"))
            .collect::<Result<Vec<_>, _>>()?;
        let axis = request.operation.descriptor().attributes().axis;
        CudaBackendImpl::<D>::concat::<f32>(&operands, axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::StackExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StackExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::StackExact;
        if request.inputs.is_empty() {
            return Err(invalid(operation, "operation expects at least one operand"));
        }
        let operands = request
            .inputs
            .iter()
            .map(|handle| downcast(handle, operation, "operand is not CUDA storage"))
            .collect::<Result<Vec<_>, _>>()?;
        let axis = request.operation.descriptor().attributes().axis;
        CudaBackendImpl::<D>::stack::<f32>(&operands, axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Chunk> for CudaBackendImpl<D> {
    type Output = Vec<CudaStorage>;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Chunk, Self>,
    ) -> Result<Vec<CudaStorage>, BackendError> {
        let operation = OperationKind::Chunk;
        let [input] = request.inputs else {
            return Err(invalid(operation, "chunk expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::chunk::<f32>(input, attrs.axis, attrs.chunks)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Split> for CudaBackendImpl<D> {
    type Output = Vec<CudaStorage>;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Split, Self>,
    ) -> Result<Vec<CudaStorage>, BackendError> {
        let operation = OperationKind::Split;
        let [input] = request.inputs else {
            return Err(invalid(operation, "split expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::split::<f32>(input, attrs.axis, attrs.split_size)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// N-D batched matrix product, whose operand rank contract differs from the
/// plain `matmul` row and so does not share its registration.
impl<D: Device> Execute<op::BatchedMatMul> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchedMatMul, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::BatchedMatMul;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "batched matmul expects 2 inputs"));
        };
        let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
        let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
        CudaBackendImpl::<D>::batched_matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// Fused `beta * mat + alpha * (mat1 @ mat2)`.
impl<D: Device> Execute<op::Addmm> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Addmm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Addmm;
        let [mat, lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "addmm expects 3 inputs"));
        };
        let mat = downcast(mat, operation, "mat is not CUDA storage")?;
        let lhs = downcast(lhs, operation, "mat1 is not CUDA storage")?;
        let rhs = downcast(rhs, operation, "mat2 is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        (|| {
            let product = CudaBackendImpl::<D>::batched_matmul::<f32>(lhs, rhs)?;
            let scaled_product =
                CudaBackendImpl::<D>::mul_scalar_float::<f32>(&product, attrs.alpha)?;
            let scaled_mat = CudaBackendImpl::<D>::mul_scalar_float::<f32>(mat, attrs.beta)?;
            crate::cuda::backend::cuda_add_storage(&scaled_mat, &scaled_product)
        })()
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// The scalar inner product of two operands of equal shape: a multiply and
/// an all-reduce, composed rather than routed to a BLAS dot for the same
/// reason CPU's own `Dot` is.
impl<D: Device> Execute<op::Dot> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dot, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Dot;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "dot expects 2 inputs"));
        };
        let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
        let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
        (|| {
            let product = crate::cuda::backend::cuda_mul_storage(lhs, rhs)?;
            CudaBackendImpl::<D>::sum_all::<f32>(&product)
        })()
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// The outer product of two vectors, as a matrix: each operand grows an axis
/// on the side the other one occupies, and the broadcast multiply fills the
/// grid that leaves.
impl<D: Device> Execute<op::Outer> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Outer, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Outer;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "outer expects 2 inputs"));
        };
        let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
        let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
        (|| {
            let column = CudaBackendImpl::<D>::unsqueeze::<f32>(lhs, 1)?;
            let row = CudaBackendImpl::<D>::unsqueeze::<f32>(rhs, 0)?;
            crate::cuda::backend::cuda_mul_storage(&column, &row)
        })()
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// Scaled dot-product attention, with the mask as an optional fourth
/// operand. The attribute set says whether a mask is present, so the operand
/// count and the declared contract have to agree before anything runs.
impl<D: Device> Execute<op::ScaledDotProductAttention> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ScaledDotProductAttention, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::ScaledDotProductAttention;
        let attrs = request.operation.descriptor().attributes();
        let (query, key, value, mask) = match request.inputs {
            [q, k, v] if !attrs.has_mask => (q, k, v, None),
            [q, k, v, m] if attrs.has_mask => (q, k, v, Some(m)),
            _ => {
                return Err(invalid(
                    operation,
                    "operand count does not match the declared mask",
                ));
            }
        };
        let query = downcast(query, operation, "query is not CUDA storage")?;
        let key = downcast(key, operation, "key is not CUDA storage")?;
        let value = downcast(value, operation, "value is not CUDA storage")?;
        let mask = mask
            .map(|m| downcast(m, operation, "mask is not CUDA storage"))
            .transpose()?;
        (|| {
            let key_rank = key.shape.len();
            let key_t = if key_rank >= 2 {
                CudaBackendImpl::<D>::transpose::<f32>(key, key_rank - 2, key_rank - 1)?
            } else {
                key.clone()
            };
            let scores = CudaBackendImpl::<D>::batched_matmul::<f32>(query, &key_t)?;
            let d_k = *query.shape.last().unwrap_or(&1) as f64;
            let scale = attrs.scale.unwrap_or_else(|| 1.0 / d_k.sqrt());
            let scaled_scores = CudaBackendImpl::<D>::mul_scalar_float::<f32>(&scores, scale)?;
            let masked_scores = match mask {
                Some(mask) => crate::cuda::backend::cuda_add_storage(&scaled_scores, mask)?,
                None => scaled_scores,
            };
            let attention_axis = masked_scores.shape.len().saturating_sub(1);
            let attention =
                crate::cuda::backend::cuda_softmax::<D>(&masked_scores, attention_axis)?;
            CudaBackendImpl::<D>::batched_matmul::<f32>(&attention, value)
        })()
        .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

macro_rules! impl_cuda_scalar_tensor {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly one operand"));
                };
                let input = downcast(input, operation, "operand is not CUDA storage")?;
                let value = request.operation.descriptor().attributes().value;
                CudaBackendImpl::<D>::$method::<f32>(input, value)
                    .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_scalar_tensor![(SubScalar, sub_scalar_float), (DivScalar, div_scalar_float),];

/// Numeric comparisons, each producing fresh `bool` storage.
///
/// Not a canonical binary op: `crate::cuda::ops::compare::launch_compare`
/// writes a different dtype than it reads, which the packed/tuned pointwise
/// strategy the arithmetic binaries share cannot do (see that module's own
/// doc). The broadcast this executor performs up front is exactly the
/// precondition `launch_compare` states it relies on: both operands
/// identically shaped and contiguous. It calls
/// `crate::cuda::ops::shape::launch_broadcast` directly rather than
/// `CudaBackendImpl::broadcast_as` - the same materializing kernel launch,
/// minus the tape entry `broadcast_as` would push on the way out. That entry
/// would have nowhere to go: `launch_compare` never links its output id back
/// to an input on the tape, so nothing can ever walk to it during backward.
/// Calling the raw launch instead of arguing that the dead entry is harmless
/// keeps that true structurally. `launch_broadcast` trusts its caller to have
/// already checked shape compatibility, which `crate::layout::broadcast_shape`
/// below does before either operand is touched.
macro_rules! impl_cuda_cmp {
    ($(($op:ident, $mode:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
                let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
                let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)
                    .map_err(|e| kernel_error("Cuda", operation, e))?;
                let lhs_b = if lhs.shape == out_shape {
                    lhs.clone()
                } else {
                    crate::cuda::ops::shape::launch_broadcast(lhs, &out_shape)
                        .map_err(|e| kernel_error("Cuda", operation, e))?
                };
                let rhs_b = if rhs.shape == out_shape {
                    rhs.clone()
                } else {
                    crate::cuda::ops::shape::launch_broadcast(rhs, &out_shape)
                        .map_err(|e| kernel_error("Cuda", operation, e))?
                };
                crate::cuda::ops::compare::launch_compare(
                    crate::cuda::ops::compare::CompareOp::$mode,
                    &lhs_b,
                    &rhs_b,
                )
                .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_cmp![
    (CmpEq, Eq),
    (CmpNe, Ne),
    (CmpLt, Lt),
    (CmpLe, Le),
    (CmpGt, Gt),
    (CmpGe, Ge),
];

/// Selects between two operands by a `bool` mask, broadcasting all three to
/// one shape first.
///
/// `on_true`/`on_false` are broadcast through `CudaBackendImpl::broadcast_as`
/// (tape-recording, since a caller can legitimately want a gradient on
/// either), `mask` through the raw `launch_broadcast` (non-recording, for the
/// same reason `impl_cuda_cmp!` above uses it: a `bool` mask has nowhere to
/// send a gradient). The tape entry this pushes names the *broadcasted*
/// `on_true`/`on_false` ids as its inputs, not the original operands', so the
/// generic backward walk continues automatically through whatever
/// `broadcast_as` entries those broadcasts pushed - the same composition
/// this crate's `batched_matmul`/`softmax` already rely on. The backward
/// itself reuses the forward kernel: routing `grad_out` to `grad_true` where
/// the mask is set and to `grad_false` where it is not is exactly what
/// `where_cond(mask, grad_out, zeros)`/`where_cond(mask, zeros, grad_out)`
/// compute.
impl<D: Device> Execute<op::WhereCond> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::WhereCond, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::WhereCond;
        let [mask, on_true, on_false] = request.inputs else {
            return Err(invalid(
                operation,
                "where_cond expects exactly three operands",
            ));
        };
        let mask = downcast(mask, operation, "mask is not CUDA storage")?;
        let on_true = downcast(on_true, operation, "on_true is not CUDA storage")?;
        let on_false = downcast(on_false, operation, "on_false is not CUDA storage")?;

        let base_shape = crate::layout::broadcast_shape(&on_true.shape, &on_false.shape)
            .map_err(|e| kernel_error("Cuda", operation, e))?;
        let out_shape = crate::layout::broadcast_shape(&mask.shape, &base_shape)
            .map_err(|e| kernel_error("Cuda", operation, e))?;

        let mask_b = if mask.shape == out_shape {
            mask.clone()
        } else {
            // Not `shape::launch_broadcast`: that launches `shape.cu`'s
            // `shape_op`, whose data pointers are a hardcoded `float*` -
            // exactly the byte-width assumption this session already
            // narrowed the `BroadcastAs` capability row over, and a `bool`
            // mask hits it just the same through a direct call. See
            // `cuda::ops::select::launch_broadcast_bool_mask`'s own doc.
            crate::cuda::ops::select::launch_broadcast_bool_mask(mask, &out_shape)
                .map_err(|e| kernel_error("Cuda", operation, e))?
        };
        let true_b = if on_true.shape == out_shape {
            on_true.clone()
        } else {
            CudaBackendImpl::<D>::broadcast_as::<f32>(on_true, &out_shape)
                .map_err(|e| kernel_error("Cuda", operation, e))?
        };
        let false_b = if on_false.shape == out_shape {
            on_false.clone()
        } else {
            CudaBackendImpl::<D>::broadcast_as::<f32>(on_false, &out_shape)
                .map_err(|e| kernel_error("Cuda", operation, e))?
        };

        let out = crate::cuda::ops::select::launch_where_cond(&mask_b, &true_b, &false_b)
            .map_err(|e| kernel_error("Cuda", operation, e))?;

        let device_id = mask_b.buffer.device_id;
        let (mask_capture, out_shape_capture) = (mask_b.clone(), out_shape.clone());
        let (true_id, false_id, out_id) = (true_b.id, false_b.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![true_id, false_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let zeros = CudaBackendImpl::<D>::zeros::<f32>(
                    &out_shape_capture,
                    incin_core::tensor::dtype::DTypeId::F32.descriptor(),
                    &incin_core::tensor::device::DeviceId::cuda(device_id),
                )?;
                let grad_true =
                    crate::cuda::ops::select::launch_where_cond(&mask_capture, grad_out, &zeros)?;
                let grad_false =
                    crate::cuda::ops::select::launch_where_cond(&mask_capture, &zeros, grad_out)?;
                Ok(vec![grad_true, grad_false])
            }),
        });

        Ok(out)
    }
}

/// Overwrites the masked positions with the declared scalar. No tape entry:
/// see `cuda::ops::select::launch_masked_fill`'s own doc for why this
/// matches CPU's existing (gradient-less) behaviour rather than diverging
/// from it.
impl<D: Device> Execute<op::MaskedFill> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaskedFill, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::MaskedFill;
        let [input, mask] = request.inputs else {
            return Err(invalid(
                operation,
                "masked_fill expects exactly two operands",
            ));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let mask = downcast(mask, operation, "mask is not CUDA storage")?;
        let value = request.operation.descriptor().attributes().value;
        crate::cuda::ops::select::launch_masked_fill(input, mask, value)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// `LogicalAnd`/`LogicalOr`: both operands and the output are `bool`, unlike
/// `where_cond`/`masked_fill`'s mixed `bool`+`f32`, so the capability row
/// this answers to is `Bool`-only rather than a union - no `F32_AND_BOOL`
/// reasoning needed. Broadcasting still goes through
/// `cuda::ops::select::launch_broadcast_bool_mask`, the same non-`shape_op`
/// path `impl_cuda_cmp!`/`Execute<op::WhereCond>` use, since `shape_op`'s
/// `float*` kernel still cannot answer a 1-byte dtype.
macro_rules! impl_cuda_logical_binary {
    ($(($op:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
                let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
                let out_shape = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape)
                    .map_err(|e| kernel_error("Cuda", operation, e))?;
                let lhs_b = if lhs.shape == out_shape {
                    lhs.clone()
                } else {
                    crate::cuda::ops::select::launch_broadcast_bool_mask(lhs, &out_shape)
                        .map_err(|e| kernel_error("Cuda", operation, e))?
                };
                let rhs_b = if rhs.shape == out_shape {
                    rhs.clone()
                } else {
                    crate::cuda::ops::select::launch_broadcast_bool_mask(rhs, &out_shape)
                        .map_err(|e| kernel_error("Cuda", operation, e))?
                };
                crate::cuda::ops::logical::$func(&lhs_b, &rhs_b)
                    .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_logical_binary![
    (LogicalAnd, launch_logical_and),
    (LogicalOr, launch_logical_or),
];

impl<D: Device> Execute<op::LogicalNot> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LogicalNot, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::LogicalNot;
        let [input] = request.inputs else {
            return Err(invalid(operation, "operation expects exactly one operand"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        crate::cuda::ops::logical::launch_logical_not(input)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

macro_rules! impl_cuda_reduction_all {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
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
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
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
    (SumAll, CudaBackendImpl::<D>::sum_all::<f32>),
    (MeanAll, CudaBackendImpl::<D>::mean_all::<f32>),
    (MaxAll, CudaBackendImpl::<D>::max_all::<f32>),
    (MinAll, CudaBackendImpl::<D>::min_all::<f32>),
];

impl_cuda_reduction_dim![
    (SumDim, CudaBackendImpl::<D>::sum_dim::<f32>),
    (SumKeepDim, |input, axis| {
        CudaBackendImpl::<D>::sum_keepdim::<f32>(input, axis)
    }),
    (MeanDim, |input, axis| {
        CudaBackendImpl::<D>::mean_dim::<f32>(input, axis)
    }),
    (MeanKeepDim, |input, axis| {
        CudaBackendImpl::<D>::mean_keepdim::<f32>(input, axis)
    }),
    (MaxDim, CudaBackendImpl::<D>::max_dim::<f32>),
    (MaxKeepDim, |input, axis| {
        CudaBackendImpl::<D>::max_keepdim::<f32>(input, axis)
    }),
    (MinDim, CudaBackendImpl::<D>::min_dim::<f32>),
    (MinKeepDim, |input, axis| {
        CudaBackendImpl::<D>::min_keepdim::<f32>(input, axis)
    }),
];

macro_rules! assert_every_advertised_cuda_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, CudaBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::tensor::device::Cuda>();
        };
    };
}

crate::capability::cuda_descriptor_operations!(assert_every_advertised_cuda_row_executes,);
