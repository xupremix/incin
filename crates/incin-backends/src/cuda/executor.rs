//! Descriptor execution for the CUDA backend.
//!
//! This mirrors the CPU vertical slice from `EXE-007`: the same sealed
//! `Validated<Descriptor<op::MatMulExact>>` binds to CUDA storage through the same
//! `StorageBackend`/`Capabilities`/`Execute` contract, so the descriptor path
//! is not a CPU-only construction.

use alloc::sync::Arc;
use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend, op};
use incin_core::error::BackendError;
use incin_core::exec::catalog::LossReduction;
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel, UnsupportedReason};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{Device, DeviceId, DeviceKind};
use incin_core::tensor::dtype::DTypeId;
use incin_core::tensor::reduction::Reduction;

use super::backend::CudaBackendImpl;
use super::storage::{CudaBuffer, CudaStorage};
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for CudaBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Cuda, query)
    }
}

impl_creation_executors!(CudaBackendImpl<D>, CudaStorage);
impl_data_creation_executors!(CudaBackendImpl<D>, CudaStorage);
impl_variable_creation_executors!(CudaBackendImpl<D>, super::backend::CudaVar);
impl_readback_executors!(CudaBackendImpl<D>, CudaStorage);

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
                // The descriptor's proof is about the *output* geometry, which
                // is what a pointwise kernel iterates, so it applies here even
                // though the two operands may broadcast from different shapes.
                let specialization = crate::kernel::KernelSpecialization::from_evidence(
                    Some(request.operation.shape_evidence()),
                );
                crate::cuda::backend::$func(lhs, rhs, specialization)
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
    (Maximum, cuda_maximum_storage),
    (Minimum, cuda_minimum_storage),
    (AbsDiff, cuda_abs_diff_storage),
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
                // The one place a pointwise kernel can learn what the *type*
                // proved rather than what the buffer happens to measure.
                // `Validated` carries the frontend's `ShapeEvidence`, and this
                // is the first consumer of it in any backend.
                let specialization = crate::kernel::KernelSpecialization::from_evidence(
                    Some(request.operation.shape_evidence()),
                );
                crate::cuda::backend::$func(input, specialization)
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
    (Step, cuda_step_storage),
    (Mish, cuda_mish_storage),
    (Elu, cuda_elu_storage),
    (Gelu, cuda_gelu_storage),
    (Abs, cuda_abs_storage),
    (Neg, cuda_neg_storage),
    (Swish, cuda_swish_storage),
    (Sign, cuda_sign_storage),
    (Floor, cuda_floor_storage),
    (Ceil, cuda_ceil_storage),
    (Round, cuda_round_storage),
    (Log2, cuda_log2_storage),
    (Log10, cuda_log10_storage),
    (Sin, cuda_sin_storage),
    (Cos, cuda_cos_storage),
    (Tan, cuda_tan_storage),
    (Asin, cuda_asin_storage),
    (Acos, cuda_acos_storage),
    (Atan, cuda_atan_storage),
    (Sinh, cuda_sinh_storage),
    (Cosh, cuda_cosh_storage),
    (Asinh, cuda_asinh_storage),
    (Acosh, cuda_acosh_storage),
    (Atanh, cuda_atanh_storage),
    (Erf, cuda_erf_storage),
    (Rsqrt, cuda_rsqrt_storage),
    (Trunc, cuda_trunc_storage),
    (Frac, cuda_frac_storage),
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
        // Bias is optional, matching the CPU binder in `cpu::canonical::nn` and
        // the `Option<&Storage>` the kernel below already accepts. Refusing the
        // three-operand form here made biased conv2d executable on CPU and
        // unreachable on CUDA, even though `conv2d_with_bias_adds_per_channel_constant`
        // covers the kernel's biased path directly.
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
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "input is not CUDA storage"))?;
        let weight = weight
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "weight is not CUDA storage"))?;
        let bias = bias
            .map(|bias| {
                bias.downcast_ref::<CudaStorage>()
                    .ok_or_else(|| invalid(OperationKind::Conv2dExact, "bias is not CUDA storage"))
            })
            .transpose()?;
        let attrs = request.operation.descriptor().attributes();
        Self::conv2d::<f32>(
            input,
            weight,
            bias,
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
        crate::cuda::ops::norm::launch_softmax(input, axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

/// Fused single-pass RMSNorm kernel: `x * rsqrt(mean(x^2) + eps) * weight` computed in
/// registers/SRAM with zero intermediate VRAM allocations.
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
        crate::cuda::ops::norm::launch_rms_norm(input, weight, epsilon as f32)
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

/// A transpose that permutes metadata instead of copying.
///
/// `TransposeExact` on this backend runs a permutation kernel into a fresh
/// contiguous buffer; this shares the buffer and permutes shape and strides, so
/// it does no device work at all. The result is genuinely non-contiguous and
/// takes the strided pointwise kernels.
///
/// Both exist because neither is universally better, and which wins is a
/// property of the consumer rather than of the transpose. Measured here for a
/// transpose plus pointwise consumption: the view is ~11% faster when the
/// result is read once and ~15% slower when it is read eight times, crossing
/// over between two and four reads. See `ops::view_cost_bench` and issue #113.
impl<D: Device> Execute<op::TransposeView> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TransposeView, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::TransposeView;
        let [input] = request.inputs else {
            return Err(invalid(operation, "transpose_view expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        crate::cuda::ops::shape::launch_transpose_view(input, attrs.first, attrs.second)
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
            crate::cuda::backend::cuda_add_storage(
                &scaled_mat,
                &scaled_product,
                crate::kernel::KernelSpecialization::NONE,
            )
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
            let product = crate::cuda::backend::cuda_mul_storage(
                lhs,
                rhs,
                crate::kernel::KernelSpecialization::NONE,
            )?;
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
            crate::cuda::backend::cuda_mul_storage(
                &column,
                &row,
                crate::kernel::KernelSpecialization::NONE,
            )
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
                Some(mask) => crate::cuda::backend::cuda_add_storage(
                    &scaled_scores,
                    mask,
                    crate::kernel::KernelSpecialization::NONE,
                )?,
                None => scaled_scores,
            };
            let attention_axis = masked_scores.shape.len().saturating_sub(1);
            let attention = crate::cuda::ops::norm::launch_softmax(&masked_scores, attention_axis)?;
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

impl_cuda_scalar_tensor![
    (AddScalar, add_scalar_float),
    (SubScalar, sub_scalar_float),
    (MulScalar, mul_scalar_float),
    (DivScalar, div_scalar_float),
];

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
    (ProdAll, CudaBackendImpl::<D>::prod_all::<f32>),
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
    (ProdDim, CudaBackendImpl::<D>::prod_dim::<f32>),
];

impl<D: Device> Execute<op::VarianceAll> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::VarianceAll, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::VarianceAll;
        let [input] = request.inputs else {
            return Err(invalid(operation, "var_all expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let unbiased = request.operation.descriptor().attributes().unbiased;
        CudaBackendImpl::<D>::var_all::<f32>(input, unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::VarianceDim> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::VarianceDim, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::VarianceDim;
        let [input] = request.inputs else {
            return Err(invalid(operation, "var_dim expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::var_dim::<f32>(input, attrs.axis, attrs.unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::VarianceKeepDim> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::VarianceKeepDim, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::VarianceKeepDim;
        let [input] = request.inputs else {
            return Err(invalid(operation, "var_keepdim expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::var_keepdim::<f32>(input, attrs.axis, attrs.unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::StdAll> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StdAll, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::StdAll;
        let [input] = request.inputs else {
            return Err(invalid(operation, "std_all expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let unbiased = request.operation.descriptor().attributes().unbiased;
        CudaBackendImpl::<D>::std_all::<f32>(input, unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::StdDim> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StdDim, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::StdDim;
        let [input] = request.inputs else {
            return Err(invalid(operation, "std_dim expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::std_dim::<f32>(input, attrs.axis, attrs.unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::StdKeepDim> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StdKeepDim, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::StdKeepDim;
        let [input] = request.inputs else {
            return Err(invalid(operation, "std_keepdim expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::std_keepdim::<f32>(input, attrs.axis, attrs.unbiased)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Cumsum> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Cumsum, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Cumsum;
        let [input] = request.inputs else {
            return Err(invalid(operation, "cumsum expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let axis = request.operation.descriptor().attributes().axis;
        CudaBackendImpl::<D>::cumsum::<f32>(input, axis)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::EmbeddingExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::EmbeddingExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::EmbeddingExact;
        let [indices, weight] = request.inputs else {
            return Err(invalid(
                operation,
                "embedding expects an index tensor and a weight table",
            ));
        };
        let indices = downcast(indices, operation, "indices is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        CudaBackendImpl::<D>::embedding::<f32, i64>(weight, indices)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Gather> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Gather, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Gather;
        let [input, index] = request.inputs else {
            return Err(invalid(operation, "gather expects 2 inputs"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let index = downcast(index, operation, "index is not CUDA storage")?;
        let axis = request.operation.descriptor().attributes().axis;
        CudaBackendImpl::<D>::gather::<f32, i64>(input, axis, index)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Scatter> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Scatter, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Scatter;
        let [input, index, src] = request.inputs else {
            return Err(invalid(operation, "scatter expects 3 inputs"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let index = downcast(index, operation, "index is not CUDA storage")?;
        let src = downcast(src, operation, "src is not CUDA storage")?;
        let axis = request.operation.descriptor().attributes().axis;
        CudaBackendImpl::<D>::scatter::<f32, i64>(input, axis, index, src)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Diag> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Diag, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Diag;
        let [input] = request.inputs else {
            return Err(invalid(operation, "diag expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let diagonal = request.operation.descriptor().attributes().offset;
        CudaBackendImpl::<D>::diag::<f32>(input, diagonal)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Pad> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Pad, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Pad;
        let [input] = request.inputs else {
            return Err(invalid(operation, "pad expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::pad::<f32>(input, &attrs.padding, attrs.value)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Repeat> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Repeat, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Repeat;
        let [input] = request.inputs else {
            return Err(invalid(operation, "repeat expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let repeats = &request.operation.descriptor().attributes().repeats;
        CudaBackendImpl::<D>::repeat::<f32>(input, repeats)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Tril> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Tril, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Tril;
        let [input] = request.inputs else {
            return Err(invalid(operation, "tril expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let diagonal = request.operation.descriptor().attributes().offset;
        CudaBackendImpl::<D>::tril::<f32>(input, diagonal)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Triu> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Triu, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Triu;
        let [input] = request.inputs else {
            return Err(invalid(operation, "triu expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let diagonal = request.operation.descriptor().attributes().offset;
        CudaBackendImpl::<D>::triu::<f32>(input, diagonal)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Powf> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Powf, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Powf;
        let [input] = request.inputs else {
            return Err(invalid(operation, "powf expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let exp = request.operation.descriptor().attributes().value;
        CudaBackendImpl::<D>::powf::<f32>(input, exp)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Clamp> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Clamp, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Clamp;
        let [input] = request.inputs else {
            return Err(invalid(operation, "clamp expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::clamp::<f32>(input, attrs.min, attrs.max)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

macro_rules! impl_cuda_binary_math {
    ($(($op:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects 2 operands"));
                };
                let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
                let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
                CudaBackendImpl::<D>::$func::<f32>(lhs, rhs)
                    .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_binary_math![(Atan2, atan2), (Fmod, fmod), (Remainder, remainder),];

impl<D: Device> Execute<op::Lerp> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Lerp, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Lerp;
        let [start, end] = request.inputs else {
            return Err(invalid(operation, "lerp expects 2 inputs"));
        };
        let start = downcast(start, operation, "start is not CUDA storage")?;
        let end = downcast(end, operation, "end is not CUDA storage")?;
        let weight = request.operation.descriptor().attributes().weight;
        crate::cuda::backend::cuda_lerp_storage(start, end, weight)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

macro_rules! impl_cuda_index_reduction {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for CudaBackendImpl<D> {
            type Output = CudaStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "operation expects 1 operand"));
                };
                let input = downcast(input, operation, "input is not CUDA storage")?;
                let attrs = request.operation.descriptor().attributes();
                match attrs.axis {
                    Some(axis) => CudaBackendImpl::<D>::$method::<i64>(input, Some(axis)),
                    None => CudaBackendImpl::<D>::$method::<i64>(input, None),
                }
                .map_err(|e| kernel_error("Cuda", operation, e))
            }
        }
    )*};
}

impl_cuda_index_reduction![(ArgMax, argmax), (ArgMin, argmin),];

impl<D: Device> Execute<op::TopK> for CudaBackendImpl<D> {
    type Output = (CudaStorage, CudaStorage);
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TopK, Self>,
    ) -> Result<(CudaStorage, CudaStorage), BackendError> {
        let operation = OperationKind::TopK;
        let [input] = request.inputs else {
            return Err(invalid(operation, "topk expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::topk::<i64>(input, attrs.k, attrs.axis, attrs.largest)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Quantize> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Quantize, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Quantize;
        let [input] = request.inputs else {
            return Err(invalid(operation, "quantize expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::Q8_0.descriptor() {
            return Err(BackendError::unsupported(
                "Cuda",
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        crate::cuda::ops::quant::launch_quantize_q8_0(input)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Dequantize> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dequantize, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Dequantize;
        let [input] = request.inputs else {
            return Err(invalid(operation, "dequantize expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::unsupported(
                "Cuda",
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        crate::cuda::ops::quant::launch_dequantize_q8_0(input)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::QuantizedMatMul> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::QuantizedMatMul, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::QuantizedMatMul;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "quantized_matmul expects 2 inputs"));
        };
        let lhs = downcast(lhs, operation, "lhs is not CUDA storage")?;
        let rhs = downcast(rhs, operation, "rhs is not CUDA storage")?;
        crate::cuda::ops::quant::launch_quantized_matmul_q8_0(lhs, rhs)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::Linear> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Linear, Self>,
    ) -> Result<CudaStorage, BackendError> {
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
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        let bias = bias
            .map(|b| downcast(b, operation, "bias is not CUDA storage"))
            .transpose()?;
        let wrap = |e| kernel_error("Cuda", operation, e);

        let unbatched = input.shape.len() == 1;
        let promoted;
        let rows = if unbatched {
            promoted =
                crate::cuda::backend::shape_ops::cuda_reshape_storage(input, &[1, input.shape[0]])
                    .map_err(wrap)?;
            &promoted
        } else {
            input
        };

        let transposed =
            crate::cuda::backend::shape_ops::cuda_transpose_storage(weight, 0, 1).map_err(wrap)?;
        let product = crate::cuda::backend::shape_ops::cuda_matmul_storage(rows, &transposed)
            .map_err(wrap)?;
        let projected = match bias {
            None => product,
            Some(bias) => crate::cuda::backend::elementwise::cuda_add_storage(
                &product,
                bias,
                crate::kernel::KernelSpecialization::NONE,
            )
            .map_err(wrap)?,
        };

        if unbatched {
            let out_dim = projected.shape[projected.shape.len() - 1];
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&projected, &[out_dim])
                .map_err(wrap)
        } else {
            Ok(projected)
        }
    }
}

impl<D: Device> Execute<op::Dropout> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dropout, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Dropout;
        let [input] = request.inputs else {
            return Err(invalid(operation, "dropout expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attributes = request.operation.descriptor().attributes();
        let wrap = |e| kernel_error("Cuda", operation, e);

        if !attributes.training || attributes.probability <= 0.0 {
            return Ok(input.clone());
        }
        if attributes.probability >= 1.0 {
            return crate::cuda::backend::elementwise::cuda_mul_scalar_float(input, 0.0)
                .map_err(wrap);
        }

        let draw = Self::rand::<f32>(
            &input.shape,
            DTypeId::F32.descriptor(),
            &DeviceId::cuda(input.buffer.device_id),
        )
        .map_err(wrap)?;
        let shifted = crate::cuda::backend::elementwise::cuda_add_scalar_float(
            &draw,
            -attributes.probability,
        )
        .map_err(wrap)?;
        let mask = crate::cuda::backend::elementwise::cuda_step_storage(
            &shifted,
            crate::kernel::KernelSpecialization::NONE,
        )
        .map_err(wrap)?;
        let kept = crate::cuda::backend::elementwise::cuda_mul_storage(
            input,
            &mask,
            crate::kernel::KernelSpecialization::NONE,
        )
        .map_err(wrap)?;
        crate::cuda::backend::elementwise::cuda_mul_scalar_float(
            &kept,
            1.0 / (1.0 - attributes.probability),
        )
        .map_err(wrap)
    }
}

impl<D: Device> Execute<op::Conv1dExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv1dExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
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
        let activation = downcast(activation, operation, "activation is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        let bias = bias
            .map(|b| downcast(b, operation, "bias is not CUDA storage"))
            .transpose()?;
        let wrap = |e| kernel_error("Cuda", operation, e);

        let unbatched = activation.shape.len() == 2;
        let (b, cin, len) = if unbatched {
            (1, activation.shape[0], activation.shape[1])
        } else {
            (
                activation.shape[0],
                activation.shape[1],
                activation.shape[2],
            )
        };
        let (cout, cin_g, k_len) = (weight.shape[0], weight.shape[1], weight.shape[2]);

        let act_2d =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(activation, &[b, cin, 1, len])
                .map_err(wrap)?;
        let weight_2d =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(weight, &[cout, cin_g, 1, k_len])
                .map_err(wrap)?;
        let out_2d = CudaBackendImpl::<D>::conv2d::<f32>(
            &act_2d,
            &weight_2d,
            bias,
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.groups,
        )
        .map_err(wrap)?;

        let out_len = out_2d.shape[3];
        if unbatched {
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&out_2d, &[cout, out_len])
                .map_err(wrap)
        } else {
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&out_2d, &[b, cout, out_len])
                .map_err(wrap)
        }
    }
}

impl<D: Device> Execute<op::ConvTranspose2d> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ConvTranspose2d, Self>,
    ) -> Result<CudaStorage, BackendError> {
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
        let activation = downcast(activation, operation, "activation is not CUDA storage")?;
        let weight = downcast(weight, operation, "weight is not CUDA storage")?;
        let bias = bias
            .map(|b| downcast(b, operation, "bias is not CUDA storage"))
            .transpose()?;
        let wrap = |e| kernel_error("Cuda", operation, e);

        let [stride_h, _stride_w] = attributes.stride;
        let [pad_h, _pad_w] = attributes.padding;
        let [out_pad_h, out_pad_w] = attributes.output_padding;
        let [dil_h, _dil_w] = attributes.dilation;

        let unbatched = activation.shape.len() == 3;
        let (b, cin, h, w) = if unbatched {
            (
                1,
                activation.shape[0],
                activation.shape[1],
                activation.shape[2],
            )
        } else {
            (
                activation.shape[0],
                activation.shape[1],
                activation.shape[2],
                activation.shape[3],
            )
        };
        let (_cin_w, cout, kh, kw) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        let h_out = (h - 1) * stride_h + dil_h * (kh - 1) + 1 + out_pad_h - 2 * pad_h;
        let w_out = (w - 1) * stride_h + dil_h * (kw - 1) + 1 + out_pad_w - 2 * pad_h;

        let act_4d = if unbatched {
            crate::cuda::backend::shape_ops::cuda_reshape_storage(activation, &[1, cin, h, w])
                .map_err(wrap)?
        } else {
            activation.clone()
        };

        let w_mat =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(weight, &[cin, cout * kh * kw])
                .map_err(wrap)?;
        let act_flat =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&act_4d, &[b, cin, h * w])
                .map_err(wrap)?;

        let mut batch_cols = Vec::with_capacity(b);
        let w_mat_t =
            crate::cuda::backend::shape_ops::cuda_transpose_storage(&w_mat, 0, 1).map_err(wrap)?;
        for bi in 0..b {
            let act_b = crate::cuda::backend::shape_ops::cuda_narrow_storage(&act_flat, 0, bi, 1)
                .map_err(wrap)?;
            let act_b_sq =
                crate::cuda::backend::shape_ops::cuda_squeeze_storage(&act_b, 0).map_err(wrap)?;
            let cols_b = crate::cuda::backend::shape_ops::cuda_matmul_storage(&w_mat_t, &act_b_sq)
                .map_err(wrap)?;
            let cols_b_unsq = crate::cuda::backend::shape_ops::cuda_unsqueeze_storage(&cols_b, 0)
                .map_err(wrap)?;
            batch_cols.push(cols_b_unsq);
        }
        let cols = if b == 1 {
            batch_cols.into_iter().next().unwrap()
        } else {
            let refs: Vec<&CudaStorage> = batch_cols.iter().collect();
            crate::cuda::backend::shape_ops::cuda_concat_storage(&refs, 0).map_err(wrap)?
        };

        let spec = crate::cuda::ops::conv::Col2Im2dSpec {
            h_out,
            w_out,
            kh,
            kw,
            stride: stride_h,
            padding: pad_h,
            dilation: dil_h,
        };
        let target_shape = vec![b, cout, h_out, w_out];
        let out_4d =
            crate::cuda::ops::conv::launch_col2im_2d(&cols, &target_shape, spec).map_err(wrap)?;

        let with_bias = match bias {
            Some(b_storage) => {
                let b_reshaped = crate::cuda::backend::shape_ops::cuda_reshape_storage(
                    b_storage,
                    &[1, cout, 1, 1],
                )
                .map_err(wrap)?;
                crate::cuda::backend::elementwise::cuda_add_storage(
                    &out_4d,
                    &b_reshaped,
                    crate::kernel::KernelSpecialization::NONE,
                )
                .map_err(wrap)?
            }
            None => out_4d,
        };

        if unbatched {
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&with_bias, &[cout, h_out, w_out])
                .map_err(wrap)
        } else {
            Ok(with_bias)
        }
    }
}

impl<D: Device> Execute<op::AdaptiveAvgPool2dExact> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AdaptiveAvgPool2dExact, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::AdaptiveAvgPool2dExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "adaptive_avg_pool2d expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let [out_h, out_w] = request.operation.descriptor().attributes().output;
        let output_size = (out_h, out_w);
        let wrap = |e| kernel_error("Cuda", operation, e);

        let unbatched = input.shape.len() == 3;
        let promoted;
        let x = if unbatched {
            promoted = crate::cuda::backend::shape_ops::cuda_reshape_storage(
                input,
                &[1, input.shape[0], input.shape[1], input.shape[2]],
            )
            .map_err(wrap)?;
            &promoted
        } else {
            input
        };

        let pooled =
            crate::cuda::ops::pool::launch_adaptive_avg_pool2d(x, output_size).map_err(wrap)?;
        if unbatched {
            crate::cuda::backend::shape_ops::cuda_reshape_storage(
                &pooled,
                &[pooled.shape[1], pooled.shape[2], pooled.shape[3]],
            )
            .map_err(wrap)
        } else {
            Ok(pooled)
        }
    }
}

impl<D: Device> Execute<op::GroupNorm> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::GroupNorm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::GroupNorm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "group_norm expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attributes = request.operation.descriptor().attributes();
        cuda_group_norm_storage(input, attributes.groups, attributes.epsilon)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

impl<D: Device> Execute<op::InstanceNorm> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::InstanceNorm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::InstanceNorm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "instance_norm expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        cuda_instance_norm_storage(input, epsilon).map_err(|e| kernel_error("Cuda", operation, e))
    }
}

pub(crate) fn cuda_group_norm_storage(
    t: &CudaStorage,
    groups: usize,
    eps: f64,
) -> Result<CudaStorage, incin_core::error::Error> {
    let total = t.shape.iter().product::<usize>();
    if groups == 0 {
        return Err(incin_core::error::Error::Msg(
            "group_norm: groups must be non-zero".into(),
        ));
    }
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    if channels % groups != 0 {
        return Err(incin_core::error::Error::Msg(
            "group_norm: channels must be divisible by groups".into(),
        ));
    }
    let (batch, spatial) = if t.shape.len() >= 2 {
        (t.shape[0], t.shape[2..].iter().product::<usize>())
    } else {
        (1, total)
    };
    let group_size = channels / groups * spatial;
    let runs = batch * groups;
    let flat = crate::cuda::backend::shape_ops::cuda_reshape_storage(t, &[runs, group_size])?;
    let mean = crate::cuda::backend::reduce::cuda_mean_dim_keepdim(&flat, 1)?;
    let centered = crate::cuda::backend::elementwise::cuda_sub_storage(
        &flat,
        &mean,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let squared = crate::cuda::backend::elementwise::cuda_mul_storage(
        &centered,
        &centered,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let variance = crate::cuda::backend::reduce::cuda_mean_dim_keepdim(&squared, 1)?;
    let guarded = crate::cuda::backend::elementwise::cuda_add_scalar_float(&variance, eps)?;
    let std = crate::cuda::backend::elementwise::cuda_sqrt_storage(
        &guarded,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let normalized = crate::cuda::backend::elementwise::cuda_div_storage(
        &centered,
        &std,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    crate::cuda::backend::shape_ops::cuda_reshape_storage(&normalized, &t.shape)
}

pub(crate) fn cuda_instance_norm_storage(
    t: &CudaStorage,
    eps: f64,
) -> Result<CudaStorage, incin_core::error::Error> {
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    cuda_group_norm_storage(t, channels, eps)
}

impl<D: Device> Execute<op::IndexSelect> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::IndexSelect, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::IndexSelect;
        let [input, index] = request.inputs else {
            return Err(invalid(operation, "index_select expects 2 inputs"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let index = downcast(index, operation, "index is not CUDA storage")?;
        let axis = request.operation.descriptor().attributes().axis;
        let wrap = |e| kernel_error("Cuda", operation, e);

        let mut idx_expanded_shape = vec![1usize; input.shape.len()];
        idx_expanded_shape[axis] = index.shape.iter().product::<usize>();
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = idx_expanded_shape[axis];
        let idx_reshaped =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(index, &idx_expanded_shape)
                .map_err(wrap)?;
        let idx_broadcasted =
            crate::cuda::backend::shape_ops::cuda_broadcast_as_storage(&idx_reshaped, &out_shape)
                .map_err(wrap)?;
        crate::cuda::ops::shape::launch_gather(input, axis, &idx_broadcasted).map_err(wrap)
    }
}

impl<D: Device> Execute<op::PixelShuffle> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::PixelShuffle, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::PixelShuffle;
        let [input] = request.inputs else {
            return Err(invalid(operation, "pixel_shuffle expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let r = request.operation.descriptor().attributes().upscale_factor;
        let wrap = |e| kernel_error("Cuda", operation, e);

        if input.shape.len() != 4 {
            return Err(invalid(
                operation,
                "pixel_shuffle expects 4D tensor (N, C, H, W)",
            ));
        }
        let (n, c, h, w) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let r_sq = r * r;
        if c % r_sq != 0 {
            return Err(invalid(
                operation,
                "pixel_shuffle channels must be divisible by upscale_factor^2",
            ));
        }
        let out_c = c / r_sq;
        let out_h = h * r;
        let out_w = w * r;

        let s1 =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(input, &[n, out_c, r, r, h, w])
                .map_err(wrap)?;
        let p1 =
            crate::cuda::backend::shape_ops::cuda_transpose_storage(&s1, 2, 4).map_err(wrap)?;
        let p2 =
            crate::cuda::backend::shape_ops::cuda_transpose_storage(&p1, 3, 4).map_err(wrap)?;
        let p3 =
            crate::cuda::backend::shape_ops::cuda_transpose_storage(&p2, 4, 5).map_err(wrap)?;
        crate::cuda::backend::shape_ops::cuda_reshape_storage(&p3, &[n, out_c, out_h, out_w])
            .map_err(wrap)
    }
}

impl<D: Device> Execute<op::Unfold> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Unfold, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Unfold;
        let [input] = request.inputs else {
            return Err(invalid(operation, "unfold expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attributes = request.operation.descriptor().attributes();
        let (axis, size, step) = (attributes.axis, attributes.size, attributes.step);
        let wrap = |e| kernel_error("Cuda", operation, e);

        let dim_len = input.shape[axis];
        if size > dim_len {
            return Err(invalid(
                operation,
                "unfold size cannot exceed dimension length",
            ));
        }
        let n_windows = (dim_len - size) / step + 1;
        let mut window_slices = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let win =
                crate::cuda::backend::shape_ops::cuda_narrow_storage(input, axis, i * step, size)
                    .map_err(wrap)?;
            let win_unsq = crate::cuda::backend::shape_ops::cuda_unsqueeze_storage(&win, axis)
                .map_err(wrap)?;
            window_slices.push(win_unsq);
        }
        let refs: Vec<&CudaStorage> = window_slices.iter().collect();
        let joined =
            crate::cuda::backend::shape_ops::cuda_concat_storage(&refs, axis).map_err(wrap)?;
        let mut curr = joined;
        for d in (axis + 1)..(curr.shape.len() - 1) {
            curr = crate::cuda::backend::shape_ops::cuda_transpose_storage(&curr, d, d + 1)
                .map_err(wrap)?;
        }
        Ok(curr)
    }
}

impl<D: Device> Execute<op::ToDType> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ToDType, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::ToDType;
        let [input] = request.inputs else {
            return Err(invalid(operation, "to_dtype expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let target_dtype = request.operation.descriptor().attributes().dtype;
        if input.buffer.dtype == target_dtype {
            return Ok(input.clone());
        }
        let total = input.shape.iter().product::<usize>();
        let byte_len = crate::bytes::byte_len(target_dtype, total, operation)
            .map_err(|e| kernel_error("Cuda", operation, e))?;
        let stream = input.buffer.device.default_stream();
        let out_buffer = CudaBuffer {
            len: total,
            dtype: target_dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                kernel_error(
                    "Cuda",
                    operation,
                    incin_core::error::Error::Msg(format!("{e:?}")),
                )
            })?),
            device: input.buffer.device.clone(),
            device_id: input.buffer.device_id,
        };
        Ok(CudaStorage::new(Arc::new(out_buffer), input.shape.to_vec()))
    }
}

fn cuda_loss_reduction(reduction: LossReduction) -> Reduction {
    match reduction {
        LossReduction::None => Reduction::None,
        LossReduction::Mean => Reduction::Mean,
        LossReduction::Sum => Reduction::Sum,
    }
}

fn cuda_reduce_loss(
    t: CudaStorage,
    reduction: incin_core::tensor::reduction::Reduction,
) -> Result<CudaStorage, incin_core::error::Error> {
    match reduction {
        incin_core::tensor::reduction::Reduction::Mean => {
            crate::cuda::backend::reduce::cuda_mean_all_storage(&t)
        }
        incin_core::tensor::reduction::Reduction::Sum => {
            crate::cuda::backend::reduce::cuda_sum_all_storage(&t)
        }
        incin_core::tensor::reduction::Reduction::None => Ok(t),
    }
}

macro_rules! cuda_loss_executors {
    ($(($operation:ident, $func:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CudaBackendImpl<D> {
            type Output = CudaStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CudaStorage, BackendError> {
                let operation = OperationKind::$operation;
                let [pred, target] = request.inputs else {
                    return Err(invalid(operation, "loss expects 2 inputs"));
                };
                let pred = downcast(pred, operation, "pred is not CUDA storage")?;
                let target = downcast(target, operation, "target is not CUDA storage")?;
                let reduction = cuda_loss_reduction(request.operation.descriptor().attributes().reduction);
                $func(pred, target, reduction)
                    .map_err(|error| kernel_error("Cuda", operation, error))
            }
        }
    )*};
}

pub(crate) fn cuda_mse_loss_storage(
    pred: &CudaStorage,
    target: &CudaStorage,
    reduction: incin_core::tensor::reduction::Reduction,
) -> Result<CudaStorage, incin_core::error::Error> {
    let diff = crate::cuda::backend::elementwise::cuda_sub_storage(
        pred,
        target,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let squared = crate::cuda::backend::elementwise::cuda_mul_storage(
        &diff,
        &diff,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    cuda_reduce_loss(squared, reduction)
}

pub(crate) fn cuda_l1_loss_storage(
    pred: &CudaStorage,
    target: &CudaStorage,
    reduction: incin_core::tensor::reduction::Reduction,
) -> Result<CudaStorage, incin_core::error::Error> {
    let diff = crate::cuda::backend::elementwise::cuda_sub_storage(
        pred,
        target,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let absolute = crate::cuda::backend::elementwise::cuda_abs_storage(
        &diff,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    cuda_reduce_loss(absolute, reduction)
}

pub(crate) fn cuda_bce_with_logits_loss_storage(
    pred: &CudaStorage,
    target: &CudaStorage,
    reduction: incin_core::tensor::reduction::Reduction,
) -> Result<CudaStorage, incin_core::error::Error> {
    let max_x_0 = crate::cuda::backend::elementwise::cuda_relu_storage(
        pred,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let x_times_z = crate::cuda::backend::elementwise::cuda_mul_storage(
        pred,
        target,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let term1 = crate::cuda::backend::elementwise::cuda_sub_storage(
        &max_x_0,
        &x_times_z,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let abs_x = crate::cuda::backend::elementwise::cuda_abs_storage(
        pred,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let neg_abs_x = crate::cuda::backend::elementwise::cuda_neg_storage(
        &abs_x,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let exp_neg_abs_x = crate::cuda::backend::elementwise::cuda_exp_storage(
        &neg_abs_x,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let one_plus_exp =
        crate::cuda::backend::elementwise::cuda_add_scalar_float(&exp_neg_abs_x, 1.0)?;
    let log_term = crate::cuda::backend::elementwise::cuda_log_storage(
        &one_plus_exp,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    let unreduced = crate::cuda::backend::elementwise::cuda_add_storage(
        &term1,
        &log_term,
        crate::kernel::KernelSpecialization::NONE,
    )?;
    cuda_reduce_loss(unreduced, reduction)
}

cuda_loss_executors![
    (MseLoss, cuda_mse_loss_storage),
    (L1Loss, cuda_l1_loss_storage),
    (BceWithLogitsLoss, cuda_bce_with_logits_loss_storage),
];

impl<D: Device> Execute<op::CrossEntropyLoss> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::CrossEntropyLoss, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::CrossEntropyLoss;
        let [logits, target] = request.inputs else {
            return Err(invalid(operation, "cross_entropy_loss expects 2 inputs"));
        };
        let logits = downcast(logits, operation, "logits is not CUDA storage")?;
        let target = downcast(target, operation, "target is not CUDA storage")?;
        let reduction = cuda_loss_reduction(request.operation.descriptor().attributes().reduction);
        let wrap = |e| kernel_error("Cuda", operation, e);

        let last_dim = logits.shape.len() - 1;
        let log_probs = crate::cuda::backend::elementwise::cuda_log_softmax::<D>(logits, last_dim)
            .map_err(wrap)?;
        let mut target_exp_shape = target.shape.to_vec();
        target_exp_shape.push(1);
        let target_exp =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(target, &target_exp_shape)
                .map_err(wrap)?;
        let gathered = crate::cuda::ops::shape::launch_gather(&log_probs, last_dim, &target_exp)
            .map_err(wrap)?;
        let neg_gathered = crate::cuda::backend::elementwise::cuda_neg_storage(
            &gathered,
            crate::kernel::KernelSpecialization::NONE,
        )
        .map_err(wrap)?;
        let squeezed_shape = target.shape.to_vec();
        let nll =
            crate::cuda::backend::shape_ops::cuda_reshape_storage(&neg_gathered, &squeezed_shape)
                .map_err(wrap)?;
        cuda_reduce_loss(nll, reduction).map_err(wrap)
    }
}

impl<D: Device> Execute<op::Norm> for CudaBackendImpl<D> {
    type Output = CudaStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Norm, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Norm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "norm expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let order = request.operation.descriptor().attributes().order;
        let wrap = |e| kernel_error("Cuda", operation, e);

        const NORM_ORDER_TOLERANCE: f64 = 1e-6;
        if (order - 1.0).abs() < NORM_ORDER_TOLERANCE {
            let magnitude = crate::cuda::backend::elementwise::cuda_abs_storage(
                input,
                crate::kernel::KernelSpecialization::NONE,
            )
            .map_err(wrap)?;
            return crate::cuda::backend::reduce::cuda_sum_all_storage(&magnitude).map_err(wrap);
        }
        if (order - 2.0).abs() < NORM_ORDER_TOLERANCE {
            let squared = crate::cuda::backend::elementwise::cuda_mul_storage(
                input,
                input,
                crate::kernel::KernelSpecialization::NONE,
            )
            .map_err(wrap)?;
            let summed =
                crate::cuda::backend::reduce::cuda_sum_all_storage(&squared).map_err(wrap)?;
            return crate::cuda::backend::elementwise::cuda_sqrt_storage(
                &summed,
                crate::kernel::KernelSpecialization::NONE,
            )
            .map_err(wrap);
        }
        let magnitude = crate::cuda::backend::elementwise::cuda_abs_storage(
            input,
            crate::kernel::KernelSpecialization::NONE,
        )
        .map_err(wrap)?;
        let raised = crate::cuda::backend::elementwise::cuda_powf_storage(&magnitude, order)
            .map_err(wrap)?;
        let summed = crate::cuda::backend::reduce::cuda_sum_all_storage(&raised).map_err(wrap)?;
        crate::cuda::backend::elementwise::cuda_powf_storage(&summed, 1.0 / order).map_err(wrap)
    }
}

impl<D: Device> Execute<op::Argsort> for CudaBackendImpl<D> {
    type Output = CudaStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Argsort, Self>,
    ) -> Result<CudaStorage, BackendError> {
        let operation = OperationKind::Argsort;
        let [input] = request.inputs else {
            return Err(invalid(operation, "argsort expects 1 input"));
        };
        let input = downcast(input, operation, "input is not CUDA storage")?;
        let attrs = request.operation.descriptor().attributes();
        CudaBackendImpl::<D>::argsort::<i64>(input, attrs.axis, attrs.descending)
            .map_err(|e| kernel_error("Cuda", operation, e))
    }
}

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
