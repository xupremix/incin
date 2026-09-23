//! Descriptor execution for the Metal backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend, op};
use incin_core::error::BackendError;
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{Device, DeviceKind};

use super::backend::MetalBackendImpl;
use super::storage::MetalStorage;
use crate::descriptor_bind::{invalid, kernel_error};

impl<D: Device> Capabilities for MetalBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Metal, query)
    }
}

impl_creation_executors!(MetalBackendImpl<D>, MetalStorage);
impl_data_creation_executors!(MetalBackendImpl<D>, MetalStorage);
impl_variable_creation_executors!(MetalBackendImpl<D>, crate::metal::MetalVar);
impl_readback_executors!(MetalBackendImpl<D>, MetalStorage);

macro_rules! impl_metal_canonical {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
                let rhs = rhs.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
                Self::$method::<f32>(lhs, rhs)
                    .map_err(|error| kernel_error("Metal", operation, error))
            }
        }
    )*};
}

impl_metal_canonical![
    (Add, add),
    (Sub, sub),
    (Mul, mul),
    (Div, div),
    // The three binary floats whose backward is CPU's own recipe (the
    // quotient rule for `atan2`, `record_modulus` for the two residues).
    // Same request shape as the four above: two operands, no attributes.
    (Atan2, atan2),
    (Fmod, fmod),
    (Remainder, remainder),
];

/// Unary floats with no attributes: one operand, one `Self::$method` call.
/// Written once so the pointwise methods in `metal/pointwise.rs` and the
/// capability rows that name them cannot drift apart — the same pattern
/// `wgpu_unary_float_operations!` uses for its WGSL modes.
macro_rules! impl_metal_unary_float {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "unary operation expects 1 input"));
                };
                let input = input
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
                Self::$method::<f32>(input)
                    .map_err(|error| kernel_error("Metal", operation, error))
            }
        }
    )*};
}

impl_metal_unary_float![
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

/// One operand and one `f64` attribute: the four scalar forms plus `powf`.
/// Mirrors CUDA's `impl_cuda_scalar_tensor!` and WGPU's
/// `impl_wgpu_scalar_tensor!` in arity and attribute read.
macro_rules! impl_metal_scalar_tensor {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly one operand"));
                };
                let input = input
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
                let value = request.operation.descriptor().attributes().value;
                Self::$method::<f32>(input, value)
                    .map_err(|error| kernel_error("Metal", operation, error))
            }
        }
    )*};
}

impl_metal_scalar_tensor![
    (AddScalar, add_scalar_float),
    (SubScalar, sub_scalar_float),
    (MulScalar, mul_scalar_float),
    (DivScalar, div_scalar_float),
    (Powf, powf),
];

/// One operand and an ordered `min`/`max` pair, so this fits neither the
/// unary macro (no attributes) nor the scalar one (two bounds, not one).
impl<D: Device> Execute<op::Clamp> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Clamp, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Clamp;
        let [input] = request.inputs else {
            return Err(invalid(operation, "clamp expects exactly one operand"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
        let attributes = request.operation.descriptor().attributes();
        Self::clamp::<f32>(input, attributes.min, attributes.max)
            .map_err(|error| kernel_error("Metal", operation, error))
    }
}

impl<D: Device> Execute<op::ReshapeExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ReshapeExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::ReshapeExact,
                "reshape expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::ReshapeExact, "input is not Metal storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        Self::reshape::<f32>(storage, shape)
            .map_err(|e| kernel_error("Metal", OperationKind::ReshapeExact, e))
    }
}

impl<D: Device> Execute<op::BroadcastAs> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BroadcastAs, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::BroadcastAs,
                "broadcast expects 1 input",
            ));
        };
        let storage = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::BroadcastAs, "input is not Metal storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        Self::broadcast_as::<f32>(storage, shape)
            .map_err(|e| kernel_error("Metal", OperationKind::BroadcastAs, e))
    }
}

impl<D: Device> Execute<op::MatMulExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMulExact,
                "matmul expects 2 inputs",
            ));
        };
        let lhs = lhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "lhs is not Metal storage"))?;
        let rhs = rhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMulExact, "rhs is not Metal storage"))?;
        Self::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Metal", OperationKind::MatMulExact, e))
    }
}

impl<D: Device> Execute<op::Conv2dExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input, weight] = request.inputs else {
            return Err(invalid(
                OperationKind::Conv2dExact,
                "conv2d expects 2 inputs",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "input is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::Conv2dExact, "weight is not Metal storage"))?;
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
        .map_err(|e| kernel_error("Metal", OperationKind::Conv2dExact, e))
    }
}

impl<D: Device> Execute<op::MaxPool2d> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaxPool2d, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::MaxPool2d,
                "max_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::MaxPool2d, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        Self::max_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
            pair(attrs.dilation),
        )
        .map_err(|e| kernel_error("Metal", OperationKind::MaxPool2d, e))
    }
}

impl<D: Device> Execute<op::AvgPool2d> for MetalBackendImpl<D> {
    type Output = MetalStorage;
    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AvgPool2d, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let [input] = request.inputs else {
            return Err(invalid(
                OperationKind::AvgPool2d,
                "avg_pool2d expects 1 input",
            ));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(OperationKind::AvgPool2d, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        let pair = |[h, w]: [usize; 2]| (h, w);
        Self::avg_pool2d::<f32>(
            input,
            pair(attrs.kernel),
            pair(attrs.stride),
            pair(attrs.padding),
        )
        .map_err(|e| kernel_error("Metal", OperationKind::AvgPool2d, e))
    }
}

macro_rules! impl_metal_reduction_all {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not Metal storage"))?;
                $func(input).map_err(|e| kernel_error("Metal", OperationKind::$op, e))
            }
        }
    )*};
}

macro_rules! impl_metal_reduction_dim {
    ($(($op:ident, $func:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;
            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let [input] = request.inputs else {
                    return Err(invalid(OperationKind::$op, "reduction expects 1 input"));
                };
                let input = input.downcast_ref::<MetalStorage>().ok_or_else(|| invalid(OperationKind::$op, "input is not Metal storage"))?;
                let axis = request.operation.descriptor().attributes().axis;
                $func(input, axis).map_err(|e| kernel_error("Metal", OperationKind::$op, e))
            }
        }
    )*};
}

impl_metal_reduction_all![
    (SumAll, MetalBackendImpl::<D>::sum_all::<f32>),
    (MeanAll, MetalBackendImpl::<D>::mean_all::<f32>),
];

impl_metal_reduction_dim![
    (SumDim, |input, axis| {
        MetalBackendImpl::<D>::sum_dim::<f32>(input, axis)
    }),
    (SumKeepDim, |input, axis| {
        MetalBackendImpl::<D>::sum_keepdim::<f32>(input, axis)
    }),
    (MeanDim, |input, axis| {
        MetalBackendImpl::<D>::mean_dim::<f32>(input, axis)
    }),
    (MeanKeepDim, |input, axis| {
        MetalBackendImpl::<D>::mean_keepdim::<f32>(input, axis)
    }),
    // Not a reduction: `softmax` maps an axis rather than collapsing it.
    // It rides this macro because its request shape — one operand plus an
    // axis — is identical, exactly as on WGPU.
    (Softmax, |input, axis| {
        MetalBackendImpl::<D>::softmax::<f32>(input, axis)
    }),
    // Same request shape, same tape-honest composition.
    (LogSoftmax, |input, axis| {
        MetalBackendImpl::<D>::log_softmax::<f32>(input, axis)
    }),
    // Also not reductions: both are views that add or drop a unit axis, and
    // both read the same `axis` attribute the reductions do (WGPU hosts
    // these two on the same macro for the same reason).
    (SqueezeExact, |input, axis| {
        MetalBackendImpl::<D>::squeeze::<f32>(input, axis)
    }),
    (UnsqueezeExact, |input, axis| {
        MetalBackendImpl::<D>::unsqueeze::<f32>(input, axis)
    }),
];

/// `transpose` reads an attribute *pair* rather than a single `axis`, so it
/// fits neither the axis macro above nor the unary/scalar ones.
impl<D: Device> Execute<op::TransposeExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TransposeExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::TransposeExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "transpose expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attributes = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::transpose::<f32>(input, attributes.first, attributes.second)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `narrow` carries a full window triple, so it cannot ride the axis macro.
impl<D: Device> Execute<op::Narrow> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Narrow, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Narrow;
        let [input] = request.inputs else {
            return Err(invalid(operation, "narrow expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attributes = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::narrow::<f32>(
            input,
            attributes.axis,
            attributes.start,
            attributes.length,
        )
        .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `slice` carries one `(start, end)` range per axis.
impl<D: Device> Execute<op::SliceExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::SliceExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::SliceExact;
        let [input] = request.inputs else {
            return Err(invalid(operation, "slice expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let ranges = &request.operation.descriptor().attributes().ranges;
        MetalBackendImpl::<D>::slice_exact::<f32>(input, ranges)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `concat` is variadic: every operand is downcast, then one host walk
/// joins them (WGPU's variadic shape, with Metal's error wording).
impl<D: Device> Execute<op::ConcatExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ConcatExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::ConcatExact;
        if request.inputs.is_empty() {
            return Err(invalid(operation, "concat expects at least 1 input"));
        }
        let mut operands = Vec::with_capacity(request.inputs.len());
        for handle in request.inputs {
            let storage = handle
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
            operands.push(storage);
        }
        let axis = request.operation.descriptor().attributes().axis;
        let refs: Vec<&_> = operands.to_vec();
        MetalBackendImpl::<D>::concat_exact::<f32>(&refs, axis)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `stack` is variadic over the same request shape as `concat`.
impl<D: Device> Execute<op::StackExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::StackExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::StackExact;
        if request.inputs.is_empty() {
            return Err(invalid(operation, "stack expects at least 1 input"));
        }
        let mut operands = Vec::with_capacity(request.inputs.len());
        for handle in request.inputs {
            let storage = handle
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| invalid(operation, "operand is not Metal storage"))?;
            operands.push(storage);
        }
        let axis = request.operation.descriptor().attributes().axis;
        let refs: Vec<&_> = operands.to_vec();
        MetalBackendImpl::<D>::stack_exact::<f32>(&refs, axis)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// Two operands and an `epsilon`, so this fits neither the axis macro nor
/// the canonical binary one. Composed from primitives Metal already
/// advertises, in the same order as CPU's, CUDA's and WGPU's.
impl<D: Device> Execute<op::RmsNorm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::RmsNorm, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::RmsNorm;
        let [input, weight] = request.inputs else {
            return Err(invalid(operation, "rms norm expects an input and a weight"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "weight is not Metal storage"))?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        MetalBackendImpl::<D>::rms_norm::<f32>(input, weight, epsilon)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `layer_norm` takes input, weight and an optional bias — arity the axis
/// macro cannot express. The bias's presence is read from the operand count
/// (the descriptor has already validated it against `has_bias`).
impl<D: Device> Execute<op::LayerNorm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LayerNorm, Self>,
    ) -> Result<MetalStorage, BackendError> {
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
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "weight is not Metal storage"))?;
        let bias = bias
            .map(|bias| {
                bias.downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "bias is not Metal storage"))
            })
            .transpose()?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        MetalBackendImpl::<D>::layer_norm::<f32>(input, weight, bias, epsilon)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `tril`/`triu` read one `i64` diagonal offset, so they fit neither the
/// axis macro (a `usize` axis) nor the unary one (no attributes). Mirrors
/// WGPU's `impl_wgpu_triangular!` in arity and attribute read.
macro_rules! impl_metal_triangular {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input] = request.inputs else {
                    return Err(invalid(operation, "expects exactly 1 input"));
                };
                let input = input
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
                let offset = request.operation.descriptor().attributes().offset;
                MetalBackendImpl::<D>::$method::<f32>(input, offset)
                    .map_err(|e| kernel_error("Metal", operation, e))
            }
        }
    )*};
}

impl_metal_triangular![(Tril, tril), (Triu, triu)];

/// `dropout` is one operand plus the `(probability, training)` pair — the
/// same request shape WGPU's executor answers.
impl<D: Device> Execute<op::Dropout> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dropout, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Dropout;
        let [input] = request.inputs else {
            return Err(invalid(operation, "dropout expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attributes = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::dropout::<f32>(input, attributes.probability, attributes.training)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `linear`: an input, a weight and an optional bias — the same operand
/// split LayerNorm above reads from the count, validated against
/// `has_bias` by the descriptor before execution.
impl<D: Device> Execute<op::Linear> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Linear, Self>,
    ) -> Result<MetalStorage, BackendError> {
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
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "weight is not Metal storage"))?;
        let bias = bias
            .map(|bias| {
                bias.downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "bias is not Metal storage"))
            })
            .transpose()?;
        MetalBackendImpl::<D>::linear::<f32>(input, weight, bias)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `scaled_dot_product_attention`: q, k, v and an optional additive mask.
/// The attribute set says whether a mask is present, so the operand count
/// and the declared contract have to agree before anything runs — WGPU's
/// check, verbatim in intent.
impl<D: Device> Execute<op::ScaledDotProductAttention> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ScaledDotProductAttention, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::ScaledDotProductAttention;
        let attributes = request.operation.descriptor().attributes();
        let (q, k, v, mask) = match request.inputs {
            [q, k, v] if !attributes.has_mask => (q, k, v, None),
            [q, k, v, mask] if attributes.has_mask => (q, k, v, Some(mask)),
            _ => {
                return Err(invalid(
                    operation,
                    "operand count does not match the declared mask",
                ));
            }
        };
        let q = q
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "q is not Metal storage"))?;
        let k = k
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "k is not Metal storage"))?;
        let v = v
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "v is not Metal storage"))?;
        let mask = mask
            .map(|mask| {
                mask.downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "mask is not Metal storage"))
            })
            .transpose()?;
        MetalBackendImpl::<D>::scaled_dot_product_attention::<f32>(q, k, v, mask, attributes.scale)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

macro_rules! assert_every_advertised_metal_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, MetalBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::tensor::device::Metal>();
        };
    };
}

crate::capability::metal_descriptor_operations!(assert_every_advertised_metal_row_executes,);
