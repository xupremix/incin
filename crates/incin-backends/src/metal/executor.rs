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
    // A scan, not a collapse: same one-operand-plus-axis request shape.
    (Cumsum, |input, axis| {
        MetalBackendImpl::<D>::cumsum::<f32>(input, axis)
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

/// Map the catalog's `LossReduction` onto the host-side `Reduction` the
/// loss methods take — WGPU's `wgpu_loss_reduction`, restated for Metal.
fn metal_loss_reduction(
    reduction: incin_core::exec::catalog::LossReduction,
) -> incin_core::tensor::reduction::Reduction {
    match reduction {
        incin_core::exec::catalog::LossReduction::None => {
            incin_core::tensor::reduction::Reduction::None
        }
        incin_core::exec::catalog::LossReduction::Mean => {
            incin_core::tensor::reduction::Reduction::Mean
        }
        incin_core::exec::catalog::LossReduction::Sum => {
            incin_core::tensor::reduction::Reduction::Sum
        }
    }
}

/// Pairwise losses: prediction, target and a `reduction` attribute — the
/// same request shape WGPU's `impl_wgpu_loss!` answers.
macro_rules! impl_metal_loss {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [pred, target] = request.inputs else {
                    return Err(invalid(operation, "expects exactly 2 inputs"));
                };
                let pred = pred
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "prediction is not Metal storage"))?;
                let target = target
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "target is not Metal storage"))?;
                let reduction = metal_loss_reduction(
                    request.operation.descriptor().attributes().reduction,
                );
                MetalBackendImpl::<D>::$method::<f32>(pred, target, reduction)
                    .map_err(|e| kernel_error("Metal", operation, e))
            }
        }
    )*};
}

impl_metal_loss![
    (MseLoss, mse_loss),
    (L1Loss, l1_loss),
    (BceWithLogitsLoss, bce_with_logits_loss)
];

/// `cross_entropy_loss` takes logits (f32) and a class-index target (i64),
/// so it cannot ride the f32-only pairwise loss macro.
impl<D: Device> Execute<op::CrossEntropyLoss> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::CrossEntropyLoss, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::CrossEntropyLoss;
        let [logits, target] = request.inputs else {
            return Err(invalid(operation, "cross_entropy_loss expects 2 inputs"));
        };
        let logits = logits
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "logits is not Metal storage"))?;
        let target = target
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "target is not Metal storage"))?;
        let reduction = metal_loss_reduction(request.operation.descriptor().attributes().reduction);
        MetalBackendImpl::<D>::cross_entropy_loss(logits, target, reduction)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// Whole-tensor variance/std: one operand plus an `unbiased` flag.
macro_rules! impl_metal_variance_all {
    ($(($op:ident, $square_root:literal)),* $(,)?) => {$(
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
                let unbiased = request.operation.descriptor().attributes().unbiased;
                if $square_root {
                    MetalBackendImpl::<D>::std_all::<f32>(input, unbiased)
                } else {
                    MetalBackendImpl::<D>::variance_all::<f32>(input, unbiased)
                }
                .map_err(|e| kernel_error("Metal", operation, e))
            }
        }
    )*};
}

impl_metal_variance_all![(VarianceAll, false), (StdAll, true)];

/// Axis variance/std: one operand plus the `(axis, unbiased)` pair.
macro_rules! impl_metal_variance_axis {
    ($(($op:ident, $keepdim:literal, $square_root:literal)),* $(,)?) => {$(
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
                let attributes = request.operation.descriptor().attributes();
                let (axis, unbiased) = (attributes.axis, attributes.unbiased);
                match ($keepdim, $square_root) {
                    (false, false) => {
                        MetalBackendImpl::<D>::variance_dim::<f32>(input, axis, unbiased)
                    }
                    (false, true) => MetalBackendImpl::<D>::std_dim::<f32>(input, axis, unbiased),
                    (true, false) => {
                        MetalBackendImpl::<D>::variance_keepdim::<f32>(input, axis, unbiased)
                    }
                    (true, true) => {
                        MetalBackendImpl::<D>::std_keepdim::<f32>(input, axis, unbiased)
                    }
                }
                .map_err(|e| kernel_error("Metal", operation, e))
            }
        }
    )*};
}

impl_metal_variance_axis![
    (VarianceDim, false, false),
    (VarianceKeepDim, true, false),
    (StdDim, false, true),
    (StdKeepDim, true, true),
];

/// `norm` reads an `order: f64`, so it fits neither the reduction macros
/// (no attributes / axis only) nor the unary one.
impl<D: Device> Execute<op::Norm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Norm, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Norm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "norm expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let order = request.operation.descriptor().attributes().order;
        MetalBackendImpl::<D>::norm::<f32>(input, order)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `embedding` is an index tensor and a weight table — no axis attribute.
impl<D: Device> Execute<op::EmbeddingExact> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::EmbeddingExact, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::EmbeddingExact;
        let [indices, weight] = request.inputs else {
            return Err(invalid(
                operation,
                "embedding expects an index tensor and a weight table",
            ));
        };
        let indices = indices
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "indices is not Metal storage"))?;
        let weight = weight
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "weight is not Metal storage"))?;
        MetalBackendImpl::<D>::embedding(indices, weight)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `gather`/`index_select`: input, index and an `axis` — the same request
/// shape WGPU's indexing Execute impls answer.
macro_rules! impl_metal_index_axis {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for MetalBackendImpl<D> {
            type Output = MetalStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> Result<MetalStorage, BackendError> {
                let operation = OperationKind::$op;
                let [input, index] = request.inputs else {
                    return Err(invalid(operation, "expects exactly 2 inputs"));
                };
                let input = input
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
                let index = index
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "index is not Metal storage"))?;
                let axis = request.operation.descriptor().attributes().axis;
                MetalBackendImpl::<D>::$method(input, axis, index)
                    .map_err(|e| kernel_error("Metal", operation, e))
            }
        }
    )*};
}

impl_metal_index_axis![(Gather, gather), (IndexSelect, index_select)];

/// `batch_norm`: presence-flag operand split (input, optional weight/bias,
/// and — for inference — a running mean/variance pair), exactly as WGPU's
/// executor reads the same `BatchNormAttributes`.
impl<D: Device> Execute<op::BatchNorm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchNorm, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::BatchNorm;
        let attributes = request.operation.descriptor().attributes();
        let Some((input, optional)) = request.inputs.split_first() else {
            return Err(invalid(operation, "batch norm expects at least the input"));
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
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let weight = weight
            .map(|h| {
                h.downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "weight is not Metal storage"))
            })
            .transpose()?;
        let bias = bias
            .map(|h| {
                h.downcast_ref::<MetalStorage>()
                    .ok_or_else(|| invalid(operation, "bias is not Metal storage"))
            })
            .transpose()?;
        if attributes.training {
            return MetalBackendImpl::<D>::batch_norm_training::<f32>(
                input,
                weight,
                bias,
                attributes.epsilon,
            )
            .map_err(|e| kernel_error("Metal", operation, e));
        }
        let (Some(running_mean), Some(running_variance)) = (running_mean, running_variance) else {
            return Err(invalid(
                operation,
                "inference batch norm needs a running mean and a running variance",
            ));
        };
        let running_mean = running_mean
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "running mean is not Metal storage"))?;
        let running_variance = running_variance
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "running variance is not Metal storage"))?;
        MetalBackendImpl::<D>::batch_norm_inference::<f32>(
            input,
            weight,
            bias,
            running_mean,
            running_variance,
            attributes.epsilon,
        )
        .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `group_norm`: one operand plus `(groups, epsilon)`.
impl<D: Device> Execute<op::GroupNorm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::GroupNorm, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::GroupNorm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "group norm expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attributes = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::group_norm::<f32>(input, attributes.groups, attributes.epsilon)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `instance_norm`: one operand plus `epsilon` — `group_norm` with one
/// group per channel (`metal/normalization.rs`).
impl<D: Device> Execute<op::InstanceNorm> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::InstanceNorm, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::InstanceNorm;
        let [input] = request.inputs else {
            return Err(invalid(operation, "instance norm expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let epsilon = request.operation.descriptor().attributes().epsilon;
        MetalBackendImpl::<D>::instance_norm::<f32>(input, epsilon)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `pad`: one operand plus `(padding, value)` — host walk in `metal/layout.rs`.
impl<D: Device> Execute<op::Pad> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Pad, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Pad;
        let [input] = request.inputs else {
            return Err(invalid(operation, "pad expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::pad::<f32>(input, &attrs.padding, attrs.value)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `repeat`: one operand plus `repeats` — host walk in `metal/layout.rs`.
impl<D: Device> Execute<op::Repeat> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Repeat, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Repeat;
        let [input] = request.inputs else {
            return Err(invalid(operation, "repeat expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let repeats = &request.operation.descriptor().attributes().repeats;
        MetalBackendImpl::<D>::repeat::<f32>(input, repeats)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `scatter`: input, index and source — ternary host walk in `metal/indexing.rs`.
impl<D: Device> Execute<op::Scatter> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Scatter, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::Scatter;
        let [input, index, source] = request.inputs else {
            return Err(invalid(operation, "scatter expects 3 inputs"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let index = index
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "index is not Metal storage"))?;
        let source = source
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "source is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::scatter(input, attrs.axis, index, source, attrs.duplicate_indices)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `one_hot`: index tensor plus `depth` — bool-mask host walk in
/// `metal/indexing.rs`. Forward-only (no tape entry).
impl<D: Device> Execute<op::OneHot> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::OneHot, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::OneHot;
        let [indices] = request.inputs else {
            return Err(invalid(operation, "one_hot expects exactly 1 input"));
        };
        let indices = indices
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "indices is not Metal storage"))?;
        let depth = request.operation.descriptor().attributes().depth;
        MetalBackendImpl::<D>::one_hot(indices, depth)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `argmax`: one operand plus an optional `axis` — i64 index result, no tape.
impl<D: Device> Execute<op::ArgMax> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ArgMax, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::ArgMax;
        let [input] = request.inputs else {
            return Err(invalid(operation, "argmax expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::argmax(input, attrs.axis)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `sort`: one operand plus `(axis, descending, index_dtype)` — returns the
/// sorted values and the permutation, both with the operand's geometry.
impl<D: Device> Execute<op::Sort> for MetalBackendImpl<D> {
    type Output = (MetalStorage, MetalStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Sort, Self>,
    ) -> Result<(MetalStorage, MetalStorage), BackendError> {
        let operation = OperationKind::Sort;
        let [input] = request.inputs else {
            return Err(invalid(operation, "sort expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::sort(input, attrs.axis, attrs.descending)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `topk`: one operand plus `(k, axis, largest, index_dtype)` — values and
/// indices with the axis shrunk to `k`.
impl<D: Device> Execute<op::TopK> for MetalBackendImpl<D> {
    type Output = (MetalStorage, MetalStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TopK, Self>,
    ) -> Result<(MetalStorage, MetalStorage), BackendError> {
        let operation = OperationKind::TopK;
        let [input] = request.inputs else {
            return Err(invalid(operation, "topk expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let attrs = request.operation.descriptor().attributes();
        MetalBackendImpl::<D>::topk(input, attrs.k, attrs.axis, attrs.largest)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `to_dtype`: one operand plus a target dtype — bidirectional host cast in
/// `metal/convert.rs`.
impl<D: Device> Execute<op::ToDType> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ToDType, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::ToDType;
        let [input] = request.inputs else {
            return Err(invalid(operation, "to_dtype expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "input is not Metal storage"))?;
        let target = request.operation.descriptor().attributes().dtype;
        MetalBackendImpl::<D>::to_dtype(input, target)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// `batched_matmul`: two operands, rewritten into the already-taped
/// `Self::matmul` (the same path `MatMulExact` rides; `matmul_metal` handles
/// equal-batch and unbatched-rhs broadcasting, and the uneven multi-dim
/// broadcast case is documented as out of scope for this host walk).
impl<D: Device> Execute<op::BatchedMatMul> for MetalBackendImpl<D> {
    type Output = MetalStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchedMatMul, Self>,
    ) -> Result<MetalStorage, BackendError> {
        let operation = OperationKind::BatchedMatMul;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "batched matmul expects 2 inputs"));
        };
        let lhs = lhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "lhs is not Metal storage"))?;
        let rhs = rhs
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| invalid(operation, "rhs is not Metal storage"))?;
        MetalBackendImpl::<D>::matmul::<f32>(lhs, rhs)
            .map_err(|e| kernel_error("Metal", operation, e))
    }
}

/// Same compile-time obligation as the CPU one: every identity the Metal
/// declaration advertises must have an `Execute` impl. Group entries are
/// bare `Op` idents or `(Op, training)` pairs (the quantization groups);
/// the training flag is a capability claim, so both spellings assert the
/// executor and neither is read here.
macro_rules! assert_every_advertised_metal_row_executes {
    (; $($group:ident = [$($entry:tt),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                macro_rules! assert_entry {
                    (($operation:ident, $training:expr)) => {
                        executes::<op::$operation, MetalBackendImpl<D>>()
                    };
                    ($operation:ident) => {
                        executes::<op::$operation, MetalBackendImpl<D>>()
                    };
                }
                $( $(assert_entry!($entry);)*)*
            }

            assert_all::<incin_core::tensor::device::Metal>();
        };
    };
}

crate::capability::metal_descriptor_operations!(assert_every_advertised_metal_row_executes,);
