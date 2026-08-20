//! Neural network layer and loss function executors for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::error::BackendError;
use incin_core::exec::catalog::{LossReduction, op};
use incin_core::exec::{TensorHandle, UnsupportedReason};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DTypeId;
use incin_core::tensor::reduction::Reduction;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{admitted, operand, reduction_operand, training_mode};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::ops::conv::{Window2d, conv_transpose2d_impl, conv1d_impl, conv2d_windowed_impl};
use crate::cpu::ops::elementwise::{
    canonical_add_scalar, canonical_mul_scalar, canonical_sqrt, canonical_step,
};
use crate::cpu::ops::embedding::embedding_impl;
use crate::cpu::ops::loss::{
    bce_with_logits_loss_storage, cross_entropy_loss_storage, l1_loss_storage, mse_loss_storage,
};
use crate::cpu::ops::norm::{batch_norm_impl, batch_norm_training_impl, layer_norm_impl};
use crate::cpu::ops::pool::{adaptive_avg_pool2d_impl, avg_pool2d_impl, max_pool2d_impl};
use crate::cpu::ops::shape_ops::{
    group_norm_storage, instance_norm_storage, scaled_dot_product_attention_storage,
};
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

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

impl<D: Device> Execute<op::Conv2dExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Conv2dExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Conv2dExact;
        let attributes = request.operation.descriptor().attributes();
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

        conv2d_windowed_impl::<D, f32>(
            activation,
            weight,
            bias,
            Window2d {
                stride: attributes.stride,
                padding: attributes.padding,
                dilation: attributes.dilation,
            },
            attributes.groups,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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

        layer_norm_impl::<D, f32>(input, weight, bias, epsilon)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
        if attributes.training {
            let input = operand(input, operation)?;
            let weight = weight.map(|value| operand(value, operation)).transpose()?;
            let bias = bias.map(|value| operand(value, operation)).transpose()?;
            admitted(self, operation, input, training_mode(request.context))?;
            f32_only(operation, &[Some(input), weight, bias])?;
            let epsilon = narrowed_epsilon(operation, attributes.epsilon)?;
            return batch_norm_training_impl::<D, f32>(input, weight, bias, epsilon)
                .map_err(|error| kernel_error(CPU_NAME, operation, error));
        }

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
        let draw = crate::cpu::creation::rand_with_total(
            total,
            input.shape.as_ref(),
            metadata.dtype(),
            &metadata.device(),
        )
        .map_err(wrap)?;
        let shifted = canonical_add_scalar(&draw, -attributes.probability).map_err(wrap)?;
        let mask = canonical_step(&shifted).map_err(wrap)?;
        let kept = crate::cpu::ops::elementwise::mul_storage(input, &mask).map_err(wrap)?;
        canonical_mul_scalar(&kept, 1.0 / (1.0 - attributes.probability)).map_err(wrap)
    }
}

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

        let axis = input.shape.len().saturating_sub(1);
        let squared = crate::cpu::ops::elementwise::mul_storage(input, input).map_err(wrap)?;
        let mean = crate::cpu::ops::reduce::mean_keepdim(&squared, axis).map_err(wrap)?;
        let guarded = canonical_add_scalar(&mean, epsilon).map_err(wrap)?;
        let scale = canonical_sqrt(&guarded).map_err(wrap)?;
        let normalized = crate::cpu::ops::elementwise::div_storage(input, &scale).map_err(wrap)?;
        crate::cpu::ops::elementwise::mul_storage(&normalized, weight).map_err(wrap)
    }
}

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
        let mut bound = alloc::vec::Vec::with_capacity(3);
        for handle in operands {
            let storage = operand(handle, operation)?;
            admitted(self, operation, storage, training_mode(request.context))?;
            bound.push(storage);
        }
        if let Some(mask) = mask {
            admitted(self, operation, mask, training_mode(request.context))?;
        }
        scaled_dot_product_attention_storage::<D>(
            bound[0],
            bound[1],
            bound[2],
            mask,
            attributes.scale,
        )
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
                $method(prediction, target, reduction)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

fn loss_reduction(reduction: LossReduction) -> Reduction {
    match reduction {
        LossReduction::None => Reduction::None,
        LossReduction::Mean => Reduction::Mean,
        LossReduction::Sum => Reduction::Sum,
    }
}

loss_executors![
    (MseLoss, mse_loss_storage),
    (L1Loss, l1_loss_storage),
    (BceWithLogitsLoss, bce_with_logits_loss_storage),
];

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
        cross_entropy_loss_storage::<D>(logits, target, reduction)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}
