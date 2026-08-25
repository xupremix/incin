//! Elementwise and pointwise unary/binary execution for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::error::BackendError;
use incin_core::exec::catalog::op;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::ConstDType;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{
    admitted, operand, reduction_operand, resolved_output_shape, training_mode,
};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::ops::elementwise::{
    canonical_abs, canonical_acos, canonical_acosh, canonical_add_scalar, canonical_asin,
    canonical_asinh, canonical_atan, canonical_atan2, canonical_atanh, canonical_clamp,
    canonical_cosh, canonical_elu, canonical_erf, canonical_exp, canonical_fmod, canonical_frac,
    canonical_gelu, canonical_log, canonical_mish, canonical_mul_scalar, canonical_neg,
    canonical_powf, canonical_relu, canonical_remainder, canonical_rsqrt, canonical_sigmoid,
    canonical_sinh, canonical_softmax, canonical_sqrt, canonical_step, canonical_swish,
    canonical_tan, canonical_tanh, canonical_trunc, canonical_unary,
};
use crate::cpu::ops::shape_ops::{div_scalar_storage, sub_scalar_storage};
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

fn binary_operands<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [incin_core::exec::TensorHandle<'a>],
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

pointwise_binary_executors![
    (Add, add_storage, add_storage_with_shape),
    (Sub, sub_storage, sub_storage_with_shape),
    (Mul, mul_storage, mul_storage_with_shape),
    (Div, div_storage, div_storage_with_shape),
];

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

// Maximum/Minimum/AbsDiff declare BinaryBroadcast gradients `Defined`, so
// each composes from tape-tracked primitives instead of the raw kernel: the
// selection mask is a constant, and where_storage's own backward routes and
// unbroadcasts both cotangents.
impl<D: Device> Execute<op::Maximum> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Maximum, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Maximum;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let mask = crate::cpu::ops::shape_ops::elementwise_cmp(lhs, rhs, |a, b| a > b)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
        crate::cpu::ops::shape_ops::where_storage(&mask, lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Minimum> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Minimum, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Minimum;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let mask = crate::cpu::ops::shape_ops::elementwise_cmp(lhs, rhs, |a, b| a < b)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
        crate::cpu::ops::shape_ops::where_storage(&mask, lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::AbsDiff> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::AbsDiff, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::AbsDiff;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let diff = crate::cpu::ops::elementwise::sub_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
        crate::cpu::ops::elementwise::canonical_abs(&diff)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

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
