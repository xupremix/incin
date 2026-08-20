//! Linear algebra, matrix multiplication, and quantization executors for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::error::BackendError;
use incin_core::exec::catalog::op;
use incin_core::exec::{TensorHandle, UnsupportedReason};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DTypeId;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{admitted, operand, reduction_operand, training_mode};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::ops::quant::{dequantize_storage, quantize_storage, quantized_matmul_storage};
use crate::cpu::ops::shape_ops::{addmm_storage, transpose_storage, unsqueeze_storage};
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

impl<D: Device> Execute<op::MatMulExact> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::MatMulExact;
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(operation, "matmul expects exactly two operands"));
        };
        let lhs = operand(lhs, operation)?;
        let rhs = operand(rhs, operation)?;
        for storage in [lhs, rhs] {
            if storage.metadata().dtype() != DTypeId::F32.descriptor() {
                return Err(BackendError::unsupported(
                    CPU_NAME,
                    UnsupportedReason::DType {
                        operation,
                        dtype: storage.metadata().dtype(),
                    },
                ));
            }
            admitted(self, operation, storage, training_mode(request.context))?;
        }
        crate::cpu::ops::shape_ops::matmul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Linear> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Linear, Self>,
    ) -> Result<CpuStorage, BackendError> {
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
        let training = training_mode(request.context);
        let input = operand(input, operation)?;
        let weight = operand(weight, operation)?;
        let bias = bias.map(|bias| operand(bias, operation)).transpose()?;
        admitted(self, operation, input, training)?;
        admitted(self, operation, weight, training)?;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        let transposed = transpose_storage(weight, 0, 1).map_err(wrap)?;
        let product =
            crate::cpu::ops::shape_ops::matmul_storage(input, &transposed).map_err(wrap)?;
        match bias {
            None => Ok(product),
            Some(bias) => {
                admitted(self, operation, bias, training)?;
                crate::cpu::ops::elementwise::add_storage(&product, bias).map_err(wrap)
            }
        }
    }
}

impl<D: Device> Execute<op::Addmm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Addmm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Addmm;
        let [mat, lhs, rhs] = ternary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        addmm_storage(mat, lhs, rhs, attributes.beta, attributes.alpha)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Quantize> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Quantize, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Quantize;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::Q8_0.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        quantize_storage(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Dequantize> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dequantize, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Dequantize;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let dtype = request.operation.descriptor().attributes().dtype;
        if dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType { operation, dtype },
            ));
        }
        dequantize_storage(input).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::QuantizedMatMul> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::QuantizedMatMul, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::QuantizedMatMul;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        quantized_matmul_storage(lhs, rhs).map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Dot> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Dot, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Dot;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let product = crate::cpu::ops::elementwise::mul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
        crate::cpu::ops::reduce::sum_all(&product)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

impl<D: Device> Execute<op::Outer> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Outer, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Outer;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);
        let column = unsqueeze_storage(lhs, 1).map_err(wrap)?;
        let row = unsqueeze_storage(rhs, 0).map_err(wrap)?;
        crate::cpu::ops::elementwise::mul_storage(&column, &row).map_err(wrap)
    }
}

impl<D: Device> Execute<op::BatchedMatMul> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::BatchedMatMul, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::BatchedMatMul;
        let (lhs, rhs) = binary_operands(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        crate::cpu::ops::shape_ops::matmul_storage(lhs, rhs)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}
