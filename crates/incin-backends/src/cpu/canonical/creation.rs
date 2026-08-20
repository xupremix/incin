//! Creation, allocation, and readback executors for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest, HostInterop};
use incin_core::error::BackendError;
use incin_core::exec::catalog::op;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{reduction_operand, training_mode};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::{invalid, kernel_error};

macro_rules! allocating_executors {
    ($(($operation:ident, $method:ident $(, $argument:ident)*)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "an allocation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let total = crate::cpu::stride::numel_for_evidence(
                    &attributes.shape,
                    request.operation.shape_evidence().static_numel(),
                )
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                crate::cpu::creation::$method(
                    total,
                    $(attributes.$argument,)*
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

allocating_executors![
    (Zeros, zeros_with_total),
    (Ones, ones_with_total),
    (UniformRandom, rand_with_total),
    (NormalRandom, randn_with_total),
    (Full, full_with_total, value),
    (Arange, arange_with_total, start, step),
    (Linspace, linspace_with_total, start, end),
];

macro_rules! data_executors {
    ($($operation:ident),* $(,)?) => {$ (
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "data creation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let bytes = request
                    .payload
                    .ok_or_else(|| invalid(operation, "data creation requires borrowed bytes"))?;
                <Self as HostInterop>::from_bytes::<f32>(
                    bytes,
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

data_executors![TensorFromData, TensorFromBytes];

macro_rules! variable_allocating_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = crate::cpu::var::CpuVar;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<crate::cpu::var::CpuVar, BackendError> {
                let operation = OperationKind::$operation;
                if !request.inputs.is_empty() {
                    return Err(invalid(operation, "an allocation takes no operand"));
                }
                let attributes = request.operation.descriptor().attributes();
                let total = crate::cpu::stride::numel_for_evidence(
                    &attributes.shape,
                    request.operation.shape_evidence().static_numel(),
                )
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                crate::cpu::creation::$method(
                    total,
                    &attributes.shape,
                    attributes.dtype,
                    &attributes.device,
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

variable_allocating_executors![
    (VariableZeros, var_zeros_with_total),
    (VariableOnes, var_ones_with_total),
    (VariableUniformRandom, var_rand_with_total),
    (VariableNormalRandom, var_randn_with_total),
];

use crate::cpu::ops::shape_ops::{
    float_to_scalar_storage, float_to_vec1_storage, int_to_scalar_storage, int_to_vec1_storage,
};

macro_rules! readback_executors {
    ($(($operation:ident, $method:ident, $output:ty)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = $output;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<$output, BackendError> {
                let operation = OperationKind::$operation;
                let training = training_mode(request.context);
                let input = reduction_operand(self, request.inputs, operation, training)?;
                $method(input)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

readback_executors![
    (ToHostFloatScalar, float_to_scalar_storage, f64),
    (ToHostFloatVec, float_to_vec1_storage, alloc::vec::Vec<f64>),
    (ToHostIntScalar, int_to_scalar_storage, i64),
    (ToHostIntVec, int_to_vec1_storage, alloc::vec::Vec<i64>),
];

impl<D: Device> Execute<op::TensorToBytes> for CpuBackendImpl<D> {
    type Output = alloc::vec::Vec<u8>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TensorToBytes, Self>,
    ) -> Result<alloc::vec::Vec<u8>, BackendError> {
        let operation = OperationKind::TensorToBytes;
        let training = training_mode(request.context);
        let input = reduction_operand(self, request.inputs, operation, training)?;
        <Self as HostInterop>::to_bytes::<f32>(input)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}
