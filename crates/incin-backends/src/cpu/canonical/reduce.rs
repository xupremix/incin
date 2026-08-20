//! Reduction, indexing extremum, and variance executors for the CPU backend.

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::error::BackendError;
use incin_core::exec::UnsupportedReason;
use incin_core::exec::catalog::{AxisVarianceAttributes, VarianceAttributes, op};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DTypeId;

use crate::cpu::CpuBackendImpl;
use crate::cpu::canonical::common::{reduction_operand, training_mode};
use crate::cpu::capability::CPU_NAME;
use crate::cpu::ops::elementwise::{
    canonical_abs, canonical_mul_scalar, canonical_powf, canonical_sqrt,
};
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::kernel_error;

macro_rules! reduce_all_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                crate::cpu::ops::reduce::$method(input)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

reduce_all_executors![
    (SumAll, sum_all),
    (MeanAll, mean_all),
    (MaxAll, max_all),
    (MinAll, min_all),
    (ProdAll, prod_all),
];

macro_rules! reduce_axis_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let axis = request.operation.descriptor().attributes().axis;
                crate::cpu::ops::reduce::$method(input, axis)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

reduce_axis_executors![
    (SumDim, sum_dim),
    (MeanDim, mean_dim),
    (MaxDim, max_dim),
    (MinDim, min_dim),
    (ProdDim, prod_dim),
    (SumKeepDim, sum_keepdim),
    (MeanKeepDim, mean_keepdim),
    (MaxKeepDim, max_keepdim),
    (MinKeepDim, min_keepdim),
];

impl<D: Device> Execute<op::Cumsum> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Cumsum, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Cumsum;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let axis = request.operation.descriptor().attributes().axis;
        crate::cpu::ops::reduce::cumsum(input, axis)
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
    }
}

macro_rules! dispatch_index_dtype {
    ($operation:expr, $dtype:expr, |$index:ident| $body:expr) => {{
        let operation = $operation;
        let desc: incin_core::tensor::dtype::DTypeDescriptor = $dtype;
        match desc.builtin_id() {
            Some(DTypeId::U8) => {
                type $index = u8;
                $body
            }
            Some(DTypeId::U32) => {
                type $index = u32;
                $body
            }
            Some(DTypeId::I64) => {
                type $index = i64;
                $body
            }
            _ => Err(BackendError::unsupported(
                CPU_NAME,
                UnsupportedReason::DType {
                    operation,
                    dtype: desc,
                },
            )),
        }
    }};
}

macro_rules! index_reduction_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let attributes = request.operation.descriptor().attributes();
                dispatch_index_dtype!(operation, attributes.dtype, |KIndex| {
                    crate::cpu::ops::reduce::$method::<KIndex>(input, attributes.axis)
                        .map_err(|error| kernel_error(CPU_NAME, operation, error))
                })
            }
        }
    )*};
}

index_reduction_executors![(ArgMax, argmax), (ArgMin, argmin)];

impl<D: Device> Execute<op::Argsort> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Argsort, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Argsort;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        dispatch_index_dtype!(operation, attributes.index_dtype, |KIndex| {
            crate::cpu::ops::reduce::argsort::<KIndex>(
                input,
                attributes.axis,
                attributes.descending,
            )
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
        })
    }
}

impl<D: Device> Execute<op::TopK> for CpuBackendImpl<D> {
    type Output = (CpuStorage, CpuStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TopK, Self>,
    ) -> Result<(CpuStorage, CpuStorage), BackendError> {
        let operation = OperationKind::TopK;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let attributes = request.operation.descriptor().attributes();
        dispatch_index_dtype!(operation, attributes.index_dtype, |KIndex| {
            crate::cpu::ops::reduce::topk::<KIndex>(
                input,
                attributes.k,
                attributes.axis,
                attributes.largest,
            )
            .map_err(|error| kernel_error(CPU_NAME, operation, error))
        })
    }
}

fn variance_scale(count: usize, unbiased: bool) -> f64 {
    let count = count as f64;
    let divisor = if unbiased {
        if count <= 1.0 { 0.0 } else { count - 1.0 }
    } else {
        count
    };
    if divisor > 0.0 { 1.0 / divisor } else { 0.0 }
}

fn squared_deviations(
    input: &CpuStorage,
    mean: &CpuStorage,
    operation: OperationKind,
) -> Result<CpuStorage, BackendError> {
    let deviation = crate::cpu::ops::elementwise::sub_storage(input, mean)
        .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
    crate::cpu::ops::elementwise::mul_storage(&deviation, &deviation)
        .map_err(|error| kernel_error(CPU_NAME, operation, error))
}

macro_rules! variance_executors {
    ($(($operation:ident, $mean:ident, $reduce:ident, $finish:expr)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$operation> for CpuBackendImpl<D> {
            type Output = CpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> Result<CpuStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = reduction_operand(self, request.inputs, operation, training_mode(request.context))?;
                let attributes = request.operation.descriptor().attributes();
                let (mean, count) = <Self as VarianceAxis<D>>::$mean(input, attributes)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let squared = squared_deviations(input, &mean, operation)?;
                let summed = <Self as VarianceAxis<D>>::$reduce(&squared, attributes)
                    .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let scaled = canonical_mul_scalar(
                    &summed,
                    variance_scale(count, attributes.unbiased),
                )
                .map_err(|error| kernel_error(CPU_NAME, operation, error))?;
                let finish: fn(&CpuStorage) -> incin_core::error::Result<CpuStorage> = $finish;
                finish(&scaled).map_err(|error| kernel_error(CPU_NAME, operation, error))
            }
        }
    )*};
}

trait VarianceAxis<D: Device> {
    fn mean_over_all(
        input: &CpuStorage,
        attributes: &VarianceAttributes,
    ) -> incin_core::error::Result<(CpuStorage, usize)>;
    fn sum_over_all(
        input: &CpuStorage,
        attributes: &VarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage>;
    fn mean_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<(CpuStorage, usize)>;
    fn sum_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage>;
    fn sum_along_axis_keeping_it(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage>;
}

impl<D: Device> VarianceAxis<D> for CpuBackendImpl<D> {
    fn mean_over_all(
        input: &CpuStorage,
        _: &VarianceAttributes,
    ) -> incin_core::error::Result<(CpuStorage, usize)> {
        let count = input.shape.iter().product::<usize>();
        Ok((crate::cpu::ops::reduce::mean_all(input)?, count))
    }

    fn sum_over_all(
        input: &CpuStorage,
        _: &VarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_all(input)
    }

    fn mean_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<(CpuStorage, usize)> {
        let count = input.shape.get(attributes.axis).copied().unwrap_or(0);
        Ok((
            crate::cpu::ops::reduce::mean_keepdim(input, attributes.axis)?,
            count,
        ))
    }

    fn sum_along_axis(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_dim(input, attributes.axis)
    }

    fn sum_along_axis_keeping_it(
        input: &CpuStorage,
        attributes: &AxisVarianceAttributes,
    ) -> incin_core::error::Result<CpuStorage> {
        crate::cpu::ops::reduce::sum_keepdim(input, attributes.axis)
    }
}

fn identity(storage: &CpuStorage) -> incin_core::error::Result<CpuStorage> {
    Ok(storage.clone())
}

fn square_root<D: Device>(storage: &CpuStorage) -> incin_core::error::Result<CpuStorage> {
    let _ = core::marker::PhantomData::<D>;
    canonical_sqrt(storage)
}

variance_executors![
    (VarianceAll, mean_over_all, sum_over_all, identity),
    (VarianceDim, mean_along_axis, sum_along_axis, identity),
    (
        VarianceKeepDim,
        mean_along_axis,
        sum_along_axis_keeping_it,
        identity
    ),
    (StdAll, mean_over_all, sum_over_all, square_root::<D>),
    (StdDim, mean_along_axis, sum_along_axis, square_root::<D>),
    (
        StdKeepDim,
        mean_along_axis,
        sum_along_axis_keeping_it,
        square_root::<D>
    ),
];

const NORM_ORDER_TOLERANCE: f64 = 1e-6;

impl<D: Device> Execute<op::Norm> for CpuBackendImpl<D> {
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Norm, Self>,
    ) -> Result<CpuStorage, BackendError> {
        let operation = OperationKind::Norm;
        let input = reduction_operand(
            self,
            request.inputs,
            operation,
            training_mode(request.context),
        )?;
        let order = request.operation.descriptor().attributes().order;
        let wrap = |error| kernel_error(CPU_NAME, operation, error);

        if (order - 1.0).abs() < NORM_ORDER_TOLERANCE {
            let magnitude = canonical_abs(input).map_err(wrap)?;
            return crate::cpu::ops::reduce::sum_all(&magnitude).map_err(wrap);
        }
        if (order - 2.0).abs() < NORM_ORDER_TOLERANCE {
            let squared = crate::cpu::ops::elementwise::mul_storage(input, input).map_err(wrap)?;
            let summed = crate::cpu::ops::reduce::sum_all(&squared).map_err(wrap)?;
            return canonical_sqrt(&summed).map_err(wrap);
        }
        let magnitude = canonical_abs(input).map_err(wrap)?;
        let raised = canonical_powf(&magnitude, order).map_err(wrap)?;
        let summed = crate::cpu::ops::reduce::sum_all(&raised).map_err(wrap)?;
        canonical_powf(&summed, 1.0 / order).map_err(wrap)
    }
}
