use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::iteration::{IterationPlan, OperandLayout, UnaryIterationPlan};
use alloc::sync::Arc;
use incin_core::exec::LayoutClass;
use incin_core::prelude::{DTypeDescriptor, DTypeId, DeviceId, Error, OperationKind, Result};

use incin_core::exec::{PrecisionCapabilities, PrecisionRequest};

fn validate_kernel_abi(
    kernel: &crate::kernel::RenderedKernel,
    dtype: DTypeDescriptor,
) -> Result<()> {
    if Some(kernel.dtype) != dtype.builtin_id()
        || kernel.element_size
            != dtype
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("invalid scalar bytes".into()))?
    {
        return Err(Error::Msg(format!(
            "CUDA kernel ABI mismatch: rendered {:?}/{} bytes for {:?}/{:?}-byte storage",
            kernel.dtype,
            kernel.element_size,
            dtype,
            dtype.encoding().scalar_bytes()
        )));
    }
    Ok(())
}

fn validate_elementwise_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    crate::cuda::backend::validate_cuda_storage_dtype(dtype, op)
}

fn checked_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Msg(format!("CUDA {field} value {value} exceeds i32 launch ABI")))
}

fn checked_i32_vec(values: &[usize], field: &'static str) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&value| checked_i32(value, field))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointwiseStrategy {
    Scalar { unroll_width: u8, block_size: u16 },
    Packed { vector_width: u8, block_size: u16 },
}

impl PointwiseStrategy {
    fn block_size(self) -> u16 {
        match self {
            Self::Scalar { block_size, .. } | Self::Packed { block_size, .. } => block_size,
        }
    }

    fn candidate(self) -> crate::tuning::LaunchCandidate {
        let access = match self {
            Self::Scalar { unroll_width, .. } => {
                crate::kernel::KernelAccess::Scalar { unroll_width }
            }
            Self::Packed { vector_width, .. } => {
                crate::kernel::KernelAccess::Packed { vector_width }
            }
        };
        crate::tuning::LaunchCandidate {
            block_size: self.block_size(),
            access,
        }
    }
}

#[derive(Clone, Debug)]
struct PointwiseLaunchSelection {
    strategy: PointwiseStrategy,
    #[cfg(feature = "autotune")]
    tuning_permit: Option<crate::tuning::TuningPermit>,
    candidates: Vec<crate::tuning::LaunchCandidate>,
}

struct PreparedPointwiseKernel {
    candidate: crate::tuning::LaunchCandidate,
    kernel: crate::kernel::RenderedKernel,
    function: cudarc::driver::CudaFunction,
}

fn pointwise_strategy(
    dtype: DTypeDescriptor,
    numel: usize,
    dense: bool,
    packed_aligned: bool,
) -> Result<PointwiseStrategy> {
    let candidate = crate::tuning::default_pointwise_candidate(
        &crate::tuning::pointwise_candidates(dtype, numel, dense, packed_aligned),
    )?;
    strategy_from_candidate(candidate)
}

fn strategy_from_candidate(candidate: crate::tuning::LaunchCandidate) -> Result<PointwiseStrategy> {
    match candidate.access {
        crate::kernel::KernelAccess::Scalar { unroll_width } => Ok(PointwiseStrategy::Scalar {
            unroll_width,
            block_size: candidate.block_size,
        }),
        crate::kernel::KernelAccess::Packed { vector_width } => Ok(PointwiseStrategy::Packed {
            vector_width,
            block_size: candidate.block_size,
        }),
        access => Err(Error::Msg(format!(
            "invalid CUDA pointwise tuning access candidate {access:?}"
        ))),
    }
}

fn pointwise_launch_selection(
    context: &cudarc::driver::CudaContext,
    kernel: &crate::kernel::RenderedKernel,
    dtype: DTypeDescriptor,
    numel: usize,
    dense: bool,
    packed_aligned: bool,
    fallback: PointwiseStrategy,
) -> Result<PointwiseLaunchSelection> {
    let candidates = crate::tuning::pointwise_candidates(dtype, numel, dense, packed_aligned);
    #[cfg(feature = "autotune")]
    {
        let key = crate::tuning::TuningKey::new(
            crate::tuning::identity::TuningEnvironmentFingerprint::<
                incin_core::prelude::Cuda,
            >::from_cuda_context(context)?
            .erase(),
            &kernel.key,
            crate::tuning::WorkloadBucket::pointwise(numel, packed_aligned),
        );
        match crate::tuning::claim_tuning(key, &candidates)? {
            crate::tuning::TuningDecision::Cached(tuned) => Ok(PointwiseLaunchSelection {
                strategy: strategy_from_candidate(tuned.candidate)?,
                tuning_permit: None,
                candidates,
            }),
            crate::tuning::TuningDecision::Measure(permit) => Ok(PointwiseLaunchSelection {
                strategy: fallback,
                tuning_permit: Some(permit),
                candidates,
            }),
        }
    }
    #[cfg(not(feature = "autotune"))]
    {
        let _ = (context, kernel);
        Ok(PointwiseLaunchSelection {
            strategy: fallback,
            candidates,
        })
    }
}

#[cfg(feature = "cuda")]
fn prepare_pointwise_kernels<F>(
    device_id: usize,
    selection: &PointwiseLaunchSelection,
    dtype: DTypeDescriptor,
    mut render: F,
) -> Result<Vec<PreparedPointwiseKernel>>
where
    F: FnMut(PointwiseStrategy) -> Result<crate::kernel::RenderedKernel>,
{
    #[cfg(feature = "autotune")]
    let tune_all = selection.tuning_permit.is_some();
    #[cfg(not(feature = "autotune"))]
    let tune_all = false;
    let candidates = if tune_all {
        selection.candidates.clone()
    } else {
        vec![selection.strategy.candidate()]
    };
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let mut prepared = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let strategy = strategy_from_candidate(candidate)?;
        let kernel = render(strategy)?;
        validate_kernel_abi(&kernel, dtype)?;
        #[cfg(feature = "autotune")]
        {
            if let Some(key) = selection
                .tuning_permit
                .as_ref()
                .and_then(|permit| permit.key())
            {
                if kernel.key.tuning_problem_id() != key.problem {
                    return Err(Error::Msg(
                        "CUDA pointwise candidate changed canonical tuning problem".into(),
                    ));
                }
            }
        }
        if crate::cuda::gpu::cuda_cache::get_module(device_id, &kernel.cache_key).is_none() {
            dispatcher.compile_and_load_kernel(
                &kernel.entry_point,
                &kernel.source,
                &kernel.cache_key,
            )?;
        }
        let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
        prepared.push(PreparedPointwiseKernel {
            candidate,
            kernel,
            function,
        });
    }
    #[cfg(feature = "autotune")]
    if tune_all {
        prune_zero_occupancy_candidates(&mut prepared);
    }
    Ok(prepared)
}

/// Queries the driver for how many blocks of `block_size` can actually be
/// resident on a multiprocessor at once for `function`. `None` means the
/// query itself failed (unrelated driver error) rather than the candidate
/// being infeasible, so callers must not treat `None` as zero occupancy.
#[cfg(all(feature = "cuda", feature = "autotune"))]
fn occupancy_active_blocks(
    function: &cudarc::driver::CudaFunction,
    block_size: u16,
) -> Option<u32> {
    function
        .occupancy_max_active_blocks_per_multiprocessor(u32::from(block_size), 0, None)
        .ok()
}

/// Drops candidates the driver reports as unable to place even one block on
/// a multiprocessor, before the expensive JIT-then-measure loop times them.
/// Only prunes when at least one other candidate is confirmed viable, and
/// never removes a candidate whose occupancy query itself failed (unknown,
/// not proven infeasible) — an optimization that must never narrow the
/// legal candidate set to zero or reject a candidate on a driver hiccup.
#[cfg(all(feature = "cuda", feature = "autotune"))]
fn prune_zero_occupancy_candidates(prepared: &mut Vec<PreparedPointwiseKernel>) {
    let occupancies: Vec<Option<u32>> = prepared
        .iter()
        .map(|prepared_kernel| {
            occupancy_active_blocks(
                &prepared_kernel.function,
                prepared_kernel.candidate.block_size,
            )
        })
        .collect();
    let any_confirmed_viable = occupancies.iter().any(|occupancy| *occupancy != Some(0));
    if !any_confirmed_viable {
        return;
    }
    let mut kept = Vec::with_capacity(prepared.len());
    for (candidate, occupancy) in prepared.drain(..).zip(occupancies) {
        if occupancy != Some(0) {
            kept.push(candidate);
        }
    }
    *prepared = kept;
}

#[cfg(feature = "cuda")]
fn execute_pointwise_selection<F>(
    stream: &cudarc::driver::CudaStream,
    selection: PointwiseLaunchSelection,
    prepared: &[PreparedPointwiseKernel],
    mut launch: F,
) -> Result<()>
where
    F: FnMut(&PreparedPointwiseKernel) -> Result<()>,
{
    #[cfg(not(feature = "autotune"))]
    let _ = stream;
    #[cfg(feature = "autotune")]
    if let Some(permit) = selection.tuning_permit {
        let mut measurements = Vec::with_capacity(prepared.len());
        for candidate in prepared {
            measurements.push(crate::tuning::measure_cuda_candidate(
                stream,
                candidate.candidate,
                || launch(candidate),
            )?);
        }
        let winner = permit.record(&measurements)?;
        let selected = prepared
            .iter()
            .find(|candidate| candidate.candidate == winner.candidate)
            .ok_or_else(|| {
                Error::Msg("CUDA pointwise tuner selected an unknown candidate".into())
            })?;
        return launch(selected);
    }

    let selected_candidate = selection.strategy.candidate();
    let selected = prepared
        .iter()
        .find(|candidate| candidate.candidate == selected_candidate)
        .ok_or_else(|| Error::Msg("CUDA pointwise selection was not prepared".into()))?;
    launch(selected)
}

fn render_unary_strategy(
    op_name: &'static str,
    op_expr: &str,
    dtype: DTypeDescriptor,
    layout: LayoutClass,
    strategy: PointwiseStrategy,
) -> Result<crate::kernel::RenderedKernel> {
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(dtype, op_name)?;
    match strategy {
        PointwiseStrategy::Scalar { unroll_width, .. } => {
            crate::kernel::render_cuda_unary_for_layout(
                op_name,
                op_expr,
                builtin_id,
                layout,
                unroll_width,
            )
        }
        PointwiseStrategy::Packed { .. } => {
            crate::kernel::render_cuda_unary_packed(op_name, op_expr, builtin_id, layout)
        }
    }
}

fn render_binary_strategy(
    op_name: &'static str,
    op_expr: &str,
    dtype: DTypeDescriptor,
    layout: LayoutClass,
    strategy: PointwiseStrategy,
) -> Result<crate::kernel::RenderedKernel> {
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(dtype, op_name)?;
    match strategy {
        PointwiseStrategy::Scalar { unroll_width, .. } => {
            crate::kernel::render_cuda_binary_for_layout(
                op_name,
                op_expr,
                builtin_id,
                layout,
                unroll_width,
            )
        }
        PointwiseStrategy::Packed { .. } => {
            crate::kernel::render_cuda_binary_packed(op_name, op_expr, builtin_id, layout)
        }
    }
}

fn select_unary_strategy(
    dtype: DTypeDescriptor,
    layout: LayoutClass,
    numel: usize,
    offset: usize,
) -> Result<PointwiseStrategy> {
    let width = crate::tuning::preferred_pointwise_width(dtype);
    let dense = layout == LayoutClass::Contiguous;
    pointwise_strategy(
        dtype,
        numel,
        dense,
        dense && offset.is_multiple_of(width.into()),
    )
}

fn select_binary_strategy(
    dtype: DTypeDescriptor,
    layout: LayoutClass,
    numel: usize,
    lhs_offset: usize,
    rhs_offset: usize,
) -> Result<PointwiseStrategy> {
    let width = crate::tuning::preferred_pointwise_width(dtype);
    let aligned = match layout {
        LayoutClass::Contiguous => {
            lhs_offset.is_multiple_of(width.into()) && rhs_offset.is_multiple_of(width.into())
        }
        LayoutClass::ScalarLeft => rhs_offset.is_multiple_of(width.into()),
        LayoutClass::ScalarRight => lhs_offset.is_multiple_of(width.into()),
        LayoutClass::Strided => false,
        other => {
            return Err(Error::Msg(format!(
                "layout {} is not valid for CUDA binary strategy selection",
                other.as_str()
            )));
        }
    };
    pointwise_strategy(dtype, numel, layout != LayoutClass::Strided, aligned)
}

fn launch_config(
    numel: usize,
    elements_per_thread: u8,
    block_size: u16,
) -> Result<cudarc::driver::LaunchConfig> {
    if !matches!(elements_per_thread, 1 | 2 | 4) {
        return Err(Error::Msg(format!(
            "unsupported CUDA pointwise elements-per-thread {elements_per_thread}"
        )));
    }
    if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
        return Err(Error::Msg(format!(
            "unsupported CUDA pointwise block size {block_size}"
        )));
    }
    let work_items = numel.div_ceil(usize::from(elements_per_thread));
    let work_items = u32::try_from(work_items).map_err(|_| {
        Error::Msg(format!(
            "CUDA pointwise work-item count {work_items} exceeds u32 launch grid"
        ))
    })?;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (work_items.div_ceil(u32::from(block_size)), 1, 1),
        block_dim: (u32::from(block_size), 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(feature = "cuda")]
/// `launch_unary_op`.
pub(crate) fn launch_unary_op(
    op_name: &'static str,
    op_expr: &str,
    t: &CudaStorage,
) -> Result<CudaStorage> {
    let b = &*t.buffer;
    validate_elementwise_dtype(b.dtype, "elementwise_unary")?;
    let plan = UnaryIterationPlan::new(OperandLayout {
        shape: &t.shape,
        strides: &t.strides,
        offset: t.offset_elements,
    })?;
    if let Some(max_index) = plan.operand.max_physical_index(&plan.output_shape)?
        && max_index >= b.len
    {
        return Err(Error::Msg(format!(
            "CUDA unary iteration accesses storage index {max_index}, but buffer length is {}",
            b.len
        )));
    }
    let device_id = b.device_id;
    let layout = plan.layout_class();
    let numel = plan.numel;
    let strategy = select_unary_strategy(b.dtype, layout, numel, plan.operand.offset)?;
    let kernel = render_unary_strategy(op_name, op_expr, b.dtype, layout, strategy)?;
    let packed_width = crate::tuning::preferred_pointwise_width(b.dtype);
    let dense = layout == LayoutClass::Contiguous;
    let selection = pointwise_launch_selection(
        &b.device,
        &kernel,
        b.dtype,
        numel,
        dense,
        dense && plan.operand.offset.is_multiple_of(packed_width.into()),
        strategy,
    )?;
    let prepared = prepare_pointwise_kernels(device_id, &selection, b.dtype, |strategy| {
        render_unary_strategy(op_name, op_expr, b.dtype, layout, strategy)
    })?;
    let dtype = prepared
        .first()
        .ok_or_else(|| Error::Msg("CUDA unary candidate set is empty".into()))?
        .kernel
        .dtype;
    let byte_len = crate::bytes::byte_len(dtype, numel, OperationKind::Pointwise)?;
    let offset_i32 = checked_i32(plan.operand.offset, "offset")?;
    let numel_i32 = checked_i32(numel, "element count")?;
    let strided_metadata = if layout == LayoutClass::Strided {
        Some((
            checked_i32_vec(&plan.output_shape, "shape")?,
            checked_i32_vec(&plan.operand.strides, "stride")?,
            checked_i32(plan.output_shape.len(), "rank")?,
        ))
    } else {
        None
    };
    let stream = b.device.default_stream();

    let mut out_b =
        CudaBuffer {
            len: numel,
            dtype: b.dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|error| {
                Error::Msg(format!("CUDA output allocation failed: {error:?}"))
            })?),
            device: b.device.clone(),
            device_id,
        };

    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), t.shape.to_vec()));
    }

    unsafe {
        let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| Error::Msg("fresh CUDA output buffer was unexpectedly shared".into()))?;

        use cudarc::driver::PushKernelArg;
        match strided_metadata {
            None => {
                let mut launch = |candidate: &PreparedPointwiseKernel| -> Result<()> {
                    let config = launch_config(
                        numel,
                        candidate.kernel.elements_per_thread(),
                        candidate.candidate.block_size,
                    )?;
                    stream
                        .launch_builder(&candidate.function)
                        .arg(&*b.data)
                        .arg(&mut *out_slice_u8)
                        .arg(&offset_i32)
                        .arg(&numel_i32)
                        .launch(config)
                        .map(|_| ())
                        .map_err(|error| Error::Msg(format!("Kernel launch failed: {error:?}")))
                };
                execute_pointwise_selection(&stream, selection, &prepared, &mut launch)
            }
            Some((shape_i32, strides_i32, ndim_i32)) => {
                let shape_dev = stream
                    .clone_htod(&shape_i32)
                    .map_err(|error| Error::Msg(format!("CUDA shape upload failed: {error:?}")))?;
                let strides_dev = stream
                    .clone_htod(&strides_i32)
                    .map_err(|error| Error::Msg(format!("CUDA stride upload failed: {error:?}")))?;
                let mut launch = |candidate: &PreparedPointwiseKernel| -> Result<()> {
                    let config = launch_config(
                        numel,
                        candidate.kernel.elements_per_thread(),
                        candidate.candidate.block_size,
                    )?;
                    stream
                        .launch_builder(&candidate.function)
                        .arg(&*b.data)
                        .arg(&mut *out_slice_u8)
                        .arg(&shape_dev)
                        .arg(&strides_dev)
                        .arg(&offset_i32)
                        .arg(&numel_i32)
                        .arg(&ndim_i32)
                        .launch(config)
                        .map(|_| ())
                        .map_err(|error| Error::Msg(format!("Kernel launch failed: {error:?}")))
                };
                execute_pointwise_selection(&stream, selection, &prepared, &mut launch)
            }
        }?;
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_b),
        t.shape.to_vec(),
    ))
}

#[cfg(feature = "cuda")]
/// `launch_binary_op`.
pub(crate) fn launch_binary_op(
    op_name: &'static str,
    op_expr: &str,
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    out_shape: &[usize],
) -> Result<CudaStorage> {
    let (lhs_b, rhs_b) = (&*lhs.buffer, &*rhs.buffer);
    if lhs_b.dtype != rhs_b.dtype {
        return Err(Error::DTypeStorageMismatch {
            expected: lhs_b.dtype,
            got: rhs_b.dtype,
        });
    }
    if lhs_b.device_id != rhs_b.device_id {
        return Err(Error::DeviceMismatch {
            left: DeviceId::cuda(lhs_b.device_id),
            right: DeviceId::cuda(rhs_b.device_id),
        });
    }
    validate_elementwise_dtype(lhs_b.dtype, "elementwise_binary")?;
    let plan = IterationPlan::binary(
        OperandLayout {
            shape: &lhs.shape,
            strides: &lhs.strides,
            offset: lhs.offset_elements,
        },
        OperandLayout {
            shape: &rhs.shape,
            strides: &rhs.strides,
            offset: rhs.offset_elements,
        },
        out_shape,
    )?;
    for (name, operand, buffer_len) in [
        ("lhs", &plan.operands[0], lhs_b.len),
        ("rhs", &plan.operands[1], rhs_b.len),
    ] {
        if let Some(max_index) = operand.max_physical_index(&plan.output_shape)?
            && max_index >= buffer_len
        {
            return Err(Error::Msg(format!(
                "CUDA binary {name} iteration accesses storage index {max_index}, but buffer length is {buffer_len}"
            )));
        }
    }
    let device_id = lhs_b.device_id;
    let layout = plan.binary_layout_class();
    let numel = plan.numel;
    let lhs_plan = &plan.operands[0];
    let rhs_plan = &plan.operands[1];
    let strategy =
        select_binary_strategy(lhs_b.dtype, layout, numel, lhs_plan.offset, rhs_plan.offset)?;
    let kernel = render_binary_strategy(op_name, op_expr, lhs_b.dtype, layout, strategy)?;
    let packed_width = crate::tuning::preferred_pointwise_width(lhs_b.dtype);
    let packed_aligned = match layout {
        LayoutClass::Contiguous => {
            lhs_plan.offset.is_multiple_of(packed_width.into())
                && rhs_plan.offset.is_multiple_of(packed_width.into())
        }
        LayoutClass::ScalarLeft => rhs_plan.offset.is_multiple_of(packed_width.into()),
        LayoutClass::ScalarRight => lhs_plan.offset.is_multiple_of(packed_width.into()),
        LayoutClass::Strided => false,
        other => {
            return Err(Error::Msg(format!(
                "layout {} is not valid for a CUDA binary launch",
                other.as_str()
            )));
        }
    };
    let selection = pointwise_launch_selection(
        &lhs_b.device,
        &kernel,
        lhs_b.dtype,
        numel,
        layout != LayoutClass::Strided,
        packed_aligned,
        strategy,
    )?;
    let prepared = prepare_pointwise_kernels(device_id, &selection, lhs_b.dtype, |strategy| {
        render_binary_strategy(op_name, op_expr, lhs_b.dtype, layout, strategy)
    })?;
    let dtype = prepared
        .first()
        .ok_or_else(|| Error::Msg("CUDA binary candidate set is empty".into()))?
        .kernel
        .dtype;
    let byte_len = crate::bytes::byte_len(dtype, numel, OperationKind::Pointwise)?;
    let lhs_offset_i32 = checked_i32(lhs_plan.offset, "lhs offset")?;
    let rhs_offset_i32 = checked_i32(rhs_plan.offset, "rhs offset")?;
    let numel_i32 = checked_i32(numel, "element count")?;
    let strided_metadata = if layout == LayoutClass::Strided {
        Some((
            checked_i32_vec(&plan.output_shape, "shape")?,
            checked_i32_vec(&lhs_plan.strides, "lhs stride")?,
            checked_i32_vec(&rhs_plan.strides, "rhs stride")?,
            checked_i32(plan.output_shape.len(), "rank")?,
        ))
    } else {
        None
    };
    let stream = lhs_b.device.default_stream();

    let mut out_b =
        CudaBuffer {
            len: numel,
            dtype: lhs_b.dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|error| {
                Error::Msg(format!("CUDA output allocation failed: {error:?}"))
            })?),
            device: lhs_b.device.clone(),
            device_id,
        };

    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), out_shape.to_vec()));
    }

    unsafe {
        let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| Error::Msg("fresh CUDA output buffer was unexpectedly shared".into()))?;

        use cudarc::driver::PushKernelArg;
        match strided_metadata {
            None => {
                let mut launch = |candidate: &PreparedPointwiseKernel| -> Result<()> {
                    let config = launch_config(
                        numel,
                        candidate.kernel.elements_per_thread(),
                        candidate.candidate.block_size,
                    )?;
                    stream
                        .launch_builder(&candidate.function)
                        .arg(&*lhs_b.data)
                        .arg(&*rhs_b.data)
                        .arg(&mut *out_slice_u8)
                        .arg(&lhs_offset_i32)
                        .arg(&rhs_offset_i32)
                        .arg(&numel_i32)
                        .launch(config)
                        .map(|_| ())
                        .map_err(|error| Error::Msg(format!("Kernel launch failed: {error:?}")))
                };
                execute_pointwise_selection(&stream, selection, &prepared, &mut launch)
            }
            Some((out_shape_i32, lhs_strides, rhs_strides, ndim_i32)) => {
                let out_shape_dev = stream
                    .clone_htod(&out_shape_i32)
                    .map_err(|error| Error::Msg(format!("CUDA shape upload failed: {error:?}")))?;
                let lhs_strides_dev = stream.clone_htod(&lhs_strides).map_err(|error| {
                    Error::Msg(format!("CUDA lhs stride upload failed: {error:?}"))
                })?;
                let rhs_strides_dev = stream.clone_htod(&rhs_strides).map_err(|error| {
                    Error::Msg(format!("CUDA rhs stride upload failed: {error:?}"))
                })?;
                let mut launch = |candidate: &PreparedPointwiseKernel| -> Result<()> {
                    let config = launch_config(
                        numel,
                        candidate.kernel.elements_per_thread(),
                        candidate.candidate.block_size,
                    )?;
                    stream
                        .launch_builder(&candidate.function)
                        .arg(&*lhs_b.data)
                        .arg(&*rhs_b.data)
                        .arg(&mut *out_slice_u8)
                        .arg(&out_shape_dev)
                        .arg(&lhs_strides_dev)
                        .arg(&rhs_strides_dev)
                        .arg(&lhs_offset_i32)
                        .arg(&rhs_offset_i32)
                        .arg(&numel_i32)
                        .arg(&ndim_i32)
                        .launch(config)
                        .map(|_| ())
                        .map_err(|error| Error::Msg(format!("Kernel launch failed: {error:?}")))
                };
                execute_pointwise_selection(&stream, selection, &prepared, &mut launch)
            }
        }?;
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_b),
        out_shape.to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_dtype_gate_matches_rendered_float_family() {
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            validate_elementwise_dtype(dtype.descriptor(), "test").unwrap();
        }
        assert!(matches!(
            validate_elementwise_dtype(DTypeId::I64.descriptor(), "test"),
            Err(Error::UnsupportedDType { .. })
        ));
    }

    #[test]
    fn rendered_kernel_abi_matches_storage_dtype() {
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            let kernel = crate::kernel::render_cuda_unary("neg", "-x", dtype).unwrap();
            validate_kernel_abi(&kernel, dtype.descriptor()).unwrap();
            assert!(
                validate_kernel_abi(&kernel, DTypeId::F32.descriptor()).is_err()
                    || dtype == DTypeId::F32
            );
        }
    }

    #[test]
    fn launch_metadata_conversion_rejects_truncation_and_overflow() {
        assert_eq!(checked_i32(i32::MAX as usize, "shape").unwrap(), i32::MAX);
        assert!(checked_i32(i32::MAX as usize + 1, "shape").is_err());
        assert_eq!(
            checked_i32_vec(&[0, 7, 31], "stride").unwrap(),
            vec![0, 7, 31]
        );
        assert!(
            crate::bytes::byte_len(
                DTypeId::F16.descriptor(),
                usize::MAX,
                OperationKind::Pointwise
            )
            .is_err()
        );
    }

    #[test]
    fn launch_grid_rounds_up_without_narrowing() {
        assert_eq!(launch_config(1, 1, 256).unwrap().grid_dim.0, 1);
        assert_eq!(launch_config(256, 1, 256).unwrap().grid_dim.0, 1);
        assert_eq!(launch_config(257, 1, 256).unwrap().grid_dim.0, 2);
        assert_eq!(launch_config(1025, 4, 256).unwrap().grid_dim.0, 2);
        assert!(launch_config(1024, 1, 48).is_err());
        if usize::BITS > u32::BITS {
            assert!(launch_config((u32::MAX as usize + 1) * 4, 4, 256).is_err());
        }
    }

    #[test]
    fn strategy_selection_separates_packed_alignment_from_scalar_unrolling() {
        assert_eq!(
            select_unary_strategy(DTypeId::F32.descriptor(), LayoutClass::Contiguous, 4096, 0)
                .unwrap(),
            PointwiseStrategy::Packed {
                vector_width: 4,
                block_size: 256
            }
        );
        assert_eq!(
            select_unary_strategy(DTypeId::F32.descriptor(), LayoutClass::Contiguous, 4096, 1)
                .unwrap(),
            PointwiseStrategy::Scalar {
                unroll_width: 4,
                block_size: 256
            }
        );
        assert_eq!(
            select_binary_strategy(
                DTypeId::F16.descriptor(),
                LayoutClass::ScalarLeft,
                4096,
                1,
                2
            )
            .unwrap(),
            PointwiseStrategy::Packed {
                vector_width: 2,
                block_size: 256
            }
        );
        assert_eq!(
            select_binary_strategy(DTypeId::F64.descriptor(), LayoutClass::Strided, 4096, 0, 0)
                .unwrap(),
            PointwiseStrategy::Scalar {
                unroll_width: 1,
                block_size: 256
            }
        );
        assert_eq!(
            select_binary_strategy(
                DTypeId::F32.descriptor(),
                LayoutClass::Contiguous,
                1023,
                0,
                0
            )
            .unwrap(),
            PointwiseStrategy::Scalar {
                unroll_width: 1,
                block_size: 256
            }
        );
    }
}
