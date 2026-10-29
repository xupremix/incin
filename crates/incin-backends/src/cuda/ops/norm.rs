//! CUDA `layer_norm` and `batch_norm`.
//!
//! `softmax` and `rms_norm` are not here: both are answered in
//! `cuda::executor` by composing already-implemented pointwise and reduction
//! `Execute<O>` calls rather than by a dedicated kernel, and their capability
//! rows say `Composed` because of it. `layer_norm` and `batch_norm` are real
//! kernels because a Welford row reduction and a per-channel affine pass are
//! each one launch; composing either from primitives would mean materializing
//! an intermediate the size of the input for no reason a fused kernel needs.
//!
//! `batch_norm` ships three launches over one rendered module: the
//! inference form (running statistics, one thread per element), the
//! training form (per-channel batch statistics, one block per channel,
//! issue #123), and the fused backward replaying the training form's
//! saved statistics.

use crate::cuda::checked_i32;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::PrecisionRequest;
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeDescriptor;

/// Contiguity and bounds as plain metadata, so the check is testable
/// without a device handle behind it.
///
/// Every kernel here addresses storage linearly, so a strided or sliced
/// view would be read as if it were packed. This is the first refusal of
/// both `launch_batch_norm` entry points, before any allocation or launch.
pub(crate) fn validate_contiguous_meta(
    shape: &[usize],
    strides: &[usize],
    offset_elements: usize,
    buffer_len: usize,
    name: &'static str,
) -> Result<usize> {
    let numel = crate::bytes::checked_numel(shape)?;
    let expected = crate::layout::contiguous_strides(shape);
    if strides != expected.strides() {
        return Err(Error::Msg(format!(
            "CUDA normalization requires contiguous {name} storage"
        )));
    }
    let end = offset_elements
        .checked_add(numel)
        .ok_or_else(|| Error::Msg(format!("CUDA normalization {name} storage bound overflow")))?;
    if end > buffer_len {
        return Err(Error::Msg(format!(
            "CUDA normalization {name} view ends at {end}, but buffer length is {buffer_len}"
        )));
    }
    Ok(numel)
}

fn validate_contiguous(storage: &CudaStorage, name: &'static str) -> Result<usize> {
    validate_contiguous_meta(
        &storage.shape,
        &storage.strides,
        storage.offset_elements,
        storage.buffer.len,
        name,
    )
}

/// An affine parameter must live in the input's own dtype: the kernel
/// loads both through one scalar spec, so a crossed wire here would
/// reinterpret the parameter's bytes as the input's element type.
pub(crate) fn ensure_matching_dtype(expected: DTypeDescriptor, got: DTypeDescriptor) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::DTypeStorageMismatch { expected, got })
    }
}

/// Input and parameter must sit on the same device: the launch passes one
/// pointer of each per kernel, with no staging between contexts.
pub(crate) fn ensure_matching_cuda_device(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::DeviceMismatch {
            left: incin_core::tensor::device::DeviceId::cuda(expected),
            right: incin_core::tensor::device::DeviceId::cuda(got),
        })
    }
}

fn validate_parameter(
    input: &CudaStorage,
    parameter: &CudaStorage,
    needed: usize,
    name: &'static str,
) -> Result<()> {
    ensure_matching_dtype(input.buffer.dtype, parameter.buffer.dtype)?;
    ensure_matching_cuda_device(input.buffer.device_id, parameter.buffer.device_id)?;
    let numel = validate_contiguous(parameter, name)?;
    if numel < needed {
        return Err(Error::Msg(format!(
            "CUDA normalization {name} has {numel} elements, but {needed} are required"
        )));
    }
    Ok(())
}

/// Per-channel geometry of a batch-norm operand (issue #123).
///
/// Statistics run over every axis but the channel one: for `[N, C, H, W]`
/// that is `N * H * W` elements into one mean and one variance per channel.
/// `channel_axis` is 1 for a batched input and 0 for the rank-one parameter
/// vectors the capability row also admits; `spatial_size` flattens axes 2
/// onward, which is the stride the kernels' `(batch, spatial)` indexing
/// walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchNormGeometry {
    /// Extent of the channel axis.
    pub(crate) num_channels: usize,
    /// Product of every axis after the channel one (1 when there are none).
    pub(crate) spatial_size: usize,
    /// Elements reduced into each channel's statistics: every axis except
    /// the channel one, i.e. the batch and all spatial positions.
    pub(crate) channel_elements: usize,
}

/// The training launches' shared geometry, refusing the shapes whose
/// statistics are undefined.
///
/// A zero count is exactly the shape CPU's `batch_norm_training_impl`
/// refuses ("training mode needs at least one element per channel"), and
/// zero channels would ask the launcher for a zero-block grid and a
/// zero-length statistics buffer. The inference path keeps its own,
/// more permissive geometry: an all-empty input there still produces an
/// empty output, and this function's contract is training only.
pub(crate) fn batch_norm_geometry(shape: &[usize]) -> Result<BatchNormGeometry> {
    if shape.is_empty() {
        return Err(Error::Msg("CUDA batch norm requires rank >= 1".into()));
    }
    let channel_axis = usize::from(shape.len() > 1);
    let num_channels = shape[channel_axis];
    let spatial_size = if shape.len() > 2 {
        crate::bytes::checked_numel(&shape[2..])?
    } else {
        1
    };
    let mut channel_elements = 1usize;
    for (axis, &extent) in shape.iter().enumerate() {
        if axis != channel_axis {
            channel_elements = channel_elements.checked_mul(extent).ok_or_else(|| {
                Error::Msg("CUDA batch norm channel element count overflow".into())
            })?;
        }
    }
    if num_channels == 0 {
        return Err(Error::Msg(
            "CUDA batch norm training requires at least one channel".into(),
        ));
    }
    if channel_elements == 0 {
        return Err(Error::Msg(
            "CUDA batch norm: training mode needs at least one element per channel".into(),
        ));
    }
    Ok(BatchNormGeometry {
        num_channels,
        spatial_size,
        channel_elements,
    })
}

/// The shapes the fused batch-norm backward allocates, named exactly as
/// the tape entry's `input_ids` order: input, then weight and bias when
/// the forward ran with them (issue #123).
///
/// Each gradient takes its operand's own shape, not the geometry-derived
/// channel extent: a `[C]` weight and a `[1, C, 1, 1]` bias both reduce
/// back to whatever shape they arrived in, which is what the CPU
/// composition's reshape/unbroadcast backward produces and what this
/// launcher must match for a gradient hand-off between backends to line
/// up. Kept as one function so a host-side test can pin the plan against
/// the CPU reference without a device.
pub(crate) fn batch_norm_grad_shapes(
    input_shape: &[usize],
    weight_shape: Option<&[usize]>,
    bias_shape: Option<&[usize]>,
) -> BatchNormGradShapes {
    BatchNormGradShapes {
        input: input_shape.to_vec(),
        weight: weight_shape.map(<[usize]>::to_vec),
        bias: bias_shape.map(<[usize]>::to_vec),
    }
}

/// Output shapes of [`batch_norm_grad_shapes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchNormGradShapes {
    pub(crate) input: alloc::vec::Vec<usize>,
    pub(crate) weight: Option<alloc::vec::Vec<usize>>,
    pub(crate) bias: Option<alloc::vec::Vec<usize>>,
}

#[cfg(feature = "cuda")]
fn ensure_normalization_loaded(
    device_id: usize,
    kernel: &crate::kernel::RenderedKernel,
) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, &kernel.cache_key).is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            &kernel.entry_point,
            &kernel.source,
            &kernel.cache_key,
        )?;
    }
    Ok(())
}

struct NormalizationLaunchSelection {
    candidate: crate::tuning::LaunchCandidate,
    #[cfg(feature = "autotune")]
    tuning_permit: Option<crate::tuning::TuningPermit>,
}

fn normalization_launch_selection(
    context: &cudarc::driver::CudaContext,
    kernel: &crate::kernel::RenderedKernel,
    batch_size: usize,
    norm_size: usize,
    is_layer_norm: bool,
) -> Result<NormalizationLaunchSelection> {
    let candidates = crate::tuning::normalization_candidates(is_layer_norm);
    let fallback = crate::tuning::default_normalization_candidate(&candidates)?;
    #[cfg(feature = "autotune")]
    {
        let key = crate::tuning::TuningKey::new(
            crate::tuning::identity::TuningEnvironmentFingerprint::<
                incin_core::tensor::device::Cuda,
            >::from_cuda_context(context)?
            .erase(),
            &kernel.key,
            crate::tuning::WorkloadBucket::normalization(batch_size, norm_size),
        );
        match crate::tuning::claim_tuning(key, &candidates)? {
            crate::tuning::TuningDecision::Cached(tuned) => Ok(NormalizationLaunchSelection {
                candidate: tuned.candidate,
                tuning_permit: None,
            }),
            crate::tuning::TuningDecision::Measure(permit) => Ok(NormalizationLaunchSelection {
                candidate: fallback,
                tuning_permit: Some(permit),
            }),
        }
    }
    #[cfg(not(feature = "autotune"))]
    {
        let _ = (context, kernel, batch_size, norm_size, is_layer_norm);
        Ok(NormalizationLaunchSelection {
            candidate: fallback,
        })
    }
}

#[cfg(feature = "cuda")]
fn empirically_select_normalization_candidate<F>(
    stream: &cudarc::driver::CudaStream,
    selection: NormalizationLaunchSelection,
    is_layer_norm: bool,
    mut launch: F,
) -> Result<crate::tuning::LaunchCandidate>
where
    F: FnMut(crate::tuning::LaunchCandidate) -> Result<()>,
{
    #[cfg(feature = "autotune")]
    if let Some(permit) = selection.tuning_permit {
        let candidates = crate::tuning::normalization_candidates(is_layer_norm);
        let mut measurements = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            measurements.push(crate::tuning::measure_cuda_candidate(
                stream,
                candidate,
                || launch(candidate),
            )?);
        }
        return Ok(permit.record(&measurements)?.candidate);
    }
    #[cfg(not(feature = "autotune"))]
    let _ = (stream, is_layer_norm, &mut launch);
    Ok(selection.candidate)
}

/// Per-row statistics a layer-norm backward replays.
///
/// The forward kernel writes these when asked; the recipe below reads them
/// back rather than recomputing mean and variance, which would silently run
/// under different numerical conditions than the Welford pass that produced
/// the output. Both buffers hold one compute-precision value per batch row.
#[cfg(feature = "cuda")]
pub(crate) struct LayerNormStats {
    pub(crate) mean: CudaStorage,
    pub(crate) rstd: CudaStorage,
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_layer_norm(
    input: &CudaStorage,
    weight: &CudaStorage,
    bias: Option<&CudaStorage>,
    eps: f32,
    save_stats: bool,
) -> Result<(CudaStorage, Option<LayerNormStats>)> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    let norm_size = *input
        .shape
        .last()
        .ok_or_else(|| Error::Msg("CUDA layer norm requires rank >= 1".into()))?;
    if norm_size == 0 {
        return Err(Error::Msg(
            "CUDA layer norm is undefined for an empty normalized axis".into(),
        ));
    }
    validate_parameter(input, weight, norm_size, "weight")?;
    if let Some(bias) = bias {
        validate_parameter(input, bias, norm_size, "bias")?;
    }
    let req = PrecisionRequest::new(
        incin_core::shapes::error::OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "layer_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("layer_norm", builtin_id)?;
    if Some(kernel.dtype) != buffer.dtype.builtin_id()
        || kernel.element_size
            != buffer
                .dtype
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("invalid scalar bytes".into()))?
    {
        return Err(Error::Msg(
            "CUDA layer norm kernel/storage ABI mismatch".into(),
        ));
    }
    let batch_size = input_numel / norm_size;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: input_numel,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    input_numel,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA layer norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if input_numel == 0 {
        // No rows produced statistics and none ever will: the recipe replays
        // an empty stat list by returning zero gradients without launching.
        return Ok((
            CudaStorage::new(Arc::new(output), input.shape.to_vec()),
            None,
        ));
    }

    // Per-row statistics live in compute precision, not storage precision: a
    // half-precision mean would round the very values backward replays.
    // Without a recording caller there is nowhere to keep them, so a
    // single-element scratch stands in behind the `save_stats` flag rather
    // than a null pointer, which the launch builder cannot spell.
    let compute_dtype = policy.compute;
    let stats_len = if save_stats { batch_size } else { 1 };
    let alloc_stat = |tag: &'static str| -> Result<CudaBuffer> {
        let bytes = crate::bytes::byte_len(compute_dtype, stats_len, OperationKind::Normalization)?;
        Ok(CudaBuffer {
            len: stats_len,
            dtype: compute_dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(bytes).map_err(|error| {
                Error::Msg(format!(
                    "CUDA layer norm {tag} allocation failed: {error:?}"
                ))
            })?),
            device: buffer.device.clone(),
            device_id: buffer.device_id,
        })
    };
    let mut mean_buf = alloc_stat("mean")?;
    let mut rstd_buf = alloc_stat("rstd")?;

    let selection =
        normalization_launch_selection(&buffer.device, &kernel, batch_size, norm_size, true)?;
    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let bias_storage = bias.unwrap_or(weight);
    let has_bias = i32::from(bias.is_some());
    let save_stats_flag = i32::from(save_stats);

    // SAFETY: the selected kernel's checked launch candidate and validated
    // tensor metadata bound all views; output and both stat buffers are fresh
    // unique allocations. The stat views stay `u8`: the kernel reinterprets
    // them as its compute type, which is what `byte_len` sized them for.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA layer norm output was unexpectedly shared".into())
        })?;
        let mean_u8: &cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut mean_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA layer norm mean buffer was unexpectedly shared".into())
            })?;
        let rstd_u8: &cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut rstd_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA layer norm rstd buffer was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
            let block_size = u32::from(candidate.block_size);
            let warp_count = (block_size as usize) / 32;
            let acc_bytes = policy
                .accumulator
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("CUDA layer norm invalid accumulator encoding".into()))?;
            let shared_bytes = warp_count
                .checked_mul(acc_bytes)
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| bytes.checked_add(warp_count * core::mem::size_of::<i32>()))
                .and_then(|bytes| u32::try_from(bytes).ok())
                .ok_or_else(|| Error::Msg("CUDA layer norm shared-memory size overflow".into()))?;
            let grid = u32::try_from(batch_size).map_err(|_| {
                Error::Msg("CUDA layer norm batch count exceeds u32 grid ABI".into())
            })?;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            stream
                .launch_builder(&function)
                .arg(&*buffer.data)
                .arg(&*weight.buffer.data)
                .arg(&*bias_storage.buffer.data)
                .arg(&mut *output_u8)
                .arg(&eps)
                .arg(&checked_i32(norm_size, "normalized axis length")?)
                .arg(&has_bias)
                .arg(&checked_i32(batch_size, "batch count")?)
                .arg(&checked_i32(input.offset_elements, "input offset")?)
                .arg(&checked_i32(weight.offset_elements, "weight offset")?)
                .arg(&checked_i32(bias_storage.offset_elements, "bias offset")?)
                .arg(mean_u8)
                .arg(rstd_u8)
                .arg(&save_stats_flag)
                .launch(config)
                .map(|_| ())
                .map_err(|error| Error::Msg(format!("CUDA layer norm launch failed: {error:?}")))
        };
        let candidate =
            empirically_select_normalization_candidate(&stream, selection, true, &mut launch)?;
        launch(candidate)?;
    }

    let out = CudaStorage::new(Arc::new(output), input.shape.to_vec());
    let stats = save_stats.then(|| LayerNormStats {
        mean: CudaStorage::new(Arc::new(mean_buf), alloc::vec![batch_size]),
        rstd: CudaStorage::new(Arc::new(rstd_buf), alloc::vec![batch_size]),
    });
    Ok((out, stats))
}

/// Gradients of a layer-norm forward: input, weight, and bias when present.
///
/// `mean`/`rstd` are the forward's own per-row statistics, never recomputed
/// here. All three gradients derive from them plus the upstream gradient, in
/// one row-per-block launch.
#[cfg(feature = "cuda")]
pub(crate) struct LayerNormGrads {
    pub(crate) input: CudaStorage,
    pub(crate) weight: CudaStorage,
    pub(crate) bias: Option<CudaStorage>,
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_layer_norm_backward(
    grad_output: &CudaStorage,
    input: &CudaStorage,
    weight: &CudaStorage,
    mean: &CudaStorage,
    rstd: &CudaStorage,
    has_bias: bool,
) -> Result<LayerNormGrads> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    validate_contiguous(grad_output, "grad_output")?;
    let norm_size = *input
        .shape
        .last()
        .ok_or_else(|| Error::Msg("CUDA layer norm backward requires rank >= 1".into()))?;
    if norm_size == 0 {
        return Err(Error::Msg(
            "CUDA layer norm backward is undefined for an empty normalized axis".into(),
        ));
    }
    if grad_output.shape != input.shape {
        return Err(Error::ShapeMismatch {
            op: "layer_norm_backward",
            expected: input.shape.to_vec(),
            got: grad_output.shape.to_vec(),
            msg: "the upstream gradient must match the forward input shape element-wise".into(),
        });
    }
    if grad_output.buffer.dtype != buffer.dtype {
        return Err(Error::DTypeStorageMismatch {
            expected: buffer.dtype,
            got: grad_output.buffer.dtype,
        });
    }
    validate_parameter(input, weight, norm_size, "weight")?;
    let batch_size = input_numel / norm_size;
    let req = PrecisionRequest::new(
        incin_core::shapes::error::OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    // The statistics must be what the forward wrote: compute-precision, one
    // value per batch row. Anything else is a crossed-wires caller, and the
    // kernel would read past it or misinterpret the bytes.
    for (stats, name) in [(mean, "mean"), (rstd, "rstd")] {
        if stats.buffer.dtype != policy.compute {
            return Err(Error::DTypeStorageMismatch {
                expected: policy.compute,
                got: stats.buffer.dtype,
            });
        }
        if stats.shape != [batch_size] {
            return Err(Error::ShapeMismatch {
                op: "layer_norm_backward",
                expected: alloc::vec![batch_size],
                got: stats.shape.to_vec(),
                msg: alloc::format!("saved layer norm {name} must hold one value per batch row"),
            });
        }
    }
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "layer_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("layer_norm", builtin_id)?;
    let stream = buffer.device.default_stream();
    let alloc = |dtype: incin_core::tensor::dtype::DTypeDescriptor,
                 len: usize,
                 tag: &'static str|
     -> Result<CudaBuffer> {
        let bytes = crate::bytes::byte_len(dtype, len, OperationKind::Normalization)?;
        Ok(CudaBuffer {
            len,
            dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(bytes).map_err(|error| {
                Error::Msg(format!(
                    "CUDA layer norm backward {tag} allocation failed: {error:?}"
                ))
            })?),
            device: buffer.device.clone(),
            device_id: buffer.device_id,
        })
    };
    let mut dx_buf = alloc(buffer.dtype, input_numel, "input gradient")?;
    let mut dw_buf = alloc(policy.compute, norm_size, "weight gradient")?;
    let mut db_buf = alloc(policy.compute, norm_size, "bias gradient")?;
    if input_numel == 0 {
        // No rows ran forward, so no kernel runs backward: zero gradients of
        // the right shapes, matching what a launch over zero rows would add
        // into (nothing).
        let shape_of = |storage: &CudaStorage| storage.shape.to_vec();
        return Ok(LayerNormGrads {
            input: CudaStorage::new(Arc::new(dx_buf), shape_of(input)),
            weight: CudaStorage::new(Arc::new(dw_buf), shape_of(weight)),
            bias: has_bias.then(|| CudaStorage::new(Arc::new(db_buf), alloc::vec![norm_size])),
        });
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let backward_entry = alloc::format!("{}_backward", kernel.entry_point);
    let function = dispatcher.get_function(&kernel.cache_key, &backward_entry)?;
    let has_bias_flag = i32::from(has_bias);
    let compute_bytes = policy
        .compute
        .encoding()
        .scalar_bytes()
        .ok_or_else(|| Error::Msg("CUDA layer norm invalid compute encoding".into()))?;
    let block_size: u32 = 256;
    let warp_count = (block_size as usize) / 32;
    let shared_bytes = (2 * warp_count)
        .checked_mul(compute_bytes)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| Error::Msg("CUDA layer norm backward shared-memory size overflow".into()))?;
    let grid = u32::try_from(batch_size)
        .map_err(|_| Error::Msg("CUDA layer norm batch count exceeds u32 grid ABI".into()))?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    // SAFETY: validated metadata bounds every view; all three outputs are
    // fresh unique allocations, zeroed before the atomic accumulation. Every
    // buffer travels as its bytes: the kernel reinterprets them as the dtypes
    // `byte_len` sized them for.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let dx_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut dx_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA layer norm input gradient was unexpectedly shared".into())
            })?;
        let dw_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut dw_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA layer norm weight gradient was unexpectedly shared".into())
            })?;
        let db_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut db_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA layer norm bias gradient was unexpectedly shared".into())
            })?;
        stream
            .launch_builder(&function)
            .arg(&*grad_output.buffer.data)
            .arg(&*buffer.data)
            .arg(&*weight.buffer.data)
            .arg(&*mean.buffer.data)
            .arg(&*rstd.buffer.data)
            .arg(&mut *dx_u8)
            .arg(&mut *dw_u8)
            .arg(&mut *db_u8)
            .arg(&checked_i32(norm_size, "normalized axis length")?)
            .arg(&checked_i32(batch_size, "batch count")?)
            .arg(&has_bias_flag)
            .arg(&checked_i32(
                grad_output.offset_elements,
                "grad_output offset",
            )?)
            .arg(&checked_i32(input.offset_elements, "input offset")?)
            .arg(&checked_i32(weight.offset_elements, "weight offset")?)
            .launch(config)
            .map(|_| ())
            .map_err(|error| {
                Error::Msg(format!("CUDA layer norm backward launch failed: {error:?}"))
            })?;
    }

    let weight_shape = weight.shape.to_vec();
    Ok(LayerNormGrads {
        input: CudaStorage::new(Arc::new(dx_buf), input.shape.to_vec()),
        weight: CudaStorage::new(Arc::new(dw_buf), weight_shape),
        bias: has_bias.then(|| CudaStorage::new(Arc::new(db_buf), alloc::vec![norm_size])),
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_batch_norm(
    input: &CudaStorage,
    weight: Option<&CudaStorage>,
    bias: Option<&CudaStorage>,
    running_mean: Option<&CudaStorage>,
    running_variance: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
    let buffer = &*input.buffer;
    let total_elements = validate_contiguous(input, "input")?;
    if input.shape.is_empty() {
        return Err(Error::Msg("CUDA batch norm requires rank >= 1".into()));
    }
    let channel_axis = usize::from(input.shape.len() > 1);
    let num_channels = input.shape[channel_axis];
    let spatial_size = if input.shape.len() > 2 {
        crate::bytes::checked_numel(&input.shape[2..])?
    } else {
        1
    };
    for (parameter, name) in [
        (weight, "weight"),
        (bias, "bias"),
        (running_mean, "running mean"),
        (running_variance, "running variance"),
    ] {
        if let Some(parameter) = parameter {
            validate_parameter(input, parameter, num_channels, name)?;
        }
    }
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "batch_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "batch_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("batch_norm", builtin_id)?;
    let selection = normalization_launch_selection(
        &buffer.device,
        &kernel,
        total_elements,
        num_channels,
        false,
    )?;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: total_elements,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    total_elements,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA batch norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if total_elements == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()));
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let weight_storage = weight.unwrap_or(input);
    let bias_storage = bias.unwrap_or(input);
    let mean_storage = running_mean.unwrap_or(input);
    let variance_storage = running_variance.unwrap_or(input);

    // SAFETY: validated batch-norm metadata and the checked launch candidate
    // bound device accesses; output is freshly allocated and uniquely owned.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA batch norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
            let block_size = u32::from(candidate.block_size);
            let work_items = u32::try_from(total_elements).map_err(|_| {
                Error::Msg("CUDA batch norm element count exceeds u32 grid ABI".into())
            })?;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (work_items.div_ceil(block_size), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            stream
                .launch_builder(&function)
                .arg(&*buffer.data)
                .arg(&*weight_storage.buffer.data)
                .arg(&*bias_storage.buffer.data)
                .arg(&*mean_storage.buffer.data)
                .arg(&*variance_storage.buffer.data)
                .arg(&mut *output_u8)
                .arg(&eps)
                .arg(&checked_i32(num_channels, "channel count")?)
                .arg(&checked_i32(spatial_size, "spatial size")?)
                .arg(&checked_i32(total_elements, "element count")?)
                .arg(&i32::from(weight.is_some()))
                .arg(&i32::from(bias.is_some()))
                .arg(&i32::from(running_mean.is_some()))
                .arg(&i32::from(running_variance.is_some()))
                .arg(&checked_i32(input.offset_elements, "input offset")?)
                .arg(&checked_i32(
                    weight_storage.offset_elements,
                    "weight offset",
                )?)
                .arg(&checked_i32(bias_storage.offset_elements, "bias offset")?)
                .arg(&checked_i32(mean_storage.offset_elements, "mean offset")?)
                .arg(&checked_i32(
                    variance_storage.offset_elements,
                    "variance offset",
                )?)
                .launch(config)
                .map(|_| ())
                .map_err(|error| Error::Msg(format!("CUDA batch norm launch failed: {error:?}")))
        };
        let candidate =
            empirically_select_normalization_candidate(&stream, selection, false, &mut launch)?;
        launch(candidate)?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()))
}

/// Per-channel statistics a batch-norm backward replays (issue #123).
///
/// The training forward writes these when asked; the backward below reads
/// them back rather than recomputing mean and variance, which would
/// silently run under different numerical conditions than the Welford
/// pass that produced the output. Both buffers hold one compute-precision
/// value per channel -- the batch statistics of *this* batch, never the
/// running estimates the inference path reads.
#[cfg(feature = "cuda")]
pub(crate) struct BatchNormStats {
    pub(crate) mean: CudaStorage,
    pub(crate) rstd: CudaStorage,
}

/// Training-mode batch-norm forward: per-channel batch statistics, fused
/// affine, and optional statistic saving for the backward (issue #123).
///
/// One block per channel reduces that channel over the batch and every
/// spatial position (`channel_elements` values) with Welford, applies
/// `y = (x - mean) * rstd * weight + bias`, and -- when `save_stats` is
/// set -- keeps mean and inverse standard deviation in compute precision
/// for [`launch_batch_norm_backward`] to replay. `save_stats` comes from
/// the ambient grad mode: a forward with no recording caller has nowhere
/// to keep the statistics, so a single-element scratch stands in behind
/// the flag rather than a null pointer, which the launch builder cannot
/// spell (the same arrangement `launch_layer_norm` uses).
///
/// Unlike the inference launch, an empty reduction is an error here, not
/// a zero output: there would be no statistics to normalize by, and CPU's
/// training kernel refuses the same shapes by name.
#[cfg(feature = "cuda")]
pub(crate) fn launch_batch_norm_training(
    input: &CudaStorage,
    weight: Option<&CudaStorage>,
    bias: Option<&CudaStorage>,
    eps: f32,
    save_stats: bool,
) -> Result<(CudaStorage, Option<BatchNormStats>)> {
    let buffer = &*input.buffer;
    let total_elements = validate_contiguous(input, "input")?;
    let BatchNormGeometry {
        num_channels,
        spatial_size,
        channel_elements,
    } = batch_norm_geometry(&input.shape)?;
    for (parameter, name) in [(weight, "weight"), (bias, "bias")] {
        if let Some(parameter) = parameter {
            validate_parameter(input, parameter, num_channels, name)?;
        }
    }
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "batch_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "batch_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("batch_norm", builtin_id)?;
    if Some(kernel.dtype) != buffer.dtype.builtin_id() {
        return Err(Error::Msg(
            "CUDA batch norm training kernel/storage ABI mismatch".into(),
        ));
    }
    let req = PrecisionRequest::new(
        OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    debug_assert_eq!(policy.accumulator, policy.compute);

    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: total_elements,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    total_elements,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!(
                        "CUDA batch norm training allocation failed: {error:?}"
                    ))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };

    // Statistics live in compute precision, not storage precision: a
    // half-precision mean would round the very values backward replays.
    let compute_dtype = policy.compute;
    let stats_len = if save_stats { num_channels } else { 1 };
    let alloc_stat = |tag: &'static str| -> Result<CudaBuffer> {
        let bytes = crate::bytes::byte_len(compute_dtype, stats_len, OperationKind::Normalization)?;
        Ok(CudaBuffer {
            len: stats_len,
            dtype: compute_dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(bytes).map_err(|error| {
                Error::Msg(format!(
                    "CUDA batch norm training {tag} allocation failed: {error:?}"
                ))
            })?),
            device: buffer.device.clone(),
            device_id: buffer.device_id,
        })
    };
    let mut mean_buf = alloc_stat("mean")?;
    let mut rstd_buf = alloc_stat("rstd")?;

    // Fixed geometry, no autotune: this launch is one block per channel,
    // a different shape from the elementwise inference grid whose
    // `WorkloadBucket` the tuning cache is keyed on, so reusing that
    // bucket's decision here would measure the wrong launch. 256 is the
    // same fallback the layer_norm launches standardize on.
    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let training_entry = alloc::format!("{}_training", kernel.entry_point);
    let function = dispatcher.get_function(&kernel.cache_key, &training_entry)?;
    let weight_storage = weight.unwrap_or(input);
    let bias_storage = bias.unwrap_or(input);
    let has_weight = i32::from(weight.is_some());
    let has_bias = i32::from(bias.is_some());
    let save_stats_flag = i32::from(save_stats);

    // SAFETY: validated batch-norm metadata bounds every view; output and
    // both statistic buffers are fresh unique allocations. The stat views
    // stay `u8`: the kernel reinterprets them as its compute type, which
    // is what `byte_len` sized them for. A substituted weight/bias pointer
    // is only ever loaded behind its own `has_*` flag.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA batch norm training output was unexpectedly shared".into())
        })?;
        let mean_u8: &cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut mean_buf.data).ok_or_else(|| {
                Error::Msg(
                    "fresh CUDA batch norm training mean buffer was unexpectedly shared".into(),
                )
            })?;
        let rstd_u8: &cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut rstd_buf.data).ok_or_else(|| {
                Error::Msg(
                    "fresh CUDA batch norm training rstd buffer was unexpectedly shared".into(),
                )
            })?;
        use cudarc::driver::PushKernelArg;
        let block_size: u32 = 256;
        let warp_count = (block_size as usize) / 32;
        let acc_bytes = policy
            .accumulator
            .encoding()
            .scalar_bytes()
            .ok_or_else(|| {
                Error::Msg("CUDA batch norm training invalid accumulator encoding".into())
            })?;
        let shared_bytes = warp_count
            .checked_mul(acc_bytes)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_add(warp_count * core::mem::size_of::<i32>()))
            .and_then(|bytes| u32::try_from(bytes).ok())
            .ok_or_else(|| {
                Error::Msg("CUDA batch norm training shared-memory size overflow".into())
            })?;
        let grid = crate::cuda::checked_u32(num_channels, "batch norm training channel count")?;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&*weight_storage.buffer.data)
            .arg(&*bias_storage.buffer.data)
            .arg(&mut *output_u8)
            .arg(&eps)
            .arg(&checked_i32(num_channels, "channel count")?)
            .arg(&checked_i32(spatial_size, "spatial size")?)
            .arg(&checked_i32(channel_elements, "channel element count")?)
            .arg(&has_weight)
            .arg(&has_bias)
            .arg(&checked_i32(input.offset_elements, "input offset")?)
            .arg(&checked_i32(
                weight_storage.offset_elements,
                "weight offset",
            )?)
            .arg(&checked_i32(bias_storage.offset_elements, "bias offset")?)
            .arg(mean_u8)
            .arg(rstd_u8)
            .arg(&save_stats_flag)
            .launch(config)
            .map(|_| ())
            .map_err(|error| {
                Error::Msg(format!("CUDA batch norm training launch failed: {error:?}"))
            })?;
    }

    let out = CudaStorage::new(Arc::new(output), input.shape.to_vec());
    let stats = save_stats.then(|| BatchNormStats {
        mean: CudaStorage::new(Arc::new(mean_buf), alloc::vec![num_channels]),
        rstd: CudaStorage::new(Arc::new(rstd_buf), alloc::vec![num_channels]),
    });
    Ok((out, stats))
}

/// Gradients of a batch-norm training forward: input, and weight/bias
/// when the forward ran with them (issue #123).
///
/// `mean`/`rstd` are the forward's own per-channel batch statistics, never
/// recomputed here. All three gradients derive from them plus the upstream
/// gradient, in one channel-per-block launch.
#[cfg(feature = "cuda")]
pub(crate) struct BatchNormGrads {
    pub(crate) input: CudaStorage,
    pub(crate) weight: Option<CudaStorage>,
    pub(crate) bias: Option<CudaStorage>,
}

/// Fused backward of [`launch_batch_norm_training`] (issue #123).
///
/// Replays the saved per-channel mean and inverse standard deviation
/// against the upstream gradient. With `xhat = (x - mean) * rstd` and
/// `gw = grad_output * weight`, the channel's means enter exactly as the
/// forward's reduction defined them:
///
/// * `dx = rstd * (gw - mean(gw) - xhat * mean(gw * xhat))`
/// * `dw = sum(grad_output * xhat)` per channel
/// * `db = sum(grad_output)` per channel
///
/// One block owns one channel, so `dw`/`db` are plain per-channel writes
/// rather than atomics: deterministic at the last ulp, unlike the
/// row-per-block layer_norm backward that shares columns across blocks.
/// Gradient shapes follow the operands (via [`batch_norm_grad_shapes`]),
/// matching what the CPU composition's reshape/unbroadcast backward
/// produces.
#[cfg(feature = "cuda")]
pub(crate) fn launch_batch_norm_backward(
    grad_output: &CudaStorage,
    input: &CudaStorage,
    weight: Option<&CudaStorage>,
    bias: Option<&CudaStorage>,
    mean: &CudaStorage,
    rstd: &CudaStorage,
) -> Result<BatchNormGrads> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    validate_contiguous(grad_output, "grad_output")?;
    let BatchNormGeometry {
        num_channels,
        spatial_size,
        channel_elements,
    } = batch_norm_geometry(&input.shape)?;
    if grad_output.shape != input.shape {
        return Err(Error::ShapeMismatch {
            op: "batch_norm_backward",
            expected: input.shape.to_vec(),
            got: grad_output.shape.to_vec(),
            msg: "the upstream gradient must match the forward input shape element-wise".into(),
        });
    }
    ensure_matching_dtype(buffer.dtype, grad_output.buffer.dtype)?;
    let mut weight_numel = num_channels;
    let mut bias_numel = num_channels;
    for (parameter, name, needed) in [
        (weight, "weight", &mut weight_numel),
        (bias, "bias", &mut bias_numel),
    ] {
        if let Some(parameter) = parameter {
            validate_parameter(input, parameter, num_channels, name)?;
            *needed = crate::bytes::checked_numel(&parameter.shape)?;
        }
    }
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "batch_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "batch_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("batch_norm", builtin_id)?;
    let req = PrecisionRequest::new(
        OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    // The statistics must be what the forward wrote: compute-precision,
    // one value per channel. Anything else is a crossed-wires caller, and
    // the kernel would read past them or misinterpret the bytes.
    for (stats, name) in [(mean, "mean"), (rstd, "rstd")] {
        ensure_matching_dtype(policy.compute, stats.buffer.dtype)?;
        if stats.shape != [num_channels] {
            return Err(Error::ShapeMismatch {
                op: "batch_norm_backward",
                expected: alloc::vec![num_channels],
                got: stats.shape.to_vec(),
                msg: alloc::format!("saved batch norm {name} must hold one value per channel"),
            });
        }
    }

    let stream = buffer.device.default_stream();
    let alloc = |dtype: DTypeDescriptor, len: usize, tag: &'static str| -> Result<CudaBuffer> {
        let bytes = crate::bytes::byte_len(dtype, len, OperationKind::Normalization)?;
        Ok(CudaBuffer {
            len,
            dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(bytes).map_err(|error| {
                Error::Msg(format!(
                    "CUDA batch norm backward {tag} allocation failed: {error:?}"
                ))
            })?),
            device: buffer.device.clone(),
            device_id: buffer.device_id,
        })
    };
    let shapes = batch_norm_grad_shapes(
        &input.shape,
        weight.map(|parameter| parameter.shape.as_ref()),
        bias.map(|parameter| parameter.shape.as_ref()),
    );
    let mut dx_buf = alloc(buffer.dtype, input_numel, "input gradient")?;
    let mut dw_buf = alloc(policy.compute, weight_numel, "weight gradient")?;
    let mut db_buf = alloc(policy.compute, bias_numel, "bias gradient")?;

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let backward_entry = alloc::format!("{}_backward", kernel.entry_point);
    let function = dispatcher.get_function(&kernel.cache_key, &backward_entry)?;
    let weight_storage = weight.unwrap_or(input);
    // The backward kernel takes no bias *values* -- `db` is a plain sum of
    // the upstream gradient -- so unlike the forward there is no bias
    // pointer to substitute when the operand is absent.
    let has_weight_flag = i32::from(weight.is_some());
    let has_bias_flag = i32::from(bias.is_some());
    let acc_bytes = policy
        .accumulator
        .encoding()
        .scalar_bytes()
        .ok_or_else(|| {
            Error::Msg("CUDA batch norm backward invalid accumulator encoding".into())
        })?;
    let block_size: u32 = 256;
    let warp_count = (block_size as usize) / 32;
    // Four sums per channel (g, g*w, g*xhat, g*w*xhat), one slot per warp.
    let shared_bytes = (warp_count * 4)
        .checked_mul(acc_bytes)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| Error::Msg("CUDA batch norm backward shared-memory size overflow".into()))?;
    let grid = crate::cuda::checked_u32(num_channels, "batch norm backward channel count")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    // SAFETY: validated metadata bounds every view; all three outputs are
    // fresh unique allocations, zeroed before any write. The absent
    // parameter buffers are real scratch sized for the full channel, so
    // the kernel's guarded writes always land somewhere valid. Every
    // buffer travels as its bytes: the kernel reinterprets them as the
    // dtypes `byte_len` sized them for.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let dx_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut dx_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA batch norm input gradient was unexpectedly shared".into())
            })?;
        let dw_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut dw_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA batch norm weight gradient was unexpectedly shared".into())
            })?;
        let db_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut db_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA batch norm bias gradient was unexpectedly shared".into())
            })?;
        stream
            .launch_builder(&function)
            .arg(&*grad_output.buffer.data)
            .arg(&*buffer.data)
            .arg(&*weight_storage.buffer.data)
            .arg(&*mean.buffer.data)
            .arg(&*rstd.buffer.data)
            .arg(&mut *dx_u8)
            .arg(&mut *dw_u8)
            .arg(&mut *db_u8)
            .arg(&checked_i32(num_channels, "channel count")?)
            .arg(&checked_i32(spatial_size, "spatial size")?)
            .arg(&checked_i32(channel_elements, "channel element count")?)
            .arg(&has_weight_flag)
            .arg(&has_bias_flag)
            .arg(&checked_i32(
                grad_output.offset_elements,
                "grad_output offset",
            )?)
            .arg(&checked_i32(input.offset_elements, "input offset")?)
            .arg(&checked_i32(
                weight_storage.offset_elements,
                "weight offset",
            )?)
            .launch(config)
            .map(|_| ())
            .map_err(|error| {
                Error::Msg(format!("CUDA batch norm backward launch failed: {error:?}"))
            })?;
    }

    Ok(BatchNormGrads {
        input: CudaStorage::new(Arc::new(dx_buf), shapes.input),
        // When the forward had no weight/bias the scratch allocation is
        // simply dropped with the closure that never ran: the kernel was
        // handed it above so no launch ever passes a null pointer, and
        // there is no gradient to hand back for an absent operand.
        weight: shapes
            .weight
            .map(|shape| CudaStorage::new(Arc::new(dw_buf), shape)),
        bias: shapes
            .bias
            .map(|shape| CudaStorage::new(Arc::new(db_buf), shape)),
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_rms_norm(
    input: &CudaStorage,
    weight: &CudaStorage,
    eps: f32,
    save_norm: bool,
) -> Result<(CudaStorage, Option<CudaStorage>)> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    let norm_size = *input
        .shape
        .last()
        .ok_or_else(|| Error::Msg("CUDA RMS norm requires rank >= 1".into()))?;
    if norm_size == 0 {
        return Err(Error::Msg(
            "CUDA RMS norm is undefined for an empty normalized axis".into(),
        ));
    }
    validate_parameter(input, weight, norm_size, "weight")?;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "rms_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "rms_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("rms_norm", builtin_id)?;
    let req = PrecisionRequest::new(
        incin_core::shapes::error::OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    let batch_size = input_numel / norm_size;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: input_numel,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    input_numel,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA RMS norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if input_numel == 0 {
        return Ok((
            CudaStorage::new(Arc::new(output), input.shape.to_vec()),
            None,
        ));
    }

    // One inverse norm factor per batch row, in compute precision, saved for
    // the backward recipe exactly like layer_norm's statistics. A scratch
    // stand-in behind the flag keeps inference launches valid.
    let compute_dtype = policy.compute;
    let norm_len = if save_norm { batch_size } else { 1 };
    let mut norm_buf = CudaBuffer {
        len: norm_len,
        dtype: compute_dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    compute_dtype,
                    norm_len,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA RMS norm factor allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    let save_norm_flag = i32::from(save_norm);

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;

    // SAFETY: Input, weight, and output buffers are validated for contiguous bounds and data types;
    // launch configuration uses checked block and grid dimensions with exclusive output buffer access.
    // The factor buffer travels as bytes sized for the compute dtype.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA RMS norm output was unexpectedly shared".into())
        })?;
        let norm_u8: &cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut norm_buf.data).ok_or_else(|| {
                Error::Msg("fresh CUDA RMS norm factor buffer was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        let block_size = 256u32;
        let warp_count = (block_size as usize).div_ceil(32);
        let shared_bytes = (warp_count * core::mem::size_of::<f32>()) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (
                crate::cuda::checked_u32(batch_size, "norm launch grid")?,
                1,
                1,
            ),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&*weight.buffer.data)
            .arg(&mut *output_u8)
            .arg(&eps)
            .arg(&checked_i32(norm_size, "norm size")?)
            .arg(&checked_i32(batch_size, "batch size")?)
            .arg(&checked_i32(input.offset_elements, "input offset")?)
            .arg(&checked_i32(weight.offset_elements, "weight offset")?)
            .arg(norm_u8)
            .arg(&save_norm_flag)
            .launch(config)
            .map_err(|error| Error::Msg(format!("CUDA RMS norm launch failed: {error:?}")))?;
    }

    let out = CudaStorage::new(Arc::new(output), input.shape.to_vec());
    let factor = save_norm.then(|| CudaStorage::new(Arc::new(norm_buf), alloc::vec![batch_size]));
    Ok((out, factor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_metadata_checks_reject_overflow() {
        assert!(crate::bytes::checked_numel(&[usize::MAX, 2]).is_err());
        assert!(
            crate::bytes::byte_len(
                incin_core::tensor::dtype::DTypeId::F16,
                usize::MAX,
                OperationKind::Normalization
            )
            .is_err()
        );
    }

    // The three refusals below are the ones `launch_batch_norm_training`
    // and `launch_batch_norm_backward` perform *before* any allocation or
    // kernel launch, factored onto plain metadata so they are testable
    // without a CUDA device (issue #123's host-side rejection contract).

    #[test]
    fn batch_norm_validations_reject_non_contiguous_metadata() {
        // A transposed view of a [2, 3] tensor: shape preserved, strides not
        // row-major, so the packed-addressing kernel would read it wrong.
        let err = validate_contiguous_meta(&[2, 3], &[2, 1], 0, 6, "input")
            .expect_err("non-row-major strides must refuse before launch");
        assert!(
            alloc::format!("{err:?}").contains("contiguous input storage"),
            "unexpected refusal: {err:?}"
        );
        // And a well-formed view whose tail runs past the allocation.
        let err = validate_contiguous_meta(&[2, 3], &[3, 1], 2, 6, "input")
            .expect_err("an offset past the buffer must refuse before launch");
        assert!(
            alloc::format!("{err:?}").contains("ends at 8"),
            "unexpected refusal: {err:?}"
        );
        // The accepting case, so the two refusals above are not vacuous.
        assert_eq!(
            validate_contiguous_meta(&[2, 3], &[3, 1], 0, 6, "input").unwrap(),
            6
        );
    }

    #[test]
    fn batch_norm_validations_reject_mismatched_dtype_metadata() {
        use incin_core::tensor::dtype::DTypeId;
        let err = ensure_matching_dtype(DTypeId::F32.into(), DTypeId::I64.into())
            .expect_err("a parameter in another dtype must refuse before launch");
        assert!(matches!(err, Error::DTypeStorageMismatch { .. }));
        ensure_matching_dtype(DTypeId::F32.into(), DTypeId::F32.into()).unwrap();
    }

    #[test]
    fn batch_norm_validations_reject_mismatched_device_metadata() {
        let err = ensure_matching_cuda_device(0, 1)
            .expect_err("a parameter on another device must refuse before launch");
        assert!(matches!(err, Error::DeviceMismatch { .. }));
        ensure_matching_cuda_device(0, 0).unwrap();
    }

    #[test]
    fn batch_norm_geometry_rejects_shapes_with_no_statistics() {
        assert!(
            batch_norm_geometry(&[]).is_err(),
            "rank zero has no channel axis"
        );
        // Zero channels: a zero-block grid and a zero-length stat buffer.
        assert!(batch_norm_geometry(&[2, 0]).is_err());
        // Zero elements per channel: the exact shape CPU's training kernel
        // refuses by name ("needs at least one element per channel").
        let err = batch_norm_geometry(&[0, 3])
            .expect_err("an empty batch has no statistics to normalize by");
        assert!(
            alloc::format!("{err:?}").contains("at least one element per channel"),
            "unexpected refusal: {err:?}"
        );
    }

    #[test]
    fn batch_norm_geometry_reduces_every_axis_but_the_channel_one() {
        // [N, C, H, W]: channel 3, spatial 4*5, batch 2 -> 40 per channel.
        let geometry = batch_norm_geometry(&[2, 3, 4, 5]).unwrap();
        assert_eq!(
            geometry,
            BatchNormGeometry {
                num_channels: 3,
                spatial_size: 20,
                channel_elements: 40,
            }
        );
        // [N, C]: no spatial axes, one element per (batch, channel) pair.
        let geometry = batch_norm_geometry(&[5, 7]).unwrap();
        assert_eq!(
            geometry,
            BatchNormGeometry {
                num_channels: 7,
                spatial_size: 1,
                channel_elements: 5,
            }
        );
        // Rank one is the per-channel parameter vector the row also admits.
        let geometry = batch_norm_geometry(&[4]).unwrap();
        assert_eq!(
            geometry,
            BatchNormGeometry {
                num_channels: 4,
                spatial_size: 1,
                channel_elements: 1,
            }
        );
    }

    #[test]
    fn batch_norm_grad_shapes_follow_the_forward_operands() {
        // The plan the backward allocates by: each gradient takes its
        // operand's own shape, in the tape's input_ids order.
        let plan = batch_norm_grad_shapes(&[2, 3, 4, 5], Some(&[3]), Some(&[3]));
        assert_eq!(plan.input, alloc::vec![2, 3, 4, 5]);
        assert_eq!(plan.weight, Some(alloc::vec![3]));
        assert_eq!(plan.bias, Some(alloc::vec![3]));
        // An operand-free forward still gets an input gradient of its shape
        // and nothing else -- no placeholder zeros for absent parameters.
        let plan = batch_norm_grad_shapes(&[2, 3], None, None);
        assert_eq!(plan.input, alloc::vec![2, 3]);
        assert!(plan.weight.is_none());
        assert!(plan.bias.is_none());
    }
}
