//! Tape-tracked layer normalization for CUDA storage.
//!
//! The forward kernel in `cuda::ops::norm` is fused; this is the method that
//! makes it differentiable. It runs the forward with statistic saving exactly
//! when the ambient [`GradMode`](incin_core::exec::GradMode) records, pushes
//! one entry replaying those statistics, and otherwise behaves like a plain
//! launch. An empty batch records no statistics and needs no kernel: the
//! recipe answers zero gradients of the recorded shapes directly.

use super::*;
use crate::cuda::ops::norm::{launch_layer_norm, launch_layer_norm_backward};
use crate::cuda::storage::CudaBuffer;
use incin_core::exec::GradMode;

impl<D: Device> CudaBackendImpl<D> {
    /// Softmax along `axis`, composed from tracked primitives.
    ///
    /// Deliberately not the fused `launch_softmax`: that kernel runs the same
    /// arithmetic but records no tape entry. `log_softmax` (max/sub/exp/sum/
    /// log, each pushing its own entry) followed by `exp` replays the full
    /// backward through the walk, which is what the training-capable
    /// capability row promises. `max_keepdim` stays untracked on purpose:
    /// softmax is shift-invariant, so the true gradient through the
    /// stabilizing max is zero and an untracked leaf gives that for free.
    pub(crate) fn softmax<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let input: &CudaStorage = input;
        let log_probs = crate::cuda::backend::elementwise::cuda_log_softmax::<D>(input, axis)?;
        crate::cuda::backend::elementwise::cuda_exp_storage(
            &log_probs,
            crate::kernel::KernelSpecialization::NONE,
        )
    }

    /// RMS normalization with a backward replaying the saved norm factor.
    ///
    /// The fused kernel stores one inverse norm factor per batch row when the
    /// grad mode records; the recipe below needs no other saved values. With
    /// `z = x * inv`, `g2 = gout * w`, and per-row means over the trailing
    /// axis, `dx = (g2 - mean(g2) - z * mean(g2 * z)) * inv` and the weight
    /// gradient sums `g2 * z` over every leading axis, all through tracked
    /// primitives.
    pub(crate) fn rms_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let input: &CudaStorage = input;
        let weight: &CudaStorage = weight;
        let recording = GradMode::current().records();
        let (out, factor) = crate::cuda::ops::norm::launch_rms_norm(input, weight, eps, recording)?;
        if !recording {
            return Ok(out);
        }
        let (input_id, weight_id, out_id) = (input.id, weight.id, out.id);
        let (input_shape, weight_shape, input_dtype) = (
            input.shape.to_vec(),
            weight.shape.to_vec(),
            input.buffer.dtype,
        );
        let norm_size = input.shape.last().copied().unwrap_or(0);
        let (input_saved, weight_saved) = (input.clone(), weight.clone());
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![input_id, weight_id],
            backward: Box::new(move |grad_out: &CudaStorage| match &factor {
                Some(factor) => Self::rms_norm_backward(
                    grad_out,
                    &input_saved,
                    &weight_saved,
                    factor,
                    &input_shape,
                ),
                None => empty_batch_grads(
                    grad_out,
                    &input_shape,
                    &weight_shape,
                    norm_size,
                    input_dtype,
                    false,
                ),
            }),
        });
        Ok(out)
    }

    /// Backward of [`Self::rms_norm`] for a non-empty batch. Every step is a
    /// tracked primitive, so a second-order walk would replay through these
    /// entries the same way the first-order one replays through the forward.
    ///
    /// With `z = x * inv`, `g2 = gout * w` and per-row means over the
    /// trailing axis, `dx = (g2 - z * mean(g2 * z)) * inv`. There is no
    /// `mean(g2)` term: unlike layer norm, nothing here subtracts a mean,
    /// and adding one is wrong everywhere except uniform weight.
    fn rms_norm_backward(
        grad_out: &CudaStorage,
        input: &CudaStorage,
        weight: &CudaStorage,
        factor: &CudaStorage,
        input_shape: &[usize],
    ) -> Result<alloc::vec::Vec<CudaStorage>> {
        use crate::cuda::backend::elementwise::{cuda_mul_storage, cuda_sub_storage};
        let none = crate::kernel::KernelSpecialization::NONE;
        let rank = input_shape.len();
        let last = rank
            .checked_sub(1)
            .ok_or_else(|| Error::Msg("CUDA rms norm backward requires rank >= 1".into()))?;
        // Per-row inverse factor broadcast back to the full shape: reshape to
        // a trailing singleton first, since a bare [B] vector would align
        // against the wrong axis.
        let mut factor_shape = input_shape.to_vec();
        factor_shape[last] = 1;
        let inv_kept = Self::reshape::<f32>(factor, &factor_shape)?;
        let inv_wide = Self::broadcast_as::<f32>(&inv_kept, input_shape)?;
        let splayed = cuda_mul_storage(input, &inv_wide, none)?;
        let gated = cuda_mul_storage(grad_out, weight, none)?;
        let gated_splayed = cuda_mul_storage(&gated, &splayed, none)?;
        let mean_gated_splayed = Self::mean_keepdim::<f32>(&gated_splayed, last)?;
        let mean_gated_splayed_wide = Self::broadcast_as::<f32>(&mean_gated_splayed, input_shape)?;
        let scaled = cuda_sub_storage(
            &gated,
            &cuda_mul_storage(&splayed, &mean_gated_splayed_wide, none)?,
            none,
        )?;
        let input_grad = cuda_mul_storage(&scaled, &inv_wide, none)?;
        // The weight gradient sums plain upstream times normalized input
        // over every leading axis, leaving [N]. Note the missing weight
        // here next to the input gradient above: d(out)/dw has no w factor
        // left (it differentiated away), while d(out)/dx keeps one inside
        // g2. Summing g2*z instead understates every lanes by its own
        // weight, exactly the defect CPU parity caught.
        let mut weight_grad = cuda_mul_storage(grad_out, &splayed, none)?;
        for _ in 0..last {
            weight_grad = Self::sum_dim::<f32>(&weight_grad, 0)?;
        }
        Ok(alloc::vec![input_grad, weight_grad])
    }

    /// Fused forward with a backward that replays the saved statistics.
    ///
    /// `eps` arrives narrowed to `f32` already; the kernel takes no wider
    /// scalar, and an `f64` handed to a `float` parameter shifts every
    /// argument after it rather than converting.
    pub(crate) fn layer_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let input: &CudaStorage = input;
        let weight: &CudaStorage = weight;
        let bias: Option<&CudaStorage> = bias;
        let recording = GradMode::current().records();
        let (out, stats) = launch_layer_norm(input, weight, bias, eps, recording)?;
        if !recording {
            return Ok(out);
        }
        let (input_id, weight_id, out_id) = (input.id, weight.id, out.id);
        let bias_id = bias.map(|storage| storage.id);
        let mut input_ids = alloc::vec![input_id, weight_id];
        if let Some(id) = bias_id {
            input_ids.push(id);
        }
        // Shapes and dtype for the empty-batch answer, which launches
        // nothing: there are no rows to replay, so every gradient is zeros.
        let (input_shape, weight_shape, input_dtype) = (
            input.shape.to_vec(),
            weight.shape.to_vec(),
            input.buffer.dtype,
        );
        let norm_size = input.shape.last().copied().unwrap_or(0);
        let has_bias = bias.is_some();
        let (input_saved, weight_saved) = (input.clone(), weight.clone());
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids,
            backward: Box::new(move |grad_out: &CudaStorage| match &stats {
                Some(stats) => {
                    let grads = launch_layer_norm_backward(
                        grad_out,
                        &input_saved,
                        &weight_saved,
                        &stats.mean,
                        &stats.rstd,
                        has_bias,
                    )?;
                    backward_outputs(grads, has_bias)
                }
                None => empty_batch_grads(
                    grad_out,
                    &input_shape,
                    &weight_shape,
                    norm_size,
                    input_dtype,
                    has_bias,
                ),
            }),
        });
        Ok(out)
    }
}

/// Order the backward outputs exactly as `input_ids` names them: input and
/// weight always, bias only when the forward ran with one. A missing bias
/// gradient here would shift every id after it onto the wrong tensor.
fn backward_outputs(
    grads: crate::cuda::ops::norm::LayerNormGrads,
    has_bias: bool,
) -> Result<alloc::vec::Vec<CudaStorage>> {
    let crate::cuda::ops::norm::LayerNormGrads {
        input,
        weight,
        bias,
    } = grads;
    let mut out = alloc::vec![input, weight];
    match (has_bias, bias) {
        (true, Some(db)) => {
            out.push(db);
            Ok(out)
        }
        (true, None) => Err(Error::Msg(
            "CUDA layer norm backward produced no bias gradient for a biased forward".into(),
        )),
        (false, _) => Ok(out),
    }
}

/// Zero gradients for an empty batch: no rows ran forward, so no kernel runs
/// backward either. Shapes come from the forward recording, the device from
/// the upstream gradient that seeds this walk.
fn empty_batch_grads(
    grad_out: &CudaStorage,
    input_shape: &[usize],
    weight_shape: &[usize],
    norm_size: usize,
    dtype: DTypeDescriptor,
    has_bias: bool,
) -> Result<alloc::vec::Vec<CudaStorage>> {
    let stream = grad_out.buffer.device.default_stream();
    let zeros = |len: usize, shape: &[usize]| -> Result<CudaStorage> {
        let bytes = crate::bytes::byte_len(dtype, len, OperationKind::Storage)?;
        Ok(CudaStorage::new(
            Arc::new(CudaBuffer {
                len,
                dtype,
                data: Arc::new(stream.alloc_zeros::<u8>(bytes).map_err(|error| {
                    Error::Msg(alloc::format!(
                        "CUDA layer norm empty-batch gradient allocation failed: {error:?}"
                    ))
                })?),
                device: grad_out.buffer.device.clone(),
                device_id: grad_out.buffer.device_id,
            }),
            shape.to_vec(),
        ))
    };
    let input_len: usize = input_shape.iter().product();
    let weight_len: usize = weight_shape.iter().product();
    let mut out = alloc::vec![
        zeros(input_len, input_shape)?,
        zeros(weight_len, weight_shape)?,
    ];
    if has_bias {
        out.push(zeros(norm_size, &[norm_size])?);
    }
    Ok(out)
}
