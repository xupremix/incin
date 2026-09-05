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
