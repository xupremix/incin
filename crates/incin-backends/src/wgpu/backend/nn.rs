//! Pooling and convolution WGPU operations, and the CPU-side im2col/
//! col2im/matmul helpers they are composed from (no WGSL kernel exists
//! for these yet).

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Conv backward helpers (CPU-side, used inside backward closures that
// read data back from WgpuBuffer and compute gradients in host memory).
// The same im2col / col2im logic as `crates/incin-backends/src/cpu/ops/conv/`
// but operating on plain `Vec<f32>` instead of `CpuStorage`.
// ─────────────────────────────────────────────────────────────────────────────

/// Gather a `[B, Cin, H, W]` buffer (row-major) into a
/// `[B, H_out*W_out, Cin*Kh*Kw]` column matrix. Out-of-bounds positions
/// (i.e. positions in the padded region) contribute 0.0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col_2d_cpu(
    input: &[f32],
    b: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let h_out = pool_output_dim(h, kh, stride, padding, dilation)?;
    let w_out = pool_output_dim(w, kw, stride, padding, dilation)?;
    let col_len = ShapeBuf::from_slice(&[cin, kh, kw]).checked_numel(OperationKind::Conv2d)?;
    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let out_elements =
        ShapeBuf::from_slice(&[b, spatial, col_len]).checked_numel(OperationKind::Conv2d)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let o_flat = oh * w_out + ow;
                for ci in 0..cin {
                    for ki_h in 0..kh {
                        for ki_w in 0..kw {
                            let src_h = oh * stride + ki_h * dilation;
                            let src_w = ow * stride + ki_w * dilation;
                            let val = if src_h >= padding
                                && src_h - padding < h
                                && src_w >= padding
                                && src_w - padding < w
                            {
                                let ih = src_h - padding;
                                let iw = src_w - padding;
                                input[bi * (cin * h * w) + ci * (h * w) + ih * w + iw]
                            } else {
                                0.0
                            };
                            let col_idx = ci * kh * kw + ki_h * kw + ki_w;
                            out[bi * (spatial * col_len) + o_flat * col_len + col_idx] = val;
                        }
                    }
                }
            }
        }
    }
    Ok((out, h_out, w_out))
}

/// Scatter-ADD a `[B, H_out*W_out, Cin*Kh*Kw]` gradient back into a
/// zero-initialized `[B, Cin, H, W]` buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn col2im_2d_cpu(
    cols_grad: &[f32],
    b: usize,
    cin: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Vec<f32>> {
    let col_len = ShapeBuf::from_slice(&[cin, kh, kw]).checked_numel(OperationKind::Conv2d)?;
    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let out_elements =
        ShapeBuf::from_slice(&[b, cin, h, w]).checked_numel(OperationKind::Conv2d)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let o_flat = oh * w_out + ow;
                for ci in 0..cin {
                    for ki_h in 0..kh {
                        for ki_w in 0..kw {
                            let src_h = oh * stride + ki_h * dilation;
                            let src_w = ow * stride + ki_w * dilation;
                            if src_h >= padding
                                && src_h - padding < h
                                && src_w >= padding
                                && src_w - padding < w
                            {
                                let ih = src_h - padding;
                                let iw = src_w - padding;
                                let col_idx = ci * kh * kw + ki_h * kw + ki_w;
                                out[bi * (cin * h * w) + ci * (h * w) + ih * w + iw] += cols_grad
                                    [bi * (spatial * col_len) + o_flat * col_len + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Batched matrix multiply on CPU: lhs `[B, M, K]` × rhs `[B, K, N]` → `[B, M, N]`.
/// Used inside conv backward closures.
pub(crate) fn cpu_bmm(
    lhs: &[f32],
    rhs: &[f32],
    b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[b, m, n]).checked_numel(OperationKind::MatMul)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += lhs[bi * (m * k) + mi * k + ki] * rhs[bi * (k * n) + ki * n + ni];
                }
                out[bi * (m * n) + mi * n + ni] = acc;
            }
        }
    }
    Ok(out)
}

/// Transpose the last two dimensions of a `[B, M, N]` tensor → `[B, N, M]`.
pub(crate) fn cpu_transpose_last2(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[b, m, n]).checked_numel(OperationKind::Transpose)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                out[bi * (n * m) + ni * m + mi] = src[bi * (m * n) + mi * n + ni];
            }
        }
    }
    Ok(out)
}

/// Sum a `[B, M, N]` buffer over its leading batch axis → `[M, N]`.
pub(crate) fn cpu_sum_batch(src: &[f32], b: usize, m: usize, n: usize) -> Result<Vec<f32>> {
    let out_elements = ShapeBuf::from_slice(&[m, n]).checked_numel(OperationKind::Reduction)?;
    let mut out = vec![0.0f32; out_elements];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                out[mi * n + ni] += src[bi * (m * n) + mi * n + ni];
            }
        }
    }
    Ok(out)
}

/// Gather a `[B, Cin, L]` buffer (row-major) into a `[B, L_out, Cin*K]`
/// column matrix. Positions in the padded region contribute 0.0 — CPU's
/// `im2col_1d` walk for walk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col_1d_cpu(
    input: &[f32],
    b: usize,
    cin: usize,
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(Vec<f32>, usize)> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Conv1d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| len.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv1d,
            expression: "convolution padded input dimension",
        })?;
    let effective = dilation
        .checked_mul(kernel_size.saturating_sub(1))
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv1d,
            expression: "convolution effective kernel",
        })?;
    let l_out = padded.saturating_sub(effective) / stride + 1;
    let columns =
        ShapeBuf::from_slice(&[cin, kernel_size]).checked_numel(OperationKind::Conv1d)?;
    let mut out = alloc::vec![0.0f32; b * l_out * columns];
    for bi in 0..b {
        for oi in 0..l_out {
            for ci in 0..cin {
                for ki in 0..kernel_size {
                    let src = oi * stride + ki * dilation;
                    let val = if src >= padding && src - padding < len {
                        input[bi * (cin * len) + ci * len + (src - padding)]
                    } else {
                        0.0
                    };
                    out[bi * (l_out * columns) + oi * columns + ci * kernel_size + ki] = val;
                }
            }
        }
    }
    Ok((out, l_out))
}

/// Exact inverse of [`im2col_1d_cpu`]: scatter-ADD each contribution of a
/// `[B, L_out, Cin*K]`-shaped gradient back into a zero-initialized
/// `[B, Cin, L]` buffer, skipping padded destinations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn col2im_1d_cpu(
    cols_grad: &[f32],
    b: usize,
    cin: usize,
    len: usize,
    l_out: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Vec<f32>> {
    let mut out = alloc::vec![0.0f32; b * cin * len];
    for bi in 0..b {
        for oi in 0..l_out {
            for ci in 0..cin {
                for ki in 0..kernel_size {
                    let src = oi * stride + ki * dilation;
                    if src >= padding && src - padding < len {
                        let dst = src - padding;
                        out[bi * (cin * len) + ci * len + dst] += cols_grad[bi
                            * (l_out * cin * kernel_size)
                            + oi * (cin * kernel_size)
                            + ci * kernel_size
                            + ki];
                    }
                }
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// WgpuBackendImpl inherent helpers
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// Forward-only conv2d (no tape entry). Used by both `conv1d` and `conv2d`
    /// so they can push exactly ONE clean tape entry each for their respective
    /// grad shapes, rather than having nested entries from the internal matmul.
    pub(crate) fn conv2d_no_tape<K: DType>(
        t: &WgpuStorage,
        weight: &WgpuStorage,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<WgpuStorage> {
        let shape = &t.shape; // [N, C_in, H, W]
        let ws = &weight.shape; // [C_out, C_in/groups, Kh, Kw]
        if shape.len() != 4 || ws.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![4],
                got: vec![shape.len()],
                msg: "expected 4D input and weight".into(),
            });
        }
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, c_in_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let g = groups.max(1);
        let c_in_g = c_in / g;
        assert_eq!(c_in_g, c_in_per_g, "groups mismatch");

        let h_out =
            (h_in + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
        let w_out =
            (w_in + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;

        let col_channels =
            ShapeBuf::from_slice(&[c_in, kh, kw]).checked_numel(OperationKind::Conv2d)?;
        let col_spatial =
            ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
        let col_elements = ShapeBuf::from_slice(&[batch, col_channels, col_spatial])
            .checked_numel(OperationKind::Conv2d)?;
        let col_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, col_elements, OperationKind::Conv2d)?;

        let params = checked_u32_array(
            [
                batch, c_in, h_in, w_in, h_out, w_out, kh, kw, stride, stride, padding, padding,
                dilation, dilation,
            ],
            "WGPU im2col kernel parameter",
        )?;
        dispatch::dispatch_im2col(&t.buffer, &col_buf, &params)?;

        let k_size =
            ShapeBuf::from_slice(&[c_in_g, kh, kw]).checked_numel(OperationKind::Conv2d)?;

        if g == 1 {
            let w_storage = WgpuStorage::new(weight.buffer.clone(), vec![c_out, k_size]);
            let col_storage = WgpuStorage::new(col_buf, vec![batch, k_size, col_spatial]);
            let out_storage = Self::matmul::<K>(&w_storage, &col_storage)?;
            return Ok(WgpuStorage::new(
                out_storage.buffer,
                vec![batch, c_out, h_out, w_out],
            ));
        }

        // groups > 1: direct kernel
        let out_elements = ShapeBuf::from_slice(&[batch, c_out, h_out, w_out])
            .checked_numel(OperationKind::Conv2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Conv2d)?;
        let conv_params = checked_u32_array(
            [
                batch, c_in, h_in, w_in, c_out, h_out, w_out, kh, kw, stride, stride, padding,
                padding, dilation, dilation, groups,
            ],
            "WGPU convolution kernel parameter",
        )?;
        dispatch::dispatch_conv2d_direct(&t.buffer, &weight.buffer, &out_buf, &conv_params)?;
        Ok(WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]))
    }
}

/// Row-major contiguous strides for a rank-4 `[N, C, H, W]` shape. WGPU
/// storage has no non-contiguous view support (`WgpuStorage::new` always
/// derives strides from shape), so pooling backward closures - which read
/// buffers back to a flat host `Vec` - can compute this directly instead of
/// pulling in `cpu::stride`.
pub(crate) fn contiguous_strides_4d(shape: &[usize]) -> Result<[usize; 4]> {
    let strides = StrideBuf::contiguous_for(&ShapeBuf::from_slice(shape), OperationKind::Storage)?;
    strides
        .strides()
        .try_into()
        .map_err(|_| Error::Msg("WGPU pooling expected rank-four storage".into()))
}

pub(crate) fn pool_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| input.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling padded input dimension",
        })?;
    let effective_kernel = dilation
        .checked_mul(kernel - 1)
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling effective kernel dimension",
        })?;
    if effective_kernel > padded {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "effective kernel exceeds padded input",
            value: effective_kernel,
        }
        .into());
    }
    (padded - effective_kernel)
        .checked_div(stride)
        .and_then(|steps| steps.checked_add(1))
        .ok_or_else(|| {
            ShapeError::ArithmeticOverflow {
                operation: OperationKind::Pool2d,
                expression: "pooling output dimension",
            }
            .into()
        })
}

// ─────────────────────────────────────────────────────────────────────────────
//
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `x / sqrt(mean(x^2, last axis) + eps) * weight`.
    ///
    /// Composed step for step the way CPU's kernel and CUDA's both compose it,
    /// so all three agree on the same input rather than each being separately
    /// plausible. Every step is already a tape-tracked primitive, so this
    /// writes no backward code: the composite's gradient is the replay of the
    /// six entries below, which is what makes `training = true` on its
    /// capability row a verified claim rather than a hopeful one.
    ///
    /// `epsilon` is added to the mean *before* the square root, not after.
    /// That ordering is the entire point of the guard: it keeps the root
    /// finite when every element of a row is zero, which adding afterwards
    /// would not.
    ///
    /// The axis is the last one, matching both references. RMS norm is defined
    /// over the feature axis and the operation carries no axis attribute to
    /// say otherwise.
    pub(crate) fn rms_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let axis = input.shape.len().saturating_sub(1);
        let squared = Self::mul::<K>(input, input)?;
        let mean = Self::mean_keepdim::<K>(&squared, axis)?;
        let guarded = Self::add_scalar_float::<K>(&mean, epsilon)?;
        let scale = Self::sqrt::<K>(&guarded)?;
        let normalized = Self::div::<K>(input, &scale)?;
        Self::mul::<K>(&normalized, weight)
    }

    /// `layer_norm(input, weight, bias?, eps)` over the trailing axis only,
    /// step for step the way CPU's `layer_norm_impl` and CUDA's both compose
    /// it: mean, center, mean-of-squares, `+eps`, sqrt, divide, affine. Every
    /// step is already a taped primitive, so this writes no backward of its
    /// own and `training = true` on its capability row is the replay of the
    /// composite. Absent bias falls through to a plain add of the weight-scaled
    /// term (CPU substitutes a zero buffer of the weight's shape; adding nothing
    /// is the same value).
    pub(crate) fn layer_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let last = input.shape.len().saturating_sub(1);
        let mean = Self::mean_keepdim::<K>(input, last)?;
        let centered = Self::sub::<K>(input, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let variance = Self::mean_keepdim::<K>(&squared, last)?;
        let guarded = Self::add_scalar_float::<K>(&variance, epsilon)?;
        let std = Self::sqrt::<K>(&guarded)?;
        let normalized = Self::div::<K>(&centered, &std)?;
        let scaled = Self::mul::<K>(&normalized, weight)?;
        match bias {
            Some(bias) => Self::add::<K>(&scaled, bias),
            None => Ok(scaled),
        }
    }

    /// `batch_norm` in inference mode: `((x - running_mean) / sqrt(running_var
    /// + eps)) * weight + bias`, with each per-channel vector reshaped to
    /// `[1, C, 1, ...]` and stretched by the elementwise path's broadcast.
    ///
    /// Matches CPU's `batch_norm_impl` exactly, including the Candle default
    /// fallbacks: absent mean becomes zeros, absent var/weight becomes ones,
    /// absent bias becomes zeros.
    pub(crate) fn batch_norm_inference<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: Option<&<Self as StorageBackend>::Storage<K>>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        running_mean: &<Self as StorageBackend>::Storage<K>,
        running_variance: &<Self as StorageBackend>::Storage<K>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = input.shape.len();
        let channel_dim = if rank > 1 { 1 } else { 0 };
        let channels = input.shape[channel_dim];
        let mut bcast = alloc::vec![1usize; rank];
        bcast[channel_dim] = channels;

        let reshape_param =
            |p: &<Self as StorageBackend>::Storage<K>| Self::reshape::<K>(p, &bcast);
        let rm = reshape_param(running_mean)?;
        let rv = reshape_param(running_variance)?;
        let w = match weight {
            Some(w) => reshape_param(w)?,
            None => {
                let n = channels;
                WgpuStorage::new(
                    WgpuBuffer::from_slice(&alloc::vec![1.0f32; n]),
                    bcast.clone(),
                )
            }
        };
        let b = match bias {
            Some(b) => reshape_param(b)?,
            None => {
                let n = channels;
                WgpuStorage::new(
                    WgpuBuffer::from_slice(&alloc::vec![0.0f32; n]),
                    bcast.clone(),
                )
            }
        };

        let centered = Self::sub::<K>(input, &rm)?;
        let rv_eps = Self::add_scalar_float::<K>(&rv, epsilon)?;
        let std = Self::sqrt::<K>(&rv_eps)?;
        let normalized = Self::div::<K>(&centered, &std)?;
        let scaled = Self::mul::<K>(&normalized, &w)?;
        Self::add::<K>(&scaled, &b)
    }

    /// `batch_norm` in training mode: batch statistics computed from the
    /// input itself, composed from the same taped primitives CPU's
    /// `batch_norm_training_impl` uses — `sum_keepdim` over every axis but
    /// the channel one (so the mean/variance path reaches the gradient),
    /// then center, scale, and the optional affine weight/bias.
    pub(crate) fn batch_norm_training<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: Option<&<Self as StorageBackend>::Storage<K>>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = input.shape.len();
        let channel_dim = if rank > 1 { 1 } else { 0 };
        let num_channels = input.shape[channel_dim];

        let mut bcast_shape = alloc::vec![1usize; rank];
        bcast_shape[channel_dim] = num_channels;

        // Every axis but the channel one is reduced away, keeping its
        // position so the result broadcasts back against the input without
        // a reshape.
        let reduced_axes: alloc::vec::Vec<usize> =
            (0..rank).filter(|axis| *axis != channel_dim).collect();
        let count: usize = reduced_axes.iter().map(|&axis| input.shape[axis]).product();
        if count == 0 {
            return Err(Error::Msg(
                "batch_norm: training mode needs at least one element per channel".into(),
            ));
        }

        // Every reduction here must go through the tape-tracked
        // `sum_keepdim`, not a raw launch: training-mode batch norm
        // normalizes by statistics that are functions of the input, so a
        // tape-silent reduction would cut the mean/variance path out of
        // every backward pass computed through this composition.
        let sum_over_reduced = |x: &<Self as StorageBackend>::Storage<K>| -> Result<
            <Self as StorageBackend>::Storage<K>,
        > {
            let mut acc = x.clone();
            for &axis in &reduced_axes {
                acc = Self::sum_keepdim::<K>(&acc, axis)?;
            }
            Ok(acc)
        };

        let inv_count = 1.0 / count as f64;
        let total = sum_over_reduced(input)?;
        let mean = Self::mul_scalar_float::<K>(&total, inv_count)?;
        let centered = Self::sub::<K>(input, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let squared_total = sum_over_reduced(&squared)?;
        let variance = Self::mul_scalar_float::<K>(&squared_total, inv_count)?;

        let variance_eps = Self::add_scalar_float::<K>(&variance, epsilon)?;
        let std = Self::sqrt::<K>(&variance_eps)?;
        let normalized = Self::div::<K>(&centered, &std)?;

        let scaled = match weight {
            Some(w) => {
                let w = Self::reshape::<K>(w, &bcast_shape)?;
                Self::mul::<K>(&normalized, &w)?
            }
            None => normalized,
        };
        match bias {
            Some(b) => {
                let b = Self::reshape::<K>(b, &bcast_shape)?;
                Self::add::<K>(&scaled, &b)
            }
            None => Ok(scaled),
        }
    }

    /// `group_norm(input, groups, eps)`: reshape to
    /// `[batch * groups, channels/groups * spatial]`, mean-center, divide by
    /// the per-group standard deviation, reshape back — CPU's
    /// `group_norm_storage` composed from the same taped primitives, so the
    /// statistical path reaches the gradient without a hand-derived kernel.
    pub(crate) fn group_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        groups: usize,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if groups == 0 {
            return Err(Error::Msg("group_norm: groups must be non-zero".into()));
        }
        let rank = input.shape.len();
        let channels = if rank >= 2 { input.shape[1] } else { 1 };
        if channels % groups != 0 {
            return Err(Error::Msg(
                "group_norm: channels must be divisible by groups".into(),
            ));
        }
        let total = num_elements(&input.shape)?;
        let (batch, spatial) = if rank >= 2 {
            (input.shape[0], input.shape[2..].iter().product::<usize>())
        } else {
            (1, total)
        };
        let group_size = (channels / groups) * spatial;
        let runs = batch * groups;

        let flat = Self::reshape::<K>(input, &[runs, group_size])?;
        let mean = Self::mean_keepdim::<K>(&flat, 1)?;
        let centered = Self::sub::<K>(&flat, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let variance = Self::mean_keepdim::<K>(&squared, 1)?;
        let guarded = Self::add_scalar_float::<K>(&variance, epsilon)?;
        let std = Self::sqrt::<K>(&guarded)?;
        let normalized = Self::div::<K>(&centered, &std)?;
        Self::reshape::<K>(&normalized, &input.shape)
    }

    /// `mse_loss(pred, target, reduction)`: `sub`, square, then the reduction
    /// mode — CPU's composition, so the gradient arrives by replay.
    pub(crate) fn mse_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = Self::sub::<K>(pred, target)?;
        let squared = Self::mul::<K>(&diff, &diff)?;
        Self::reduce_loss::<K>(squared, reduction)
    }

    /// `l1_loss(pred, target, reduction)`: `sub`, `abs`, then the reduction
    /// mode — CPU's composition.
    pub(crate) fn l1_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = Self::sub::<K>(pred, target)?;
        let absolute = Self::abs::<K>(&diff)?;
        Self::reduce_loss::<K>(absolute, reduction)
    }

    /// Apply a loss reduction mode: `Mean` → `mean_all`, `Sum` → `sum_all`,
    /// `None` → identity. Same mapping CPU's `reduce_loss` uses.
    fn reduce_loss<K: DType>(
        t: <Self as StorageBackend>::Storage<K>,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        match reduction {
            incin_core::tensor::reduction::Reduction::Mean => Self::mean_all::<K>(&t),
            incin_core::tensor::reduction::Reduction::Sum => Self::sum_all::<K>(&t),
            incin_core::tensor::reduction::Reduction::None => Ok(t),
        }
    }

    /// `instance_norm(t, eps)` = `group_norm(t, channels, eps)`, the same
    /// rewrite CPU's `instance_norm_storage` performs: one group per channel.
    pub(crate) fn instance_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
        Self::group_norm::<K>(t, channels, epsilon)
    }

    /// `bce_with_logits_loss(pred, target, reduction)`: CPU's stable recipe —
    /// `relu(pred)` under `GradMode::Disabled` with a custom slope-0.5 tape
    /// entry at `x == 0`, then `max(x,0) - x*z + log(1 + exp(-|x|))`.
    pub(crate) fn bce_with_logits_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        // The forward relu must not record its own step-style entry: CPU
        // replaces it with a custom slope (0.5 at the kink). Running under
        // `Disabled` silences the built-in push; the explicit `push_with`
        // below re-records the real recipe under the ambient mode.
        let max_x_0 = incin_core::exec::GradMode::Disabled.restrict(|| Self::relu::<K>(pred))?;
        let pred_capture = pred.clone();
        let shape = pred.shape.to_vec();
        let (pred_id, out_id) = (pred.id, max_x_0.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![pred_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = pred_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let mut grad = alloc::vec![0.0f32; input_data.len()];
                for (i, &x) in input_data.iter().enumerate() {
                    let slope = if x == 0.0 {
                        0.5
                    } else if x > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    grad[i] = grad_data[i] * slope;
                }
                Ok(alloc::vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad),
                    shape.clone()
                )])
            }),
        });
        let x_times_z = Self::mul::<K>(pred, target)?;
        let term1 = Self::sub::<K>(&max_x_0, &x_times_z)?;
        let abs_x = Self::abs::<K>(pred)?;
        let neg_abs = Self::neg::<K>(&abs_x)?;
        let exp_neg = Self::exp::<K>(&neg_abs)?;
        let one_plus = Self::add_scalar_float::<K>(&exp_neg, 1.0)?;
        let term2 = Self::log::<K>(&one_plus)?;
        let loss = Self::add::<K>(&term1, &term2)?;
        Self::reduce_loss::<K>(loss, reduction)
    }

    /// `cross_entropy_loss(logits, target, reduction)`: CPU's composition —
    /// `log_softmax` on the class axis, a tape-tracked `gather` of the
    /// target class from each row, negate, then the reduction mode. The
    /// integer target operand is off the tape (no gradient flows into it);
    /// the gather's scatter-based backward is what carries the gradient
    /// back into the logits.
    pub(crate) fn cross_entropy_loss(
        logits: &WgpuStorage,
        target: &WgpuStorage,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<WgpuStorage> {
        if logits.shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "cross_entropy_loss",
                expected: alloc::vec![0, 0],
                got: logits.shape.to_vec(),
                msg: alloc::format!(
                    "cross_entropy_loss: logits must be rank-2 [batch, classes], got shape {:?}",
                    logits.shape
                ),
            });
        }
        let batch = logits.shape[0];
        if target.shape.as_ref() != [batch] {
            return Err(Error::ShapeMismatch {
                op: "cross_entropy_loss",
                expected: alloc::vec![batch],
                got: target.shape.to_vec(),
                msg: alloc::format!(
                    "cross_entropy_loss: target must be shape [{batch}] matching the logits \
                     batch, got shape {:?}",
                    target.shape
                ),
            });
        }
        require_f32(logits, "cross_entropy_logits")?;
        match target.dtype.builtin_id() {
            Some(DTypeId::I64) | Some(DTypeId::U32) | Some(DTypeId::U8) => {}
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: target.dtype,
                    backend: "Wgpu",
                    op: "cross_entropy_target",
                });
            }
        }

        let log_probs = Self::log_softmax::<f32>(logits, 1)?;
        // Reshape the integer target to `[batch, 1]` for the gather without
        // pushing a tape entry: the index operand is off the tape by
        // construction, and `reshape`'s own entry would be dead weight
        // nothing ever backprops through.
        let target_exp =
            WgpuStorage::new_with_dtype(target.buffer.clone(), alloc::vec![batch, 1], target.dtype);
        // The tape-tracked gather, not a raw launch: its scatter-based
        // backward is what carries the gradient back into the logits.
        let gathered = Self::gather(&log_probs, 1, &target_exp)?;
        let neg_gathered = Self::neg::<f32>(&gathered)?;
        let nll = Self::reshape::<f32>(&neg_gathered, &[batch])?;
        Self::reduce_loss::<f32>(nll, reduction)
    }

    /// `dropout(t, p, training)`: identity when off, zero when `p >= 1`,
    /// otherwise a host-LCG keep-mask (same LCG as `creation::rand`) scaled
    /// by `1 / (1 - p)`. The mask is constant; the tape rides `mul` and
    /// `mul_scalar_float`, so the gradient multiplies by the same mask.
    pub(crate) fn dropout<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        probability: f64,
        training: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if !training || probability <= 0.0 {
            // Clone-links-identity: the operand itself, same tensor id, so a
            // gradient arriving here needs no entry of its own — CPU, CUDA
            // and Metal all return `t.clone()` for the eval/`p <= 0` path.
            return Ok(t.clone());
        }
        if probability >= 1.0 {
            return Self::mul_scalar_float::<K>(t, 0.0);
        }
        let n = num_elements(&t.shape)?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let p = probability as f32;
        let mask_data: alloc::vec::Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let draw = ((state >> 33) as f32) / (u32::MAX as f32);
                // `step(draw - p)`: keep when draw > p, matching CPU's
                // `canonical_step` on the shifted draw.
                if draw > p { 1.0 } else { 0.0 }
            })
            .collect();
        let mask = WgpuStorage::new(WgpuBuffer::from_slice(&mask_data), t.shape.to_vec());
        let kept = Self::mul::<K>(t, &mask)?;
        Self::mul_scalar_float::<K>(&kept, 1.0 / (1.0 - probability))
    }

    /// `scaled_dot_product_attention(q, k, v, mask?, scale?)`: CPU's
    /// composition — transpose k, matmul, scale, optional additive mask,
    /// softmax on the last axis, matmul with v. Every step is taped.
    pub(crate) fn scaled_dot_product_attention<K: DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let k_t = if k.shape.len() >= 2 {
            let rank = k.shape.len();
            Self::transpose::<K>(k, rank - 2, rank - 1)?
        } else {
            k.clone()
        };
        let scores = Self::matmul::<K>(q, &k_t)?;
        let d_k = *q.shape.last().unwrap_or(&1) as f64;
        let scaled =
            Self::mul_scalar_float::<K>(&scores, scale.unwrap_or_else(|| 1.0 / d_k.sqrt()))?;
        let masked = match mask {
            Some(mask) => Self::add::<K>(&scaled, mask)?,
            None => scaled,
        };
        let axis = scores.shape.len().saturating_sub(1);
        let attention = Self::softmax::<K>(&masked, axis)?;
        Self::matmul::<K>(&attention, v)
    }
}

impl<D: Device> WgpuBackendImpl<D> {
    /// `avg_pool2d`.
    pub(crate) fn avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let oh = pool_output_dim(h, kh, sh, ph, 1)?;
        let ow = pool_output_dim(w, kw, sw, pw, 1)?;

        let out_elements =
            ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Pool2d)?;
        let [
            n_u32,
            c_u32,
            h_u32,
            w_u32,
            oh_u32,
            ow_u32,
            kh_u32,
            kw_u32,
            sh_u32,
            sw_u32,
            ph_u32,
            pw_u32,
        ] = checked_u32_array(
            [n, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw],
            "WGPU average-pooling kernel parameter",
        )?;

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 1, // mode 1 = avg
            n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32, kh_u32, kw_u32, sh_u32, sw_u32, ph_u32,
            pw_u32, 1, 1,
        )?;

        let out = WgpuStorage::new(out_buf, vec![n, c, oh, ow]);

        // Backward: distributes grad_out's per-position value uniformly
        // (divided by the FIXED kh*kw divisor - count_include_pad=True,
        // PyTorch's default, matching this op's forward, which sums the
        // padded region as 0.0 but still divides by kh*kw) into every input
        // position the window covered (padded positions are skipped, never
        // written), `+=`-accumulating across overlapping windows. Mirrors
        // the CPU backend's `avg_pool2d_impl` exactly.
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        let window_count = (kh * kw) as f32;
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let (b, c, h, w) = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_elements =
                    ShapeBuf::from_slice(&input_shape).checked_numel(OperationKind::Pool2d)?;
                let mut grad_input = vec![0.0f32; grad_elements];
                let in_strides = contiguous_strides_4d(&input_shape)?;
                let h_out = grad_out.shape[2];
                let w_out = grad_out.shape[3];
                for bi in 0..b {
                    for ci in 0..c {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                                let g = grad_data[flat_out] / window_count;
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let src_h = oh * sh + khi;
                                        let src_w = ow * sw + kwi;
                                        if src_h >= ph
                                            && src_h - ph < h
                                            && src_w >= pw
                                            && src_w - pw < w
                                        {
                                            let ih = src_h - ph;
                                            let iw = src_w - pw;
                                            let flat = bi * in_strides[0]
                                                + ci * in_strides[1]
                                                + ih * in_strides[2]
                                                + iw * in_strides[3];
                                            grad_input[flat] += g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_input),
                    input_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// Variable window for output position `i`: `start = floor(i*in/out)`,
    /// `end = ceil((i+1)*in/out)` — CPU's `adaptive_window_bounds` bound for
    /// bound, including its overflow errors.
    fn adaptive_bounds(
        in_size: usize,
        out_size: usize,
        i: usize,
    ) -> Result<(usize, usize)> {
        if out_size == 0 {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::AdaptiveAvgPool2d,
                reason: "adaptive_avg_pool2d output size must be non-zero",
            }));
        }
        let start = i
            .checked_mul(in_size)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::AdaptiveAvgPool2d,
                expression: "adaptive-pooling start index",
            })?
            / out_size;
        let end = i
            .checked_add(1)
            .and_then(|next| next.checked_mul(in_size))
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::AdaptiveAvgPool2d,
                expression: "adaptive-pooling end index",
            })?
            .div_ceil(out_size);
        Ok((start, end))
    }

    /// `adaptive_avg_pool2d`: variable windows from the output size, walked
    /// host-side exactly as CPU's `adaptive_avg_pool2d_impl` does (same
    /// floor/ceil window bounds, same `f64` accumulation narrowed once,
    /// same `g/count` spread backward). Rank 3 rides without a batch axis
    /// like CPU's `batched_spatial`. ONE `TapeEntry`, `input_ids = vec![t.id]`.
    pub(crate) fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = t.shape.len();
        if rank != 3 && rank != 4 {
            return Err(Error::ShapeMismatch {
                op: "adaptive_avg_pool2d",
                expected: alloc::vec![3, 4],
                got: alloc::vec![rank],
                msg: "adaptive_avg_pool2d accepts rank 3 or 4".to_string(),
            });
        }
        let (h_out, w_out) = output_size;
        if h_out == 0 || w_out == 0 {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::AdaptiveAvgPool2d,
                reason: "adaptive_avg_pool2d output size must be non-zero",
            }));
        }
        let (b, c, h, w) = if rank == 4 {
            (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        } else {
            (1, t.shape[0], t.shape[1], t.shape[2])
        };
        let data = t.buffer.to_vec::<f32>()?;
        let out_shape = if rank == 4 {
            alloc::vec![b, c, h_out, w_out]
        } else {
            alloc::vec![c, h_out, w_out]
        };
        let out_total = num_elements(&out_shape)?;
        let mut out_vals = alloc::vec![0.0f32; out_total];
        // Flat base per (batch, channel) in the downloaded buffer, which is
        // row-major contiguous: readback refuses strided operands before a
        // host walk ever sees one.
        let in_plane = h * w;
        for bi in 0..b {
            for ci in 0..c {
                let base = (bi * c + ci) * in_plane;
                for oh in 0..h_out {
                    let (h_start, h_end) = Self::adaptive_bounds(h, h_out, oh)?;
                    for ow in 0..w_out {
                        let (w_start, w_end) = Self::adaptive_bounds(w, w_out, ow)?;
                        let mut sum = 0.0f64;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                sum += f64::from(data[base + ih * w + iw]);
                            }
                        }
                        let count = ((h_end - h_start) * (w_end - w_start)) as f64;
                        out_vals[((bi * c + ci) * h_out + oh) * w_out + ow] =
                            (sum / count) as f32;
                    }
                }
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_vals), out_shape);

        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let input_total = num_elements(&input_shape)?;
                let mut grads = alloc::vec![0.0f32; input_total];
                let (gb, gc, gh, gw) = if rank == 4 {
                    (
                        input_shape[0],
                        input_shape[1],
                        input_shape[2],
                        input_shape[3],
                    )
                } else {
                    (1, input_shape[0], input_shape[1], input_shape[2])
                };
                let gh_out = grad_out.shape[grad_out.shape.len() - 2];
                let gw_out = grad_out.shape[grad_out.shape.len() - 1];
                for bi in 0..gb {
                    for ci in 0..gc {
                        let base = (bi * gc + ci) * (gh * gw);
                        let grad_base = (bi * gc + ci) * (gh_out * gw_out);
                        for oh in 0..gh_out {
                            let (h_start, h_end) = Self::adaptive_bounds(gh, gh_out, oh)?;
                            for ow in 0..gw_out {
                                let (w_start, w_end) =
                                    Self::adaptive_bounds(gw, gw_out, ow)?;
                                let count =
                                    ((h_end - h_start) * (w_end - w_start)) as f32;
                                let g = grad_data[grad_base + oh * gw_out + ow] / count;
                                for ih in h_start..h_end {
                                    for iw in w_start..w_end {
                                        grads[base + ih * gw + iw] += g;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(alloc::vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grads),
                    input_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }

    /// `max_pool2d`.
    pub(crate) fn max_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let oh = pool_output_dim(h, kh, sh, ph, dh)?;
        let ow = pool_output_dim(w, kw, sw, pw, dw)?;

        let out_elements =
            ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_elements, OperationKind::Pool2d)?;
        let [
            n_u32,
            c_u32,
            h_u32,
            w_u32,
            oh_u32,
            ow_u32,
            kh_u32,
            kw_u32,
            sh_u32,
            sw_u32,
            ph_u32,
            pw_u32,
            dh_u32,
            dw_u32,
        ] = checked_u32_array(
            [n, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw],
            "WGPU max-pooling kernel parameter",
        )?;

        dispatch::dispatch_pool2d(
            &t.buffer, &out_buf, 2, // mode 2 = max
            n_u32, c_u32, h_u32, w_u32, oh_u32, ow_u32, kh_u32, kw_u32, sh_u32, sw_u32, ph_u32,
            pw_u32, dh_u32, dw_u32,
        )?;

        let out = WgpuStorage::new(out_buf, vec![n, c, oh, ow]);

        // Backward: recomputes each output position's winning (first-argmax,
        // strict `>`) source position from the captured input (padded
        // positions are never candidates, never substituted with 0.0 -
        // matches the WGSL forward's `-FLT_MAX` init and its bounds-checked
        // skip), then `+=`-accumulates grad_out's value there - never `=`,
        // since overlapping windows (stride < kernel_size) can share a
        // winning input position. Mirrors the CPU backend's
        // `max_window_2d`/`scatter_pool_grad_2d` exactly.
        let input_shape = t.shape.to_vec();
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let (b, c, h, w) = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                let input_data = t_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_elements =
                    ShapeBuf::from_slice(&input_shape).checked_numel(OperationKind::Pool2d)?;
                let mut grad_input = vec![0.0f32; grad_elements];
                let in_strides = contiguous_strides_4d(&input_shape)?;
                let h_out = grad_out.shape[2];
                let w_out = grad_out.shape[3];
                for bi in 0..b {
                    for ci in 0..c {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let mut best_val = f32::NEG_INFINITY;
                                let mut best_flat = 0usize;
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let src_h = oh * sh + khi * dh;
                                        let src_w = ow * sw + kwi * dw;
                                        if src_h < ph
                                            || src_h - ph >= h
                                            || src_w < pw
                                            || src_w - pw >= w
                                        {
                                            continue;
                                        }
                                        let ih = src_h - ph;
                                        let iw = src_w - pw;
                                        let flat = bi * in_strides[0]
                                            + ci * in_strides[1]
                                            + ih * in_strides[2]
                                            + iw * in_strides[3];
                                        let v = input_data[flat];
                                        if v > best_val {
                                            best_val = v;
                                            best_flat = flat;
                                        }
                                    }
                                }
                                let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                                grad_input[best_flat] += grad_data[flat_out];
                            }
                        }
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_input),
                    input_shape.clone(),
                )])
            }),
        });

        Ok(out)
    }

    /// `conv2d`.
    pub(crate) fn conv2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let conv_out = Self::conv2d_no_tape::<K>(t, weight, stride, padding, dilation, groups)?;
        let shape = &t.shape; // [N, C_in, H, W]
        let ws = &weight.shape; // [C_out, C_in/groups, Kh, Kw]
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, c_in_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let c_out_g = c_out / groups.max(1);
        let h_out = conv_out.shape[2];
        let w_out = conv_out.shape[3];

        // Wire autograd tape entry.
        let (inp_capture, w_capture) = (t.clone(), weight.clone());
        let (inp_id, w_id, out_id) = (t.id, weight.id, conv_out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![inp_id, w_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = inp_capture.buffer.to_vec::<f32>()?;
                let weight_data = w_capture.buffer.to_vec::<f32>()?;
                // grad_out: [N, C_out, H_out, W_out] → flatten to row-major vec
                let grad_data = grad_out.buffer.to_vec::<f32>()?;

                let grad_input_elements = ShapeBuf::from_slice(&[batch, c_in, h_in, w_in])
                    .checked_numel(OperationKind::Conv2d)?;
                let grad_weight_elements = ShapeBuf::from_slice(&[c_out, c_in_g, kh, kw])
                    .checked_numel(OperationKind::Conv2d)?;
                let mut grad_input_data = vec![0.0f32; grad_input_elements];
                let mut grad_weight_data = vec![0.0f32; grad_weight_elements];

                for g in 0..groups {
                    // Slice input group [N, C_in_g, H, W]
                    let input_group_elements = ShapeBuf::from_slice(&[batch, c_in_g, h_in, w_in])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut input_g = vec![0.0f32; input_group_elements];
                    for bi in 0..batch {
                        for ci in 0..c_in_g {
                            for hi in 0..h_in {
                                for wi in 0..w_in {
                                    input_g[bi * (c_in_g * h_in * w_in)
                                        + ci * (h_in * w_in)
                                        + hi * w_in
                                        + wi] = input_data[bi * (c_in * h_in * w_in)
                                        + (g * c_in_g + ci) * (h_in * w_in)
                                        + hi * w_in
                                        + wi];
                                }
                            }
                        }
                    }

                    // Slice weight group [C_out_g, C_in_g, Kh, Kw]
                    let weight_group_elements = ShapeBuf::from_slice(&[c_out_g, c_in_g, kh, kw])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut weight_g = vec![0.0f32; weight_group_elements];
                    for co in 0..c_out_g {
                        let src_co = g * c_out_g + co;
                        for rest in 0..c_in_g * kh * kw {
                            weight_g[co * c_in_g * kh * kw + rest] =
                                weight_data[src_co * c_in_g * kh * kw + rest];
                        }
                    }

                    // Slice grad_out group [N, C_out_g, H_out, W_out]
                    let go_group_elements = ShapeBuf::from_slice(&[batch, c_out_g, h_out, w_out])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut go_g = vec![0.0f32; go_group_elements];
                    for bi in 0..batch {
                        for co in 0..c_out_g {
                            for hi in 0..h_out {
                                for wi in 0..w_out {
                                    go_g[bi * (c_out_g * h_out * w_out)
                                        + co * (h_out * w_out)
                                        + hi * w_out
                                        + wi] = grad_data[bi * (c_out * h_out * w_out)
                                        + (g * c_out_g + co) * (h_out * w_out)
                                        + hi * w_out
                                        + wi];
                                }
                            }
                        }
                    }

                    let (cols, ..) = im2col_2d_cpu(
                        &input_g, batch, c_in_g, h_in, w_in, kh, kw, stride, padding, dilation,
                    )?;
                    // cols: [N, H_out*W_out, C_in_g*Kh*Kw]
                    // go_g: [N, C_out_g, H_out*W_out] → [N, H_out*W_out, C_out_g]
                    let spatial = ShapeBuf::from_slice(&[h_out, w_out])
                        .checked_numel(OperationKind::Conv2d)?;
                    let go_transposed_elements = ShapeBuf::from_slice(&[batch, spatial, c_out_g])
                        .checked_numel(OperationKind::Conv2d)?;
                    let mut go_t = vec![0.0f32; go_transposed_elements];
                    for bi in 0..batch {
                        for co in 0..c_out_g {
                            for s in 0..spatial {
                                go_t[bi * spatial * c_out_g + s * c_out_g + co] =
                                    go_g[bi * c_out_g * spatial + co * spatial + s];
                            }
                        }
                    }

                    // grad_cols = go_t @ weight_g: [N, spatial, C_out_g] @ [C_out_g, C_in_g*Kh*Kw]
                    let grad_cols =
                        cpu_bmm(&go_t, &weight_g, batch, spatial, c_out_g, c_in_g * kh * kw)?;
                    // col2im → grad for input_g [N, C_in_g, H, W]
                    let grad_input_g = col2im_2d_cpu(
                        &grad_cols, batch, c_in_g, h_in, w_in, h_out, w_out, kh, kw, stride,
                        padding, dilation,
                    )?;

                    // Accumulate into grad_input_data
                    for bi in 0..batch {
                        for ci in 0..c_in_g {
                            for hi in 0..h_in {
                                for wi in 0..w_in {
                                    grad_input_data[bi * (c_in * h_in * w_in)
                                        + (g * c_in_g + ci) * (h_in * w_in)
                                        + hi * w_in
                                        + wi] += grad_input_g[bi * (c_in_g * h_in * w_in)
                                        + ci * (h_in * w_in)
                                        + hi * w_in
                                        + wi];
                                }
                            }
                        }
                    }

                    // grad_weight_g: go_t^T @ cols → [N, C_out_g, C_in_g*Kh*Kw] → sum over batch
                    let go_t2 = cpu_transpose_last2(&go_t, batch, spatial, c_out_g)?;
                    let gw_mat = cpu_bmm(&go_t2, &cols, batch, c_out_g, spatial, c_in_g * kh * kw)?;
                    let gw_summed = cpu_sum_batch(&gw_mat, batch, c_out_g, c_in_g * kh * kw)?;

                    for co in 0..c_out_g {
                        for rest in 0..c_in_g * kh * kw {
                            grad_weight_data[(g * c_out_g + co) * c_in_g * kh * kw + rest] +=
                                gw_summed[co * c_in_g * kh * kw + rest];
                        }
                    }
                }

                Ok(vec![
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_input_data),
                        inp_capture.shape.to_vec(),
                    ),
                    WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_weight_data),
                        w_capture.shape.to_vec(),
                    ),
                ])
            }),
        });

        // The bias is stretched to the output shape *before* the add: WGPU's
        // elementwise kernels require equal shapes and do not broadcast, so
        // handing `add` a `[1, C_out, 1, 1]` operand fails for every biased
        // convolution. Both steps are tape-tracked, so grad_bias still flows
        // back through the broadcast.
        match bias {
            Some(b) => {
                let b_shaped = Self::reshape::<K>(b, &[1, c_out, 1, 1])?;
                let b_stretched = Self::broadcast_as::<K>(&b_shaped, &conv_out.shape)?;
                Self::add::<K>(&conv_out, &b_stretched)
            }
            None => Ok(conv_out),
        }
    }

    /// `conv1d`: im2col host walk plus the GPU matmul, exactly CPU's
    /// `conv1d_impl` composition (narrow per group, unfold, weight reshape
    /// and transpose, batched matmul, concat, transpose back, reshape).
    /// Rank 2 rides with a batch of one like CPU's `with_batch_axis`; bias
    /// follows conv2d's reshape/broadcast/add tail so its gradient flows
    /// through tracked ops. ONE `TapeEntry` (`input`, `weight`); the piece
    /// calls below run under a disabled recording scope so the composition
    /// leaves no nested entries, and the backward closure recomputes through
    /// the same impls (safe: the backward walk itself runs disabled).
    pub(crate) fn conv1d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = t.shape.len();
        if rank != 2 && rank != 3 {
            return Err(Error::ShapeMismatch {
                op: "conv1d",
                expected: alloc::vec![2, 3],
                got: alloc::vec![rank],
                msg: "conv1d accepts rank 2 or 3".to_string(),
            });
        }
        let unbatched = rank == 2;
        // Work on the batched form throughout; the axis comes back off the
        // output and the input gradient below.
        let activation = if unbatched {
            let mut batched = alloc::vec![1usize];
            batched.extend_from_slice(&t.shape);
            Self::reshape::<K>(t, &batched)?
        } else {
            WgpuStorage::new(t.buffer.clone(), t.shape.to_vec())
        };
        let (b, cin, len) = (activation.shape[0], activation.shape[1], activation.shape[2]);
        if weight.shape.len() != 3 {
            return Err(Error::ShapeMismatch {
                op: "conv1d",
                expected: alloc::vec![3],
                got: alloc::vec![weight.shape.len()],
                msg: "conv1d weight must be [Cout, Cin/groups, K]".to_string(),
            });
        }
        let (cout, cin_g, kernel_size) = (weight.shape[0], weight.shape[1], weight.shape[2]);
        let g = groups.max(1);
        if g == 0 || cin % g != 0 || cout % g != 0 || cin / g != cin_g {
            return Err(Error::ShapeMismatch {
                op: "conv1d",
                expected: alloc::vec![g],
                got: alloc::vec![cin, cout],
                msg: "conv1d: groups must evenly divide Cin and Cout, and weight Cin must equal Cin/groups".to_string(),
            });
        }
        let cout_g = cout / g;
        let input_columns =
            ShapeBuf::from_slice(&[cin_g, kernel_size]).checked_numel(OperationKind::Conv1d)?;

        let (out_storage, l_out) = incin_core::exec::GradMode::Disabled.scope(|| {
            let mut group_outputs: Vec<WgpuStorage> = Vec::new();
            let mut l_out = 0usize;
            for group in 0..g {
                let input_g = Self::narrow::<K>(&activation, 1, group * (cin / g), cin / g)?;
                let weight_g = Self::narrow::<K>(weight, 0, group * cout_g, cout_g)?;
                let input_data = input_g.buffer.to_vec::<f32>()?;
                let (cols, l) = im2col_1d_cpu(
                    &input_data,
                    b,
                    cin / g,
                    len,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?;
                l_out = l;
                let cols_storage = WgpuStorage::new(
                    WgpuBuffer::from_slice(&cols),
                    alloc::vec![b, l_out, input_columns],
                );
                let weight_mat = Self::reshape::<K>(&weight_g, &[cout_g, input_columns])?;
                let weight_t = Self::transpose::<K>(&weight_mat, 0, 1)?;
                let out_g = Self::batched_matmul::<K>(&cols_storage, &weight_t)?;
                group_outputs.push(out_g);
            }
            let refs: Vec<&WgpuStorage> = group_outputs.iter().collect();
            let matmul_out = if g == 1 {
                refs[0].clone()
            } else {
                Self::concat_exact::<K>(&refs, 2)?
            };
            // [B, L_out, Cout] -> [B, Cout, L_out].
            let transposed = Self::transpose::<K>(&matmul_out, 1, 2)?;
            let conv_out = Self::reshape::<K>(&transposed, &[b, cout, l_out])?;
            Ok::<_, Error>((conv_out, l_out))
        })?;
        let conv_out = if unbatched {
            Self::reshape::<K>(&out_storage, &[cout, l_out])?
        } else {
            out_storage
        };

        // ONE tape entry over the composition above; the piece calls ran
        // disabled so they left none. Backward recomputes through the same
        // impls (safe: the backward walk itself runs disabled), mirroring
        // CPU's `conv1d_impl` group loop.
        let (input_capture, weight_capture) = (t.clone(), weight.clone());
        let (input_id, weight_id, out_id) = (t.id, weight.id, conv_out.id);
        let input_shape = t.shape.to_vec();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![input_id, weight_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_batched = if unbatched {
                    Self::reshape::<K>(grad_out, &[1, cout, l_out])?
                } else {
                    WgpuStorage::new(grad_out.buffer.clone(), grad_out.shape.to_vec())
                };
                let grad_t = Self::transpose::<K>(&grad_batched, 1, 2)?;
                // The captured input is already row-major flat; the batched
                // view below is metadata over the same buffer, exactly as
                // forward's reshape was.
                let act_shape = if unbatched {
                    alloc::vec![1, input_shape[0], input_shape[1]]
                } else {
                    input_shape.clone()
                };
                let act_storage =
                    WgpuStorage::new(input_capture.buffer.clone(), act_shape.clone());
                let (ab, acin, alen) = (act_shape[0], act_shape[1], act_shape[2]);
                let mut grad_input_groups: Vec<WgpuStorage> = Vec::new();
                let mut grad_weight_groups: Vec<WgpuStorage> = Vec::new();
                for group in 0..g {
                    let input_g =
                        Self::narrow::<K>(&act_storage, 1, group * (acin / g), acin / g)?;
                    let weight_g =
                        Self::narrow::<K>(&weight_capture, 0, group * cout_g, cout_g)?;
                    let grad_out_g =
                        Self::narrow::<K>(&grad_t, 2, group * cout_g, cout_g)?;
                    let weight_mat =
                        Self::reshape::<K>(&weight_g, &[cout_g, input_columns])?;
                    let grad_cols = Self::batched_matmul::<K>(&grad_out_g, &weight_mat)?;
                    let grad_cols_data = grad_cols.buffer.to_vec::<f32>()?;
                    let grad_cols_shape = grad_cols.shape.to_vec();
                    let grad_input_vec = col2im_1d_cpu(
                        &grad_cols_data,
                        ab,
                        acin / g,
                        alen,
                        grad_cols_shape[1],
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    )?;
                    grad_input_groups.push(WgpuStorage::new(
                        WgpuBuffer::from_slice(&grad_input_vec),
                        alloc::vec![ab, acin / g, alen],
                    ));
                    let input_g_data = input_g.buffer.to_vec::<f32>()?;
                    let (cols, _) = im2col_1d_cpu(
                        &input_g_data,
                        ab,
                        acin / g,
                        alen,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    )?;
                    let cols_storage = WgpuStorage::new(
                        WgpuBuffer::from_slice(&cols),
                        alloc::vec![ab, grad_cols_shape[1], input_columns],
                    );
                    let grad_out_g_t = Self::transpose::<K>(&grad_out_g, 1, 2)?;
                    let grad_weight_mat =
                        Self::batched_matmul::<K>(&grad_out_g_t, &cols_storage)?;
                    let grad_weight_summed = Self::sum_dim::<K>(&grad_weight_mat, 0)?;
                    let grad_weight_g = Self::reshape::<K>(
                        &grad_weight_summed,
                        &[cout_g, acin / g, kernel_size],
                    )?;
                    grad_weight_groups.push(grad_weight_g);
                }
                let grad_input = if g == 1 {
                    grad_input_groups.into_iter().next().ok_or(Error::Backend(
                        BackendError::InvalidInput {
                            operation: OperationKind::Conv1d,
                            reason: "the single convolution group produced no input gradient",
                        },
                    ))?
                } else {
                    let refs: Vec<&WgpuStorage> = grad_input_groups.iter().collect();
                    Self::concat_exact::<K>(&refs, 1)?
                };
                let grad_input = if unbatched {
                    Self::reshape::<K>(&grad_input, &input_shape)?
                } else {
                    grad_input
                };
                let grad_weight = if g == 1 {
                    grad_weight_groups.into_iter().next().ok_or(Error::Backend(
                        BackendError::InvalidInput {
                            operation: OperationKind::Conv1d,
                            reason: "the single convolution group produced no weight gradient",
                        },
                    ))?
                } else {
                    let refs: Vec<&WgpuStorage> = grad_weight_groups.iter().collect();
                    Self::concat_exact::<K>(&refs, 0)?
                };
                Ok(alloc::vec![grad_input, grad_weight])
            }),
        });

        match bias {
            Some(bi) => {
                let bcast_shape: Vec<usize> = if unbatched {
                    alloc::vec![cout, 1]
                } else {
                    alloc::vec![1, cout, 1]
                };
                let b_shaped = Self::reshape::<K>(bi, &bcast_shape)?;
                let b_stretched = Self::broadcast_as::<K>(&b_shaped, &conv_out.shape)?;
                Self::add::<K>(&conv_out, &b_stretched)
            }
            None => Ok(conv_out),
        }
    }

    /// `conv_transpose2d` (groups == 1, like CPU): input unfold plus weight
    /// matmul folded back with `col2im`, exactly CPU's
    /// `conv_transpose2d_impl` composition. Rank 3 rides with a batch of one;
    /// bias follows conv2d's reshape/broadcast/add tail. ONE `TapeEntry`
    /// (`input`, `weight`); piece calls run disabled so the composition
    /// leaves no nested entries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_transpose2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if groups != 1 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: alloc::vec![1],
                got: alloc::vec![groups],
                msg: "conv_transpose2d: only groups == 1 is supported".to_string(),
            });
        }
        let rank = t.shape.len();
        if rank != 3 && rank != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: alloc::vec![3, 4],
                got: alloc::vec![rank],
                msg: "conv_transpose2d accepts rank 3 or 4".to_string(),
            });
        }
        let unbatched = rank == 3;
        let activation = if unbatched {
            let mut batched = alloc::vec![1usize];
            batched.extend_from_slice(&t.shape);
            Self::reshape::<K>(t, &batched)?
        } else {
            WgpuStorage::new(t.buffer.clone(), t.shape.to_vec())
        };
        let (b, cin, h, w) = (
            activation.shape[0],
            activation.shape[1],
            activation.shape[2],
            activation.shape[3],
        );
        if weight.shape.len() != 4 || weight.shape[0] != cin {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: alloc::vec![cin],
                got: weight.shape.to_vec(),
                msg: "conv_transpose2d: weight Cin must match input Cin".to_string(),
            });
        }
        let (cout, kh, kw) = (weight.shape[1], weight.shape[2], weight.shape[3]);
        // Natural output size: conv2d's forward-shape formula inverted,
        // saturating rather than underflowing on pathological inputs.
        let h_nat = h
            .saturating_sub(1)
            .saturating_mul(stride.0)
            .saturating_add(dilation.0.saturating_mul(kh.saturating_sub(1)))
            .saturating_add(1)
            .saturating_sub(2 * padding.0);
        let w_nat = w
            .saturating_sub(1)
            .saturating_mul(stride.1)
            .saturating_add(dilation.1.saturating_mul(kw.saturating_sub(1)))
            .saturating_add(1)
            .saturating_sub(2 * padding.1);
        let (op_h, op_w) = output_padding;
        let (final_h, final_w) = (h_nat + op_h, w_nat + op_w);
        let input_spatial = h * w;
        let weight_columns = cout * kh * kw;

        let conv_out = incin_core::exec::GradMode::Disabled.scope(|| {
            let input_flat = Self::reshape::<K>(&activation, &[b, cin, input_spatial])?;
            let input_t = Self::transpose::<K>(&input_flat, 1, 2)?;
            let weight_mat = Self::reshape::<K>(weight, &[cin, weight_columns])?;
            let cols = Self::batched_matmul::<K>(&input_t, &weight_mat)?;
            // Fold [B, H*W, Cout*Kh*Kw] into the natural output via col2im.
            let cols_data = cols.buffer.to_vec::<f32>()?;
            let mut natural_data = alloc::vec![0.0f32; b * cout * h_nat * w_nat];
            for bi in 0..b {
                for co in 0..cout {
                    for hi in 0..h {
                        for wi in 0..w {
                            let col_base =
                                (bi * (h * w) + hi * w + wi) * weight_columns;
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let sh = hi * stride.0 + khi * dilation.0;
                                    let sw = wi * stride.1 + kwi * dilation.1;
                                    if sh >= padding.0
                                        && sh - padding.0 < h_nat
                                        && sw >= padding.1
                                        && sw - padding.1 < w_nat
                                    {
                                        natural_data[(bi * cout + co) * (h_nat * w_nat)
                                            + (sh - padding.0) * w_nat
                                            + (sw - padding.1)] += cols_data[col_base
                                            + co * (kh * kw)
                                            + khi * kw
                                            + kwi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let natural_storage = WgpuStorage::new(
                WgpuBuffer::from_slice(&natural_data),
                alloc::vec![b, cout, h_nat, w_nat],
            );
            let padded = if op_h == 0 && op_w == 0 {
                natural_storage
            } else {
                let _final_storage = WgpuStorage::new(
                    WgpuBuffer::new_zeros_for(
                        DTypeId::F32,
                        b * cout * final_h * final_w,
                        OperationKind::ConvTranspose2d,
                    )?,
                    alloc::vec![b, cout, final_h, final_w],
                );
                // Paste the natural fold into the zeroed final buffer.
                let mut final_data =
                    alloc::vec![0.0f32; b * cout * final_h * final_w];
                let natural_data = natural_storage.buffer.to_vec::<f32>()?;
                for bi in 0..b {
                    for co in 0..cout {
                        for hi in 0..h_nat {
                            for wi in 0..w_nat {
                                final_data[(bi * cout + co) * (final_h * final_w)
                                    + hi * final_w
                                    + wi] = natural_data[(bi * cout + co)
                                    * (h_nat * w_nat)
                                    + hi * w_nat
                                    + wi];
                            }
                        }
                    }
                }
                WgpuStorage::new(
                    WgpuBuffer::from_slice(&final_data),
                    alloc::vec![b, cout, final_h, final_w],
                )
            };
            Ok::<_, Error>(padded)
        })?;
        let conv_out = if unbatched {
            Self::reshape::<K>(&conv_out, &[cout, final_h, final_w])?
        } else {
            conv_out
        };

        // ONE tape entry; backward mirrors CPU's swap (this op's backward
        // w.r.t. input is conv2d's forward formula on the cotangent).
        let (input_capture, weight_capture) = (t.clone(), weight.clone());
        let (input_id, weight_id, out_id) = (t.id, weight.id, conv_out.id);
        let input_shape = t.shape.to_vec();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![input_id, weight_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_batched = if unbatched {
                    Self::reshape::<K>(grad_out, &[1, cout, final_h, final_w])?
                } else {
                    WgpuStorage::new(grad_out.buffer.clone(), grad_out.shape.to_vec())
                };
                // Crop output padding: it contributed nothing.
                let grad_nat = if op_h == 0 && op_w == 0 {
                    grad_batched
                } else {
                    let cropped_h =
                        Self::narrow::<K>(&grad_batched, 2, 0, h_nat)?;
                    Self::narrow::<K>(&cropped_h, 3, 0, w_nat)?
                };
                // grad_out_nat: unfold with the forward fold's window, then
                // matmul against the weight matrix transposed.
                let grad_nat_data = grad_nat.buffer.to_vec::<f32>()?;
                let (cols, _) = {
                    // [B, Cout, H_nat, W_nat] -> im2col over the INPUT
                    // geometry windows the forward fold used.
                    let mut unfolded = alloc::vec![0.0f32; b * h * w * cout * kh * kw];
                    for bi in 0..b {
                        for co in 0..cout {
                            for hi in 0..h {
                                for wi in 0..w {
                                    for khi in 0..kh {
                                        for kwi in 0..kw {
                                            let sh = hi * stride.0 + khi * dilation.0;
                                            let sw = wi * stride.1 + kwi * dilation.1;
                                            let val = if sh >= padding.0
                                                && sh - padding.0 < h_nat
                                                && sw >= padding.1
                                                && sw - padding.1 < w_nat
                                            {
                                                grad_nat_data[(bi * cout + co)
                                                    * (h_nat * w_nat)
                                                    + (sh - padding.0) * w_nat
                                                    + (sw - padding.1)]
                                            } else {
                                                0.0
                                            };
                                            unfolded[((bi * h + hi) * w + wi) * (cout * kh * kw)
                                                + co * (kh * kw)
                                                + khi * kw
                                                + kwi] = val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    (unfolded, ())
                };
                let cols_storage = WgpuStorage::new(
                    WgpuBuffer::from_slice(&cols),
                    alloc::vec![b, h * w, cout * kh * kw],
                );
                let weight_mat =
                    Self::reshape::<K>(&weight_capture, &[cin, weight_columns])?;
                let weight_t = Self::transpose::<K>(&weight_mat, 0, 1)?;
                let grad_input_flat =
                    Self::batched_matmul::<K>(&cols_storage, &weight_t)?;
                let grad_input_t =
                    Self::transpose::<K>(&grad_input_flat, 1, 2)?;
                let grad_input_full =
                    Self::reshape::<K>(&grad_input_t, &[b, cin, h, w])?;
                let grad_input = if unbatched {
                    Self::reshape::<K>(&grad_input_full, &input_shape)?
                } else {
                    grad_input_full
                };
                // grad_weight: input_flat^T @ grad_out_cols, summed over batch.
                // input_flat is already contiguous [B,Cin,H*W]: reshape only.
                let input_flat =
                    Self::reshape::<K>(&input_capture, &[b, cin, h * w])?;
                let in_cols_t = Self::transpose::<K>(&input_flat, 1, 2)?;
                let in_cols_storage = Self::transpose::<K>(&in_cols_t, 1, 2)?;
                // grad_out as [B, H*W, Cout*Kh*Kw]: reuse the forward fold's
                // window by unfolding grad_nat the same way.
                let grad_nat_data2 = grad_nat.buffer.to_vec::<f32>()?;
                let mut go_cols = alloc::vec![0.0f32; b * h * w * cout * kh * kw];
                for bi in 0..b {
                    for hi in 0..h {
                        for wi in 0..w {
                            for co in 0..cout {
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let sh = hi * stride.0 + khi * dilation.0;
                                        let sw = wi * stride.1 + kwi * dilation.1;
                                        let val = if sh >= padding.0
                                            && sh - padding.0 < h_nat
                                            && sw >= padding.1
                                            && sw - padding.1 < w_nat
                                        {
                                            grad_nat_data2[(bi * cout + co)
                                                * (h_nat * w_nat)
                                                + (sh - padding.0) * w_nat
                                                + (sw - padding.1)]
                                        } else {
                                            0.0
                                        };
                                        go_cols[((bi * h + hi) * w + wi) * (cout * kh * kw)
                                            + co * (kh * kw)
                                            + khi * kw
                                            + kwi] = val;
                                    }
                                }
                            }
                        }
                    }
                }
                let go_cols_storage = WgpuStorage::new(
                    WgpuBuffer::from_slice(&go_cols),
                    alloc::vec![b, h * w, cout * kh * kw],
                );
                let grad_weight_mat =
                    Self::batched_matmul::<K>(&in_cols_storage, &go_cols_storage)?;
                let grad_weight_summed = Self::sum_dim::<K>(&grad_weight_mat, 0)?;
                let grad_weight = Self::reshape::<K>(
                    &grad_weight_summed,
                    &[cin, cout, kh, kw],
                )?;
                Ok(alloc::vec![grad_input, grad_weight])
            }),
        });

        match bias {
            Some(bi) => {
                let bcast_shape: Vec<usize> = if unbatched {
                    alloc::vec![cout, 1, 1]
                } else {
                    alloc::vec![1, cout, 1, 1]
                };
                let b_shaped = Self::reshape::<K>(bi, &bcast_shape)?;
                let b_stretched = Self::broadcast_as::<K>(&b_shaped, &conv_out.shape)?;
                Self::add::<K>(&conv_out, &b_stretched)
            }
            None => Ok(conv_out),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss helper (cross_entropy is composed from float/reduce operations).
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Quantization helpers (Q8_0: CPU-side encode/decode, GPU matmul via dequant)
// ─────────────────────────────────────────────────────────────────────────────
//
// WgpuStorage stores raw bytes in a WgpuBuffer. For Q8_0 quantized tensors,
// the buffer holds packed BlockQ8_0 structs (34 bytes each):
//   [0..1]  = f16 scale `d` (little-endian)
//   [2..33] = 32 × i8 quantized weights
//
// This mirrors the NativeBackend's `BlockQ8_0` layout, allowing byte-level
// interoperability.  The encode/decode runs on the CPU (WgpuBuffer::to_vec /
// from_slice); a GPU-native WGSL kernel is deferred post-0.1.0.

impl<D: Device> WgpuBackendImpl<D> {
    /// `quantize` to Q8_0 blocks: download `f32`, encode CPU-side with the
    /// same block math as `cpu::ops::quant::quantize_storage` (per-block
    /// `max_abs / 127` scale, round-to-nearest `i8`), upload the packed
    /// 34-byte blocks under the Q8_0 label. Only Q8_0 is admitted (CPU
    /// routes FP4 targets too; that admission lights up here when the
    /// catalog rows do). Straight-through estimator: the backward passes
    /// the cotangent through unchanged, one tape entry.
    pub(crate) fn quantize<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dtype != DTypeId::Q8_0.descriptor() {
            return Err(Error::Backend(BackendError::unsupported(
                "Wgpu",
                UnsupportedReason::DType {
                    operation: OperationKind::Quantize,
                    dtype,
                },
            )));
        }
        if t.dtype != DTypeId::F32.descriptor() {
            return Err(Error::UnsupportedDType {
                dtype: t.dtype,
                backend: "Wgpu",
                op: "quantize",
            });
        }
        let data = t.buffer.to_vec::<f32>()?;
        if data.len() % 32 != 0 {
            return Err(Error::Msg(alloc::format!(
                "quantize Q8_0 requires buffer length multiple of 32, got {}",
                data.len()
            )));
        }
        let mut bytes = alloc::vec::Vec::with_capacity(data.len() / 32 * 34);
        for chunk in data.chunks_exact(32) {
            let max_abs = chunk.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let d = max_abs / 127.0;
            let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
            let mut qs = [0i8; 32];
            for (index, value) in chunk.iter().enumerate() {
                qs[index] = (*value * inv_d).round() as i8;
            }
            let block = crate::quant::BlockQ8_0 {
                d: half::f16::from_f32(d),
                qs,
            };
            bytes.extend_from_slice(&block.d.to_le_bytes());
            bytes.extend_from_slice(&qs.map(|q| q as u8));
        }
        let out = WgpuStorage::new_with_dtype(
            WgpuBuffer::from_slice(&bytes),
            t.shape.to_vec(),
            DTypeId::Q8_0.descriptor(),
        );
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(alloc::vec![grad_out.clone()])
            }),
        });
        Ok(out)
    }

    /// `dequantize` from Q8_0 blocks to `f32`: download, decode CPU-side,
    /// upload. The dual of `quantize`'s straight-through node: decoding has
    /// no derivative either, so the cotangent passes to the quantized
    /// operand unchanged, one tape entry.
    pub(crate) fn dequantize<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dtype != DTypeId::F32.descriptor() {
            return Err(Error::Backend(BackendError::unsupported(
                "Wgpu",
                UnsupportedReason::DType {
                    operation: OperationKind::Dequantize,
                    dtype,
                },
            )));
        }
        if t.dtype != DTypeId::Q8_0.descriptor() {
            return Err(Error::UnsupportedDType {
                dtype: t.dtype,
                backend: "Wgpu",
                op: "dequantize",
            });
        }
        let bytes = t.buffer.to_vec::<u8>()?;
        if bytes.len() % 34 != 0 {
            return Err(Error::Msg(alloc::format!(
                "dequantize Q8_0 requires a whole number of 34-byte blocks, got {} bytes",
                bytes.len()
            )));
        }
        let mut out = alloc::vec::Vec::with_capacity(bytes.len() / 34 * 32);
        for block in bytes.chunks_exact(34) {
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for &q in &block[2..34] {
                out.push((q as i8) as f32 * d);
            }
        }
        let out_storage = WgpuStorage::new(WgpuBuffer::from_slice(&out), t.shape.to_vec());
        let (t_id, out_id) = (t.id, out_storage.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(alloc::vec![grad_out.clone()])
            }),
        });
        Ok(out_storage)
    }

    /// `quantized_matmul`: dequantize both Q8_0 operands on the host, then
    /// the GPU matmul. Records nothing and has `GradientRule::None` by
    /// design, like CPU/CUDA.
    pub(crate) fn quantized_matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        for operand in [lhs, rhs] {
            if operand.dtype != DTypeId::Q8_0.descriptor() {
                return Err(Error::UnsupportedDType {
                    dtype: operand.dtype,
                    backend: "Wgpu",
                    op: "quantized_matmul",
                });
            }
        }
        if lhs.shape.len() < 2 || rhs.shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "quantized_matmul",
                expected: alloc::vec![2],
                got: alloc::vec![lhs.shape.len(), rhs.shape.len()],
                msg: "quantized_matmul requires lhs rank >= 2 and rhs rank 2".to_string(),
            });
        }
        let k = lhs.shape[lhs.shape.len() - 1];
        if k != rhs.shape[1] || !k.is_multiple_of(32) {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::QuantizedMatMul,
                reason: "quantized_matmul K must match and be a multiple of 32",
            }));
        }
        // Dequantize pushed straight-through entries above; this op records
        // nothing by design, so everything below runs disabled.
        incin_core::exec::GradMode::Disabled.scope(|| {
            let lhs_f32 = Self::dequantize::<K>(lhs, DTypeId::F32.descriptor())?;
            let rhs_f32 = Self::dequantize::<K>(rhs, DTypeId::F32.descriptor())?;
            // The quantized rhs is stored pre-transposed ([N,K]); the matmul
            // needs [K,N].
            let rhs_t = Self::transpose::<K>(&rhs_f32, 0, 1)?;
            Self::matmul::<K>(&lhs_f32, &rhs_t)
        })
    }
}
