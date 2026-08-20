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
/// derives strides from shape), so pooling backward closures — which read
/// buffers back to a flat host `Vec` — can compute this directly instead of
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
        // (divided by the FIXED kh*kw divisor — count_include_pad=True,
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
        // positions are never candidates, never substituted with 0.0 —
        // matches the WGSL forward's `-FLT_MAX` init and its bounds-checked
        // skip), then `+=`-accumulates grad_out's value there — never `=`,
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
