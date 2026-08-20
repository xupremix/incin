//! Pooling and convolution CUDA operations, plus the tape-tracked im2col
//! helper they are composed from.

use super::*;

/// Tape-tracked wrapper pairing `launch_im2col_2d`/`launch_col2im_2d` as each
/// other's forward/backward (they are exact inverses of one another). Once
/// this is a proper tape op, `conv1d`/`conv2d`'s own forward can be composed
/// entirely from already-tape-tracked primitives (`narrow`/`reshape`/
/// `matmul`/`concat` plus this) with NO hand-written backward closure of
/// their own — mirroring the free loss helpers' "free via composition"
/// discovery documented by the backend conformance audit.
pub(crate) fn im2col_2d_tape(
    t: &CudaStorage,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let out = crate::cuda::ops::conv::launch_im2col_2d(t, kh, kw, stride, padding, dilation)?;
    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let h_out =
                crate::cuda::ops::conv::out_size(original_shape[2], kh, stride, padding, dilation)?;
            let w_out =
                crate::cuda::ops::conv::out_size(original_shape[3], kw, stride, padding, dilation)?;
            Ok(vec![crate::cuda::ops::conv::launch_col2im_2d(
                grad_out,
                &original_shape,
                crate::cuda::ops::conv::Col2Im2dSpec {
                    h_out,
                    w_out,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                },
            )?])
        }),
    });
    Ok(out)
}

/// Matches `cpu/ops/conv/helpers.rs::validate_groups` exactly.
pub(crate) fn validate_conv_groups(
    op: &'static str,
    cin: usize,
    cout: usize,
    groups: usize,
) -> Result<()> {
    if groups == 0 || !cin.is_multiple_of(groups) || !cout.is_multiple_of(groups) {
        return Err(Error::ShapeMismatch {
            op,
            expected: vec![groups],
            got: vec![cin, cout],
            msg: format!("{op}: groups={groups} must evenly divide both Cin={cin} and Cout={cout}"),
        });
    }
    Ok(())
}

impl<D: Device> CudaBackendImpl<D> {
    /// Backward replays `max_indices` (captured from the forward pass)
    /// through `scatter_pool_grad_2d` — no forward recomputation needed,
    /// mirrors CPU's `max_window_2d`/`scatter_pool_grad_2d` pairing exactly.
    pub(crate) fn max_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let (out, max_indices) = crate::cuda::ops::pool::launch_max_pool2d_forward(
            t,
            kernel_size,
            stride,
            padding,
            dilation,
        )?;
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::pool::launch_scatter_pool_grad_2d(
                    grad_out,
                    &max_indices,
                    &input_shape,
                )?])
            }),
        });
        Ok(out)
    }

    /// Backward is a real CUDA kernel (`avg_pool2d_backward`), unlike
    /// WGPU's host-readback-and-Rust-loop approach — see this file's
    /// module doc.
    pub(crate) fn avg_pool2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out =
            crate::cuda::ops::pool::launch_avg_pool2d_forward(t, kernel_size, stride, padding)?;
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::pool::launch_avg_pool2d_backward(
                    grad_out,
                    &input_shape,
                    kernel_size,
                    stride,
                    padding,
                )?])
            }),
        });
        Ok(out)
    }

    /// Mirrors `conv1d`'s exact structure generalized to two spatial axes.
    /// CUDA's `im2col_2d` kernel lays cols out channel-major
    /// (`[B, Cin_g*Kh*Kw, H_out*W_out]` — see `cuda/ops/conv.rs`'s module
    /// doc), so this computes `weight_mat @ cols_b` directly per batch, no
    /// transpose of either operand needed (unlike CPU/WGPU's
    /// spatial-major `cols @ weight_mat^T`).
    pub(crate) fn conv2d<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let (t, w): (&CudaStorage, &CudaStorage) = (t, w);
        let bias = bias.map(|b| b as &CudaStorage);
        let (batch, cin, h, wid) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
        let (cout, cin_g, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        validate_conv_groups("conv2d", cin, cout, groups)?;
        if cin / groups != cin_g {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![cin / groups],
                got: vec![cin_g],
                msg: format!(
                    "conv2d: weight's Cin/groups ({cin_g}) does not match input Cin/groups ({})",
                    cin / groups
                ),
            });
        }
        let cout_g = cout / groups;
        let h_out = crate::cuda::ops::conv::out_size(h, kh, stride, padding, dilation)?;
        let w_out = crate::cuda::ops::conv::out_size(wid, kw, stride, padding, dilation)?;

        let mut group_outputs: Vec<CudaStorage> = Vec::with_capacity(groups);
        for g in 0..groups {
            let input_g = Self::narrow::<K>(t, 1, g * cin_g, cin_g)?;
            let weight_g = Self::narrow::<K>(w, 0, g * cout_g, cout_g)?;
            let cols = im2col_2d_tape(&input_g, kh, kw, stride, padding, dilation)?;
            let weight_mat = Self::reshape::<K>(&weight_g, &[cout_g, cin_g * kh * kw])?;

            let mut batch_outs: Vec<CudaStorage> = Vec::with_capacity(batch);
            for bi in 0..batch {
                let cols_b = Self::narrow::<K>(&cols, 0, bi, 1)?;
                let cols_b = Self::squeeze::<K>(&cols_b, 0)?;
                let out_b = Self::matmul::<K>(&weight_mat, &cols_b)?;
                let out_b = Self::reshape::<K>(&out_b, &[1, cout_g, h_out * w_out])?;
                batch_outs.push(out_b);
            }
            let group_out = if batch == 1 {
                batch_outs.into_iter().next().unwrap()
            } else {
                let refs: Vec<&CudaStorage> = batch_outs.iter().collect();
                Self::concat::<K>(&refs, 0)?
            };
            group_outputs.push(group_out);
        }
        let conv_out = if groups == 1 {
            group_outputs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&CudaStorage> = group_outputs.iter().collect();
            Self::concat::<K>(&refs, 1)?
        };
        let conv_out = Self::reshape::<K>(&conv_out, &[batch, cout, h_out, w_out])?;

        match bias {
            Some(bv) => {
                let bias_shaped = Self::reshape::<K>(bv, &[1, cout, 1, 1])?;
                Self::add::<K>(&conv_out, &bias_shaped)
            }
            None => Ok(conv_out),
        }
    }
}
