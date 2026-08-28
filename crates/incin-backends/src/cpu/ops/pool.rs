//! `max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d` for `CpuBackendImpl<D>` -
//! generalizes `ops/reduce.rs`'s `max_axis_with_indices`/`scatter_axis_grad`
//! 1D-axis-reduction pattern to a 2D sliding window (D-01/D-02).
//!
//! `max_window_2d`/`scatter_pool_grad_2d` are the 2D generalization of
//! `max_axis_with_indices`/`scatter_axis_grad`. Unlike `scatter_axis_grad`'s
//! bare `=` overwrite (correct there - each axis-reduce output position has
//! exactly one winning source position, contributed to by only that one
//! output), `scatter_pool_grad_2d` uses `+=` accumulation: pooling windows
//! can overlap (stride < kernel_size), so the SAME input position can be the
//! winner for two or more output windows, and each must contribute its own
//! gradient share (Pitfall 5 / T-04-14).
//!
//! Padding: any window position landing in the padded region is treated as
//! NOT a max-pooling candidate (skipped entirely, never substituted with
//! `0.0`), mirroring PyTorch/Candle's "padding contributes -inf to max-pool"
//! convention - a real negative-valued input must not lose to an artificial
//! zero. `avg_pool2d`/`adaptive_avg_pool2d`, by contrast, treat the padded
//! region as `0.0` contributing to BOTH the sum and the divisor
//! (`count_include_pad=True`, PyTorch's default).
//!
//! `adaptive_avg_pool2d` computes per-output-position variable window
//! boundaries (`start = floor(i*input_size/output_size)`,
//! `end = ceil((i+1)*input_size/output_size)`), independently per H/W axis -
//! NOT a fixed kernel_size/stride derivation, which produces wrong results
//! whenever `input_size` doesn't evenly divide `output_size` (Pitfall 6 /
//! T-04-15's sibling correctness concern for adaptive's own window sizing).

use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf, ShapeError};
use incin_core::tensor::dtype::DType;

use crate::cpu::storage::{CpuBuffer, CpuStorage, increment_index};
use crate::cpu::tape::{self, TapeEntry};

// ---------------------------------------------------------------------------
// max_pool2d
// ---------------------------------------------------------------------------

/// Split a pooling activation into a batch, a channel count and the two
/// spatial extents.
///
/// The capability row admits rank three, and `inference.rs` rewrites only the
/// trailing two axes, so the unbatched `[C, H, W]` form is advertised and its
/// output shape is already inferred. At rank three the batch is one, and at
/// rank four it is the leading extent, which leaves the flat layout, and so
/// the max-pool index vector, unchanged from before unbatched input existed.
///
/// The catalog pins `accepted_ranks` at `3..=4` for all three pooling
/// operations, so those are the only two cases. Anything else is refused here
/// rather than folded, because the fixed `[batch, channel, y, x]` index the
/// call sites build could not address a deeper operand anyway.
fn batched_spatial(
    shape: &[usize],
    operation: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let rank = shape.len();
    if !(3..=4).contains(&rank) {
        return Err(Error::UnsupportedBackendOperation {
            op: operation,
            backend: "Cpu pooling outside rank three or four",
        });
    }
    let batch = if rank == 4 { shape[0] } else { 1 };
    Ok((batch, shape[rank - 3], shape[rank - 2], shape[rank - 1]))
}

/// How many leading entries of a `[batch, channel, y, x]` index to skip for an
/// activation of this rank. One for the unbatched `[C, H, W]` form, zero for
/// the batched one, so the same fixed array serves both without allocating.
fn index_skip(shape: &[usize]) -> usize {
    4usize.saturating_sub(shape.len())
}

/// The pooled output shape: the input's, with the two spatial extents replaced.
fn pooled_shape(shape: &[usize], h_out: usize, w_out: usize) -> Vec<usize> {
    let mut output = shape.to_vec();
    let rank = output.len();
    output[rank - 2] = h_out;
    output[rank - 1] = w_out;
    output
}

/// 2D generalization of `ops::reduce::max_axis_with_indices`: for each output
/// position `(b, c, h_out, w_out)`, scan the `kernel_size` window (accounting
/// for stride/padding/dilation), skipping any position landing in the padded
/// region entirely (not a candidate - never treated as `0.0`), and track the
/// winning flat-index-into-`input` (strict `>`, first-encountered wins,
/// matching `max_axis_with_indices`'s tie convention).
fn max_window_2d(
    input: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<(CpuStorage, Vec<usize>)> {
    let (b, c, h, w) = batched_spatial(&input.shape, "max_pool2d")?;
    let in_skip = index_skip(&input.shape);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    let h_out = out_size(h, kh, sh, ph, dh)?;
    let w_out = out_size(w, kw, sw, pw, dw)?;

    let out_total =
        ShapeBuf::from_slice(&[b, c, h_out, w_out]).checked_numel(OperationKind::Pool2d)?;
    let mut best_val = vec![f64::NEG_INFINITY; out_total];
    let mut best_flat_src_idx = vec![0usize; out_total];
    let input_strides = crate::cpu::stride::contiguous_strides(&input.shape);

    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let src_h = oh * sh + khi * dh;
                            let src_w = ow * sw + kwi * dw;
                            // Padded-region positions are NOT candidates -
                            // skip entirely rather than substitute 0.0
                            // (max-pool's "-inf padding" convention).
                            if src_h < ph || src_h - ph >= h || src_w < pw || src_w - pw >= w {
                                continue;
                            }
                            let ih = src_h - ph;
                            let iw = src_w - pw;
                            let v = input.get(&[bi, ci, ih, iw][in_skip..]);
                            if v > best_val[flat_out] {
                                best_val[flat_out] = v;
                                best_flat_src_idx[flat_out] = [bi, ci, ih, iw][in_skip..]
                                    .iter()
                                    .zip(input_strides.iter())
                                    .map(|(index, stride)| index * stride)
                                    .sum();
                            }
                        }
                    }
                }
            }
        }
    }

    let out = CpuStorage::from_contiguous(
        CpuBuffer::F32(best_val.iter().map(|&v| v as f32).collect()),
        pooled_shape(&input.shape, h_out, w_out),
    );
    Ok((out, best_flat_src_idx))
}

/// Backward helper for `max_pool2d`: build a zero-filled buffer sized to
/// `input_shape`, then for each output position `+=` (NEVER `=`)
/// `grad_out`'s value at that position into
/// `vals[winning_flat_src_idx[flat_out]]`. This is the Pitfall 5 fix -
/// explicitly diverges from `ops::reduce::scatter_axis_grad`'s bare
/// assignment, since overlapping pooling windows can share a winning input
/// position and each contribution must be summed, not overwritten.
fn scatter_pool_grad_2d(
    grad_out: &CpuStorage,
    winning_flat_src_idx: &[usize],
    input_shape: &[usize],
) -> CpuStorage {
    let total: usize = crate::cpu::stride::validated_numel(input_shape);
    let mut vals = vec![0.0f32; total];
    let out_total: usize = crate::cpu::stride::validated_numel(&(grad_out.shape));
    let mut out_idx = vec![0usize; grad_out.shape.len()];
    for flat_out in 0..out_total {
        let g = grad_out.get(&out_idx);
        vals[winning_flat_src_idx[flat_out]] += g as f32;
        increment_index(&mut out_idx, &grad_out.shape);
    }
    CpuStorage::from_contiguous(CpuBuffer::F32(vals), input_shape)
}

/// Canonical max-pool implementation shared by the CPU executor.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn max_pool2d_impl<D: incin_core::tensor::device::Device, K: DType>(
    t: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<CpuStorage> {
    let (out, winning_flat_src_idx) = max_window_2d(t, kernel_size, stride, padding, dilation)?;

    let input_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![scatter_pool_grad_2d(
                grad_out,
                &winning_flat_src_idx,
                &input_shape,
            )])
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// avg_pool2d
// ---------------------------------------------------------------------------

/// Canonical avg-pool implementation: for each output
/// position, sums the window's values (padded-region positions contribute
/// `0.0` to both the sum and the fixed `kernel_size.0 * kernel_size.1`
/// divisor - PyTorch's `count_include_pad=True` default) divided by the
/// window element count. Backward distributes `grad_out`'s per-position
/// value UNIFORMLY (divided by the window's element count) into every input
/// position the window covered, `+=`-accumulating across overlapping
/// windows.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn avg_pool2d_impl<D: incin_core::tensor::device::Device, K: DType>(
    t: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<CpuStorage> {
    let (b, c, h, w) = batched_spatial(&t.shape, "avg_pool2d")?;
    let in_skip = index_skip(&t.shape);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = out_size(h, kh, sh, ph, 1)?;
    let w_out = out_size(w, kw, sw, pw, 1)?;

    let window_count = ShapeBuf::from_slice(&[kh, kw]).checked_numel(OperationKind::Pool2d)? as f64;
    let out_total =
        ShapeBuf::from_slice(&[b, c, h_out, w_out]).checked_numel(OperationKind::Pool2d)?;
    let mut out_vals = vec![0.0f32; out_total];
    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f64;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let src_h = oh * sh + khi;
                            let src_w = ow * sw + kwi;
                            let v =
                                if src_h >= ph && src_h - ph < h && src_w >= pw && src_w - pw < w {
                                    t.get(&[bi, ci, src_h - ph, src_w - pw][in_skip..])
                                } else {
                                    0.0
                                };
                            sum += v;
                        }
                    }
                    let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                    out_vals[flat_out] = (sum / window_count) as f32;
                }
            }
        }
    }
    let out = CpuStorage::from_contiguous(
        CpuBuffer::F32(out_vals),
        pooled_shape(&t.shape, h_out, w_out),
    );

    let input_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let (b, c, h, w) = batched_spatial(&input_shape, "pool2d backward")
                .expect("the forward pass already accepted this rank");
            let out_skip = index_skip(&grad_out.shape);
            let input_total =
                ShapeBuf::from_slice(&input_shape).checked_numel(OperationKind::Pool2d)?;
            let mut vals = vec![0.0f32; input_total];
            let in_strides = crate::cpu::stride::contiguous_strides(&input_shape);
            let h_out = grad_out.shape[2];
            let w_out = grad_out.shape[3];
            for bi in 0..b {
                for ci in 0..c {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let g = grad_out.get(&[bi, ci, oh, ow][out_skip..]) / window_count;
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
                                        vals[flat] += g as f32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                input_shape.clone(),
            )])
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d
// ---------------------------------------------------------------------------

/// Per RESEARCH.md Pitfall 6: computes PER-OUTPUT-POSITION window boundaries
/// via `start = floor(i * input_size / output_size)`,
/// `end = ceil((i+1) * input_size / output_size)`, independently per axis.
/// Never derives an equivalent fixed `kernel_size`/`stride` - that produces
/// wrong results whenever `input_size` does not evenly divide `output_size`
/// (e.g. 5 -> 3 produces window sizes `[2, 3, 2]`, not a uniform kernel).
fn adaptive_window_bounds(
    input_size: usize,
    output_size: usize,
    i: usize,
) -> Result<(usize, usize)> {
    if output_size == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::AdaptiveAvgPool2d,
            parameter: "output size",
            value: output_size,
        }
        .into());
    }
    let start = i
        .checked_mul(input_size)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::AdaptiveAvgPool2d,
            expression: "adaptive-pooling start index",
        })?
        / output_size;
    let end = i
        .checked_add(1)
        .and_then(|next| next.checked_mul(input_size))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::AdaptiveAvgPool2d,
            expression: "adaptive-pooling end index",
        })?
        .div_ceil(output_size);
    Ok((start, end))
}

/// Canonical adaptive-average-pool implementation.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn adaptive_avg_pool2d_impl<D: incin_core::tensor::device::Device, K: DType>(
    t: &CpuStorage,
    output_size: (usize, usize),
) -> Result<CpuStorage> {
    let (b, c, h, w) = batched_spatial(&t.shape, "adaptive_avg_pool2d")?;
    let in_skip = index_skip(&t.shape);
    let (h_out, w_out) = output_size;

    let out_total = ShapeBuf::from_slice(&[b, c, h_out, w_out])
        .checked_numel(OperationKind::AdaptiveAvgPool2d)?;
    let mut out_vals = vec![0.0f32; out_total];
    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..h_out {
                let (h_start, h_end) = adaptive_window_bounds(h, h_out, oh)?;
                for ow in 0..w_out {
                    let (w_start, w_end) = adaptive_window_bounds(w, w_out, ow)?;
                    let mut sum = 0.0f64;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += t.get(&[bi, ci, ih, iw][in_skip..]);
                        }
                    }
                    let count = ((h_end - h_start) * (w_end - w_start)) as f64;
                    let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                    out_vals[flat_out] = (sum / count) as f32;
                }
            }
        }
    }
    let out = CpuStorage::from_contiguous(
        CpuBuffer::F32(out_vals),
        pooled_shape(&t.shape, h_out, w_out),
    );

    let input_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let (b, c, h, w) = batched_spatial(&input_shape, "pool2d backward")
                .expect("the forward pass already accepted this rank");
            let out_skip = index_skip(&grad_out.shape);
            let input_total = ShapeBuf::from_slice(&input_shape)
                .checked_numel(OperationKind::AdaptiveAvgPool2d)?;
            let mut vals = vec![0.0f32; input_total];
            let in_strides = crate::cpu::stride::contiguous_strides(&input_shape);
            let h_out = grad_out.shape[2];
            let w_out = grad_out.shape[3];
            for bi in 0..b {
                for ci in 0..c {
                    for oh in 0..h_out {
                        let (h_start, h_end) = adaptive_window_bounds(h, h_out, oh)?;
                        for ow in 0..w_out {
                            let (w_start, w_end) = adaptive_window_bounds(w, w_out, ow)?;
                            let count = ((h_end - h_start) * (w_end - w_start)) as f64;
                            let g = grad_out.get(&[bi, ci, oh, ow][out_skip..]) / count;
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let flat = bi * in_strides[0]
                                        + ci * in_strides[1]
                                        + ih * in_strides[2]
                                        + iw * in_strides[3];
                                    vals[flat] += g as f32;
                                }
                            }
                        }
                    }
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                input_shape.clone(),
            )])
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared output-size arithmetic (mirrors ops::conv's out_size)
// ---------------------------------------------------------------------------

/// `out_size`.
fn out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| len.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling padded input dimension",
        })?;
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "pooling effective kernel",
        })?;
    Ok(padded.saturating_sub(effective_kernel) / stride + 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::gradcheck::{F32_STEP, GRAD_TOL, gradcheck};
    use incin_core::tensor::device::Cpu;

    /// `tensor`.
    fn tensor(v: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), shape)
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- unbatched activations ---
    //
    // The capability rows admit rank three and the descriptor's shape
    // inference is rank agnostic, so [C, H, W] is an advertised input. Each of
    // these pins the unbatched result against the batched one it must equal,
    // rather than only against a hand computation, so a future change to the
    // batch folding cannot drift the two apart.

    #[test]
    /// `max_pool2d_unbatched_matches_the_batched_form`.
    fn max_pool2d_unbatched_matches_the_batched_form() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let unbatched = tensor(data.clone(), vec![1, 4, 4]);
        let batched = tensor(data, vec![1, 1, 4, 4]);

        let thin = max_pool2d_impl::<Cpu, f32>(&unbatched, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        let wide = max_pool2d_impl::<Cpu, f32>(&batched, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();

        assert_eq!(
            thin.shape,
            vec![1, 2, 2],
            "the output keeps the input's rank"
        );
        assert_eq!(wide.shape, vec![1, 1, 2, 2]);
        assert_eq!(f32_vec(&thin), f32_vec(&wide));
        assert_eq!(f32_vec(&thin), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    /// `avg_pool2d_unbatched_matches_the_batched_form`.
    fn avg_pool2d_unbatched_matches_the_batched_form() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let unbatched = tensor(data.clone(), vec![1, 4, 4]);
        let batched = tensor(data, vec![1, 1, 4, 4]);

        let thin = avg_pool2d_impl::<Cpu, f32>(&unbatched, (2, 2), (2, 2), (0, 0)).unwrap();
        let wide = avg_pool2d_impl::<Cpu, f32>(&batched, (2, 2), (2, 2), (0, 0)).unwrap();

        assert_eq!(thin.shape, vec![1, 2, 2]);
        assert_eq!(f32_vec(&thin), f32_vec(&wide));
        assert_eq!(f32_vec(&thin), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    /// `adaptive_avg_pool2d_unbatched_matches_the_batched_form`.
    fn adaptive_avg_pool2d_unbatched_matches_the_batched_form() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let unbatched = tensor(data.clone(), vec![1, 4, 4]);
        let batched = tensor(data, vec![1, 1, 4, 4]);

        let thin = adaptive_avg_pool2d_impl::<Cpu, f32>(&unbatched, (2, 2)).unwrap();
        let wide = adaptive_avg_pool2d_impl::<Cpu, f32>(&batched, (2, 2)).unwrap();

        assert_eq!(thin.shape, vec![1, 2, 2]);
        assert_eq!(f32_vec(&thin), f32_vec(&wide));
    }

    #[test]
    /// `pooling_outside_rank_three_or_four_is_refused_rather_than_panicking`.
    fn pooling_outside_rank_three_or_four_is_refused_rather_than_panicking() {
        let flat = tensor(vec![1.0, 2.0], vec![2]);
        assert!(max_pool2d_impl::<Cpu, f32>(&flat, (1, 1), (1, 1), (0, 0), (1, 1)).is_err());
        assert!(avg_pool2d_impl::<Cpu, f32>(&flat, (1, 1), (1, 1), (0, 0)).is_err());
        assert!(adaptive_avg_pool2d_impl::<Cpu, f32>(&flat, (1, 1)).is_err());

        // The catalog stops at rank four, so a deeper operand is refused
        // rather than folded: the fixed four-element index the loops build
        // could not address it.
        let deep = tensor(vec![1.0; 16], vec![1, 1, 1, 4, 4]);
        assert!(max_pool2d_impl::<Cpu, f32>(&deep, (2, 2), (2, 2), (0, 0), (1, 1)).is_err());
        assert!(avg_pool2d_impl::<Cpu, f32>(&deep, (2, 2), (2, 2), (0, 0)).is_err());
        assert!(adaptive_avg_pool2d_impl::<Cpu, f32>(&deep, (2, 2)).is_err());
    }

    // --- output-size arithmetic edge cases ---

    /// The pooling window shares `conv2d`'s output-size arithmetic, so it
    /// shares the same underflow hazard: a 5x5 window with dilation 3 spans
    /// 13 elements against a 2x2 input. It must saturate rather than
    /// underflow the `usize` subtraction.
    #[test]
    fn max_pool2d_with_a_window_larger_than_its_input_yields_an_empty_output_not_a_panic() {
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let out = max_pool2d_impl::<Cpu, f32>(&input, (5, 5), (1, 1), (0, 0), (3, 3)).unwrap();

        assert_eq!(out.shape.as_ref(), &[1, 1, 1, 1]);
        assert_eq!(f32_vec(&out).len(), 1);
    }

    // --- max_pool2d forward ---

    #[test]
    /// `max_pool2d_forward_no_overlap_hand_computed`.
    fn max_pool2d_forward_no_overlap_hand_computed() {
        // [1,1,4,4] input, kernel=2x2, stride=2x2 -> [1,1,2,2], non-overlapping.
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = max_pool2d_impl::<Cpu, f32>(&input, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // window(0,0)=rows0-1,cols0-1={1,2,5,6}->6
        // window(0,1)=rows0-1,cols2-3={3,4,7,8}->8
        // window(1,0)=rows2-3,cols0-1={9,10,13,14}->14
        // window(1,1)=rows2-3,cols2-3={11,12,15,16}->16
        assert_eq!(f32_vec(&out), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    /// `max_pool2d_forward_with_padding_zero_boundary`.
    fn max_pool2d_forward_with_padding_zero_boundary() {
        // [1,1,2,2] input, kernel=2x2, stride=1x1, padding=1x1.
        let input = tensor(vec![1.0, -2.0, -3.0, 4.0], vec![1, 1, 2, 2]);
        let out = max_pool2d_impl::<Cpu, f32>(&input, (2, 2), (1, 1), (1, 1), (1, 1)).unwrap();
        // padded region: -inf-candidate skip, not 0.0 - confirms real values
        // (including negatives) win over padding rather than losing to an
        // artificial 0.0.
        // H_out = W_out = (2+2-2)/1+1 = 3
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        let vals = f32_vec(&out);
        // Corner window (0,0): only padded + input[0,0]=1.0 is a candidate -> 1.0
        assert_eq!(vals[0], 1.0);
        // Center window (1,1): all 4 real values {1,-2,-3,4} -> max = 4.0
        assert_eq!(vals[4], 4.0);
    }

    // --- max_pool2d backward ---

    #[test]
    /// `max_pool2d_backward_non_overlapping_routes_grad_to_winner_only`.
    fn max_pool2d_backward_non_overlapping_routes_grad_to_winner_only() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = max_pool2d_impl::<Cpu, f32>(&input, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(input.id).expect("grad_input should exist");
        let vals = f32_vec(g);
        // Winners: 6 (idx 5), 8 (idx 7), 14 (idx 13), 16 (idx 15). All others 0.
        let expected = [
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        assert_eq!(vals, expected);
    }

    /// Overlap test (Pitfall 5 / T-04-14 - the load-bearing test for this
    /// plan): construct a small input where ONE specific input position is
    /// the argmax winner for TWO adjacent overlapping output windows. The
    /// backward gradient at that position must equal the SUM of both
    /// windows' incoming gradient (2.0 total from a ones-seed), not just 1.0
    /// (which would indicate the anti-pattern bare `=` overwrite bug).
    #[test]
    fn max_pool2d_backward_overlapping_windows_shared_winner_accumulates() {
        // [1,1,1,3] input: single global max at the middle position, so it
        // wins BOTH overlapping windows (stride=1 < kernel_size=2).
        let input = tensor(vec![1.0, 100.0, 1.0], vec![1, 1, 1, 3]);
        let out = max_pool2d_impl::<Cpu, f32>(&input, (1, 2), (1, 1), (0, 0), (1, 1)).unwrap();
        // H_out=1, W_out = (3-2)/1+1 = 2: window0=[1,100]->100, window1=[100,1]->100.
        assert_eq!(out.shape, vec![1, 1, 1, 2]);
        assert_eq!(f32_vec(&out), vec![100.0, 100.0]);

        let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(input.id).expect("grad_input should exist");
        let vals = f32_vec(g);
        // Middle position (idx 1) wins both windows -> gradient = 1.0 + 1.0 = 2.0.
        // Both edge positions never win -> gradient = 0.0.
        assert_eq!(vals, vec![0.0, 2.0, 0.0]);
    }

    #[test]
    /// `max_pool2d_gradcheck_overlapping`.
    fn max_pool2d_gradcheck_overlapping() {
        let input = tensor(
            vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6, 0.8],
            vec![1, 1, 3, 3],
        );
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out =
                max_pool2d_impl::<Cpu, f32>(&inputs[0], (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();
            crate::cpu::ops::reduce::sum_all(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], F32_STEP);
        assert!(
            max_rel_err < GRAD_TOL,
            "max_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- avg_pool2d forward ---

    #[test]
    /// `avg_pool2d_forward_no_overlap_hand_computed`.
    fn avg_pool2d_forward_no_overlap_hand_computed() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = avg_pool2d_impl::<Cpu, f32>(&input, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // window(0,0) mean of {1,2,5,6} = 3.5
        // window(0,1) mean of {3,4,7,8} = 5.5
        // window(1,0) mean of {9,10,13,14} = 11.5
        // window(1,1) mean of {11,12,15,16} = 13.5
        assert_eq!(f32_vec(&out), vec![3.5, 5.5, 11.5, 13.5]);
    }

    // --- avg_pool2d backward ---

    #[test]
    /// `avg_pool2d_backward_overlapping_windows_sums_grad_contributions`.
    fn avg_pool2d_backward_overlapping_windows_sums_grad_contributions() {
        // [1,1,1,3] input, kernel=1x2, stride=1x1 (overlapping): 2 output
        // windows, middle position covered by both.
        let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 1, 3]);
        let out = avg_pool2d_impl::<Cpu, f32>(&input, (1, 2), (1, 1), (0, 0)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 1, 2]);
        // window0 mean{1,2}=1.5, window1 mean{2,3}=2.5
        assert_eq!(f32_vec(&out), vec![1.5, 2.5]);

        let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(input.id).expect("grad_input should exist");
        let vals = f32_vec(g);
        // grad_input[0] = 1/2 (only window0) = 0.5
        // grad_input[1] = 1/2 (window0) + 1/2 (window1) = 1.0 (overlap sum)
        // grad_input[2] = 1/2 (only window1) = 0.5
        assert_eq!(vals, vec![0.5, 1.0, 0.5]);
    }

    #[test]
    /// `avg_pool2d_gradcheck_overlapping`.
    fn avg_pool2d_gradcheck_overlapping() {
        let input = tensor(
            vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6, 0.8],
            vec![1, 1, 3, 3],
        );
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = avg_pool2d_impl::<Cpu, f32>(&inputs[0], (2, 2), (1, 1), (0, 0)).unwrap();
            crate::cpu::ops::reduce::sum_all(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], F32_STEP);
        assert!(
            max_rel_err < GRAD_TOL,
            "avg_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- adaptive_avg_pool2d forward ---

    #[test]
    /// `adaptive_avg_pool2d_evenly_dividing_matches_avg_pool2d`.
    fn adaptive_avg_pool2d_evenly_dividing_matches_avg_pool2d() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let adaptive = adaptive_avg_pool2d_impl::<Cpu, f32>(&input, (2, 2)).unwrap();
        let fixed = avg_pool2d_impl::<Cpu, f32>(&input, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(adaptive.shape, fixed.shape);
        assert_eq!(f32_vec(&adaptive), f32_vec(&fixed));
    }

    /// Non-evenly-dividing case (Pitfall 6): input H=5, output H=3 must
    /// produce per-output-position window sizes [2,3,2] (not a uniform
    /// fixed kernel), matching PyTorch's documented
    /// `start=floor(i*in/out), end=ceil((i+1)*in/out)` formula. (Using
    /// input=5/output=3 here rather than 7/3, since 7/3's own boundaries -
    /// `start=floor(i*7/3), end=ceil((i+1)*7/3)` - evaluate to windows
    /// [0,3),[2,5),[4,7), i.e. sizes [3,3,3] with genuine inter-window
    /// overlap, not the [3,2,2] figure RESEARCH.md's prose used as its
    /// illustrative example; 5/3 is the textbook non-uniform case and
    /// exercises the exact same variable-boundary formula.)
    #[test]
    fn adaptive_avg_pool2d_non_evenly_dividing_produces_variable_windows() {
        // H=5 -> output 3: windows [0,2), [1,4), [3,5) -> sizes [2,3,2].
        assert_eq!(adaptive_window_bounds(5, 3, 0).unwrap(), (0, 2));
        assert_eq!(adaptive_window_bounds(5, 3, 1).unwrap(), (1, 4));
        assert_eq!(adaptive_window_bounds(5, 3, 2).unwrap(), (3, 5));

        // Build a [1,1,5,1] input (W axis trivial, size 1) with distinct
        // values so each H-window's mean is hand-verifiable.
        let input_data: Vec<f32> = (1..=5).map(|x| x as f32).collect(); // 1..5
        let input = tensor(input_data, vec![1, 1, 5, 1]);
        let out = adaptive_avg_pool2d_impl::<Cpu, f32>(&input, (3, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 1]);
        let vals = f32_vec(&out);
        // window0 = mean(1,2) = 1.5
        // window1 = mean(2,3,4) = 3.0
        // window2 = mean(4,5) = 4.5
        assert_eq!(vals, vec![1.5, 3.0, 4.5]);
    }

    // --- adaptive_avg_pool2d backward ---

    #[test]
    /// `adaptive_avg_pool2d_gradcheck_non_evenly_dividing`.
    fn adaptive_avg_pool2d_gradcheck_non_evenly_dividing() {
        let input_data: Vec<f32> = (1..=7).map(|x| x as f32 * 0.1).collect();
        let input = tensor(input_data, vec![1, 1, 7, 1]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = adaptive_avg_pool2d_impl::<Cpu, f32>(&inputs[0], (3, 1)).unwrap();
            crate::cpu::ops::reduce::sum_all(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], F32_STEP);
        assert!(
            max_rel_err < GRAD_TOL,
            "adaptive_avg_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }
}
