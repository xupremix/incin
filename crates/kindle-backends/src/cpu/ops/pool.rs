//! `max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d` for `CpuBackend<T, D>` —
//! generalizes `ops/reduce.rs`'s `max_axis_with_indices`/`scatter_axis_grad`
//! 1D-axis-reduction pattern to a 2D sliding window (D-01/D-02).
//!
//! `max_window_2d`/`scatter_pool_grad_2d` are the 2D generalization of
//! `max_axis_with_indices`/`scatter_axis_grad`. Unlike `scatter_axis_grad`'s
//! bare `=` overwrite (correct there — each axis-reduce output position has
//! exactly one winning source position, contributed to by only that one
//! output), `scatter_pool_grad_2d` uses `+=` accumulation: pooling windows
//! can overlap (stride < kernel_size), so the SAME input position can be the
//! winner for two or more output windows, and each must contribute its own
//! gradient share (Pitfall 5 / T-04-14).
//!
//! Padding: any window position landing in the padded region is treated as
//! NOT a max-pooling candidate (skipped entirely, never substituted with
//! `0.0`), mirroring PyTorch/Candle's "padding contributes -inf to max-pool"
//! convention — a real negative-valued input must not lose to an artificial
//! zero. `avg_pool2d`/`adaptive_avg_pool2d`, by contrast, treat the padded
//! region as `0.0` contributing to BOTH the sum and the divisor
//! (`count_include_pad=True`, PyTorch's default).
//!
//! `adaptive_avg_pool2d` computes per-output-position variable window
//! boundaries (`start = floor(i*input_size/output_size)`,
//! `end = ceil((i+1)*input_size/output_size)`), independently per H/W axis —
//! NOT a fixed kernel_size/stride derivation, which produces wrong results
//! whenever `input_size` doesn't evenly divide `output_size` (Pitfall 6 /
//! T-04-15's sibling correctness concern for adaptive's own window sizing).

use kindle_core::prelude::{DType, Result};

use crate::cpu::storage::{CpuBuffer, CpuStorage, increment_index};
use crate::cpu::tape::{self, TapeEntry};

// ---------------------------------------------------------------------------
// max_pool2d
// ---------------------------------------------------------------------------

/// 2D generalization of `ops::reduce::max_axis_with_indices`: for each output
/// position `(b, c, h_out, w_out)`, scan the `kernel_size` window (accounting
/// for stride/padding/dilation), skipping any position landing in the padded
/// region entirely (not a candidate — never treated as `0.0`), and track the
/// winning flat-index-into-`input` (strict `>`, first-encountered wins,
/// matching `max_axis_with_indices`'s tie convention).
fn max_window_2d(
    input: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> (CpuStorage, Vec<usize>) {
    let (b, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    let h_out = out_size(h, kh, sh, ph, dh);
    let w_out = out_size(w, kw, sw, pw, dw);

    let out_total = b * c * h_out * w_out;
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
                            // Padded-region positions are NOT candidates —
                            // skip entirely rather than substitute 0.0
                            // (max-pool's "-inf padding" convention).
                            if src_h < ph || src_h - ph >= h || src_w < pw || src_w - pw >= w {
                                continue;
                            }
                            let ih = src_h - ph;
                            let iw = src_w - pw;
                            let v = input.get(&[bi, ci, ih, iw]);
                            if v > best_val[flat_out] {
                                best_val[flat_out] = v;
                                best_flat_src_idx[flat_out] = bi * input_strides[0]
                                    + ci * input_strides[1]
                                    + ih * input_strides[2]
                                    + iw * input_strides[3];
                            }
                        }
                    }
                }
            }
        }
    }

    let out = CpuStorage::from_contiguous(
        CpuBuffer::F32(best_val.iter().map(|&v| v as f32).collect()),
        vec![b, c, h_out, w_out],
    );
    (out, best_flat_src_idx)
}

/// Backward helper for `max_pool2d`: build a zero-filled buffer sized to
/// `input_shape`, then for each output position `+=` (NEVER `=`)
/// `grad_out`'s value at that position into
/// `vals[winning_flat_src_idx[flat_out]]`. This is the Pitfall 5 fix —
/// explicitly diverges from `ops::reduce::scatter_axis_grad`'s bare
/// assignment, since overlapping pooling windows can share a winning input
/// position and each contribution must be summed, not overwritten.
fn scatter_pool_grad_2d(
    grad_out: &CpuStorage,
    winning_flat_src_idx: &[usize],
    input_shape: &[usize],
) -> CpuStorage {
    let total: usize = input_shape.iter().product();
    let mut vals = vec![0.0f32; total];
    let out_total: usize = grad_out.shape.iter().product();
    let mut out_idx = vec![0usize; grad_out.shape.len()];
    for flat_out in 0..out_total {
        let g = grad_out.get(&out_idx);
        vals[winning_flat_src_idx[flat_out]] += g as f32;
        increment_index(&mut out_idx, &grad_out.shape);
    }
    CpuStorage::from_contiguous(CpuBuffer::F32(vals), input_shape.to_vec())
}

/// `ModuleOps::max_pool2d`'s `CpuBackend` implementation.
pub(crate) fn max_pool2d_impl<T: DType, D: kindle_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<CpuStorage> {
    let (out, winning_flat_src_idx) = max_window_2d(t, kernel_size, stride, padding, dilation);

    let input_shape = t.shape.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            vec![scatter_pool_grad_2d(
                grad_out,
                &winning_flat_src_idx,
                &input_shape,
            )]
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// avg_pool2d
// ---------------------------------------------------------------------------

/// `ModuleOps::avg_pool2d`'s `CpuBackend` implementation: for each output
/// position, sums the window's values (padded-region positions contribute
/// `0.0` to both the sum and the fixed `kernel_size.0 * kernel_size.1`
/// divisor — PyTorch's `count_include_pad=True` default) divided by the
/// window element count. Backward distributes `grad_out`'s per-position
/// value UNIFORMLY (divided by the window's element count) into every input
/// position the window covered, `+=`-accumulating across overlapping
/// windows.
pub(crate) fn avg_pool2d_impl<T: DType, D: kindle_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<CpuStorage> {
    let (b, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = out_size(h, kh, sh, ph, 1);
    let w_out = out_size(w, kw, sw, pw, 1);

    let window_count = (kh * kw) as f64;
    let mut out_vals = vec![0.0f32; b * c * h_out * w_out];
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
                                    t.get(&[bi, ci, src_h - ph, src_w - pw])
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
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(out_vals), vec![b, c, h_out, w_out]);

    let input_shape = t.shape.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let (b, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );
            let mut vals = vec![0.0f32; b * c * h * w];
            let in_strides = crate::cpu::stride::contiguous_strides(&input_shape);
            let h_out = grad_out.shape[2];
            let w_out = grad_out.shape[3];
            for bi in 0..b {
                for ci in 0..c {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let g = grad_out.get(&[bi, ci, oh, ow]) / window_count;
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
            vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                input_shape.clone(),
            )]
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
/// Never derives an equivalent fixed `kernel_size`/`stride` — that produces
/// wrong results whenever `input_size` does not evenly divide `output_size`
/// (e.g. 5 -> 3 produces window sizes `[2, 3, 2]`, not a uniform kernel).
fn adaptive_window_bounds(input_size: usize, output_size: usize, i: usize) -> (usize, usize) {
    let start = (i * input_size) / output_size;
    let end = ((i + 1) * input_size).div_ceil(output_size);
    (start, end)
}

/// `ModuleOps::adaptive_avg_pool2d`'s `CpuBackend` implementation.
pub(crate) fn adaptive_avg_pool2d_impl<T: DType, D: kindle_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    output_size: (usize, usize),
) -> Result<CpuStorage> {
    let (b, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (h_out, w_out) = output_size;

    let mut out_vals = vec![0.0f32; b * c * h_out * w_out];
    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..h_out {
                let (h_start, h_end) = adaptive_window_bounds(h, h_out, oh);
                for ow in 0..w_out {
                    let (w_start, w_end) = adaptive_window_bounds(w, w_out, ow);
                    let mut sum = 0.0f64;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += t.get(&[bi, ci, ih, iw]);
                        }
                    }
                    let count = ((h_end - h_start) * (w_end - w_start)) as f64;
                    let flat_out = ((bi * c + ci) * h_out + oh) * w_out + ow;
                    out_vals[flat_out] = (sum / count) as f32;
                }
            }
        }
    }
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(out_vals), vec![b, c, h_out, w_out]);

    let input_shape = t.shape.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let (b, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );
            let mut vals = vec![0.0f32; b * c * h * w];
            let in_strides = crate::cpu::stride::contiguous_strides(&input_shape);
            let h_out = grad_out.shape[2];
            let w_out = grad_out.shape[3];
            for bi in 0..b {
                for ci in 0..c {
                    for oh in 0..h_out {
                        let (h_start, h_end) = adaptive_window_bounds(h, h_out, oh);
                        for ow in 0..w_out {
                            let (w_start, w_end) = adaptive_window_bounds(w, w_out, ow);
                            let count = ((h_end - h_start) * (w_end - w_start)) as f64;
                            let g = grad_out.get(&[bi, ci, oh, ow]) / count;
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
            vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                input_shape.clone(),
            )]
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared output-size arithmetic (mirrors ops::conv's out_size)
// ---------------------------------------------------------------------------

/// Auto-generated documentation for out_size.
fn out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let padded = len + 2 * padding;
    let effective_kernel = dilation * kernel_size.saturating_sub(1) + 1;
    padded.saturating_sub(effective_kernel) / stride + 1
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::cpu::gradcheck::gradcheck;
    use kindle_core::prelude::{Cpu, ReductionOps};

    /// Auto-generated documentation for TestBackend.
    type TestBackend = CpuBackend<f32, Cpu>;

    /// Auto-generated documentation for tensor.
    fn tensor(v: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), shape)
    }

    /// Auto-generated documentation for f32_vec.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- max_pool2d forward ---

    #[test]
    /// Auto-generated documentation for max_pool2d_forward_no_overlap_hand_computed.
    fn max_pool2d_forward_no_overlap_hand_computed() {
        // [1,1,4,4] input, kernel=2x2, stride=2x2 -> [1,1,2,2], non-overlapping.
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = max_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // window(0,0)=rows0-1,cols0-1={1,2,5,6}->6
        // window(0,1)=rows0-1,cols2-3={3,4,7,8}->8
        // window(1,0)=rows2-3,cols0-1={9,10,13,14}->14
        // window(1,1)=rows2-3,cols2-3={11,12,15,16}->16
        assert_eq!(f32_vec(&out), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    /// Auto-generated documentation for max_pool2d_forward_with_padding_zero_boundary.
    fn max_pool2d_forward_with_padding_zero_boundary() {
        // [1,1,2,2] input, kernel=2x2, stride=1x1, padding=1x1.
        let input = tensor(vec![1.0, -2.0, -3.0, 4.0], vec![1, 1, 2, 2]);
        let out = max_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2), (1, 1), (1, 1), (1, 1)).unwrap();
        // padded region: -inf-candidate skip, not 0.0 — confirms real values
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
    /// Auto-generated documentation for max_pool2d_backward_non_overlapping_routes_grad_to_winner_only.
    fn max_pool2d_backward_non_overlapping_routes_grad_to_winner_only() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = max_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        let loss = TestBackend::sum_all::<f32>(&out).unwrap();
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

    /// Overlap test (Pitfall 5 / T-04-14 — the load-bearing test for this
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
        let out = max_pool2d_impl::<f32, Cpu, f32>(&input, (1, 2), (1, 1), (0, 0), (1, 1)).unwrap();
        // H_out=1, W_out = (3-2)/1+1 = 2: window0=[1,100]->100, window1=[100,1]->100.
        assert_eq!(out.shape, vec![1, 1, 1, 2]);
        assert_eq!(f32_vec(&out), vec![100.0, 100.0]);

        let loss = TestBackend::sum_all::<f32>(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(input.id).expect("grad_input should exist");
        let vals = f32_vec(g);
        // Middle position (idx 1) wins both windows -> gradient = 1.0 + 1.0 = 2.0.
        // Both edge positions never win -> gradient = 0.0.
        assert_eq!(vals, vec![0.0, 2.0, 0.0]);
    }

    #[test]
    /// Auto-generated documentation for max_pool2d_gradcheck_overlapping.
    fn max_pool2d_gradcheck_overlapping() {
        let input = tensor(
            vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6, 0.8],
            vec![1, 1, 3, 3],
        );
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = max_pool2d_impl::<f32, Cpu, f32>(&inputs[0], (2, 2), (1, 1), (0, 0), (1, 1))
                .unwrap();
            TestBackend::sum_all::<f32>(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "max_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- avg_pool2d forward ---

    #[test]
    /// Auto-generated documentation for avg_pool2d_forward_no_overlap_hand_computed.
    fn avg_pool2d_forward_no_overlap_hand_computed() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let out = avg_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // window(0,0) mean of {1,2,5,6} = 3.5
        // window(0,1) mean of {3,4,7,8} = 5.5
        // window(1,0) mean of {9,10,13,14} = 11.5
        // window(1,1) mean of {11,12,15,16} = 13.5
        assert_eq!(f32_vec(&out), vec![3.5, 5.5, 11.5, 13.5]);
    }

    // --- avg_pool2d backward ---

    #[test]
    /// Auto-generated documentation for avg_pool2d_backward_overlapping_windows_sums_grad_contributions.
    fn avg_pool2d_backward_overlapping_windows_sums_grad_contributions() {
        // [1,1,1,3] input, kernel=1x2, stride=1x1 (overlapping): 2 output
        // windows, middle position covered by both.
        let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 1, 3]);
        let out = avg_pool2d_impl::<f32, Cpu, f32>(&input, (1, 2), (1, 1), (0, 0)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 1, 2]);
        // window0 mean{1,2}=1.5, window1 mean{2,3}=2.5
        assert_eq!(f32_vec(&out), vec![1.5, 2.5]);

        let loss = TestBackend::sum_all::<f32>(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(input.id).expect("grad_input should exist");
        let vals = f32_vec(g);
        // grad_input[0] = 1/2 (only window0) = 0.5
        // grad_input[1] = 1/2 (window0) + 1/2 (window1) = 1.0 (overlap sum)
        // grad_input[2] = 1/2 (only window1) = 0.5
        assert_eq!(vals, vec![0.5, 1.0, 0.5]);
    }

    #[test]
    /// Auto-generated documentation for avg_pool2d_gradcheck_overlapping.
    fn avg_pool2d_gradcheck_overlapping() {
        let input = tensor(
            vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6, 0.8],
            vec![1, 1, 3, 3],
        );
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = avg_pool2d_impl::<f32, Cpu, f32>(&inputs[0], (2, 2), (1, 1), (0, 0)).unwrap();
            TestBackend::sum_all::<f32>(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "avg_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- adaptive_avg_pool2d forward ---

    #[test]
    /// Auto-generated documentation for adaptive_avg_pool2d_evenly_dividing_matches_avg_pool2d.
    fn adaptive_avg_pool2d_evenly_dividing_matches_avg_pool2d() {
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let adaptive = adaptive_avg_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2)).unwrap();
        let fixed = avg_pool2d_impl::<f32, Cpu, f32>(&input, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(adaptive.shape, fixed.shape);
        assert_eq!(f32_vec(&adaptive), f32_vec(&fixed));
    }

    /// Non-evenly-dividing case (Pitfall 6): input H=5, output H=3 must
    /// produce per-output-position window sizes [2,3,2] (not a uniform
    /// fixed kernel), matching PyTorch's documented
    /// `start=floor(i*in/out), end=ceil((i+1)*in/out)` formula. (Using
    /// input=5/output=3 here rather than 7/3, since 7/3's own boundaries —
    /// `start=floor(i*7/3), end=ceil((i+1)*7/3)` — evaluate to windows
    /// [0,3),[2,5),[4,7), i.e. sizes [3,3,3] with genuine inter-window
    /// overlap, not the [3,2,2] figure RESEARCH.md's prose used as its
    /// illustrative example; 5/3 is the textbook non-uniform case and
    /// exercises the exact same variable-boundary formula.)
    #[test]
    fn adaptive_avg_pool2d_non_evenly_dividing_produces_variable_windows() {
        // H=5 -> output 3: windows [0,2), [1,4), [3,5) -> sizes [2,3,2].
        assert_eq!(adaptive_window_bounds(5, 3, 0), (0, 2));
        assert_eq!(adaptive_window_bounds(5, 3, 1), (1, 4));
        assert_eq!(adaptive_window_bounds(5, 3, 2), (3, 5));

        // Build a [1,1,5,1] input (W axis trivial, size 1) with distinct
        // values so each H-window's mean is hand-verifiable.
        let input_data: Vec<f32> = (1..=5).map(|x| x as f32).collect(); // 1..5
        let input = tensor(input_data, vec![1, 1, 5, 1]);
        let out = adaptive_avg_pool2d_impl::<f32, Cpu, f32>(&input, (3, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 1]);
        let vals = f32_vec(&out);
        // window0 = mean(1,2) = 1.5
        // window1 = mean(2,3,4) = 3.0
        // window2 = mean(4,5) = 4.5
        assert_eq!(vals, vec![1.5, 3.0, 4.5]);
    }

    // --- adaptive_avg_pool2d backward ---

    #[test]
    /// Auto-generated documentation for adaptive_avg_pool2d_gradcheck_non_evenly_dividing.
    fn adaptive_avg_pool2d_gradcheck_non_evenly_dividing() {
        let input_data: Vec<f32> = (1..=7).map(|x| x as f32 * 0.1).collect();
        let input = tensor(input_data, vec![1, 1, 7, 1]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = adaptive_avg_pool2d_impl::<f32, Cpu, f32>(&inputs[0], (3, 1)).unwrap();
            TestBackend::sum_all::<f32>(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[input], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "adaptive_avg_pool2d gradcheck max relative error too high: {max_rel_err}"
        );
    }
}
