//! `max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d` for `NativeBackend<T, D>` —
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
//! zero.

use kindle_core::prelude::{DType, Result};

use crate::storage::{NativeBuffer, NativeStorage, increment_index};
use crate::tape::{self, TapeEntry};

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
    input: &NativeStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> (NativeStorage, Vec<usize>) {
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
    let input_strides = crate::stride::contiguous_strides(&input.shape);

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

    let out = NativeStorage::from_contiguous(
        NativeBuffer::F32(best_val.iter().map(|&v| v as f32).collect()),
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
    grad_out: &NativeStorage,
    winning_flat_src_idx: &[usize],
    input_shape: &[usize],
) -> NativeStorage {
    let total: usize = input_shape.iter().product();
    let mut vals = vec![0.0f32; total];
    let out_total: usize = grad_out.shape.iter().product();
    let mut out_idx = vec![0usize; grad_out.shape.len()];
    for flat_out in 0..out_total {
        let g = grad_out.get(&out_idx);
        vals[winning_flat_src_idx[flat_out]] += g as f32;
        increment_index(&mut out_idx, &grad_out.shape);
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(vals), input_shape.to_vec())
}

/// `ModuleOps::max_pool2d`'s `NativeBackend` implementation.
pub(crate) fn max_pool2d_impl<T: DType, D: kindle_core::prelude::Device, K: DType>(
    t: &NativeStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<NativeStorage> {
    let (out, winning_flat_src_idx) = max_window_2d(t, kernel_size, stride, padding, dilation);

    let input_shape = t.shape.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &NativeStorage| {
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
// Shared output-size arithmetic (mirrors ops::conv's out_size)
// ---------------------------------------------------------------------------

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
mod tests {
    use super::*;
    use crate::NativeBackend;
    use crate::testutil::gradcheck;
    use kindle_core::prelude::{Cpu, ReductionOps};

    type TestBackend = NativeBackend<f32, Cpu>;

    fn tensor(v: Vec<f32>, shape: Vec<usize>) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), shape)
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- max_pool2d forward ---

    #[test]
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
    fn max_pool2d_gradcheck_overlapping() {
        let input = tensor(
            vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6, 0.8],
            vec![1, 1, 3, 3],
        );
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
}
