//! Free-function `embedding` helper for `NativeBackend<T, D>`.
//!
//! `embedding_impl` is called by `ops/module.rs`'s `ModuleOps::embedding`
//! trait method, mirroring `ops/norm.rs`'s free-function-called-from-
//! module.rs pattern.
//!
//! Forward is a genuine materializing per-row gather (like im2col's forward)
//! — not a `NativeStorage` view op, since gathered rows are not contiguous/
//! stridable from arbitrary integer indices. Backward is a hand-composed
//! scatter-add: repeated indices in the input tensor must ACCUMULATE (sum)
//! their gradient contributions into the corresponding weight row, never
//! overwrite — the embedding-specific analogue of NATBACK-05's tape-level
//! accumulate-not-overwrite requirement, done inside ONE backward closure
//! rather than via multiple `TapeEntry` pushes.

use kindle_core::prelude::Error;
use kindle_core::prelude::{DType, Result};

use crate::storage::{NativeBuffer, NativeStorage};
use crate::tape::{self, TapeEntry};

/// Gather rows of `w` (shape `[vocab_size, hidden_size]`) addressed by the
/// integer indices in `t` (any rank), producing an output of shape
/// `t.shape ++ [hidden_size]`.
///
/// Backward pushes exactly ONE `TapeEntry` whose `input_ids` is `vec![w.id]`
/// only — the integer indices tensor `t` is not a differentiable input (it
/// has no gradient, mirroring `cross_entropy_loss`'s treatment of its
/// integer `target` argument). The backward closure scatter-adds each
/// gathered position's incoming gradient slice into the corresponding row
/// of a zero-filled buffer sized to `w`, summing contributions when the same
/// row index appears more than once in `t`.
pub(crate) fn embedding_impl<T: DType, D: kindle_core::prelude::Device, K: DType, KInt: DType>(
    t: &NativeStorage,
    w: &NativeStorage,
) -> Result<NativeStorage> {
    if w.shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            op: "embedding",
            expected: vec![0, 0],
            got: w.shape.clone(),
            msg: format!(
                "embedding: weight table must be rank-2 [vocab_size, hidden_size], got shape {:?}",
                w.shape
            ),
        });
    }
    let vocab_size = w.shape[0];
    let hidden_size = w.shape[1];

    #[cfg(feature = "cuda")]
    if matches!(&*w.buffer, NativeBuffer::Cuda(_)) && matches!(&*t.buffer, NativeBuffer::Cuda(_)) {
        let out = crate::ops::cuda_embedding::launch_embedding_fwd(t, w)?;
        let total_indices: usize = t.shape.iter().product();

        let w_id = w.id;
        let out_id = out.id;
        let w_shape = w.shape.clone();

        if let NativeBuffer::Cuda(idx_b) = &*t.buffer {
            let indices_dev_clone = idx_b.data.clone();
            let device_clone = idx_b.device.clone();
            let device_id = idx_b.device_id;

            tape::push(TapeEntry {
                output_id: out_id,
                input_ids: vec![w_id],
                backward: Box::new(move |grad_out: &NativeStorage| {
                    let gw = crate::ops::cuda_embedding::launch_embedding_bwd(
                        grad_out,
                        &indices_dev_clone,
                        &device_clone,
                        device_id,
                        vocab_size,
                        hidden_size,
                        total_indices,
                    ).unwrap();
                    vec![gw]
                }),
            });
        }
        return Ok(out);
    }

    let total_indices: usize = t.shape.iter().product();
    let mut idx = vec![0usize; t.shape.len()];

    // Resolved row index (into `w`) for each of the `total_indices` gathered
    // positions, in forward-iteration order. Captured by the backward
    // closure so it never needs to re-read/re-cast `t`.
    let mut row_indices: Vec<usize> = Vec::with_capacity(total_indices);
    let mut out_vals: Vec<f32> = Vec::with_capacity(total_indices * hidden_size);

    for _ in 0..total_indices {
        let raw = t.get(&idx);
        let row_idx = raw as i64 as usize;
        if raw < 0.0 || row_idx >= vocab_size {
            return Err(Error::ShapeMismatch {
                op: "embedding",
                expected: vec![vocab_size],
                got: vec![row_idx],
                msg: format!("embedding: index {raw} out of range for vocab_size {vocab_size}"),
            });
        }
        for h in 0..hidden_size {
            out_vals.push(w.get(&[row_idx, h]) as f32);
        }
        row_indices.push(row_idx);
        if !t.shape.is_empty() {
            crate::ops::elementwise::increment_index(&mut idx, &t.shape);
        }
    }

    let mut out_shape = t.shape.clone();
    out_shape.push(hidden_size);
    let out = NativeStorage::from_contiguous(NativeBuffer::F32(out_vals), out_shape);

    let w_total: usize = w.shape.iter().product();
    let t_shape = t.shape.clone();
    let (w_id, out_id) = (w.id, out.id);
    let w_shape_for_backward = w.shape.clone();
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![w_id],
        backward: Box::new(move |grad_out: &NativeStorage| {
            let mut grad_w = vec![0.0f32; w_total];
            // Walk the LEADING axes (matching `t_shape`) with the shared
            // odometer helper, appending the trailing hidden-size axis
            // explicitly for each leading position — this mirrors how `out`
            // was populated during forward (row-major over `t`'s positions,
            // contiguous `hidden_size` run per position).
            let mut leading_idx = vec![0usize; t_shape.len()];
            for &row_idx in &row_indices {
                let mut full_idx = leading_idx.clone();
                full_idx.push(0);
                for h in 0..hidden_size {
                    *full_idx.last_mut().unwrap() = h;
                    let g = grad_out.get(&full_idx);
                    grad_w[row_idx * hidden_size + h] += g as f32;
                }
                if !t_shape.is_empty() {
                    crate::ops::elementwise::increment_index(&mut leading_idx, &t_shape);
                }
            }
            vec![NativeStorage::from_contiguous(
                NativeBuffer::F32(grad_w),
                w_shape_for_backward.clone(),
            )]
        }),
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use crate::NativeBackend;
    use crate::tape;
    use kindle_core::prelude::{Cpu, ReductionOps};

    /// Auto-generated documentation for B.
    type B = NativeBackend<f32, Cpu>;

    /// Auto-generated documentation for weight.
    fn weight(v: Vec<f32>, vocab: usize, hidden: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![vocab, hidden])
    }

    /// Auto-generated documentation for indices_i64.
    fn indices_i64(v: Vec<i64>, shape: Vec<usize>) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::I64(v), shape)
    }

    /// Auto-generated documentation for f32_vec.
    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    /// Auto-generated documentation for forward_1d_indices_gathers_correct_rows_with_repeats.
    fn forward_1d_indices_gathers_correct_rows_with_repeats() {
        // weight [3,2]: row0=[1,2], row1=[3,4], row2=[5,6]
        let w = weight(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let idx = indices_i64(vec![0, 2, 0], vec![3]);
        let out = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        let vals = f32_vec(&out);
        // row 0 -> [1,2], row 2 -> [5,6], row 0 again -> [1,2]
        assert_eq!(vals, vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    /// Auto-generated documentation for forward_2d_indices_gathers_correct_rows.
    fn forward_2d_indices_gathers_correct_rows() {
        // weight [4,3]
        let w = weight(
            vec![
                0.0, 1.0, 2.0, // row0
                10.0, 11.0, 12.0, // row1
                20.0, 21.0, 22.0, // row2
                30.0, 31.0, 32.0, // row3
            ],
            4,
            3,
        );
        let idx = indices_i64(vec![1, 3, 0, 2], vec![2, 2]);
        let out = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w).unwrap();
        assert_eq!(out.shape, vec![2, 2, 3]);
        let vals = f32_vec(&out);
        assert_eq!(
            vals,
            vec![
                10.0, 11.0, 12.0, // idx=1
                30.0, 31.0, 32.0, // idx=3
                0.0, 1.0, 2.0, // idx=0
                20.0, 21.0, 22.0, // idx=2
            ]
        );
    }

    #[test]
    /// Auto-generated documentation for backward_repeated_index_accumulates_not_overwrites.
    fn backward_repeated_index_accumulates_not_overwrites() {
        // weight [2,2]: row0, row1
        let w = weight(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let idx = indices_i64(vec![0, 1, 0], vec![3]);
        let out = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w).unwrap();
        // Manually seed grad_out as ones (mirrors ones_like seeding via a
        // sum_all-style consumer) by summing the output through the real
        // ReductionOps::sum_all so tape::backward seeds correctly.
        let loss = B::sum_all::<f32>(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(w.id).expect("weight should have gradient");
        assert_eq!(g.shape, vec![2, 2]);
        let vals = f32_vec(g);
        // row 0 is addressed twice (positions 0 and 2) -> gradient = 1+1 = 2
        // for each of its 2 columns. row 1 addressed once -> gradient = 1.
        assert_eq!(vals, vec![2.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    /// Auto-generated documentation for backward_unaddressed_rows_get_exactly_zero_gradient.
    fn backward_unaddressed_rows_get_exactly_zero_gradient() {
        // weight [3,2]; only row 1 is addressed.
        let w = weight(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let idx = indices_i64(vec![1], vec![1]);
        let out = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w).unwrap();
        let loss = B::sum_all::<f32>(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(w.id).expect("weight should have gradient");
        let vals = f32_vec(g);
        // row 0 (unaddressed) = [0,0]; row 1 (addressed once) = [1,1]; row 2
        // (unaddressed) = [0,0]
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    /// Auto-generated documentation for out_of_range_index_returns_shape_mismatch_error.
    fn out_of_range_index_returns_shape_mismatch_error() {
        let w = weight(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let idx = indices_i64(vec![5], vec![1]);
        let result = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    /// Auto-generated documentation for negative_index_returns_shape_mismatch_error_not_panic.
    fn negative_index_returns_shape_mismatch_error_not_panic() {
        let w = weight(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let idx = indices_i64(vec![-1], vec![1]);
        let result = embedding_impl::<f32, Cpu, f32, i64>(&idx, &w);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }
}
