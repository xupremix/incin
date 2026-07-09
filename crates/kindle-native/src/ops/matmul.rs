//! Naive, stride-aware, unbatched-2D-only `matmul` for `NativeBackend`.
//!
//! Per CONTEXT.md's explicit Phase 1 scope decision, batch-broadcast matmul
//! (>2D operands) is out of scope here (deferred to Phase 3 / NATBACK-07).
//! The inner loop reads through each operand's own shape/strides/offset via
//! `NativeStorage::get` directly — it never forces a `.contiguous()`
//! materialization first, so a transposed (non-contiguous) view produces
//! correct values without a hidden copy (NATBACK-02 / Pitfall 3).
//!
//! This file only contributes a plain function, `matmul_impl`, rather than
//! its own `impl TensorOps` block: Rust does not allow two separate `impl
//! TensorOps<..> for NativeBackend<..>` blocks for the same trait+type
//! across two files, so `ops/shape_ops.rs`'s single `TensorOps` impl block
//! calls into `matmul_impl` for its `matmul` method.

use kindle_core::err::Error;
use kindle_core::prelude::Result;

use crate::storage::{NativeBuffer, NativeStorage};
use crate::tape::{self, TapeEntry};

/// Swap the two axes of a 2D `NativeStorage` (thin wrapper over
/// `NativeStorage::transpose(0, 1)`, reused by the backward closure so the
/// gradient composition is built from already-tested primitives rather than
/// a bespoke derivation).
fn transpose_2d(t: &NativeStorage) -> NativeStorage {
    t.transpose(0, 1)
        .expect("2D transpose of a 2D matmul operand cannot fail")
}

/// Naive triple-nested-loop 2D matmul: `lhs` (`[M,K]`) @ `rhs` (`[K,N]`) ->
/// `[M,N]`. Reads through each operand's own strides/offset (via
/// `NativeStorage::get`), so a transposed (non-contiguous) operand is
/// handled correctly without an implicit contiguous materialization.
///
/// Pushes a `TapeEntry` whose backward closure computes
/// `grad_lhs = grad_out @ rhs^T` and `grad_rhs = lhs^T @ grad_out`, composed
/// by recursing into `matmul_impl` itself plus `transpose_2d` — not a
/// bespoke hand-derived kernel.
pub(crate) fn matmul_impl(lhs: &NativeStorage, rhs: &NativeStorage) -> Result<NativeStorage> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![lhs.shape[0], rhs.shape.first().copied().unwrap_or(0)],
            got: rhs.shape.clone(),
            msg: format!(
                "matmul requires unbatched 2D operands with lhs.shape[1] == rhs.shape[0]; got lhs={:?}, rhs={:?}",
                lhs.shape, rhs.shape
            ),
        });
    }

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];

    let mut out = Vec::with_capacity(m * n);
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += lhs.get(&[mi, ki]) * rhs.get(&[ki, ni]);
            }
            out.push(acc as f32);
        }
    }

    let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), vec![m, n]);

    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &NativeStorage| {
            let grad_lhs = matmul_impl(grad_out, &transpose_2d(&rhs_capture))
                .expect("grad_lhs = grad_out @ rhs^T cannot fail (shapes proven compatible)");
            let grad_rhs = matmul_impl(&transpose_2d(&lhs_capture), grad_out)
                .expect("grad_rhs = lhs^T @ grad_out cannot fail (shapes proven compatible)");
            vec![grad_lhs, grad_rhs]
        }),
    });

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    fn matmul_forward_hand_computed_2x3_times_3x4() {
        // lhs = [[1,2,3],[4,5,6]] (2x3)
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        // rhs = [[7,8,9,10],[11,12,13,14],[15,16,17,18]] (3x4)
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
        //      = [7+22+45, 8+24+48, 9+26+51, 10+28+54] = [74, 80, 86, 92]
        // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
        //      = [28+55+90, 32+60+96, 36+65+102, 40+70+108] = [173, 188, 203, 218]
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn matmul_forward_transposed_lhs_view_is_correct_without_materializing() {
        // Original storage is [3,2] = [[1,4],[2,5],[3,6]]; transpose(0,1)
        // gives a non-contiguous [2,3] view = [[1,2,3],[4,5,6]] (same
        // logical values as the previous test's `lhs`), read directly
        // through strides (no .contiguous() call in matmul_impl itself).
        let original = matrix(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        let lhs = original.transpose(0, 1).unwrap(); // [2,3], non-contiguous
        assert!(!crate::stride::is_contiguous(&lhs.shape, &lhs.strides));

        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn matmul_backward_matches_hand_computed_gradients() {
        // lhs [2,3], rhs [3,4] as above; grad_out is a synthetic [2,4] all-ones-ish
        // matrix with distinct values so the composition is unambiguous.
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
        let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

        // grad_out = ones_like(out) = [2,4] all ones.
        // grad_lhs = grad_out @ rhs^T : [2,4] @ [4,3] -> [2,3]
        // rhs^T rows are rhs's columns: col0=[7,11,15], col1=[8,12,16],
        // col2=[9,13,17], col3=[10,14,18]. Each output row of grad_lhs is the
        // sum of rhs^T's rows (since grad_out row is all ones):
        // sum over rhs^T's 4 rows (its columns as rows) = per-column sums of rhs:
        // col0: 7+11+15=33, col1: 8+12+16=36, col2: 9+13+17=39, col3: 10+14+18=42
        // Wait: rhs^T is [4,3] (rows = rhs's 4 columns transposed to rows length 3).
        // rhs^T row i = rhs's column i as a length-3 vector: [rhs[0][i], rhs[1][i], rhs[2][i]]
        // grad_lhs[m][k] = sum_n grad_out[m][n] * rhs^T[n][k] = sum_n rhs[k][n] (since grad_out=1)
        //               = sum over n of rhs[k][n] = row-sum of rhs's row k.
        // rhs row 0 = [7,8,9,10] sum=34; row1=[11,12,13,14] sum=50; row2=[15,16,17,18] sum=66
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![34.0, 50.0, 66.0, 34.0, 50.0, 66.0]);

        // grad_rhs = lhs^T @ grad_out : [3,2] @ [2,4] -> [3,4]
        // grad_rhs[k][n] = sum_m lhs^T[k][m] * grad_out[m][n] = sum_m lhs[m][k] (since grad_out=1)
        //               = column-sum of lhs's column k.
        // lhs col0 = [1,4] sum=5; col1=[2,5] sum=7; col2=[3,6] sum=9
        assert_eq!(rhs_grad.shape, vec![3, 4]);
        assert_eq!(
            f32_vec(rhs_grad),
            vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]
        );
    }

    #[test]
    fn matmul_shape_incompatible_returns_err_not_panic() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(vec![0.0; 20], 4, 5);
        let result = matmul_impl(&lhs, &rhs);
        assert!(result.is_err());
    }
}
