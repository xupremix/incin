use super::*;

/// Batched matmul: broadcasts both operands' batch dims (every axis except
/// the trailing 2) and runs one GEMM per output batch slice.
///
/// Handles the unbatched case too as the degenerate `batch_total == 1` case,
/// with no separate code path: an operand with no batch axes normalizes to an
/// empty stride list whose only `physical_index` is its own offset.
pub(crate) fn batched_matmul_impl(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out = batched_gemm(lhs, rhs)?;

    // Backward is hand-composed from transpose_last2 + recursive forward
    // calls + tape::unbroadcast per operand (Pattern 2), not a bespoke
    // gradient derivation. The operands are captured as they were given,
    // NOT batch-expanded: `batched_gemm` broadcasts batch axes itself, so
    // the expanded copies the previous implementation had to capture never
    // have to exist. `unbroadcast`'s target is still each operand's OWN full
    // original shape, trailing [M,K]/[K,N] dims included (Pitfall 2).
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // grad_lhs = grad_out @ rhs^T, summed back down to lhs's own shape.
            let grad_lhs = batched_gemm(grad_out, &transpose_last2(&rhs_capture))
                .and_then(|grad| tape::unbroadcast(&grad, &lhs_shape))?;
            // grad_rhs = lhs^T @ grad_out, summed back down to rhs's own shape.
            let grad_rhs = batched_gemm(&transpose_last2(&lhs_capture), grad_out)
                .and_then(|grad| tape::unbroadcast(&grad, &rhs_shape))?;
            Ok(vec![grad_lhs, grad_rhs])
        }),
    });

    Ok(out)
}

/// The batched forward computation on its own, with no tape recording.
///
/// Keeping this separate from `batched_matmul_impl` is what stops a batch of
/// `B` slices from leaving `B + 1` entries on the tape, and stops the backward
/// closure from pushing entries during a walk that has already drained it.
fn batched_gemm(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let (lhs_rank, rhs_rank) = (lhs.shape.len(), rhs.shape.len());
    if lhs_rank < 2 || rhs_rank < 2 {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![2],
            got: vec![lhs_rank, rhs_rank],
            msg: format!(
                "batched matmul requires both operands to have rank >= 2; got lhs.shape={:?}, rhs.shape={:?}",
                lhs.shape, rhs.shape
            ),
        });
    }

    let (m, lhs_k) = (lhs.shape[lhs_rank - 2], lhs.shape[lhs_rank - 1]);
    let (rhs_k, n) = (rhs.shape[rhs_rank - 2], rhs.shape[rhs_rank - 1]);
    if lhs_k != rhs_k {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![lhs_k],
            got: vec![rhs_k],
            msg: format!(
                "matmul inner dims must match: lhs.shape={:?} (K={lhs_k}), rhs.shape={:?} (K={rhs_k})",
                lhs.shape, rhs.shape
            ),
        });
    }

    // Batch dims = every axis except the trailing 2, right-aligned per
    // stride::broadcast_shape's existing NumPy-style rule (REUSED, not
    // reimplemented).
    let lhs_batch = &lhs.shape[..lhs_rank - 2];
    let rhs_batch = &rhs.shape[..rhs_rank - 2];
    let out_batch = stride::broadcast_shape(lhs_batch, rhs_batch)?;

    // The plan carries the broadcast rule for the batch axes only. Its numel
    // is the batch count: an empty batch shape gives the empty product 1, the
    // unbatched case, while a genuine size-0 batch axis gives 0 and is NOT
    // conflated with it (Pitfall 6).
    let batch = IterationPlan::binary(
        OperandLayout {
            shape: lhs_batch,
            strides: &lhs.strides[..lhs_rank - 2],
            offset: lhs.offset_elements,
        },
        OperandLayout {
            shape: rhs_batch,
            strides: &rhs.strides[..rhs_rank - 2],
            offset: rhs.offset_elements,
        },
        &out_batch,
    )?;

    let tile = m
        .checked_mul(n)
        .ok_or_else(|| Error::Msg(format!("matmul output overflow for [{m}, {n}]")))?;
    let total = batch.numel.checked_mul(tile).ok_or_else(|| {
        Error::Msg(format!(
            "matmul output overflow for {out_batch:?} of [{m}, {n}]"
        ))
    })?;

    let lhs_matrix = MatrixView::trailing(lhs);
    let rhs_matrix = MatrixView::trailing(rhs);
    let mut out_shape = out_batch;
    out_shape.extend_from_slice(&[m, n]);

    // A widened operand keeps its dtype, by the same split as the unbatched
    // path above: the `f32` kernels write `f32`, and anything else stays in
    // `f64` until it is converted through the operand's own buffer.
    if !writes_f32(lhs, rhs) {
        let mut out_data = vec![0f64; total];
        if tile != 0 {
            for (index, out_tile) in out_data.chunks_mut(tile).enumerate() {
                let lhs_base = batch.operands[0].physical_index(index, &batch.output_shape);
                let rhs_base = batch.operands[1].physical_index(index, &batch.output_shape);
                gemm_f64(
                    m,
                    lhs_k,
                    n,
                    lhs_matrix.at(lhs_base),
                    rhs_matrix.at(rhs_base),
                    out_tile,
                );
            }
        }
        return CpuStorage::try_from_contiguous(lhs.buffer.from_f64_values(out_data)?, out_shape);
    }

    let mut out_data = vec![0f32; total];
    // `chunks_mut` requires a nonzero chunk size, and a zero-sized tile has
    // nothing to compute anyway: the output shape already carries the zero.
    if tile != 0 {
        for (index, out_tile) in out_data.chunks_mut(tile).enumerate() {
            let lhs_base = batch.operands[0].physical_index(index, &batch.output_shape);
            let rhs_base = batch.operands[1].physical_index(index, &batch.output_shape);
            gemm(
                m,
                lhs_k,
                n,
                lhs_matrix.at(lhs_base),
                rhs_matrix.at(rhs_base),
                out_tile,
            );
        }
    }

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out_data), out_shape)
}
