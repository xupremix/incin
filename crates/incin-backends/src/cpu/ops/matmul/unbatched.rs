use super::*;

/// Unbatched 2D matmul: `lhs` (`[M,K]`) @ `rhs` (`[K,N]`) -> `[M,N]`, reading
/// through each operand's own strides and offset so a transposed
/// (non-contiguous) operand is handled without an implicit contiguous
/// materialization.
///
/// Pushes a `TapeEntry` whose backward closure computes
/// `grad_lhs = grad_out @ rhs^T` and `grad_rhs = lhs^T @ grad_out`, composed
/// from the forward kernel itself plus `transpose_2d` — not a bespoke
/// hand-derived kernel.
pub(crate) fn matmul_impl(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out = matmul_forward(lhs, rhs)?;

    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad_lhs = matmul_forward(grad_out, &transpose_2d(&rhs_capture))?;
            let grad_rhs = matmul_forward(&transpose_2d(&lhs_capture), grad_out)?;
            Ok(vec![grad_lhs, grad_rhs])
        }),
    });

    Ok(out)
}

/// The unbatched forward computation on its own, with no tape recording.
fn matmul_forward(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![lhs.shape[0], rhs.shape.first().copied().unwrap_or(0)],
            got: rhs.shape.to_vec(),
            msg: format!(
                "matmul requires unbatched 2D operands with lhs.shape[1] == rhs.shape[0]; got lhs={:?}, rhs={:?}",
                lhs.shape, rhs.shape
            ),
        });
    }

    let (m, k, n) = (lhs.shape[0], lhs.shape[1], rhs.shape[1]);
    let out_total = ShapeBuf::from_slice(&[m, n]).checked_numel(OperationKind::MatMul)?;
    if !writes_f32(lhs, rhs) {
        let mut out_data = vec![0f64; out_total];
        gemm_f64(
            m,
            k,
            n,
            MatrixView::trailing(lhs),
            MatrixView::trailing(rhs),
            &mut out_data,
        );
        return CpuStorage::try_from_contiguous(lhs.buffer.from_f64_values(out_data)?, vec![m, n]);
    }
    let mut out_data = vec![0f32; out_total];
    gemm(
        m,
        k,
        n,
        MatrixView::trailing(lhs),
        MatrixView::trailing(rhs),
        &mut out_data,
    );
    CpuStorage::try_from_contiguous(CpuBuffer::F32(out_data), vec![m, n])
}
