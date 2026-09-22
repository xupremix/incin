//! Grouped (expert-tiled) matmul for the CPU backend.
//!
//! `lhs [T, K]`, stacked `rhs [E, K, N]`, and an i64 `offsets [E+1]` that
//! tiles `[0, T)` into per-expert row spans. Each expert's block of rows is
//! multiplied by its own `[K, N]` slice; an empty span contributes nothing and
//! is skipped rather than erroring, which is the "empty expert" case the
//! issue names.
//!
//! Forward is tape-free per expert: `matmul_forward` and the raw
//! `CpuStorage::narrow`/`reshape` methods run without recording, and a single
//! `TapeEntry` at the end composes the backward from `transpose_2d` and
//! `matmul_forward` over the same slices, so the tape sees one entry for the
//! whole grouped product rather than one per expert.
//!
//! `input_ids` carries only `lhs` and `rhs`: `offsets` is an integer tile
//! with no cotangent, the same exclusion `scatter_add` applies to its index
//! operand.

use super::*;

/// Validate that `offsets` is a non-decreasing i64 tile of `[0, T)` with
/// exactly `E + 1` entries, then return the per-expert `(start, end)` spans.
fn expert_spans(
    offsets: &CpuStorage,
    experts: usize,
    tokens: usize,
) -> Result<Vec<(usize, usize)>> {
    if offsets.shape != [experts + 1] {
        return Err(Error::ShapeMismatch {
            op: "grouped_matmul",
            expected: vec![experts + 1],
            got: offsets.shape.to_vec(),
            msg: format!(
                "grouped_matmul offsets must have length E+1 = {}, got shape {:?}",
                experts + 1,
                offsets.shape
            ),
        });
    }
    let mut values = Vec::with_capacity(experts + 1);
    let mut previous = 0i64;
    for expert in 0..=experts {
        let value = offsets.get_i64_checked(&[expert], "grouped_matmul")?;
        if value < 0 || value as usize > tokens {
            return Err(Error::Msg(format!(
                "grouped_matmul offsets[{expert}] = {value} is outside [0, {tokens}]"
            )));
        }
        if value < previous {
            return Err(Error::Msg(format!(
                "grouped_matmul offsets must be non-decreasing; offsets[{expert}] = {value} < {previous}"
            )));
        }
        previous = value;
        values.push(value as usize);
    }
    if values.first() != Some(&0) || values.last() != Some(&tokens) {
        return Err(Error::Msg(format!(
            "grouped_matmul offsets must tile [0, {tokens}); got {values:?}"
        )));
    }
    Ok(values.windows(2).map(|w| (w[0], w[1])).collect())
}

/// Tape-free expert slice of `rhs`: `[E, K, N]` at `expert` -> `[K, N]`.
///
/// Uses the raw storage methods so no tape entry is pushed per expert; the
/// caller composes one over the whole tile instead.
fn expert_weight(rhs: &CpuStorage, expert: usize) -> Result<CpuStorage> {
    let slice = rhs.narrow(0, expert, 1)?;
    slice.reshape(&[rhs.shape[1], rhs.shape[2]])
}

/// Tape-free row slice of `lhs`: `[T, K]` rows `start..end` -> `[rows, K]`.
fn expert_rows(lhs: &CpuStorage, start: usize, end: usize) -> Result<CpuStorage> {
    if start == end {
        // A zero-length narrow is a valid view; reshape keeps rank two so
        // `matmul_forward` still sees `[0, K] @ [K, N]`.
        return lhs.narrow(0, start, 0);
    }
    lhs.narrow(0, start, end - start)
}

/// Grouped matmul forward + one composed tape entry.
pub(crate) fn grouped_matmul_impl(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    offsets: &CpuStorage,
) -> Result<CpuStorage> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 3 || offsets.shape.len() != 1 {
        return Err(Error::ShapeMismatch {
            op: "grouped_matmul",
            expected: vec![2, 3, 1],
            got: vec![lhs.shape.len(), rhs.shape.len(), offsets.shape.len()],
            msg: format!(
                "grouped_matmul requires lhs [T,K], rhs [E,K,N], offsets [E+1]; got lhs={:?}, rhs={:?}, offsets={:?}",
                lhs.shape, rhs.shape, offsets.shape
            ),
        });
    }
    let (tokens, contracting) = (lhs.shape[0], lhs.shape[1]);
    let (experts, _, out_cols) = (rhs.shape[0], rhs.shape[1], rhs.shape[2]);
    if contracting != rhs.shape[1] {
        return Err(Error::ShapeMismatch {
            op: "grouped_matmul",
            expected: vec![contracting],
            got: vec![rhs.shape[1]],
            msg: format!(
                "grouped_matmul contracting dimensions differ: lhs K = {contracting}, rhs K = {}",
                rhs.shape[1]
            ),
        });
    }
    let spans = expert_spans(offsets, experts, tokens)?;
    let total = tokens * out_cols;

    let out = if writes_f32(lhs, rhs) {
        let mut out_data = vec![0f32; total];
        for (expert, &(start, end)) in spans.iter().enumerate() {
            if start == end {
                continue;
            }
            let rows = expert_rows(lhs, start, end)?;
            let weight = expert_weight(rhs, expert)?;
            let product = matmul_forward(&rows, &weight)?;
            for row in 0..(end - start) {
                for col in 0..out_cols {
                    out_data[(start + row) * out_cols + col] = product.get(&[row, col]) as f32;
                }
            }
        }
        CpuStorage::try_from_contiguous(CpuBuffer::F32(out_data), vec![tokens, out_cols])?
    } else {
        let mut out_data = vec![0f64; total];
        for (expert, &(start, end)) in spans.iter().enumerate() {
            if start == end {
                continue;
            }
            let rows = expert_rows(lhs, start, end)?;
            let weight = expert_weight(rhs, expert)?;
            let product = matmul_forward(&rows, &weight)?;
            for row in 0..(end - start) {
                for col in 0..out_cols {
                    out_data[(start + row) * out_cols + col] = product.get(&[row, col]);
                }
            }
        }
        CpuStorage::try_from_contiguous(
            lhs.buffer.from_f64_values(out_data)?,
            vec![tokens, out_cols],
        )?
    };

    record(lhs, rhs, out, &spans)
}

/// Push the single composed tape entry covering every expert's slice.
///
/// `input_ids` is `[lhs_id, rhs_id]` only: `offsets` is an integer tile with
/// no cotangent, so the backward returns exactly two gradients.
fn record(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out: CpuStorage,
    spans: &[(usize, usize)],
) -> Result<CpuStorage> {
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let spans = spans.to_vec();
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    let contracting = lhs.shape[1];
    let out_cols = rhs.shape[2];
    let use_f32 = writes_f32(lhs, rhs);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let tokens = lhs_capture.shape[0];
            let experts = rhs_capture.shape[0];
            if use_f32 {
                let mut grad_lhs = vec![0f32; tokens * contracting];
                let mut grad_rhs = vec![0f32; experts * contracting * out_cols];
                for (expert, &(start, end)) in spans.iter().enumerate() {
                    if start == end {
                        continue;
                    }
                    // grad_lhs block = g_rows @ weight^T
                    let g_rows = grad_out.narrow(0, start, end - start)?;
                    let weight = expert_weight(&rhs_capture, expert)?;
                    let grad_block = matmul_forward(&g_rows, &transpose_2d(&weight))?;
                    for row in 0..(end - start) {
                        for col in 0..contracting {
                            grad_lhs[(start + row) * contracting + col] =
                                grad_block.get(&[row, col]) as f32;
                        }
                    }
                    // grad_rhs block = lhs_rows^T @ g_rows
                    let lhs_rows = expert_rows(&lhs_capture, start, end)?;
                    let grad_weight = matmul_forward(&transpose_2d(&lhs_rows), &g_rows)?;
                    for k in 0..contracting {
                        for n in 0..out_cols {
                            grad_rhs[(expert * contracting + k) * out_cols + n] =
                                grad_weight.get(&[k, n]) as f32;
                        }
                    }
                }
                let grad_lhs = CpuStorage::try_from_contiguous(
                    CpuBuffer::F32(grad_lhs),
                    vec![tokens, contracting],
                )?;
                let grad_rhs = CpuStorage::try_from_contiguous(
                    CpuBuffer::F32(grad_rhs),
                    rhs_capture.shape.clone(),
                )?;
                Ok(vec![grad_lhs, grad_rhs])
            } else {
                let mut grad_lhs = vec![0f64; tokens * contracting];
                let mut grad_rhs = vec![0f64; experts * contracting * out_cols];
                for (expert, &(start, end)) in spans.iter().enumerate() {
                    if start == end {
                        continue;
                    }
                    let g_rows = grad_out.narrow(0, start, end - start)?;
                    let weight = expert_weight(&rhs_capture, expert)?;
                    let grad_block = matmul_forward(&g_rows, &transpose_2d(&weight))?;
                    for row in 0..(end - start) {
                        for col in 0..contracting {
                            grad_lhs[(start + row) * contracting + col] = grad_block.get(&[row, col]);
                        }
                    }
                    let lhs_rows = expert_rows(&lhs_capture, start, end)?;
                    let grad_weight = matmul_forward(&transpose_2d(&lhs_rows), &g_rows)?;
                    for k in 0..contracting {
                        for n in 0..out_cols {
                            grad_rhs[(expert * contracting + k) * out_cols + n] =
                                grad_weight.get(&[k, n]);
                        }
                    }
                }
                let grad_lhs = CpuStorage::try_from_contiguous(
                    lhs_capture.buffer.from_f64_values(grad_lhs)?,
                    vec![tokens, contracting],
                )?;
                let grad_rhs = CpuStorage::try_from_contiguous(
                    rhs_capture.buffer.from_f64_values(grad_rhs)?,
                    rhs_capture.shape.clone(),
                )?;
                Ok(vec![grad_lhs, grad_rhs])
            }
        }),
    });
    Ok(out)
}
