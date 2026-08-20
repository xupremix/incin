use super::*;

// Canonical executors and dynamic dispatch share these helpers so each
// operation has one CPU implementation.

/// Reshape a view and record the inverse for backward.
pub(crate) fn reshape_storage(t: &CpuStorage, shape: &[usize]) -> Result<CpuStorage> {
    let out = t.reshape(shape)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![grad_out.reshape(&original_shape)?])
        }),
    });
    Ok(out)
}

pub(crate) fn transpose_storage(t: &CpuStorage, dim1: usize, dim2: usize) -> Result<CpuStorage> {
    let out = t.transpose(dim1, dim2)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| Ok(vec![grad_out.transpose(dim1, dim2)?])),
    });
    Ok(out)
}

pub(crate) fn narrow_storage(
    t: &CpuStorage,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<CpuStorage> {
    let out = t.narrow(dim, start, len)?;
    let original_shape = t.shape.to_vec();
    let mut region_start = vec![0usize; original_shape.len()];
    region_start[dim] = start;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![crate::cpu::storage::scatter_into_zeros(
                &original_shape,
                &region_start,
                grad_out,
            )?])
        }),
    });
    Ok(out)
}

pub(crate) fn slice_storage(t: &CpuStorage, ranges: &[(usize, usize)]) -> Result<CpuStorage> {
    let mut out = t.clone();
    for (dim, &(start, end)) in ranges.iter().enumerate() {
        out = narrow_storage(&out, dim, start, end - start)?;
    }
    Ok(out)
}

pub(crate) fn squeeze_storage(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() || t.shape[dim] != 1 {
        return Err(Error::ShapeMismatch {
            op: "squeeze",
            expected: vec![1],
            got: t.shape.to_vec(),
            msg: format!(
                "squeeze requires axis {dim} to have size 1, got size {} in shape {:?}",
                t.shape.get(dim).copied().unwrap_or(0),
                t.shape
            ),
        });
    }
    let mut target_shape = t.shape.to_vec();
    target_shape.remove(dim);
    reshape_storage(t, &target_shape)
}

pub(crate) fn unsqueeze_storage(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    let mut target_shape = t.shape.to_vec();
    if dim <= target_shape.len() {
        target_shape.insert(dim, 1);
    } else {
        target_shape.push(1);
    }
    reshape_storage(t, &target_shape)
}

pub(crate) fn flatten_storage(
    t: &CpuStorage,
    start_dim: usize,
    end_dim: usize,
) -> Result<CpuStorage> {
    if start_dim > end_dim || end_dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "flatten",
            expected: t.shape.to_vec(),
            got: vec![start_dim, end_dim],
            msg: format!(
                "flatten(start_dim={start_dim}, end_dim={end_dim}) out of bounds for shape {:?}",
                t.shape
            ),
        });
    }
    let merged = crate::cpu::stride::checked_numel(&t.shape[start_dim..=end_dim])?;
    let mut target_shape = t.shape[..start_dim].to_vec();
    target_shape.push(merged);
    target_shape.extend_from_slice(&t.shape[end_dim + 1..]);
    reshape_storage(t, &target_shape)
}

/// Broadcast a view and record the reducing inverse for backward.
pub(crate) fn broadcast_as_storage(t: &CpuStorage, shape: &[usize]) -> Result<CpuStorage> {
    let out = t.broadcast_as(shape)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![tape::unbroadcast(grad_out, &original_shape)?])
        }),
    });
    Ok(out)
}

pub(crate) fn broadcast_left_storage(t: &CpuStorage, shape: &[usize]) -> Result<CpuStorage> {
    let mut target_shape = shape.to_vec();
    target_shape.extend_from_slice(&t.shape);
    broadcast_as_storage(t, &target_shape)
}
