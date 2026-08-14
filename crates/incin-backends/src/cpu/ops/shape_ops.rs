//! `TensorOps` for `CpuBackendImpl<D>`: real `reshape`/`transpose`/
//! `broadcast_as`/`matmul`/`float_to_scalar`/`float_to_vec1`; every other
//! method is a typed stub returning `Error::UnsupportedBackendOperation`.
//!
//! This is the single `impl TensorOps<..> for CpuBackendImpl<..>` block for
//! the whole crate — `matmul`'s method body delegates to
//! `ops::matmul::matmul_impl` (see that file's module doc for why the naive
//! loop lives in its own file as a plain function rather than its own impl
//! block). `reshape`/`transpose`/`broadcast_as` are thin wrappers over
//! `CpuStorage`'s own already-O(1) view methods (Plan 01) — they do not
//! duplicate that logic, only add tape tracking (D-05: every op is a graph
//! node, unconditionally recorded).

use crate::cpu::CpuBackendImpl;
use incin_core::prelude::{
    Backend, BackendError, DType, DTypeDescriptor, DTypeId, Device, Error, OperationKind, Result,
    ShapeBuf, ShapeError, StorageBackend,
};
use incin_core::__backend_compat::legacy::{FloatOps, NumericOps, TensorOps};

use crate::cpu::ops::elementwise::elementwise_unary;
use crate::cpu::ops::matmul::{batched_matmul_impl, matmul_impl};
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape::{self, TapeEntry};

// `reshape`, `broadcast_as` and `matmul` are free functions so that the
// canonical `Execute<op::ReshapeExact>` executors in
// `cpu::canonical` and the legacy `TensorOps` methods below run the same body.
// One implementation is the point: a descriptor path that re-derived the view
// would be a second semantics to keep in agreement.

/// Reshape a view and record the inverse for backward.
pub(crate) fn reshape_storage(t: &CpuStorage, shape: &[usize]) -> Result<CpuStorage> {
    let out = t.reshape(shape)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
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
    tape::push(TapeEntry {
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
    tape::push(TapeEntry {
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
    tape::push(TapeEntry {
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

pub(crate) fn float_to_scalar_storage(t: &CpuStorage) -> Result<f64> {
    if crate::cpu::stride::checked_numel(&t.shape)? != 1 {
        return Err(Error::ShapeMismatch {
            op: "float_to_scalar",
            expected: vec![1],
            got: t.shape.to_vec(),
            msg: alloc::string::String::from("float_to_scalar requires a single-element tensor"),
        });
    }
    Ok(t.get(&vec![0usize; t.shape.len()]))
}

pub(crate) fn float_to_vec1_storage(t: &CpuStorage) -> Result<alloc::vec::Vec<f64>> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = alloc::vec::Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(t.get(&idx));
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(out)
}

pub(crate) fn int_to_scalar_storage(t: &CpuStorage) -> Result<i64> {
    if crate::cpu::stride::checked_numel(&t.shape)? != 1 {
        return Err(Error::ShapeMismatch {
            op: "int_to_scalar",
            expected: vec![1],
            got: t.shape.to_vec(),
            msg: alloc::string::String::from("int_to_scalar requires a single-element tensor"),
        });
    }
    t.get_i64_checked(&vec![0usize; t.shape.len()], "int_to_scalar")
}

pub(crate) fn int_to_vec1_storage(t: &CpuStorage) -> Result<alloc::vec::Vec<i64>> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = alloc::vec::Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(t.get_i64_checked(&idx, "int_to_vec1")?);
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(out)
}

pub(crate) fn masked_fill_storage(
    t: &CpuStorage,
    mask: &CpuStorage,
    value: f64,
) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(if mask.get_bool(&idx) { value } else { t.get(&idx) });
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, t.shape.to_vec()))
}

pub(crate) fn repeat_storage(t: &CpuStorage, repeats: &[usize]) -> Result<CpuStorage> {
    if repeats.len() != t.shape.len() {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Repeat,
            reason: "repeat factors must match tensor rank",
        }));
    }
    let out_shape: Vec<usize> = t
        .shape
        .iter()
        .zip(repeats.iter())
        .map(|(size, repeat)| size * repeat)
        .collect();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let src_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .map(|(axis, &value)| value % t.shape[axis])
            .collect();
        out.push(t.get(&src_idx));
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, out_shape))
}

pub(crate) fn pad_storage(
    t: &CpuStorage,
    padding: &[(usize, usize)],
    value: f64,
) -> Result<CpuStorage> {
    let out_shape: Vec<usize> = t
        .shape
        .iter()
        .zip(padding.iter())
        .map(|(size, &(before, after))| size + before + after)
        .collect();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let mut inside = true;
        let mut src_idx = Vec::with_capacity(idx.len());
        for (axis, &position) in idx.iter().enumerate() {
            let (before, _) = padding[axis];
            if position < before || position >= before + t.shape[axis] {
                inside = false;
                break;
            }
            src_idx.push(position - before);
        }
        out.push(if inside { t.get(&src_idx) } else { value });
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, out_shape))
}

/// Plain or batched matrix multiplication, chosen by operand rank.
pub(crate) fn matmul_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
        matmul_impl(lhs, rhs)
    } else {
        batched_matmul_impl(lhs, rhs)
    }
}

pub(crate) fn elementwise_cmp(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    f: impl Fn(f64, f64) -> bool,
) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let total: usize = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    if lhs.shape == rhs.shape {
        let mut idx = vec![0usize; lhs.shape.len()];
        for _ in 0..total {
            let v = if f(lhs.get(&idx), rhs.get(&idx)) {
                1u8
            } else {
                0u8
            };
            out.push(v);
            if !lhs.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &lhs.shape);
            }
        }
    } else {
        let plan = crate::iteration::IterationPlan::binary(
            crate::iteration::OperandLayout {
                shape: &lhs.shape,
                strides: &lhs.strides,
                offset: lhs.offset_elements,
            },
            crate::iteration::OperandLayout {
                shape: &rhs.shape,
                strides: &rhs.strides,
                offset: rhs.offset_elements,
            },
            &out_shape,
        )?;
        let l_plan = &plan.operands[0];
        let r_plan = &plan.operands[1];
        for flat_idx in 0..plan.numel {
            let a = lhs
                .buffer
                .get_f64(l_plan.physical_index(flat_idx, &plan.output_shape));
            let b = rhs
                .buffer
                .get_f64(r_plan.physical_index(flat_idx, &plan.output_shape));
            out.push(if f(a, b) { 1u8 } else { 0u8 });
        }
    }
    Ok(CpuStorage::from_contiguous(CpuBuffer::Bool(out), out_shape))
}

pub(crate) fn sub_scalar_storage(t: &CpuStorage, val: f64) -> Result<CpuStorage> {
    elementwise_unary(t, |value| value - val)
}

pub(crate) fn div_scalar_storage(t: &CpuStorage, val: f64) -> Result<CpuStorage> {
    elementwise_unary(t, |value| value / val)
}

pub(crate) fn triu_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    triangular_storage(t, k, true)
}

pub(crate) fn tril_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    triangular_storage(t, k, false)
}

pub(crate) fn diag_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    let rank = t.shape.len();
    if rank == 1 {
        let n = t.shape[0];
        let k_abs = k.unsigned_abs() as usize;
        let out_dim = n.checked_add(k_abs).ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Storage,
            expression: "diagonal output dimension",
        })?;
        let out_total =
            ShapeBuf::from_slice(&[out_dim, out_dim]).checked_numel(OperationKind::Storage)?;
        let mut out = vec![0.0f64; out_total];
        for i in 0..n {
            let row = if k >= 0 { i } else { i + k_abs };
            let col = if k >= 0 { i + k_abs } else { i };
            if row < out_dim && col < out_dim {
                out[row * out_dim + col] = t.get(&[i]);
            }
        }
        return Ok(CpuStorage::from_contiguous(
            t.buffer.from_f64_values(out)?,
            vec![out_dim, out_dim],
        ));
    }
    let row_len = t.shape[rank - 2];
    let col_len = t.shape[rank - 1];
    let mut values = Vec::new();
    for row in 0..row_len {
        let Some(col) = (row as i64).checked_add(k).filter(|&col| col >= 0) else {
            continue;
        };
        let col = col as usize;
        if col < col_len {
            let mut index = vec![0; rank];
            index[rank - 2] = row;
            index[rank - 1] = col;
            values.push(t.get(&index));
        }
    }
    let len = values.len();
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(values)?,
        vec![len],
    ))
}

pub(crate) fn group_norm_storage(t: &CpuStorage, groups: usize, eps: f64) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    if groups == 0 {
        return Err(Error::Msg("group_norm: groups must be non-zero".into()));
    }
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    if channels % groups != 0 {
        return Err(Error::Msg(
            "group_norm: channels must be divisible by groups".into(),
        ));
    }
    let (batch, spatial) = if t.shape.len() >= 2 {
        (t.shape[0], t.shape[2..].iter().product::<usize>())
    } else {
        (1, total)
    };
    let group_size = channels / groups * spatial;
    let mut out = Vec::with_capacity(total);
    for run in 0..batch * groups {
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for i in 0..group_size {
            let index = crate::cpu::ops::elementwise::flat_to_nd(run * group_size + i, &t.shape);
            let value = t.get(&index);
            sum += value;
            sq_sum += value * value;
        }
        let mean = sum / group_size as f64;
        let variance = (sq_sum / group_size as f64 - mean * mean).max(0.0);
        let inv_std = 1.0 / (variance + eps).sqrt();
        for i in 0..group_size {
            let index = crate::cpu::ops::elementwise::flat_to_nd(run * group_size + i, &t.shape);
            out.push((t.get(&index) - mean) * inv_std);
        }
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        t.shape.to_vec(),
    ))
}

pub(crate) fn instance_norm_storage(t: &CpuStorage, eps: f64) -> Result<CpuStorage> {
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    group_norm_storage(t, channels, eps)
}

fn triangular_storage(t: &CpuStorage, k: i64, upper: bool) -> Result<CpuStorage> {
    let rank = t.shape.len();
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let (row, col) = if rank >= 2 {
            (idx[rank - 2] as i64, idx[rank - 1] as i64)
        } else {
            (0, idx[0] as i64)
        };
        let keep = if upper {
            col >= row + k
        } else {
            col <= row + k
        };
        out.push(if keep { t.get(&idx) } else { 0.0 });
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        t.shape.to_vec(),
    ))
}

impl<D: Device> TensorOps<Self> for CpuBackendImpl<D> {
    /// `reshape`.
    fn reshape<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        reshape_storage(t, shape)
    }

    /// `transpose`.
    fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        transpose_storage(t, dim1, dim2)
    }

    /// `broadcast_as`.
    fn broadcast_as<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        broadcast_as_storage(t, shape)
    }

    /// `matmul`.
    fn matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        matmul_storage(lhs, rhs)
    }

    /// `narrow`.
    fn narrow<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        narrow_storage(t, dim, start, len)
    }

    /// `squeeze`.
    fn squeeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        squeeze_storage(t, dim)
    }

    /// `stack`.
    fn stack<K: DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: vec![],
                got: vec![],
                msg: alloc::string::String::from("stack requires at least one input tensor"),
            });
        }

        let rank = tensors[0].shape.len();
        if dim > rank {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: tensors[0].shape.to_vec(),
                got: vec![dim],
                msg: format!(
                    "stack dim {dim} out of range for rank-{rank} shape {:?} (dim may equal rank to append at the end)",
                    tensors[0].shape
                ),
            });
        }

        for t in tensors.iter().skip(1) {
            if t.shape != tensors[0].shape {
                return Err(Error::ShapeMismatch {
                    op: "stack",
                    expected: tensors[0].shape.to_vec(),
                    got: t.shape.to_vec(),
                    msg: format!(
                        "stack requires every input to have an IDENTICAL shape; expected {:?}, got {:?}",
                        tensors[0].shape, t.shape
                    ),
                });
            }
        }

        // Unsqueeze each input by reshaping to a target shape with a new
        // size-1 axis spliced in at `dim` (the TensorOps trait has no
        // dedicated `unsqueeze` method), then delegate to Self::concat —
        // this composition needs zero new backward code: reshape's and
        // concat's own tape entries compose correctly on their own.
        let mut unsqueezed = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let mut target_shape = t.shape.to_vec();
            target_shape.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &target_shape)?);
        }

        let refs: Vec<&<Self as StorageBackend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// `concat`.
    fn concat<K: DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: vec![],
                got: vec![],
                msg: alloc::string::String::from("concat requires at least one input tensor"),
            });
        }

        let rank = tensors[0].shape.len();
        if dim >= rank {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: tensors[0].shape.to_vec(),
                got: vec![dim],
                msg: format!(
                    "concat dim {dim} out of range for rank-{rank} shape {:?}",
                    tensors[0].shape
                ),
            });
        }

        for t in tensors.iter().skip(1) {
            if t.shape.len() != rank {
                return Err(Error::ShapeMismatch {
                    op: "concat",
                    expected: tensors[0].shape.to_vec(),
                    got: t.shape.to_vec(),
                    msg: format!(
                        "concat requires every input to have the same rank; expected rank {rank}, got shape {:?}",
                        t.shape
                    ),
                });
            }
            // Every axis EXCEPT `dim` must match EXACTLY — never
            // broadcast-compatible (Pitfall 5: a size-1-vs-larger mismatch
            // here must be REJECTED, not silently accepted the way
            // stride::broadcast_shape would treat it).
            for (axis, (&a, &b)) in tensors[0].shape.iter().zip(t.shape.iter()).enumerate() {
                if axis != dim && a != b {
                    return Err(Error::ShapeMismatch {
                        op: "concat",
                        expected: tensors[0].shape.to_vec(),
                        got: t.shape.to_vec(),
                        msg: format!(
                            "concat requires exact equality on every non-concat axis; axis {axis} has size {a} vs {b}"
                        ),
                    });
                }
            }
        }

        let mut out_shape = tensors[0].shape.to_vec();
        out_shape[dim] = tensors.iter().try_fold(0usize, |total, tensor| {
            total.checked_add(tensor.shape[dim]).ok_or(
                incin_core::prelude::ShapeError::ArithmeticOverflow {
                    operation: incin_core::prelude::OperationKind::Concat,
                    expression: "sum of concatenated axis dimensions",
                },
            )
        })?;
        let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;

        // Cumulative offset of each input along `dim`, needed by both the
        // forward copy and the backward narrow-based scatter.
        let mut cumulative_offsets = Vec::with_capacity(tensors.len());
        let mut running = 0usize;
        for t in tensors.iter() {
            cumulative_offsets.push(running);
            running = running.checked_add(t.shape[dim]).ok_or(
                incin_core::prelude::ShapeError::ArithmeticOverflow {
                    operation: incin_core::prelude::OperationKind::Concat,
                    expression: "cumulative concatenation offset",
                },
            )?;
        }

        macro_rules! concat_variant {
            ($variant:ident, $ty:ty) => {{
                let mut out: Vec<$ty> = vec![Default::default(); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    // Read this input through ITS OWN strides directly — no
                    // prior `.contiguous()` materialization.
                    let value_count: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = t.get(&multi_idx) as $ty;
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::$variant(out)
            }};
        }

        let new_buffer = match &*tensors[0].buffer {
            CpuBuffer::F32(_) => concat_variant!(F32, f32),
            CpuBuffer::F64(_) => concat_variant!(F64, f64),
            CpuBuffer::U8(_) => concat_variant!(U8, u8),
            CpuBuffer::Bool(_) => concat_variant!(Bool, u8),
            CpuBuffer::U32(_) => concat_variant!(U32, u32),
            CpuBuffer::I64(_) => concat_variant!(I64, i64),
            CpuBuffer::F16(_) => {
                let mut out: Vec<half::f16> = vec![half::f16::from_f64(0.0); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    let value_count: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = half::f16::from_f64(t.get(&multi_idx));
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::F16(out)
            }
            CpuBuffer::BF16(_) => {
                let mut out: Vec<half::bf16> = vec![half::bf16::from_f64(0.0); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    let value_count: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = half::bf16::from_f64(t.get(&multi_idx));
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::BF16(out)
            }
            CpuBuffer::Q8_0(_) => {
                return Err(Error::UnsupportedDType {
                    dtype: DTypeId::Q8_0.descriptor(),
                    backend: "cpu",
                    op: "concat",
                });
            }
        };

        let out = CpuStorage::from_contiguous(new_buffer, out_shape);

        let out_id = out.id;
        let input_ids: Vec<_> = tensors.iter().map(|t| t.id).collect();
        let input_dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape[dim]).collect();
        let offsets = cumulative_offsets.clone();
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids,
            // Collecting an iterator of `Result` straight into
            // `Result<Vec<_>>` is the whole conversion here: the recipe's
            // return type is now exactly what `collect` already produced.
            backward: Box::new(move |grad_out: &CpuStorage| {
                offsets
                    .iter()
                    .zip(input_dim_sizes.iter())
                    .map(|(&offset, &len)| grad_out.narrow(dim, offset, len))
                    .collect()
            }),
        });

        Ok(out)
    }

    /// `slice`.
    fn slice<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut out = t.clone();
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            out = Self::narrow::<K>(&out, dim, start, end - start)?;
        }
        Ok(out)
    }

    /// `flatten`.
    fn flatten<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        flatten_storage(t, start_dim, end_dim)
    }

    /// `broadcast_left`.
    fn broadcast_left<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        broadcast_left_storage(t, shape)
    }

    /// `float_to_scalar`.
    fn float_to_scalar<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<f64> {
        float_to_scalar_storage(t)
    }

    /// `float_to_vec1`.
    fn float_to_vec1<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        float_to_vec1_storage(t)
    }

    /// `int_to_scalar`.
    fn int_to_scalar<K: DType>(t: &<Self as StorageBackend>::Storage<K>) -> Result<i64> {
        int_to_scalar_storage(t)
    }

    /// `int_to_vec1`.
    fn int_to_vec1<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        int_to_vec1_storage(t)
    }

    /// `tensor_to_dtype`.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K2>> {
        let total: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
        let mut multi_idx = vec![0usize; t.shape.len()];

        macro_rules! convert_variant {
            ($variant:ident, $ty:ty) => {{
                let mut out: alloc::vec::Vec<$ty> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(t.get(&multi_idx) as $ty);
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::$variant(out)
            }};
        }

        let new_buffer = match dtype.builtin_id() {
            Some(DTypeId::F32) => convert_variant!(F32, f32),
            Some(DTypeId::F64) => convert_variant!(F64, f64),
            Some(DTypeId::U8) => convert_variant!(U8, u8),
            Some(DTypeId::U32) => convert_variant!(U32, u32),
            Some(DTypeId::I64) => convert_variant!(I64, i64),
            Some(DTypeId::F16) => {
                let mut out: alloc::vec::Vec<half::f16> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(half::f16::from_f64(t.get(&multi_idx)));
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::F16(out)
            }
            Some(DTypeId::BF16) => {
                let mut out: alloc::vec::Vec<half::bf16> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(half::bf16::from_f64(t.get(&multi_idx)));
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::BF16(out)
            }
            Some(DTypeId::Q8_0) => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "tensor_to_dtype(Q8_0)",
                    backend: "Cpu",
                });
            }
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "tensor_to_dtype(unknown)",
                    backend: "Cpu",
                });
            }
        };

        Ok(CpuStorage::from_contiguous(new_buffer, t.shape.to_vec()))
    }

    fn where_cond<K: DType>(
        mask: &<Self as StorageBackend>::Storage<bool>,
        on_true: &<Self as StorageBackend>::Storage<K>,
        on_false: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&on_true.shape, &on_false.shape)?;
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let is_true = mask.get_bool(&idx);
            let val = if is_true {
                on_true.get(&idx)
            } else {
                on_false.get(&idx)
            };
            out.push(val);
            if !out_shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &out_shape);
            }
        }
        let buffer = on_true.buffer.from_f64_values(out)?;
        let out_storage = CpuStorage::from_contiguous(buffer, out_shape);

        let (mask_cap, on_true_cap, on_false_cap) =
            (mask.clone(), on_true.clone(), on_false.clone());
        let (true_id, false_id, out_id) = (on_true.id, on_false.id, out_storage.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![true_id, false_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let total: usize = crate::cpu::stride::checked_numel(&(grad_out.shape))?;
                let mut grad_true = Vec::with_capacity(total);
                let mut grad_false = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let m = mask_cap.get_bool(&idx);
                    let g = grad_out.get(&idx);
                    if m {
                        grad_true.push(g);
                        grad_false.push(0.0);
                    } else {
                        grad_true.push(0.0);
                        grad_false.push(g);
                    }
                    if !grad_out.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut idx, &grad_out.shape);
                    }
                }
                let g_true = CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(grad_true)?,
                    grad_out.shape.to_vec(),
                );
                let g_false = CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(grad_false)?,
                    grad_out.shape.to_vec(),
                );
                Ok(vec![
                    tape::unbroadcast(&g_true, &on_true_cap.shape)?,
                    tape::unbroadcast(&g_false, &on_false_cap.shape)?,
                ])
            }),
        });
        Ok(out_storage)
    }

    fn gather<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out_shape = index.shape.to_vec();
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let target_i = index.get(&idx) as usize;
            let mut src_idx = idx.clone();
            src_idx[dim] = target_i;
            out.push(t.get(&src_idx));
            if !out_shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &out_shape);
            }
        }
        let buffer = t.buffer.from_f64_values(out)?;
        let out_storage = CpuStorage::from_contiguous(buffer, out_shape);

        let (t_cap, index_cap) = (t.clone(), index.clone());
        let (t_id, out_id) = (t.id, out_storage.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let t_total: usize = crate::cpu::stride::checked_numel(&(t_cap.shape))?;
                let mut grad_t_data = vec![0.0; t_total];
                let index_total: usize = crate::cpu::stride::checked_numel(&(index_cap.shape))?;
                let mut idx = vec![0usize; index_cap.shape.len()];
                for _ in 0..index_total {
                    let target_i = index_cap.get(&idx) as usize;
                    let mut src_idx = idx.clone();
                    src_idx[dim] = target_i;
                    let out_strides = crate::cpu::stride::contiguous_strides(&t_cap.shape);
                    let mut flat_dst = 0;
                    for (i, s) in src_idx.iter().zip(out_strides.iter()) {
                        flat_dst += i * s;
                    }
                    grad_t_data[flat_dst] += grad_out.get(&idx);
                    if !index_cap.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut idx, &index_cap.shape);
                    }
                }
                Ok(vec![CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(grad_t_data)?,
                    t_cap.shape.to_vec(),
                )])
            }),
        });
        Ok(out_storage)
    }

    fn scatter<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
        src: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t_total: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
        let mut out_data: Vec<f64> = (0..t_total)
            .map(|i| {
                let nd = crate::cpu::ops::elementwise::flat_to_nd(i, &t.shape);
                t.get(&nd)
            })
            .collect();
        let index_total: usize = crate::cpu::stride::checked_numel(&(index.shape))?;
        let mut idx = vec![0usize; index.shape.len()];
        for _ in 0..index_total {
            let target_i = index.get(&idx) as usize;
            let src_val = src.get(&idx);
            let mut dest_idx = idx.clone();
            dest_idx[dim] = target_i;
            let strides = crate::cpu::stride::contiguous_strides(&t.shape);
            let flat_dest: usize = dest_idx
                .iter()
                .zip(strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            if flat_dest < out_data.len() {
                out_data[flat_dest] = src_val;
            }
            if !index.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &index.shape);
            }
        }
        let buffer = t.buffer.from_f64_values(out_data)?;
        Ok(CpuStorage::from_contiguous(buffer, t.shape.to_vec()))
    }

    fn index_select<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let idx_total: usize = crate::cpu::stride::checked_numel(&(index.shape))?;
        let idx_vec: Vec<f64> = (0..idx_total)
            .map(|i| index.get(&crate::cpu::ops::elementwise::flat_to_nd(i, &index.shape)))
            .collect();
        let mut out_shape = t.shape.to_vec();
        out_shape[dim] = idx_vec.len();
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;
        let mut out = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let selected_pos = idx_vec[out_idx[dim]] as usize;
            let mut src_idx = out_idx.clone();
            src_idx[dim] = selected_pos;
            out.push(t.get(&src_idx));
            if !out_shape.is_empty() {
                crate::cpu::storage::increment_index(&mut out_idx, &out_shape);
            }
        }
        let buffer = t.buffer.from_f64_values(out)?;
        Ok(CpuStorage::from_contiguous(buffer, out_shape))
    }

    fn masked_fill<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        mask: &<Self as StorageBackend>::Storage<bool>,
        value: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        masked_fill_storage(t, mask, value)
    }

    fn unsqueeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        unsqueeze_storage(t, dim)
    }

    fn repeat<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        repeat_storage(t, repeats)
    }

    fn pad<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        pad_storage(t, padding, val)
    }

    fn triu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        triu_storage(t, k)
    }

    fn tril<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        tril_storage(t, k)
    }

    fn diag<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        diag_storage(t, k)
    }

    fn cmp_eq<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a == b)
    }

    fn cmp_ne<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a != b)
    }

    fn cmp_lt<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a < b)
    }

    fn cmp_le<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a <= b)
    }

    fn cmp_gt<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a > b)
    }

    fn cmp_ge<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a >= b)
    }

    fn logical_and(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a != 0.0 && b != 0.0)
    }

    fn logical_or(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        elementwise_cmp(lhs, rhs, |a, b| a != 0.0 || b != 0.0)
    }

    fn logical_not(
        t: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let total: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            let v = if t.get(&idx) == 0.0 { 1u8 } else { 0u8 };
            out.push(v);
            if !t.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &t.shape);
            }
        }
        Ok(CpuStorage::from_contiguous(
            CpuBuffer::Bool(out),
            t.shape.to_vec(),
        ))
    }

    fn sub_scalar<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        sub_scalar_storage(t, val)
    }

    fn div_scalar<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        div_scalar_storage(t, val)
    }

    fn maximum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(&(lhs.shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; lhs.shape.len()];
        for _ in 0..total {
            let v = lhs.get(&idx).max(rhs.get(&idx));
            out.push(v);
            if !lhs.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &lhs.shape);
            }
        }
        Ok(CpuStorage::from_contiguous(
            lhs.buffer.from_f64_values(out)?,
            lhs.shape.to_vec(),
        ))
    }

    fn minimum<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(&(lhs.shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; lhs.shape.len()];
        for _ in 0..total {
            let v = lhs.get(&idx).min(rhs.get(&idx));
            out.push(v);
            if !lhs.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &lhs.shape);
            }
        }
        Ok(CpuStorage::from_contiguous(
            lhs.buffer.from_f64_values(out)?,
            lhs.shape.to_vec(),
        ))
    }

    fn abs_diff<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(&(lhs.shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; lhs.shape.len()];
        for _ in 0..total {
            let v = (lhs.get(&idx) - rhs.get(&idx)).abs();
            out.push(v);
            if !lhs.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &lhs.shape);
            }
        }
        Ok(CpuStorage::from_contiguous(
            lhs.buffer.from_f64_values(out)?,
            lhs.shape.to_vec(),
        ))
    }

    fn lerp<K: DType>(
        start: &<Self as StorageBackend>::Storage<K>,
        end: &<Self as StorageBackend>::Storage<K>,
        weight: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(&(start.shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; start.shape.len()];
        for _ in 0..total {
            let s = start.get(&idx);
            let e = end.get(&idx);
            out.push(s + weight * (e - s));
            if !start.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &start.shape);
            }
        }
        Ok(CpuStorage::from_contiguous(
            start.buffer.from_f64_values(out)?,
            start.shape.to_vec(),
        ))
    }

    fn addmm<K: DType>(
        mat: &<Self as StorageBackend>::Storage<K>,
        mat1: &<Self as StorageBackend>::Storage<K>,
        mat2: &<Self as StorageBackend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mm = Self::matmul::<K>(mat1, mat2)?;
        let mm_alpha = Self::mul_scalar_float::<K>(&mm, alpha)?;
        let mat_beta = Self::mul_scalar_float::<K>(mat, beta)?;
        Self::add::<K>(&mat_beta, &mm_alpha)
    }

    fn bmm<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::matmul::<K>(lhs, rhs)
    }

    fn scaled_dot_product_attention<K: DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let k_rank = k.shape.len();
        let k_t = if k_rank >= 2 {
            Self::transpose::<K>(k, k_rank - 2, k_rank - 1)?
        } else {
            k.clone()
        };
        let scores = Self::matmul::<K>(q, &k_t)?;
        let d_k = *q.shape.last().unwrap_or(&1) as f64;
        let s = scale.unwrap_or_else(|| 1.0 / d_k.sqrt());
        let scaled_scores = Self::mul_scalar_float::<K>(&scores, s)?;
        let masked_scores = if let Some(m) = mask {
            Self::add::<K>(&scaled_scores, m)?
        } else {
            scaled_scores
        };
        let attn = Self::softmax::<K>(&masked_scores, scores.shape.len() - 1)?;
        Self::matmul::<K>(&attn, v)
    }

    fn unfold<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dim_len = t.shape[dim];
        if size > dim_len {
            return Err(Error::Msg(
                "unfold size cannot exceed dimension length".into(),
            ));
        }
        let n_windows = (dim_len - size) / step + 1;
        let mut out_shape = t.shape.to_vec();
        out_shape[dim] = n_windows;
        out_shape.push(size);
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let win_idx = idx[dim];
            let offset_idx = idx[out_shape.len() - 1];
            let mut src_idx = idx[..t.shape.len()].to_vec();
            src_idx[dim] = win_idx * step + offset_idx;
            out.push(t.get(&src_idx));
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
        Ok(CpuStorage::from_contiguous(
            t.buffer.from_f64_values(out)?,
            out_shape,
        ))
    }

    fn pixel_shuffle<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        upscale_factor: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if t.shape.len() != 4 {
            return Err(Error::Msg(
                "pixel_shuffle expects 4D tensor (N, C, H, W)".into(),
            ));
        }
        let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
        let r = upscale_factor;
        let r_sq = r * r;
        if c % r_sq != 0 {
            return Err(Error::Msg(
                "pixel_shuffle channels must be divisible by upscale_factor^2".into(),
            ));
        }
        let out_c = c / r_sq;
        let out_h = h * r;
        let out_w = w * r;
        let out_shape = vec![n, out_c, out_h, out_w];
        let total: usize = crate::cpu::stride::checked_numel(&(out_shape))?;
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; 4];
        for _ in 0..total {
            let (b, c_out, h_out, w_out) = (idx[0], idx[1], idx[2], idx[3]);
            let h_in = h_out / r;
            let w_in = w_out / r;
            let r_h = h_out % r;
            let r_w = w_out % r;
            let c_in = c_out * r_sq + r_h * r + r_w;
            out.push(t.get(&[b, c_in, h_in, w_in]));
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
        Ok(CpuStorage::from_contiguous(
            t.buffer.from_f64_values(out)?,
            out_shape,
        ))
    }

    fn group_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        groups: usize,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        group_norm_storage(t, groups, eps)
    }

    fn instance_norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        instance_norm_storage(t, eps)
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<incin_core::prelude::Cpu>;

    /// `matrix`.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    /// `reshape_through_trait_matches_direct_storage_call`.
    fn reshape_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let direct = t.reshape(&[3, 2]).unwrap();
        let via_trait = TestBackend::reshape::<f32>(&t, &[3, 2]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    #[test]
    /// `transpose_through_trait_matches_direct_storage_call`.
    fn transpose_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let direct = t.transpose(0, 1).unwrap();
        let via_trait = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    /// `broadcast_as_through_trait_matches_direct_storage_call`.
    fn broadcast_as_through_trait_matches_direct_storage_call() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let direct = t.broadcast_as(&[4, 3]).unwrap();
        let via_trait = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    /// `float_to_scalar_reads_single_element`.
    fn float_to_scalar_reads_single_element() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![42.0]), vec![]);
        let v = TestBackend::float_to_scalar::<f32>(&t).unwrap();
        assert_eq!(v, 42.0);
    }

    #[test]
    /// `float_to_vec1_reads_all_elements_row_major`.
    fn float_to_vec1_reads_all_elements_row_major() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let v = TestBackend::float_to_vec1::<f32>(&t).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    /// `reshape_backward_reshapes_grad_back_to_original_shape`.
    fn reshape_backward_reshapes_grad_back_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::reshape::<f32>(&t, &[6]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    /// `transpose_backward_reapplies_same_transpose`.
    fn transpose_backward_reapplies_same_transpose() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    /// `broadcast_as_backward_unbroadcasts_to_original_shape`.
    fn broadcast_as_backward_unbroadcasts_to_original_shape() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let out = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![1, 3]);
        // ones_like(out) [4,3] summed over the broadcast axis -> [4,4,4]
        assert_eq!(f32_vec(g), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    /// `matmul_via_trensor_ops_delegates_to_matmul_impl`.
    fn matmul_via_trensor_ops_delegates_to_matmul_impl() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    /// `unsupported_methods_return_typed_error_not_silent_placeholder`.
    fn unsupported_methods_return_typed_error_not_silent_placeholder() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        // All other TensorOps methods are now fully implemented. We prove that
        // unsupported operations return typed errors by attempting to convert
        // to Q8_0, which is intentionally left unsupported in the Cpu backend.
        let result = TestBackend::tensor_to_dtype::<f32, f32>(&t, DTypeId::Q8_0.descriptor());
        assert!(matches!(
            result,
            Err(Error::UnsupportedBackendOperation {
                op: "tensor_to_dtype(Q8_0)",
                ..
            })
        ));
    }

    /// Task 1 Test 1: `TensorOps::narrow` called through the trait matches
    /// calling `CpuStorage::narrow` directly (thin-wrapper equivalence).
    #[test]
    fn narrow_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let direct = t.narrow(0, 1, 1).unwrap();
        let via_trait = TestBackend::narrow::<f32>(&t, 0, 1, 1).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    /// Task 1 Test 2: `narrow`'s backward zero-pads `grad_out` back to the
    /// original shape at the correct region.
    #[test]
    fn narrow_backward_zero_pads_grad_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let out = TestBackend::narrow::<f32>(&t, 0, 1, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![3, 2]);
        assert_eq!(f32_vec(g), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    /// Task 1 Test 3: out-of-bounds narrow range returns `Err`, not a panic.
    #[test]
    fn narrow_out_of_bounds_returns_err_not_panic() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let result = TestBackend::narrow::<f32>(&t, 0, 2, 2);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 4: `narrow`'s forward value on a pre-transposed
    /// (non-contiguous) input still produces correct values.
    #[test]
    fn narrow_on_transposed_input_produces_correct_values_without_materializing() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        // transposed is logically [[1,4],[2,5],[3,6]], shape [3,2]
        let narrowed = TestBackend::narrow::<f32>(&transposed, 0, 1, 1).unwrap();
        assert_eq!(narrowed.shape, vec![1, 2]);
        assert_eq!(narrowed.get(&[0, 0]), 2.0);
        assert_eq!(narrowed.get(&[0, 1]), 5.0);
    }

    /// Task 2 Test 1: `slice(t, &[(1,3),(0,2)])` on a `[4,3]` matrix matches
    /// manually narrowing dim 0 to `(1,3)` then dim 1 to `(0,2)` in sequence.
    #[test]
    fn slice_matches_manual_sequential_narrow_calls() {
        let t = matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let manual = TestBackend::narrow::<f32>(&t, 0, 1, 2).unwrap();
        let manual = TestBackend::narrow::<f32>(&manual, 1, 0, 2).unwrap();

        let via_slice = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 2)]).unwrap();
        assert_eq!(via_slice.shape, manual.shape);
        assert_eq!(f32_vec(&via_slice), f32_vec(&manual));
    }

    /// Task 2 Test 2: `slice` on a pre-transposed (non-contiguous) input,
    /// across multiple dims, produces correct values without a
    /// `.contiguous()` call happening internally.
    #[test]
    fn slice_on_transposed_input_across_multiple_dims_produces_correct_values() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
        // slice rows [1,3) and cols [0,1) -> [[2],[3]]
        let out = TestBackend::slice::<f32>(&transposed, &[(1, 3), (0, 1)]).unwrap();
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.get(&[0, 0]), 2.0);
        assert_eq!(out.get(&[1, 0]), 3.0);
    }

    /// Task 2 Test 3: `slice`'s backward correctly zero-pads back to the
    /// original shape, composed entirely from `narrow`'s own backward.
    #[test]
    fn slice_backward_zero_pads_grad_to_original_shape() {
        let t = matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let out = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 2)]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![4, 3]);
        assert_eq!(
            f32_vec(g),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    /// Task 2 Test 4: an out-of-bounds range in any dim of a multi-dim
    /// `slice` call returns `Err`, not a panic.
    #[test]
    fn slice_out_of_bounds_range_returns_err_not_panic() {
        let t = matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let result = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 5)]);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// `tensor3`.
    fn tensor3(v: Vec<f32>, d0: usize, d1: usize, d2: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![d0, d1, d2])
    }

    /// Task 3 Test 1: `squeeze(t, 1)` on a `[3,1,4]` storage produces shape
    /// `[3,4]` with unchanged (row-major) values.
    #[test]
    fn squeeze_removes_size_one_axis_and_preserves_values() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = tensor3(data.clone(), 3, 1, 4);
        let out = TestBackend::squeeze::<f32>(&t, 1).unwrap();
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 2: `squeeze(t, 0)` on a `[3,1,4]` storage (dim 0 has size
    /// 3, not 1) returns a clear squeeze-specific `Error::ShapeMismatch`.
    #[test]
    fn squeeze_on_non_one_sized_axis_returns_shape_mismatch() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = tensor3(data, 3, 1, 4);
        let result = TestBackend::squeeze::<f32>(&t, 0);
        match result {
            Err(Error::ShapeMismatch { op, .. }) => assert_eq!(op, "squeeze"),
            other => panic!("expected squeeze-specific ShapeMismatch, got {other:?}"),
        }
    }

    /// Task 3 Test 3: `squeeze`'s backward reshapes `grad_out` back to the
    /// original `[3,1,4]` shape, delegated entirely to `reshape`'s backward.
    #[test]
    fn squeeze_backward_reshapes_grad_to_original_shape() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = tensor3(data, 3, 1, 4);
        let out = TestBackend::squeeze::<f32>(&t, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![3, 1, 4]);
        assert_eq!(f32_vec(g), vec![1.0; 12]);
    }

    /// Task 3 Test 4: `flatten(t, 1, 2)` on a `[2,3,4]` storage produces
    /// shape `[2,12]` (merging dims 1..=2).
    #[test]
    fn flatten_merges_middle_dims() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = tensor3(data.clone(), 2, 3, 4);
        let out = TestBackend::flatten::<f32>(&t, 1, 2).unwrap();
        assert_eq!(out.shape, vec![2, 12]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 5: `flatten(t, 0, 2)` on a `[2,3,4]` storage (flattening
    /// all dims) produces shape `[24]`.
    #[test]
    fn flatten_all_dims_produces_1d_shape() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = tensor3(data.clone(), 2, 3, 4);
        let out = TestBackend::flatten::<f32>(&t, 0, 2).unwrap();
        assert_eq!(out.shape, vec![24]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 6: `flatten`'s backward reshapes `grad_out` back to the
    /// original shape, delegated entirely to `reshape`'s backward.
    #[test]
    fn flatten_backward_reshapes_grad_to_original_shape() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = tensor3(data, 2, 3, 4);
        let out = TestBackend::flatten::<f32>(&t, 1, 2).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3, 4]);
        assert_eq!(f32_vec(g), vec![1.0; 24]);
    }

    /// Test 6: `TensorOps::matmul` called through the trait on two rank-2
    /// operands still produces identical values to a direct `matmul_impl`
    /// call (dispatch does not change the unbatched path's behavior).
    #[test]
    fn matmul_dispatch_rank2_matches_matmul_impl_directly() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let direct = matmul_impl(&lhs, &rhs).unwrap();
        let via_trait = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    /// Test 7: `TensorOps::matmul` called through the trait on two rank-3
    /// (or higher) operands correctly dispatches to `batched_matmul_impl`
    /// and produces the same values a direct `batched_matmul_impl` call
    /// would.
    #[test]
    fn matmul_dispatch_rank3_matches_batched_matmul_impl_directly() {
        let lhs_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32).collect();
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(lhs_data), vec![2, 3, 4]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(rhs_data), vec![2, 4, 5]);

        let direct = batched_matmul_impl(&lhs, &rhs).unwrap();
        let via_trait = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    /// Task 1 Test 1: `concat(&[a, b], 0)` where `a` is `[2,3]` and `b` is
    /// `[3,3]` produces shape `[5,3]`, rows 0-1 matching `a`, rows 2-4
    /// matching `b`.
    #[test]
    fn concat_dim0_stacks_rows_in_input_order() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            3,
            3,
        );
        let out = TestBackend::concat::<f32>(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape, vec![5, 3]);
        assert_eq!(
            f32_vec(&out),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
            ]
        );
    }

    /// Task 1 Test 2: `concat(&[a, b], 1)` where `a` is `[2,3]` and `b` is
    /// `[2,2]` produces shape `[2,5]`, columns correctly interleaved by row.
    #[test]
    fn concat_dim1_interleaves_columns_by_row() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![7.0, 8.0, 9.0, 10.0], 2, 2);
        let out = TestBackend::concat::<f32>(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape, vec![2, 5]);
        assert_eq!(
            f32_vec(&out),
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 4.0, 5.0, 6.0, 9.0, 10.0]
        );
    }

    /// Task 1 Test 3 (Pitfall 5 regression): a size-1-vs-size-larger
    /// mismatch at a NON-concat axis is REJECTED with `Err(ShapeMismatch)`,
    /// proving the validation uses exact equality, not
    /// `stride::broadcast_shape`'s size-1-is-compatible-with-anything rule.
    #[test]
    fn concat_rejects_non_concat_axis_size_mismatch_even_when_broadcast_compatible() {
        // a: [3,1], b: [3,4] -- dim 1 sizes differ (1 vs 4), concatenating on
        // dim 0. stride::broadcast_shape would treat size-1 as compatible
        // with anything; concat must NOT.
        let a = matrix(vec![1.0, 2.0, 3.0], 3, 1);
        let b = matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            3,
            4,
        );
        let result = TestBackend::concat::<f32>(&[&a, &b], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 4: `concat(&[], 0)` (empty input list) returns
    /// `Err(Error::ShapeMismatch)`, not a panic.
    #[test]
    fn concat_empty_input_list_returns_err_not_panic() {
        let result: Result<CpuStorage> = TestBackend::concat::<f32>(&[], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 5: `concat` called with `dim >= rank` returns
    /// `Err(Error::ShapeMismatch)`.
    #[test]
    fn concat_dim_out_of_bounds_returns_err() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = TestBackend::concat::<f32>(&[&a, &b], 2);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 6: `concat`'s backward correctly narrows `grad_out` back
    /// to each input's own shape at its cumulative `dim`-offset, with 2
    /// inputs of DIFFERENT sizes along the concat dim.
    #[test]
    fn concat_backward_narrows_grad_to_each_inputs_own_shape_and_values() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            3,
            3,
        );
        let out = TestBackend::concat::<f32>(&[&a, &b], 0).unwrap();
        let grads = tape::backward(&out).unwrap();

        let ga = grads.get(a.id).expect("a should have a gradient");
        assert_eq!(ga.shape, vec![2, 3]);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(ga.get(&[r, c]), 1.0);
            }
        }

        let gb = grads.get(b.id).expect("b should have a gradient");
        assert_eq!(gb.shape, vec![3, 3]);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(gb.get(&[r, c]), 1.0);
            }
        }
    }

    /// Task 1 Test 7: each input to `concat` is read through its OWN
    /// strides without being materialized first — one input is a
    /// TRANSPOSED (non-contiguous) view, output values are still correct.
    #[test]
    fn concat_on_transposed_input_produces_correct_values_without_materializing() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&a, 0, 1).unwrap();
        // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
        let b = matrix(vec![100.0, 200.0], 1, 2);
        let out = TestBackend::concat::<f32>(&[&transposed, &b], 0).unwrap();
        assert_eq!(out.shape, vec![4, 2]);
        assert_eq!(
            f32_vec(&out),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 100.0, 200.0]
        );
    }

    /// Task 2 Test 1: `stack(&[a, b], 0)` where `a`/`b` are both `[2,3]`
    /// produces shape `[2,2,3]`, with the new axis-0 slices matching `a`/`b`
    /// respectively.
    #[test]
    fn stack_dim0_inserts_new_leading_axis() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let out = TestBackend::stack::<f32>(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape, vec![2, 2, 3]);
        assert_eq!(
            f32_vec(&out),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    /// Task 2 Test 2: `stack(&[a, b], 2)` (dim equal to rank, appending at
    /// the very end) where `a`/`b` are both `[2,3]` produces shape `[2,3,2]`.
    #[test]
    fn stack_dim_equal_to_rank_appends_new_trailing_axis() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let out = TestBackend::stack::<f32>(&[&a, &b], 2).unwrap();
        assert_eq!(out.shape, vec![2, 3, 2]);
        // Element [r,c,0] == a[r,c], [r,c,1] == b[r,c]
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(out.get(&[r, c, 0]), a.get(&[r, c]));
                assert_eq!(out.get(&[r, c, 1]), b.get(&[r, c]));
            }
        }
    }

    /// Task 2 Test 3: `stack` with mismatched-shape inputs returns
    /// `Err(Error::ShapeMismatch)` — stack requires IDENTICAL shapes,
    /// stricter than concat's "all-but-one-axis" rule.
    #[test]
    fn stack_rejects_mismatched_shapes() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
        let result = TestBackend::stack::<f32>(&[&a, &b], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 2 Test 4: `stack(&[], 0)` (empty input list) returns
    /// `Err(Error::ShapeMismatch)`, not a panic.
    #[test]
    fn stack_empty_input_list_returns_err_not_panic() {
        let result: Result<CpuStorage> = TestBackend::stack::<f32>(&[], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 2 Test 5: `stack`'s backward correctly narrows-then-squeezes
    /// `grad_out` back to each input's own ORIGINAL shape (the inserted
    /// axis removed), with 2 distinct inputs.
    #[test]
    fn stack_backward_narrows_and_squeezes_grad_to_original_shape() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let out = TestBackend::stack::<f32>(&[&a, &b], 0).unwrap();
        let grads = tape::backward(&out).unwrap();

        let ga = grads.get(a.id).expect("a should have a gradient");
        assert_eq!(ga.shape, vec![2, 3]);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(ga.get(&[r, c]), 1.0);
            }
        }

        let gb = grads.get(b.id).expect("b should have a gradient");
        assert_eq!(gb.shape, vec![2, 3]);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(gb.get(&[r, c]), 1.0);
            }
        }
    }

    /// Task 3 Test 1: `broadcast_left(t, &[4])` on a `[3]` vector produces
    /// shape `[4,3]` (the `[4]` prepended as a new leading dim, `t`'s own
    /// `[3]` shape unchanged and trailing).
    #[test]
    fn broadcast_left_prepends_single_new_leading_dim() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let out = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        for row in 0..4 {
            assert_eq!(out.get(&[row, 0]), 1.0);
            assert_eq!(out.get(&[row, 1]), 2.0);
            assert_eq!(out.get(&[row, 2]), 3.0);
        }
    }

    /// Task 3 Test 2: `broadcast_left(t, &[2,4])` on a `[3]` vector produces
    /// shape `[2,4,3]` (multiple new leading dims prepended at once).
    #[test]
    fn broadcast_left_prepends_multiple_new_leading_dims() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let out = TestBackend::broadcast_left::<f32>(&t, &[2, 4]).unwrap();
        assert_eq!(out.shape, vec![2, 4, 3]);
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(out.get(&[i, j, 0]), 1.0);
                assert_eq!(out.get(&[i, j, 1]), 2.0);
                assert_eq!(out.get(&[i, j, 2]), 3.0);
            }
        }
    }

    /// Task 3 Test 3: `broadcast_left`'s backward correctly unbroadcasts
    /// `grad_out` back to `t`'s own original shape, with ZERO new backward
    /// code (delegates entirely to `Self::broadcast_as`).
    #[test]
    fn broadcast_left_backward_unbroadcasts_to_original_shape() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let out = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![3]);
        // ones_like(out) [4,3] summed over the broadcast axis -> [4,4,4]
        assert_eq!(f32_vec(g), vec![4.0, 4.0, 4.0]);
    }

    /// Task 3 Test 4: `broadcast_left` called through the trait matches
    /// calling `CpuStorage::broadcast_as` directly with the manually
    /// prepended target shape (thin-wrapper equivalence).
    #[test]
    fn broadcast_left_through_trait_matches_direct_broadcast_as_call() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let direct = t.broadcast_as(&[4, 3]).unwrap();
        let via_trait = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    /// Every pre-existing `group_norm` test used a batch of 1, which is the
    /// one size at which grouping over the whole flattened buffer and grouping
    /// per sample agree. Two samples are the smallest case that tells them
    /// apart: sample 1 is sample 0 shifted by a constant, and normalization
    /// removes a constant offset, so a per-sample result has to be identical
    /// for both. Grouping across the batch cannot produce that.
    #[test]
    fn group_norm_statistics_are_per_sample_not_across_the_batch() {
        let first: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let second: Vec<f32> = first.iter().map(|v| v + 100.0).collect();
        let data = first.iter().copied().chain(second).collect::<Vec<f32>>();
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(data), vec![2, 4, 1, 2]);

        let out = f32_vec(&TestBackend::group_norm::<f32>(&t, 2, 1e-5).unwrap());

        assert_eq!(out[..8], out[8..], "the two samples must normalize alike");
        // Group 0 of sample 0 is [0,1,2,3]: mean 1.5, population variance 1.25.
        let inv_std = 1.0 / (1.25f64 + 1e-5).sqrt();
        for (i, value) in [0.0f64, 1.0, 2.0, 3.0].iter().enumerate() {
            let expected = ((value - 1.5) * inv_std) as f32;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "element {i}: got {}, want {expected}",
                out[i]
            );
        }
    }

    /// `instance_norm` is `group_norm` with one group per channel, so each
    /// channel of each sample normalizes alone. A channel holding a single
    /// repeated value therefore has zero variance and normalizes to zero,
    /// whatever the other channels hold.
    #[test]
    fn instance_norm_normalizes_each_channel_of_each_sample_alone() {
        let t = CpuStorage::from_contiguous(
            CpuBuffer::F32(vec![
                1.0, 1.0, 5.0, 7.0, // sample 0: channel 0 flat, channel 1 varies
                2.0, 2.0, 9.0, 3.0, // sample 1: channel 0 flat, channel 1 varies
            ]),
            vec![2, 2, 2],
        );

        let out = f32_vec(&TestBackend::instance_norm::<f32>(&t, 1e-5).unwrap());

        for flat in [0, 1, 4, 5] {
            assert!(
                out[flat].abs() < 1e-5,
                "constant channel at {flat} must normalize to zero, got {}",
                out[flat]
            );
        }
        // A two-element channel normalizes to the symmetric pair -1, +1.
        assert!((out[2] + 1.0).abs() < 1e-3, "got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-3, "got {}", out[3]);
        assert!((out[6] - 1.0).abs() < 1e-3, "got {}", out[6]);
        assert!((out[7] + 1.0).abs() < 1e-3, "got {}", out[7]);
    }
    /// `scaled_dot_product_attention` is composed from `matmul`, `softmax` and
    /// `add`, so the f32 result it used to return for every operand dtype was
    /// the matmul mislabel showing through. Asserted here rather than only at
    /// matmul because this is the composition a caller actually reaches.
    #[test]
    fn attention_keeps_the_operand_dtype() {
        let operand =
            || CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 0.0, 0.0, 1.0]), vec![2, 2]);
        let out = TestBackend::scaled_dot_product_attention::<f64>(
            &operand(),
            &operand(),
            &operand(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(out.dtype, incin_core::prelude::DTypeId::F64.descriptor());
        assert_eq!(out.shape, vec![2, 2]);
    }
}
