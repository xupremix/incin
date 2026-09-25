use super::*;

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

/// Convert dtype and, when both sides are floating, record the conversion.
///
/// A mid-graph dtype change is an ordinary research move: cast to `f64` for a
/// numerically delicate step, or to `bf16` for a cheap one, and cast back.
/// Until this recorded, doing that detached the graph silently. Everything
/// upstream of the cast received no gradient at all, and nothing said so.
///
/// The rule is the identity, in the input's dtype: `d(cast(x))/dx = 1`, so the
/// incoming gradient is passed through and cast back to what the input was.
/// The cast itself is lossy in one direction (`f64` to `bf16` and back does
/// not round-trip), and that loss is the honest gradient of a lossy forward.
///
/// Only float to float records. A cast to an integer dtype truncates, so its
/// derivative is zero almost everywhere, and passing a gradient through one
/// would report a sensitivity the forward pass does not have.
pub(crate) fn canonical_to_dtype(t: &CpuStorage, dtype: DTypeDescriptor) -> Result<CpuStorage> {
    let source = t.metadata().dtype();
    let out = tensor_to_dtype_storage(t, dtype)?;

    let both_float = source.builtin_id().is_some_and(DTypeId::is_float)
        && dtype.builtin_id().is_some_and(DTypeId::is_float);
    if both_float {
        let (input_id, out_id) = (t.id, out.id);
        crate::cpu::tape::push_with(|| crate::cpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![input_id],
            backward: alloc::boxed::Box::new(move |grad_out: &CpuStorage| {
                Ok(alloc::vec![tensor_to_dtype_storage(grad_out, source)?])
            }),
        });
    }
    Ok(out)
}

pub(crate) fn tensor_to_dtype_storage(
    t: &CpuStorage,
    dtype: DTypeDescriptor,
) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut multi_idx = vec![0usize; t.shape.len()];
    macro_rules! convert_variant {
        ($variant:ident, $ty:ty) => {{
            let mut out: Vec<$ty> = Vec::with_capacity(total);
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
            let mut out = Vec::with_capacity(total);
            for _ in 0..total {
                out.push(half::f16::from_f64(t.get(&multi_idx)));
                if !t.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                }
            }
            CpuBuffer::F16(out)
        }
        Some(DTypeId::BF16) => {
            let mut out = Vec::with_capacity(total);
            for _ in 0..total {
                out.push(half::bf16::from_f64(t.get(&multi_idx)));
                if !t.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                }
            }
            CpuBuffer::BF16(out)
        }
        Some(DTypeId::F8E4M3) => {
            let mut out = Vec::with_capacity(total);
            for _ in 0..total {
                out.push(incin_core::tensor::dtype::F8E4M3::from_f64(
                    t.get(&multi_idx),
                ));
                if !t.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                }
            }
            CpuBuffer::F8E4M3(out)
        }
        Some(DTypeId::F8E5M2) => {
            let mut out = Vec::with_capacity(total);
            for _ in 0..total {
                out.push(incin_core::tensor::dtype::F8E5M2::from_f64(
                    t.get(&multi_idx),
                ));
                if !t.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                }
            }
            CpuBuffer::F8E5M2(out)
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
    Ok(CpuStorage::from_contiguous(new_buffer, &t.shape))
}
