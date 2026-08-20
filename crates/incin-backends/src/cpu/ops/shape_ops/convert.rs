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
