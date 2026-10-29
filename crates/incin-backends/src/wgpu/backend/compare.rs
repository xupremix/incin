//! WGPU comparison and logical ops: the nine bool-output identities (#91).
//!
//! The six numeric comparisons (`cmp_eq`..`cmp_ge`) ride `binary.wgsl`'s
//! modes 7–12 and the two binary logicals ride modes 13/14; `logical_not`
//! rides `unary.wgsl` mode 34. Every one of those shader branches predates
//! this module — what was missing was the representation decision and the
//! executor: the catalog types all nine outputs `Bool`
//! (`catalog/descriptor.rs::trace_output_dtype`), while the kernels write
//! physical `f32` 0.0/1.0. That encoding is settled in
//! `wgpu/storage.rs::physical_element_bytes` and is exactly what
//! `masked_fill`/`where_cond` already consume as a mask, so each result
//! here is wrapped with `WgpuStorage::try_new_with_dtype(.., Bool)`: the
//! API-boundary dtype stays honest over the physical storage.
//!
//! Broadcasts go through `broadcast_storage_raw`, the non-recording form:
//! these operations push no `TapeEntry` (CPU and CUDA push none either —
//! `descriptor_training` resolves every one of them `false`, so a
//! training-mode invocation fails at admission long before this code), and
//! a dead entry a backward walk could never reach would only be a lie the
//! tape tells about itself. Operand dtypes are re-checked by name here
//! (`f32` for comparisons, `bool` for logicals) because a direct
//! `Execute` call bypasses the capability rows that already refuse the
//! other widths.

use super::*;
use crate::descriptor_bind::{invalid, kernel_error};

// `binary.wgsl` op_mode values, quoted from its header comment.
const CMP_EQ: u32 = 7;
const CMP_NE: u32 = 8;
const CMP_LT: u32 = 9;
const CMP_LE: u32 = 10;
const CMP_GT: u32 = 11;
const CMP_GE: u32 = 12;
const LOGICAL_AND: u32 = 13;
const LOGICAL_OR: u32 = 14;
// `unary.wgsl` op_mode value, quoted from its header comment.
const LOGICAL_NOT: u32 = 34;

/// Refuse any operand that is not `bool` for a logical operation.
fn require_bool(t: &WgpuStorage, op: &'static str) -> Result<()> {
    if t.dtype == DTypeId::Bool.descriptor() {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype: t.dtype,
            backend: "Wgpu",
            op,
        })
    }
}

/// One elementwise pass over two operands, producing `Bool` storage.
///
/// The shared body of the comparisons and the binary logicals: `check`
/// names the operand dtype (`f32` for comparisons, `bool` for logicals),
/// `mode` selects the `binary.wgsl` branch, and the result is freshly
/// allocated at the broadcast shape — `elements × 4` physical bytes under
/// a `Bool` label (see `physical_element_bytes`) — with nothing recorded
/// on the tape.
///
/// The broadcast uses the same resolution sequence
/// `elementwise.rs::binary_op` runs — equal shapes skip the kernel, a
/// mismatch resolves through `crate::layout::broadcast_shape` so an
/// incompatible pair fails with the axes named — except the stretch goes
/// through `broadcast_storage_raw`, the non-recording form: nothing
/// downstream of a non-differentiable op can ever walk back to these
/// temporaries, so pushing an entry a backward walk would never reach
/// would only be a lie the tape tells about itself.
fn elementwise_bool_out(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    mode: u32,
    op_name: &'static str,
    check: fn(&WgpuStorage, &'static str) -> Result<()>,
) -> Result<WgpuStorage> {
    check(lhs, op_name)?;
    check(rhs, op_name)?;
    let (lhs_owned, rhs_owned);
    let (lhs, rhs) = if lhs.shape == rhs.shape {
        (lhs, rhs)
    } else {
        let target = crate::layout::broadcast_shape(&lhs.shape, &rhs.shape).map_err(|_| {
            Error::ShapeMismatch {
                op: op_name,
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "operands do not broadcast against each other".to_string(),
            }
        })?;
        lhs_owned = if lhs.shape[..] == target[..] {
            None
        } else {
            Some(broadcast_storage_raw(lhs, &target)?)
        };
        rhs_owned = if rhs.shape[..] == target[..] {
            None
        } else {
            Some(broadcast_storage_raw(rhs, &target)?)
        };
        (
            lhs_owned.as_ref().unwrap_or(lhs),
            rhs_owned.as_ref().unwrap_or(rhs),
        )
    };
    let elements = num_elements(&lhs.shape)?;
    let n = checked_u32(elements, "WGPU comparison element count")?;
    let out_buf =
        WgpuBuffer::new_zeros_for(DTypeId::Bool.descriptor(), elements, OperationKind::Storage)?;
    let params = [mode, n];
    dispatch::dispatch_binary(&lhs.buffer, &rhs.buffer, &out_buf, &params)?;
    WgpuStorage::try_new_with_dtype(out_buf, lhs.shape.to_vec(), DTypeId::Bool.descriptor())
}

/// `cmp_eq`..`cmp_ge`: two `f32` operands in, `Bool` out.
fn compare(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    elementwise_bool_out(lhs, rhs, mode, op_name, require_f32)
}

/// `logical_and`/`logical_or`: two `bool` operands in, `Bool` out.
fn logical_binary(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    elementwise_bool_out(lhs, rhs, mode, op_name, require_bool)
}

impl<D: Device> WgpuBackendImpl<D> {
    /// `cmp_eq`: `lhs == rhs`, `Bool` out.
    pub(crate) fn cmp_eq(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_EQ, "cmp_eq")
    }
    /// `cmp_ne`: `lhs != rhs`, `Bool` out.
    pub(crate) fn cmp_ne(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_NE, "cmp_ne")
    }
    /// `cmp_lt`: `lhs < rhs`, `Bool` out.
    pub(crate) fn cmp_lt(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_LT, "cmp_lt")
    }
    /// `cmp_le`: `lhs <= rhs`, `Bool` out.
    pub(crate) fn cmp_le(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_LE, "cmp_le")
    }
    /// `cmp_gt`: `lhs > rhs`, `Bool` out.
    pub(crate) fn cmp_gt(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_GT, "cmp_gt")
    }
    /// `cmp_ge`: `lhs >= rhs`, `Bool` out.
    pub(crate) fn cmp_ge(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        compare(lhs, rhs, CMP_GE, "cmp_ge")
    }
    /// `logical_and`: both `bool` in, `Bool` out.
    pub(crate) fn logical_and(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        logical_binary(lhs, rhs, LOGICAL_AND, "logical_and")
    }
    /// `logical_or`: both `bool` in, `Bool` out.
    pub(crate) fn logical_or(lhs: &WgpuStorage, rhs: &WgpuStorage) -> Result<WgpuStorage> {
        logical_binary(lhs, rhs, LOGICAL_OR, "logical_or")
    }
    /// `logical_not`: one `bool` in, `Bool` out, through `unary.wgsl`.
    pub(crate) fn logical_not(input: &WgpuStorage) -> Result<WgpuStorage> {
        require_bool(input, "logical_not")?;
        let elements = num_elements(&input.shape)?;
        let n = checked_u32(elements, "WGPU logical_not element count")?;
        let out_buf = WgpuBuffer::new_zeros_for(
            DTypeId::Bool.descriptor(),
            elements,
            OperationKind::Storage,
        )?;
        let params = [LOGICAL_NOT, n];
        dispatch::dispatch_unary(&input.buffer, &out_buf, &params)?;
        WgpuStorage::try_new_with_dtype(out_buf, input.shape.to_vec(), DTypeId::Bool.descriptor())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute impls
// ─────────────────────────────────────────────────────────────────────────────

/// Two operands in, one `Bool` result out, no attributes and no tape —
/// the shared request shape of the six comparisons. Same structure as
/// CUDA's `impl_cuda_cmp!`, including the raw (non-recording) broadcast.
macro_rules! impl_wgpu_cmp {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> core::result::Result<WgpuStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs
                    .downcast_ref::<WgpuStorage>()
                    .ok_or_else(|| invalid(operation, "lhs is not WGPU storage"))?;
                let rhs = rhs
                    .downcast_ref::<WgpuStorage>()
                    .ok_or_else(|| invalid(operation, "rhs is not WGPU storage"))?;
                Self::$method(lhs, rhs).map_err(|e| kernel_error("Wgpu", operation, e))
            }
        }
    )*};
}

impl_wgpu_cmp![
    (CmpEq, cmp_eq),
    (CmpNe, cmp_ne),
    (CmpLt, cmp_lt),
    (CmpLe, cmp_le),
    (CmpGt, cmp_gt),
    (CmpGe, cmp_ge),
];

/// The two binary logicals: identical request shape to the comparisons
/// above, `bool` operands instead of `f32` — CUDA's
/// `impl_cuda_logical_binary!` with the downcasts spelled the WGPU way.
macro_rules! impl_wgpu_logical_binary {
    ($(($op:ident, $method:ident)),* $(,)?) => {$(
        impl<D: Device> Execute<op::$op> for WgpuBackendImpl<D> {
            type Output = WgpuStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$op, Self>,
            ) -> core::result::Result<WgpuStorage, BackendError> {
                let operation = OperationKind::$op;
                let [lhs, rhs] = request.inputs else {
                    return Err(invalid(operation, "operation expects exactly two operands"));
                };
                let lhs = lhs
                    .downcast_ref::<WgpuStorage>()
                    .ok_or_else(|| invalid(operation, "lhs is not WGPU storage"))?;
                let rhs = rhs
                    .downcast_ref::<WgpuStorage>()
                    .ok_or_else(|| invalid(operation, "rhs is not WGPU storage"))?;
                Self::$method(lhs, rhs).map_err(|e| kernel_error("Wgpu", operation, e))
            }
        }
    )*};
}

impl_wgpu_logical_binary![(LogicalAnd, logical_and), (LogicalOr, logical_or),];

impl<D: Device> Execute<op::LogicalNot> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::LogicalNot, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::LogicalNot;
        let [input] = request.inputs else {
            return Err(invalid(operation, "operation expects exactly one operand"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        Self::logical_not(input).map_err(|e| kernel_error("Wgpu", operation, e))
    }
}
