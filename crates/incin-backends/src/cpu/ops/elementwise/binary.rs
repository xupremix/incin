use super::*;

pub(crate) fn canonical_fmod(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary(lhs, rhs, &lhs.shape, |a, b| a % b)
}

pub(crate) fn canonical_remainder(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary(lhs, rhs, &lhs.shape, |a, b| a.rem_euclid(b))
}

pub(crate) fn canonical_atan2(y: &CpuStorage, x: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_binary(y, x, &y.shape, |y_value, x_value| y_value.atan2(x_value))?;
    let (y_id, x_id, out_id) = (y.id, x.id, out.id);
    let (y_capture, x_capture) = (y.clone(), x.clone());
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![y_id, x_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let x2 =
                elementwise_binary_numeric(BinaryOp::Mul, &x_capture, &x_capture, &grad_out.shape)?;
            let y2 =
                elementwise_binary_numeric(BinaryOp::Mul, &y_capture, &y_capture, &grad_out.shape)?;
            let denominator = elementwise_binary_numeric(BinaryOp::Add, &x2, &y2, &grad_out.shape)?;
            let numer_y =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &x_capture, &grad_out.shape)?;
            let grad_y =
                elementwise_binary_numeric(BinaryOp::Div, &numer_y, &denominator, &grad_out.shape)?;
            let neg_y = negate(&y_capture);
            let numer_x =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &neg_y, &grad_out.shape)?;
            let grad_x =
                elementwise_binary_numeric(BinaryOp::Div, &numer_x, &denominator, &grad_out.shape)?;
            Ok(vec![
                tape::unbroadcast(&grad_y, &y_capture.shape)?,
                tape::unbroadcast(&grad_x, &x_capture.shape)?,
            ])
        }),
    });
    Ok(out)
}

// The four pointwise binary kernels below are free functions rather than trait
// bodies because two entry points now need them: `` for the legacy
// tensor surface, and the canonical `Execute<op::Add>` executor in
// `cpu::canonical`. Keeping one body means the descriptor path cannot drift
// from the path it is replacing, and when `` is deleted these
// functions stay exactly as they are.

/// Broadcast elementwise addition, with its gradient recorded.
pub(crate) fn add_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    add_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`add_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn add_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Add, lhs, rhs, out_shape)?;

    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: {
            // Allocated here rather than at the call site: an unrecorded
            // operation must not pay for the shapes its gradient would need.
            let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
            Box::new(move |grad_out: &CpuStorage| {
                Ok(vec![
                    tape::unbroadcast(grad_out, &lhs_shape)?,
                    tape::unbroadcast(grad_out, &rhs_shape)?,
                ])
            })
        },
    });
    Ok(out)
}

/// Broadcast elementwise subtraction, with its gradient recorded.
pub(crate) fn sub_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    sub_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`sub_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn sub_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Sub, lhs, rhs, out_shape)?;

    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| {
        // Allocated inside the closure so an unrecorded operation does not
        // pay for the shapes its gradient would have needed.
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                Ok(vec![
                    tape::unbroadcast(grad_out, &lhs_shape)?,
                    tape::unbroadcast(&negate(grad_out), &rhs_shape)?,
                ])
            }),
        }
    });
    Ok(out)
}

/// Broadcast elementwise multiplication, with its gradient recorded.
pub(crate) fn mul_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    mul_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`mul_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn mul_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Mul, lhs, rhs, out_shape)?;

    // Capture cloned copies of lhs/rhs's CpuStorage (cheap, Rc-backed)
    // since the backward closure needs their VALUES, not just shapes.
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| {
        // Allocated inside the closure so an unrecorded operation does not
        // pay for the shapes its gradient would have needed.
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad_lhs = elementwise_binary_numeric(
                    BinaryOp::Mul,
                    grad_out,
                    &rhs_capture,
                    &grad_out.shape,
                )?;
                let grad_rhs = elementwise_binary_numeric(
                    BinaryOp::Mul,
                    grad_out,
                    &lhs_capture,
                    &grad_out.shape,
                )?;
                Ok(vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        }
    });
    Ok(out)
}

/// Broadcast elementwise division, with its gradient recorded.
pub(crate) fn div_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    div_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`div_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn div_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Div, lhs, rhs, out_shape)?;

    // Per Assumption A2 (RESEARCH.md): implemented for trait-completeness
    // via the standard quotient rule (1/rhs, -lhs/rhs^2), each
    // unbroadcast — best-effort correctness, not exercised by this
    // phase's example/tests.
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push_with(|| {
        // Allocated inside the closure so an unrecorded operation does not
        // pay for the shapes its gradient would have needed.
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs = elementwise_binary_numeric(
                    BinaryOp::Div,
                    grad_out,
                    &rhs_capture,
                    &grad_out.shape,
                )?;
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = -grad_lhs * (lhs/rhs)
                let lhs_div_rhs = elementwise_binary_numeric(
                    BinaryOp::Div,
                    &lhs_capture,
                    &rhs_capture,
                    &grad_out.shape,
                )?;
                let neg_grad_lhs = negate(&grad_lhs);
                let grad_rhs = elementwise_binary_numeric(
                    BinaryOp::Mul,
                    &neg_grad_lhs,
                    &lhs_div_rhs,
                    &grad_out.shape,
                )?;
                Ok(vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        }
    });
    Ok(out)
}
