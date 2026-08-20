use super::*;

pub(crate) fn dense_range(
    storage: &CpuStorage,
    buffer_len: usize,
    output_shape: &[usize],
) -> Option<Range<usize>> {
    if storage.shape.dims() != output_shape
        || !stride::is_contiguous(&storage.shape, &storage.strides)
    {
        return None;
    }
    let end = storage
        .offset_elements
        .checked_add(try_numel(output_shape)?)?;
    (end <= buffer_len).then_some(storage.offset_elements..end)
}

pub(super) fn scalar_value<T: Copy>(storage: &CpuStorage, values: &[T]) -> Option<T> {
    if try_numel(&storage.shape)? != 1 {
        return None;
    }
    values.get(storage.offset_elements).copied()
}

pub(super) fn validate_bounds(
    operand: &OperandIteration,
    output_shape: &[usize],
    buffer_len: usize,
) -> Result<()> {
    if let Some(max_index) = operand.max_physical_index(output_shape)?
        && max_index >= buffer_len
    {
        return Err(Error::Msg(format!(
            "iteration plan accesses storage index {max_index}, but buffer length is {buffer_len}"
        )));
    }
    Ok(())
}

/// The element count of `shape`, or `None` on overflow.
///
/// This is a fast-path check for two `Option`-returning callers that treat
/// overflow as "this shortcut does not apply" rather than as an error to
/// report; [`crate::bytes::checked_numel`] is the crate's answer to the
/// question "what is this shape's element count, and is it representable".
fn try_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |numel, &dim| numel.checked_mul(dim))
}

pub(super) fn erf_approx_f64(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let value = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * value);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-value * value).exp())
}
