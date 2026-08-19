use super::*;

/// Borrowed shape attributes used by the common, storage-free inference path.
/// This is deliberately not public API: callers provide typed attributes and
/// cannot choose a transform independently of the descriptor type.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub enum ShapeTransform<'a> {
    Axis(usize),
    Transpose(usize, usize),
    Narrow {
        axis: usize,
        length: usize,
    },
    Slice(&'a [(usize, usize)]),
    Flatten {
        start: usize,
        end: usize,
    },
    Repeat(&'a [usize]),
    Pad(&'a [(usize, usize)]),
    Diagonal(i64),
    Unfold {
        axis: usize,
        size: usize,
        step: usize,
    },
    PixelShuffle(usize),
    AdaptivePool2d([usize; 2]),
    TopK {
        axis: usize,
        k: usize,
    },
    Chunk {
        chunks: usize,
        axis: usize,
    },
    Split {
        split_size: usize,
        axis: usize,
    },
    Conv1d(&'a Conv1dAttributes),
    Conv2d(&'a Conv2dAttributes),
    ConvTranspose2d(&'a ConvTranspose2dAttributes),
    Pool2d(&'a Pool2dAttributes),
    AvgPool2d(&'a AvgPool2dAttributes),
    Rnn(&'a RecurrentAttributes),
}

pub(super) fn invalid(
    operation: OperationKind,
    attribute: &'static str,
    reason: &'static str,
) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation,
        attribute,
        reason,
    }
}

pub(super) fn first_shape(inputs: &[LogicalTensorMeta]) -> Option<&[usize]> {
    inputs.first()?.shape.as_deref()
}

/// An index-producing operation must declare an integer output dtype.
///
/// `argmax`, `argmin`, `topk`, and `argsort` carry their index dtype as a typed
/// attribute so the descriptor can infer the output exactly. Nothing else
/// constrained that field, so a caller could declare `F32` and have the
/// descriptor certify a floating-point "index" tensor. The family default
/// (`DTypeRule::IndexResult`) states the intent; this enforces it.
pub(super) fn validate_index_dtype(
    operation: OperationKind,
    attribute: &'static str,
    dtype: DTypeDescriptor,
) -> Result<(), DescriptorError> {
    if dtype.is_integer() {
        return Ok(());
    }
    Err(invalid(
        operation,
        attribute,
        "an index output requires an integer dtype",
    ))
}

/// Reject an unbiased (Bessel-corrected) estimate over fewer than two elements.
///
/// The correction divides by `n - 1`. The `Reduction` family default only
/// rejects an *empty* domain, which is not enough here: a single-element domain
/// is non-empty and still degenerate. Refusing it in the descriptor keeps the
/// division out of every backend.
pub(super) fn validate_unbiased_domain(
    operation: OperationKind,
    unbiased: bool,
    extent: Option<usize>,
) -> Result<(), DescriptorError> {
    if !unbiased {
        return Ok(());
    }
    match extent {
        Some(count) if count < 2 => Err(invalid(
            operation,
            "unbiased",
            "an unbiased variance or standard deviation requires at least two elements",
        )),
        _ => Ok(()),
    }
}

pub(super) fn validate_shape(
    operation: OperationKind,
    shape: &[usize],
) -> Result<(), DescriptorError> {
    crate::shapes::ShapeBuf::from_slice(shape).checked_numel(operation)?;
    Ok(())
}

macro_rules! unconstrained_attributes {
    ($($ty:ty),* $(,)?) => {$(
        impl AttributeContract for $ty {
            fn validate(&self, _operation: OperationKind, _inputs: &[LogicalTensorMeta]) -> Result<(), DescriptorError> { Ok(()) }
        }
    )*};
}

unconstrained_attributes!(NoAttributes, ScalarAttributes, LerpAttributes,);
