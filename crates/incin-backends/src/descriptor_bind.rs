//! Descriptor-to-metadata checks shared by every native executor.
//!
//! A binder's job is to prove that the storage it was handed is the storage the
//! sealed descriptor authorizes. For matmul each backend spells that out for
//! itself, because the checks are entangled with the backend's own transpose
//! handling. Convolution's are not: they are pure arithmetic on `TensorMeta`
//! and `Conv2dSpec`, identical on every device, and three copies of them would
//! be three places for the channel-group arithmetic to drift.
//!
//! Device residency and dtype stay with each backend, since those messages name
//! the backend that refused.

use alloc::string::ToString;

use incin_core::exec::{
    Conv2dSpec, Pool2dSpec, PoolOp, ReductionSpec, TensorMeta, UnsupportedReason,
};
use incin_core::prelude::{BackendError, Error, OperationKind};

/// Build an `InvalidInput` error for a descriptor binder.
pub(crate) const fn invalid(operation: OperationKind, reason: &'static str) -> BackendError {
    BackendError::InvalidInput { operation, reason }
}

/// Classify a legacy kernel's failure for the descriptor path.
///
/// The op traits report a gap the backend declared at its own impl site as
/// [`Error::UnsupportedBackendOperation`]. That is a capability answer, not an
/// execution failure: nothing ran and nothing faulted, the backend simply never
/// had a kernel. Reporting it as [`BackendError::Execution`] would tell a caller
/// the device failed, and would split "this backend cannot do that" across two
/// error variants depending on whether the registry or the kernel noticed first.
/// It becomes the same [`BackendError::Unsupported`] a capability query returns.
pub(crate) fn kernel_error(operation: OperationKind, error: Error) -> BackendError {
    match error {
        Error::UnsupportedBackendOperation { .. } => {
            UnsupportedReason::Operation { operation }.into()
        }
        other => BackendError::Execution {
            operation,
            message: other.to_string(),
        },
    }
}

/// The isotropic window the legacy `ModuleOps::conv2d` entry point accepts.
#[derive(Debug)]
pub(crate) struct Conv2dWindow {
    /// Stride shared by both spatial axes.
    pub stride: usize,
    /// Padding shared by both spatial axes.
    pub padding: usize,
    /// Dilation shared by both spatial axes.
    pub dilation: usize,
    /// Channel groups.
    pub groups: usize,
}

/// Reduce a descriptor's per-axis window to the one the legacy kernel takes.
///
/// `Conv2dSpec` records a value per spatial axis, while `ModuleOps::conv2d`
/// takes one value for both. Every descriptor `Conv2dRule` lowers is isotropic —
/// the typed frontend fixes the window with a single `typenum` per parameter —
/// but `Conv2dSpec::new` is public and an anisotropic one is constructible. It
/// is refused here rather than silently executed with one axis' value applied
/// to both.
pub(crate) fn conv2d_window(spec: &Conv2dSpec) -> Result<Conv2dWindow, BackendError> {
    for ([first, second], reason) in [
        (
            spec.kernel,
            "conv2d kernel extents differ per axis; the routed kernel takes one window for both",
        ),
        (
            spec.stride,
            "conv2d strides differ per axis; the routed kernel takes one stride for both",
        ),
        (
            spec.padding,
            "conv2d paddings differ per axis; the routed kernel takes one padding for both",
        ),
        (
            spec.dilation,
            "conv2d dilations differ per axis; the routed kernel takes one dilation for both",
        ),
    ] {
        if first != second {
            return Err(invalid(OperationKind::Conv2d, reason));
        }
    }

    Ok(Conv2dWindow {
        stride: spec.stride[0],
        padding: spec.padding[0],
        dilation: spec.dilation[0],
        groups: spec.groups,
    })
}

/// Check convolution operand metadata against the sealed descriptor.
///
/// The weight is checked against `c_in / groups` rather than `c_in`:
/// `Conv2dSpec::new` has already rejected a `groups` that does not divide both
/// channel counts, so this division is exact and the descriptor is the authority
/// on the filter bank's shape.
pub(crate) fn check_conv2d_operands(
    spec: &Conv2dSpec,
    input: &TensorMeta,
    weight: &TensorMeta,
    bias: Option<&TensorMeta>,
) -> Result<(), BackendError> {
    if input.shape().dims() != [spec.n, spec.c_in, spec.h_in, spec.w_in] {
        return Err(invalid(
            OperationKind::Conv2d,
            "conv2d input metadata does not match the validated descriptor",
        ));
    }
    if weight.shape().dims()
        != [
            spec.c_out,
            spec.c_in / spec.groups,
            spec.kernel[0],
            spec.kernel[1],
        ]
    {
        return Err(invalid(
            OperationKind::Conv2d,
            "conv2d weight metadata does not match the validated descriptor",
        ));
    }
    if let Some(bias) = bias
        && bias.shape().dims() != [spec.c_out]
    {
        return Err(invalid(
            OperationKind::Conv2d,
            "conv2d bias metadata does not match the validated descriptor",
        ));
    }

    Ok(())
}

/// The window the legacy pooling entry points accept.
///
/// Per-axis throughout, unlike [`Conv2dWindow`]: `ModuleOps::max_pool2d` takes a
/// `(usize, usize)` for each parameter, so a descriptor's anisotropic window
/// routes as it stands and nothing has to be flattened.
#[derive(Debug)]
pub(crate) struct Pool2dWindow {
    /// Window extent, `(height, width)`.
    pub kernel: (usize, usize),
    /// Stride, `(height, width)`.
    pub stride: (usize, usize),
    /// Padding, `(height, width)`.
    pub padding: (usize, usize),
    /// Dilation, `(height, width)`.
    pub dilation: (usize, usize),
}

/// Reduce a descriptor's window to the one the legacy pooling kernel takes.
///
/// Refuses a dilated average pool. `ModuleOps::avg_pool2d` has no dilation
/// parameter at all, so there is no value to pass one as; executing the
/// descriptor anyway would silently average a dense window where the caller
/// asked for a dilated one, and produce a plausible wrong answer rather than an
/// error.
pub(crate) fn pool2d_window(spec: &Pool2dSpec) -> Result<Pool2dWindow, BackendError> {
    if spec.op == PoolOp::Average && spec.dilation != [1, 1] {
        return Err(invalid(
            OperationKind::Pool2d,
            "average pooling has no dilated form; the routed kernel takes no dilation",
        ));
    }
    let pair = |[height, width]: [usize; 2]| (height, width);

    Ok(Pool2dWindow {
        kernel: pair(spec.kernel),
        stride: pair(spec.stride),
        padding: pair(spec.padding),
        dilation: pair(spec.dilation),
    })
}

/// Check the pooling operand's metadata against the sealed descriptor.
pub(crate) fn check_pool2d_operand(
    spec: &Pool2dSpec,
    input: &TensorMeta,
) -> Result<(), BackendError> {
    if input.shape().dims() != [spec.n, spec.channels, spec.h_in, spec.w_in] {
        return Err(invalid(
            OperationKind::Pool2d,
            "pool2d input metadata does not match the validated descriptor",
        ));
    }
    Ok(())
}

/// The single axis a routed reduction collapses.
///
/// `ReductionSpec` describes a contiguous *run* of axes, while every legacy
/// reduction entry point takes one `dim`. The lowering rules only ever produce a
/// one-axis run — `ReduceDim<D>` names a single axis — so a wider run reaching a
/// backend means the descriptor came from somewhere else, and collapsing it with
/// repeated single-axis calls would change the accumulation order for `mean` and
/// the intermediate range for `prod`. It is refused instead.
pub(crate) fn reduction_axis(spec: &ReductionSpec) -> Result<usize, BackendError> {
    let mut axes = spec.axes.axes();
    let (Some(axis), None) = (axes.next(), axes.next()) else {
        return Err(invalid(
            OperationKind::Reduction,
            "the routed kernel reduces one axis; this descriptor names none or several",
        ));
    };
    Ok(axis)
}

/// Check the reduction operand's metadata against the sealed descriptor.
///
/// `ReductionSpec` does not keep the input shape, only the three extents it
/// collapsed into. Those are enough: the reduced axis must be exactly `reduced`
/// long, and everything before and after it must multiply out to `outer` and
/// `inner`. An operand that satisfies all three is one the descriptor's loop
/// nest walks correctly.
pub(crate) fn check_reduction_operand(
    spec: &ReductionSpec,
    axis: usize,
    input: &TensorMeta,
) -> Result<(), BackendError> {
    let dims = input.shape().dims();
    let region = |run: &[usize]| {
        run.iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    };

    if dims.get(axis) != Some(&spec.reduced)
        || region(&dims[..axis]) != Some(spec.outer)
        || region(&dims[axis + 1..]) != Some(spec.inner)
    {
        return Err(invalid(
            OperationKind::Reduction,
            "reduction input metadata does not match the validated descriptor",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use incin_core::exec::ReduceOp;
    use incin_core::prelude::ShapeBuf;

    use super::*;

    fn spec() -> Conv2dSpec {
        Conv2dSpec::new(
            &ShapeBuf::from_slice(&[1, 4, 5, 5]),
            6,
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            2,
        )
        .expect("a dense 3x3 grouped convolution is valid geometry")
    }

    #[test]
    fn an_anisotropic_window_is_refused_rather_than_flattened() {
        let mut anisotropic = spec();
        anisotropic.stride = [1, 2];

        let error = conv2d_window(&anisotropic).expect_err("per-axis strides must not route");

        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::Conv2d,
                reason: "conv2d strides differ per axis; the routed kernel takes one stride for both"
            }
        ));
    }

    #[test]
    fn the_weight_is_checked_against_the_grouped_channel_count() {
        let spec = spec();
        let meta = |dims: &[usize]| {
            TensorMeta::contiguous(
                ShapeBuf::from_slice(dims),
                incin_core::prelude::DTypeId::F32,
                incin_core::prelude::DeviceId::cpu(),
                incin_core::exec::Alignment::of::<f32>(),
                dims.iter().product(),
            )
            .expect("test metadata must be valid")
        };

        // 4 input channels in 2 groups means each filter sees 2 of them.
        let input = meta(&[1, 4, 5, 5]);
        let grouped = meta(&[6, 2, 3, 3]);
        let ungrouped = meta(&[6, 4, 3, 3]);

        check_conv2d_operands(&spec, &input, &grouped, None)
            .expect("a correctly grouped weight must bind");
        let error = check_conv2d_operands(&spec, &input, &ungrouped, None)
            .expect_err("an ungrouped weight must not bind");
        assert!(matches!(
            error,
            BackendError::InvalidInput {
                reason: "conv2d weight metadata does not match the validated descriptor",
                ..
            }
        ));
    }

    #[test]
    fn a_bias_is_checked_only_when_one_is_supplied() {
        let spec = spec();
        let meta = |dims: &[usize]| {
            TensorMeta::contiguous(
                ShapeBuf::from_slice(dims),
                incin_core::prelude::DTypeId::F32,
                incin_core::prelude::DeviceId::cpu(),
                incin_core::exec::Alignment::of::<f32>(),
                dims.iter().product(),
            )
            .expect("test metadata must be valid")
        };
        let input = meta(&[1, 4, 5, 5]);
        let weight = meta(&[6, 2, 3, 3]);

        check_conv2d_operands(&spec, &input, &weight, None).expect("no bias is legal");
        check_conv2d_operands(&spec, &input, &weight, Some(&meta(&[6])))
            .expect("a bias of c_out is legal");
        let error = check_conv2d_operands(&spec, &input, &weight, Some(&meta(&[4])))
            .expect_err("a bias of the wrong width must not bind");
        assert!(matches!(
            error,
            BackendError::InvalidInput {
                reason: "conv2d bias metadata does not match the validated descriptor",
                ..
            }
        ));
    }

    fn pool(op: PoolOp, dilation: [usize; 2]) -> Pool2dSpec {
        Pool2dSpec::new(
            &ShapeBuf::from_slice(&[1, 3, 8, 8]),
            [2, 2],
            [2, 2],
            [0, 0],
            dilation,
            op,
        )
        .expect("a 2x2 window strided by 2 tiles an 8x8 input")
    }

    #[test]
    fn a_dilated_average_pool_is_refused_because_the_kernel_cannot_express_it() {
        // The same dilation is fine for max pooling, whose entry point takes one.
        pool2d_window(&pool(PoolOp::Max, [2, 2])).expect("max pooling is dilatable");

        let error = pool2d_window(&pool(PoolOp::Average, [2, 2]))
            .expect_err("average pooling has nowhere to put a dilation");

        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::Pool2d,
                reason: "average pooling has no dilated form; the routed kernel takes no dilation"
            }
        ));
    }

    #[test]
    fn a_pooling_window_keeps_both_axes_rather_than_flattening_them() {
        let mut anisotropic = pool(PoolOp::Max, [1, 1]);
        anisotropic.stride = [2, 1];

        let window = pool2d_window(&anisotropic).expect("the kernel takes a stride per axis");

        assert_eq!(window.stride, (2, 1));
        assert_eq!(window.kernel, (2, 2));
    }

    #[test]
    fn a_reduction_over_several_axes_does_not_route_to_a_one_axis_kernel() {
        let input = ShapeBuf::from_slice(&[2, 3, 4]);
        let single = ReductionSpec::over_axes(&input, [1], false, ReduceOp::Sum).unwrap();
        assert_eq!(reduction_axis(&single).expect("one axis routes"), 1);

        for axes in [vec![], vec![0, 1]] {
            let spec = ReductionSpec::over_axes(&input, axes, false, ReduceOp::Sum).unwrap();
            let error = reduction_axis(&spec).expect_err("only a one-axis run routes");
            assert!(matches!(
                error,
                BackendError::InvalidInput {
                    operation: OperationKind::Reduction,
                    ..
                }
            ));
        }
    }

    #[test]
    fn the_operand_is_checked_against_the_three_extents_not_a_stored_shape() {
        let meta = |dims: &[usize]| {
            TensorMeta::contiguous(
                ShapeBuf::from_slice(dims),
                incin_core::prelude::DTypeId::F32,
                incin_core::prelude::DeviceId::cpu(),
                incin_core::exec::Alignment::of::<f32>(),
                dims.iter().product(),
            )
            .expect("test metadata must be valid")
        };
        let spec =
            ReductionSpec::over_axes(&ShapeBuf::from_slice(&[2, 3, 4]), [1], false, ReduceOp::Sum)
                .unwrap();

        check_reduction_operand(&spec, 1, &meta(&[2, 3, 4])).expect("the original operand binds");
        // Same element count, wrong axis lengths: `outer` and `inner` catch it
        // where a bare `numel` comparison would not.
        let error = check_reduction_operand(&spec, 1, &meta(&[4, 3, 2]))
            .expect_err("a permuted operand must not bind");
        assert!(matches!(
            error,
            BackendError::InvalidInput {
                reason: "reduction input metadata does not match the validated descriptor",
                ..
            }
        ));
    }
}
