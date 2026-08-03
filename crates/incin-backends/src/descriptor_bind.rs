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

use incin_core::backend_authoring::{ReductionOps, TensorOps};
use incin_core::exec::{
    Conv2dSpec, Pool2dSpec, PoolOp, ReduceOp, ReductionSpec, TensorMeta, UnsupportedReason,
};
use incin_core::prelude::{Backend, BackendError, DType, Error, OperationKind};

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
            message: other.to_string().into(),
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

/// The contiguous axis run a reduction collapses, or `None` for no axes.
///
/// `ReductionSpec::new` rejects a non-contiguous axis set, so a descriptor that
/// names any axes names a half-open range. Returning the endpoints rather than a
/// single axis is what lets a backend collapse a wider run; an empty set is
/// legal and reduces nothing, which the spec's own constructor documents.
pub(crate) fn reduction_run(spec: &ReductionSpec) -> Result<Option<(usize, usize)>, BackendError> {
    let mut axes = spec.axes.axes();
    let Some(first) = axes.next() else {
        return Ok(None);
    };
    let last = axes.last().unwrap_or(first);
    if last < first {
        return Err(invalid(
            OperationKind::Reduction,
            "reduction axis run is not ordered",
        ));
    }
    Ok(Some((first, last + 1)))
}

/// Collapse a run of two or more axes with repeated single-axis calls.
///
/// This was previously refused, on the grounds that repeating a single-axis call
/// would change the accumulation order for `mean` and the intermediate range for
/// `prod`. Refusing it left a hole instead: `ReductionSpec` accepts any
/// contiguous run and the lowering rules can produce one, so a validated
/// descriptor existed that no backend would execute.
///
/// Every accumulation `ReduceOp` names is associative, so collapsing the axes in
/// sequence gives the same result as collapsing them at once in exact
/// arithmetic. `Mean` is included: averaging over a run of length `a` and then
/// one of length `b` divides by `a` then by `b`, which is a division by `a * b`.
/// What differs is floating-point rounding, since the divisions and the
/// summation groupings happen in a different order, and that is a ULP-level
/// difference rather than a semantic one.
///
/// Each step keeps the axis it reduced as length 1, which is what makes the loop
/// safe: axis indices never shift underneath it, so the run can be walked in
/// order without renumbering. The single reshape at the end drops those axes
/// when the descriptor asked for them dropped, moving no element.
pub(crate) fn reduce_axis_run<B, K>(
    spec: &ReductionSpec,
    input: &B::Storage<K>,
    input_dims: &[usize],
    start: usize,
    end: usize,
) -> Result<B::Storage<K>, Error>
where
    B: Backend + ReductionOps<B> + TensorOps<B>,
    K: DType,
{
    let mut dims = input_dims.to_vec();
    let mut current: Option<B::Storage<K>> = None;

    for axis in start..end {
        let source = current.as_ref().unwrap_or(input);
        let reduced = match spec.op {
            ReduceOp::Sum => <B as ReductionOps<B>>::sum_keepdim::<K>(source, axis)?,
            ReduceOp::Mean => <B as ReductionOps<B>>::mean_keepdim::<K>(source, axis)?,
            ReduceOp::Max => <B as ReductionOps<B>>::max_keepdim::<K>(source, axis)?,
            ReduceOp::Min => <B as ReductionOps<B>>::min_keepdim::<K>(source, axis)?,
            // `ReductionOps` has no `prod_keepdim`, so the axis is dropped and
            // reinserted. The reshape moves no element, which is why this is the
            // same operation rather than an approximation of it.
            ReduceOp::Prod => {
                let dropped = <B as ReductionOps<B>>::prod_dim::<K>(source, axis)?;
                let mut kept = dims.clone();
                kept[axis] = 1;
                <B as TensorOps<B>>::reshape::<K>(&dropped, &kept)?
            }
        };
        dims[axis] = 1;
        current = Some(reduced);
    }

    let collapsed = match current {
        Some(storage) => storage,
        // An empty run reduces nothing; the reshape below still normalizes the
        // handle to the descriptor's output shape.
        None => input.clone(),
    };
    if dims == spec.output.dims() {
        return Ok(collapsed);
    }
    <B as TensorOps<B>>::reshape::<K>(&collapsed, spec.output.dims())
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
    run: Option<(usize, usize)>,
    input: &TensorMeta,
) -> Result<(), BackendError> {
    let dims = input.shape().dims();
    let region = |run: &[usize]| {
        run.iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    };

    // An empty run collapses nothing, so the spec records the whole input as
    // `outer` with `reduced` and `inner` both 1.
    let (start, end) = run.unwrap_or((dims.len(), dims.len()));
    if end > dims.len()
        || region(&dims[start..end]) != Some(spec.reduced)
        || region(&dims[..start]) != Some(spec.outer)
        || region(&dims[end..]) != Some(spec.inner)
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

    /// Every axis set `ReductionSpec` accepts resolves to a run a backend can walk.
    ///
    /// The previous binder answered only a one-axis run and refused the rest,
    /// which meant a descriptor the schema had already validated reached a
    /// backend that would not execute it. The endpoints are what a caller needs,
    /// and an empty set is `None` rather than an error because reducing no axes
    /// is a legal identity that `ReductionSpec::new` documents.
    #[test]
    fn every_axis_set_the_schema_accepts_resolves_to_a_run() {
        let input = ShapeBuf::from_slice(&[2, 3, 4]);

        for (axes, expected) in [
            (vec![1], Some((1, 2))),
            (vec![0, 1], Some((0, 2))),
            (vec![0, 1, 2], Some((0, 3))),
            (vec![], None),
        ] {
            let spec = ReductionSpec::over_axes(&input, axes.clone(), false, ReduceOp::Sum)
                .expect("a contiguous run is a legal descriptor");
            assert_eq!(
                reduction_run(&spec).expect("a validated run resolves"),
                expected,
                "axes {axes:?}"
            );
        }
    }

    /// The operand check follows the whole run, not one axis of it.
    ///
    /// A two-axis run over `[2, 3, 4]` collapses `2 * 3` into `reduced`, leaves
    /// `outer` empty and `inner` at 4. Checking only the first axis would accept
    /// an operand whose second reduced axis disagreed with the descriptor.
    #[test]
    fn the_operand_check_spans_the_whole_reduced_run() {
        let input = ShapeBuf::from_slice(&[2, 3, 4]);
        let spec = ReductionSpec::over_axes(&input, [0, 1], false, ReduceOp::Sum).unwrap();
        let run = reduction_run(&spec).unwrap();
        assert_eq!((spec.outer, spec.reduced, spec.inner), (1, 6, 4));

        let meta = |dims: &[usize]| {
            TensorMeta::contiguous(
                ShapeBuf::from_slice(dims),
                incin_core::prelude::DTypeId::F32,
                incin_core::prelude::DeviceId::cpu(),
                incin_core::exec::Alignment::of::<f32>(),
                dims.iter().product(),
            )
            .unwrap()
        };

        check_reduction_operand(&spec, run, &meta(&[2, 3, 4])).expect("the descriptor's own shape");
        let error = check_reduction_operand(&spec, run, &meta(&[2, 4, 3]))
            .expect_err("a run whose second axis disagrees is not this descriptor's operand");
        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::Reduction,
                ..
            }
        ));
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

        check_reduction_operand(&spec, Some((1, 2)), &meta(&[2, 3, 4]))
            .expect("the original operand binds");
        // Same element count, wrong axis lengths: `outer` and `inner` catch it
        // where a bare `numel` comparison would not.
        let error = check_reduction_operand(&spec, Some((1, 2)), &meta(&[4, 3, 2]))
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
