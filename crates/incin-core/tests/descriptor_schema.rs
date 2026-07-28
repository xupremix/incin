//! The frozen descriptor schema (`EXE-001`).
//!
//! Two kinds of test live here, and they exist for different reasons.
//!
//! The **pinning** tests assert values that are supposed to be stable:
//! `DescriptorSchemaVersion::CURRENT`, and the `OperationKind` each descriptor
//! claims. Neither is a truth about the world — both are contracts. Kernel
//! caches, autotune records, and specialization keys are all derived from
//! descriptor contents, so a field that changes meaning without a version bump
//! is a stale cache entry replayed against a kernel that reads it differently.
//! A failing pin means "bump the version", not "fix the test".
//!
//! The **derivation** tests assert the property that makes a descriptor worth
//! trusting: every field a constructor did not receive is computed from the
//! ones it did. A broadcast mask is checked against its own strides, an output
//! shape against its own operands, `outer * reduced * inner` against the input's
//! element count. Backends may skip validation only to the extent that these
//! hold.

use incin_core::exec::{
    AxisMask, BroadcastSpec, Conv2dSpec, DescriptorSchemaVersion, MatMulSpec, OperationSpec,
    ReduceOp, ReductionSpec,
};
use incin_core::prelude::{
    Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeBuf, ShapeError, StrideBuf,
};

fn shape(dims: &[usize]) -> ShapeBuf {
    ShapeBuf::from_slice(dims)
}

fn strides(values: &[usize]) -> StrideBuf {
    StrideBuf::from_slice(values)
}

/// Row-major strides, for the operand layouts these tests build by hand.
fn dense(dims: &[usize]) -> StrideBuf {
    StrideBuf::contiguous_for(&shape(dims), OperationKind::Storage).unwrap()
}

// --- schema pinning ---------------------------------------------------------

#[test]
fn schema_version_is_pinned() {
    assert_eq!(
        DescriptorSchemaVersion::CURRENT.get(),
        2,
        "the descriptor field layout changed; bump CURRENT and invalidate any \
         cache keyed on descriptor contents"
    );
}

#[test]
fn every_descriptor_reports_the_current_schema() {
    assert_eq!(BroadcastSpec::SCHEMA, DescriptorSchemaVersion::CURRENT);
    assert_eq!(MatMulSpec::SCHEMA, DescriptorSchemaVersion::CURRENT);
    assert_eq!(ReductionSpec::SCHEMA, DescriptorSchemaVersion::CURRENT);
    assert_eq!(Conv2dSpec::SCHEMA, DescriptorSchemaVersion::CURRENT);
}

#[test]
fn each_descriptor_claims_a_distinct_operation() {
    let kinds = [
        BroadcastSpec::KIND,
        MatMulSpec::KIND,
        ReductionSpec::KIND,
        Conv2dSpec::KIND,
    ];
    assert_eq!(
        kinds,
        [
            OperationKind::Broadcast,
            OperationKind::MatMul,
            OperationKind::Reduction,
            OperationKind::Conv2d,
        ]
    );
    for (i, left) in kinds.iter().enumerate() {
        for right in &kinds[i + 1..] {
            assert_ne!(left, right, "two descriptors claim the same operation");
        }
    }
}

#[test]
fn schema_compatibility_is_exact() {
    let current = DescriptorSchemaVersion::CURRENT;
    assert!(current.is_compatible_with(current));
    assert!(!current.is_compatible_with(DescriptorSchemaVersion::new(current.get() + 1)));
    assert_eq!(current.to_string(), "v2");
}

// --- the promoted operation vocabulary --------------------------------------

#[test]
fn operation_family_is_idempotent_and_lands_on_a_coarse_variant() {
    const COARSE: [OperationKind; 6] = [
        OperationKind::Storage,
        OperationKind::Fill,
        OperationKind::Random,
        OperationKind::Pointwise,
        OperationKind::Reduction,
        OperationKind::Normalization,
    ];
    const ALL: [OperationKind; 23] = [
        OperationKind::Storage,
        OperationKind::Fill,
        OperationKind::Random,
        OperationKind::Pointwise,
        OperationKind::Reduction,
        OperationKind::Normalization,
        OperationKind::Broadcast,
        OperationKind::Reshape,
        OperationKind::Flatten,
        OperationKind::Squeeze,
        OperationKind::Unsqueeze,
        OperationKind::Permute,
        OperationKind::Transpose,
        OperationKind::Slice,
        OperationKind::Concat,
        OperationKind::Stack,
        OperationKind::MatMul,
        OperationKind::Conv1d,
        OperationKind::Conv2d,
        OperationKind::Pool1d,
        OperationKind::Pool2d,
        OperationKind::AdaptiveAvgPool2d,
        OperationKind::Embedding,
    ];

    for kind in ALL {
        let family = kind.family();
        assert!(
            COARSE.contains(&family),
            "{kind} folds to {family}, which is not one of the six policy families"
        );
        assert_eq!(
            family.family(),
            family,
            "family() is not idempotent at {kind}"
        );
    }
    for coarse in COARSE {
        assert_eq!(coarse.family(), coarse, "{coarse} is not its own family");
    }
}

#[test]
fn accumulating_operations_are_reductions_and_reindexing_ones_are_storage() {
    // The grouping that matters: what earns a widened accumulator, and what is
    // dtype-agnostic because it never does arithmetic.
    for kind in [
        OperationKind::MatMul,
        OperationKind::Conv1d,
        OperationKind::Conv2d,
        OperationKind::Pool1d,
        OperationKind::Pool2d,
        OperationKind::AdaptiveAvgPool2d,
    ] {
        assert_eq!(kind.family(), OperationKind::Reduction, "{kind}");
    }
    for kind in [
        OperationKind::Broadcast,
        OperationKind::Reshape,
        OperationKind::Permute,
        OperationKind::Concat,
        OperationKind::Embedding,
    ] {
        assert_eq!(kind.family(), OperationKind::Storage, "{kind}");
    }
}

// --- axis mask --------------------------------------------------------------

#[test]
fn axis_mask_holds_every_rank_the_frontend_can_express() {
    assert!(
        AxisMask::MAX_AXES >= incin_core::prelude::MAX_RANK,
        "a shape the type system can express must fit in a mask"
    );
}

#[test]
fn axis_mask_membership_and_iteration_agree() {
    let mask = AxisMask::try_from_axes(OperationKind::Reduction, 5, [3, 0, 2]).unwrap();

    assert_eq!(mask.count(), 3);
    assert!(!mask.is_empty());
    // Ascending, whatever order the axes were given in.
    assert_eq!(mask.axes().collect::<Vec<_>>(), vec![0, 2, 3]);
    for axis in 0..5 {
        assert_eq!(
            mask.contains(axis),
            [0, 2, 3].contains(&axis),
            "axis {axis}"
        );
    }
    assert_eq!(
        mask.complement_within(5)
            .unwrap()
            .axes()
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
    assert_eq!(mask.remove(2).axes().collect::<Vec<_>>(), vec![0, 3]);
}

#[test]
fn axis_mask_rejects_out_of_range_and_repeated_axes() {
    let past_rank = AxisMask::try_from_axes(OperationKind::Reduction, 3, [3]).unwrap_err();
    assert!(matches!(
        past_rank,
        ShapeError::InvalidParameter {
            parameter: "axis",
            value: 3,
            ..
        }
    ));

    // A duplicate is not absorbed: `sum(dims = [1, 1])` says one thing and
    // would do another.
    let repeated = AxisMask::try_from_axes(OperationKind::Reduction, 3, [1, 1]).unwrap_err();
    assert!(matches!(
        repeated,
        ShapeError::InvalidParameter {
            parameter: "axis",
            value: 1,
            ..
        }
    ));

    let too_wide =
        AxisMask::try_from_axes(OperationKind::Reduction, AxisMask::MAX_AXES + 1, [0]).unwrap_err();
    assert!(matches!(
        too_wide,
        ShapeError::RankMismatch {
            expected: RankExpectation::AtMost(_),
            ..
        }
    ));
}

#[test]
fn axis_mask_boundaries_do_not_overflow_the_shift() {
    assert_eq!(AxisMask::all_below(0).unwrap(), AxisMask::EMPTY);
    let full = AxisMask::all_below(AxisMask::MAX_AXES).unwrap();
    assert_eq!(full.count(), AxisMask::MAX_AXES);
    assert_eq!(full.bits(), u32::MAX);
    assert_eq!(AxisMask::all_below(AxisMask::MAX_AXES + 1), None);

    assert_eq!(AxisMask::EMPTY.insert(AxisMask::MAX_AXES), None);
    assert!(!AxisMask::EMPTY.contains(AxisMask::MAX_AXES));
    // Removing an axis that could never be present is a no-op, not an error.
    assert_eq!(full.remove(AxisMask::MAX_AXES), full);
}

#[test]
fn axis_mask_recognizes_unbroken_runs() {
    let run = |axes: &[usize]| {
        axes.iter()
            .fold(AxisMask::EMPTY, |mask, &axis| mask.insert(axis).unwrap())
            .is_contiguous_run()
    };
    assert!(run(&[]));
    assert!(run(&[0]));
    assert!(run(&[2, 3, 4]));
    assert!(!run(&[0, 2]));
    assert!(!run(&[0, 1, 3]));
}

#[test]
fn axis_mask_debug_names_its_axes() {
    let mask = AxisMask::try_from_axes(OperationKind::Reduction, 4, [1, 3]).unwrap();
    assert_eq!(format!("{mask:?}"), "AxisMask{1, 3}");
}

// --- broadcast --------------------------------------------------------------

#[test]
fn broadcast_derives_output_and_masks_from_operand_layouts() {
    // Rank 2 against rank 3: the shorter operand right-aligns, and every axis
    // it is length 1 along is stretched.
    let spec = BroadcastSpec::contiguous(&shape(&[4, 1, 3]), &shape(&[5, 3])).unwrap();

    assert_eq!(spec.output.dims(), &[4, 5, 3]);
    assert_eq!(spec.output_elements().unwrap(), 60);

    // lhs [4,1,3] strides [3,3,1]: axis 1 is stretched, so its stride is 0.
    assert_eq!(spec.lhs_strides.strides(), &[3, 0, 1]);
    assert_eq!(spec.lhs_broadcast_mask.axes().collect::<Vec<_>>(), vec![1]);

    // rhs [5,3] has no axis 0 at all, so it is stretched there.
    assert_eq!(spec.rhs_strides.strides(), &[0, 3, 1]);
    assert_eq!(spec.rhs_broadcast_mask.axes().collect::<Vec<_>>(), vec![0]);
}

#[test]
fn broadcast_masks_are_exactly_the_zero_strides_that_stretch() {
    // The invariant a kernel relies on: mask bit set <=> stride 0 at an axis
    // the output is longer than 1 along.
    let cases = [
        (vec![1usize], vec![1usize]),
        (vec![8], vec![1]),
        (vec![2, 1, 4], vec![1, 3, 4]),
        (vec![1, 1], vec![6, 7]),
        (vec![3, 1, 1, 5], vec![1, 2, 4, 1]),
    ];
    for (lhs, rhs) in cases {
        let spec = BroadcastSpec::contiguous(&shape(&lhs), &shape(&rhs)).unwrap();
        for (name, strides, mask) in [
            ("lhs", &spec.lhs_strides, spec.lhs_broadcast_mask),
            ("rhs", &spec.rhs_strides, spec.rhs_broadcast_mask),
        ] {
            for axis in 0..spec.output.rank() {
                let stretched = strides.strides()[axis] == 0 && spec.output.dims()[axis] != 1;
                assert_eq!(
                    mask.contains(axis),
                    stretched,
                    "{name} axis {axis} of {lhs:?} x {rhs:?}"
                );
            }
        }
    }
}

#[test]
fn broadcast_keeps_non_contiguous_view_strides() {
    // A view over a larger buffer: strides that are not row-major must survive
    // untouched on the axes that are not stretched.
    let spec = BroadcastSpec::new(
        &shape(&[3, 1]),
        &strides(&[10, 1]),
        &shape(&[3, 4]),
        &strides(&[100, 7]),
    )
    .unwrap();

    assert_eq!(spec.output.dims(), &[3, 4]);
    assert_eq!(spec.lhs_strides.strides(), &[10, 0]);
    assert_eq!(spec.rhs_strides.strides(), &[100, 7]);
    assert!(spec.rhs_broadcast_mask.is_empty());
}

#[test]
fn broadcast_scalar_operand_reads_one_element_throughout() {
    let spec = BroadcastSpec::contiguous(&shape(&[2, 3]), &ShapeBuf::scalar()).unwrap();

    assert_eq!(spec.output.dims(), &[2, 3]);
    assert_eq!(spec.rhs_strides.strides(), &[0, 0]);
    assert_eq!(
        spec.rhs_broadcast_mask.axes().collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(spec.lhs_broadcast_mask.is_empty());
}

#[test]
fn broadcast_rejects_incompatible_dimensions() {
    let error = BroadcastSpec::contiguous(&shape(&[2, 3]), &shape(&[2, 4])).unwrap_err();
    assert_eq!(
        error,
        ShapeError::DimensionMismatch {
            operation: OperationKind::Broadcast,
            axis: Axis::Index(1),
            lhs: 3,
            rhs: 4,
            constraint: DimensionConstraint::Broadcastable,
        }
    );
}

#[test]
fn broadcast_rejects_a_stride_count_that_does_not_match_its_shape() {
    let error = BroadcastSpec::new(
        &shape(&[2, 3]),
        &strides(&[3]),
        &shape(&[2, 3]),
        &dense(&[2, 3]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ShapeError::RankMismatch {
            operation: OperationKind::Broadcast,
            expected: RankExpectation::SameAs {
                operand: "lhs",
                rank: 2
            },
            actual: 1,
        }
    ));
}

// --- matrix multiplication --------------------------------------------------

#[test]
fn matmul_resolves_gemm_extents_and_output() {
    let spec = MatMulSpec::new(&shape(&[7, 3]), &shape(&[3, 5])).unwrap();

    assert_eq!((spec.m, spec.k, spec.n), (7, 3, 5));
    assert_eq!(spec.output.dims(), &[7, 5]);
    assert!(spec.batch.is_empty());
    assert_eq!(spec.lhs_rank, 2);
    assert_eq!(spec.rhs_rank, 2);
    assert!(!spec.transpose_lhs && !spec.transpose_rhs);
}

#[test]
fn matmul_batch_strides_step_whole_matrices() {
    // [2,4] x [2,3] batch, matrices 3x5 and 5x6.
    let spec = MatMulSpec::new(&shape(&[2, 4, 3, 5]), &shape(&[2, 4, 5, 6])).unwrap();

    assert_eq!(spec.batch.dims(), &[2, 4]);
    assert_eq!(spec.output.dims(), &[2, 4, 3, 6]);
    // lhs matrix is 3*5 = 15 elements; axis 1 steps one matrix, axis 0 steps four.
    assert_eq!(spec.lhs_batch_strides.strides(), &[60, 15]);
    // rhs matrix is 5*6 = 30.
    assert_eq!(spec.rhs_batch_strides.strides(), &[120, 30]);
    // output matrix is 3*6 = 18.
    assert_eq!(spec.output_batch_strides.strides(), &[72, 18]);
}

#[test]
fn matmul_broadcast_batch_axis_gets_a_zero_stride() {
    // The right operand is one shared matrix reused across the batch.
    let spec = MatMulSpec::new(&shape(&[8, 3, 4]), &shape(&[4, 2])).unwrap();

    assert_eq!(spec.batch.dims(), &[8]);
    assert_eq!(spec.output.dims(), &[8, 3, 2]);
    assert_eq!(spec.lhs_batch_strides.strides(), &[12]);
    assert_eq!(
        spec.rhs_batch_strides.strides(),
        &[0],
        "a reused operand must not advance"
    );

    // Same when the batch axis is present but length 1.
    let unit = MatMulSpec::new(&shape(&[8, 3, 4]), &shape(&[1, 4, 2])).unwrap();
    assert_eq!(unit.rhs_batch_strides.strides(), &[0]);
    assert_eq!(unit.output.dims(), &[8, 3, 2]);
}

#[test]
fn matmul_rejects_a_contraction_mismatch_as_a_matmul_error() {
    let error = MatMulSpec::new(&shape(&[7, 3]), &shape(&[4, 5])).unwrap_err();
    assert_eq!(
        error,
        ShapeError::DimensionMismatch {
            operation: OperationKind::MatMul,
            axis: Axis::Named("contraction"),
            lhs: 3,
            rhs: 4,
            constraint: DimensionConstraint::Equal,
        }
    );
}

#[test]
fn matmul_reports_batch_conflicts_against_matmul_not_broadcast() {
    // The batch axes broadcast, but the caller asked for a matmul; an error
    // that says "broadcast" points at an operation they never called.
    let error = MatMulSpec::new(&shape(&[2, 3, 4]), &shape(&[3, 4, 5])).unwrap_err();
    assert_eq!(error.operation(), OperationKind::MatMul);
    assert_eq!(error.axis(), Some(Axis::Index(0)));
}

#[test]
fn matmul_rejects_operands_below_rank_two() {
    let error = MatMulSpec::new(&shape(&[3]), &shape(&[3, 5])).unwrap_err();
    assert!(matches!(
        error,
        ShapeError::RankMismatch {
            operation: OperationKind::MatMul,
            expected: RankExpectation::AtLeast(2),
            actual: 1,
        }
    ));
}

#[test]
fn matmul_transpose_flags_are_layout_only() {
    let plain = MatMulSpec::new(&shape(&[7, 3]), &shape(&[3, 5])).unwrap();
    let flagged = plain.clone().transposed(true, false);

    assert!(flagged.transpose_lhs && !flagged.transpose_rhs);
    // Everything logical is untouched: the flags say how storage is read, not
    // what the operation computes.
    assert_eq!(flagged.output, plain.output);
    assert_eq!(
        (flagged.m, flagged.k, flagged.n),
        (plain.m, plain.k, plain.n)
    );
    assert_eq!(flagged.lhs_batch_strides, plain.lhs_batch_strides);
}

// --- reduction --------------------------------------------------------------

#[test]
fn reduction_decomposition_multiplies_back_to_the_input() {
    let input = shape(&[2, 3, 4, 5]);
    for (axes, expected) in [
        (vec![0usize], (1usize, 2usize, 60usize)),
        (vec![1], (2, 3, 20)),
        (vec![3], (24, 5, 1)),
        (vec![1, 2], (2, 12, 5)),
        (vec![0, 1, 2, 3], (1, 120, 1)),
    ] {
        let spec = ReductionSpec::over_axes(&input, axes.clone(), false, ReduceOp::Sum).unwrap();
        assert_eq!(
            (spec.outer, spec.reduced, spec.inner),
            expected,
            "axes {axes:?}"
        );
        assert_eq!(
            spec.outer * spec.reduced * spec.inner,
            input.numel().unwrap(),
            "decomposition lost elements for axes {axes:?}"
        );
        assert_eq!(
            spec.output.numel().unwrap(),
            spec.outer * spec.inner,
            "output disagrees with the kept regions for axes {axes:?}"
        );
    }
}

#[test]
fn reduction_output_respects_keep_dims() {
    let input = shape(&[2, 3, 4]);

    let dropped = ReductionSpec::over_axes(&input, [1], false, ReduceOp::Sum).unwrap();
    assert_eq!(dropped.output.dims(), &[2, 4]);
    assert!(!dropped.keep_dims);

    let kept = ReductionSpec::over_axes(&input, [1], true, ReduceOp::Sum).unwrap();
    assert_eq!(kept.output.dims(), &[2, 1, 4]);
    assert!(kept.keep_dims);
    // The two describe the same work, only shaped differently.
    assert_eq!(
        (kept.outer, kept.reduced, kept.inner),
        (dropped.outer, dropped.reduced, dropped.inner)
    );
}

#[test]
fn reduction_over_all_axes_produces_a_scalar() {
    let spec = ReductionSpec::over_all(&shape(&[2, 3, 4]), false, ReduceOp::Sum).unwrap();
    assert_eq!(spec.output.rank(), 0);
    assert_eq!(spec.output_elements().unwrap(), 1);
    assert_eq!((spec.outer, spec.reduced, spec.inner), (1, 24, 1));

    let kept = ReductionSpec::over_all(&shape(&[2, 3, 4]), true, ReduceOp::Sum).unwrap();
    assert_eq!(kept.output.dims(), &[1, 1, 1]);
}

#[test]
fn reduction_over_no_axes_is_the_identity() {
    // Total rather than a special case, so a caller building an axis list
    // dynamically does not need one either.
    let input = shape(&[2, 3]);
    let spec = ReductionSpec::new(&input, AxisMask::EMPTY, false, ReduceOp::Sum).unwrap();

    assert_eq!(spec.output, input);
    assert_eq!((spec.outer, spec.reduced, spec.inner), (6, 1, 1));
}

#[test]
fn reduction_rejects_scattered_axes() {
    // {0, 2} has no outer/reduced/inner decomposition without a permutation
    // first, so it is refused rather than mis-lowered.
    let error =
        ReductionSpec::over_axes(&shape(&[2, 3, 4]), [0, 2], false, ReduceOp::Sum).unwrap_err();
    assert!(matches!(
        error,
        ShapeError::InvalidAxisRange {
            operation: OperationKind::Reduction,
            start: 0,
            end: 3,
            rank: 3,
        }
    ));
}

#[test]
fn reduction_rejects_an_axis_past_the_input_rank() {
    let listed = ReductionSpec::over_axes(&shape(&[2, 3]), [2], false, ReduceOp::Sum).unwrap_err();
    assert!(matches!(
        listed,
        ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: 2,
        }
    ));

    // Also when the mask is built directly and skips `try_from_axes`.
    let masked = ReductionSpec::new(
        &shape(&[2, 3]),
        AxisMask::EMPTY.insert(5).unwrap(),
        false,
        ReduceOp::Sum,
    )
    .unwrap_err();
    assert!(matches!(
        masked,
        ShapeError::InvalidParameter {
            parameter: "axis",
            value: 5,
            ..
        }
    ));
}

#[test]
fn reduction_keeps_an_empty_input_empty() {
    // The zero axis survives into the output, so it collapses every product it
    // takes part in and nothing overflows despite the `usize::MAX` axis.
    let spec =
        ReductionSpec::over_axes(&shape(&[usize::MAX, 4, 0]), [1], false, ReduceOp::Sum).unwrap();
    assert_eq!(spec.output.dims(), &[usize::MAX, 0]);
    assert_eq!((spec.outer, spec.reduced, spec.inner), (usize::MAX, 4, 0));
    assert_eq!(spec.output_elements().unwrap(), 0);
}

#[test]
fn an_output_too_large_to_index_is_rejected_at_resolution() {
    // Reducing away the only zero axis leaves `[MAX, MAX]`: a shape whose
    // element count overflows `usize`, so no backend could allocate or index
    // it. The useful place to say so is here, while the operands are still in
    // hand — not at launch, where the diagnostic is a kernel argument.
    let error = ReductionSpec::over_axes(
        &shape(&[usize::MAX, 0, usize::MAX]),
        [1],
        false,
        ReduceOp::Sum,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ShapeError::ArithmeticOverflow {
            operation: OperationKind::Reduction,
            expression: "product of dimensions",
        }
    ));

    let broadcast =
        BroadcastSpec::contiguous(&shape(&[usize::MAX, 1]), &shape(&[usize::MAX])).unwrap_err();
    assert_eq!(broadcast.operation(), OperationKind::Broadcast);

    let matmul =
        MatMulSpec::new(&shape(&[usize::MAX, 2, 2]), &shape(&[usize::MAX, 2, 2])).unwrap_err();
    assert_eq!(matmul.operation(), OperationKind::MatMul);
}

/// Established by [`an_output_too_large_to_index_is_rejected_at_resolution`]:
/// a descriptor that exists has a representable output, so
/// [`OperationSpec::output_elements`] cannot fail on one.
#[test]
fn every_constructed_descriptor_has_a_representable_output() {
    let cases: [Box<dyn Fn() -> Result<usize, ShapeError>>; 4] = [
        Box::new(|| {
            BroadcastSpec::contiguous(&shape(&[4, 1, 3]), &shape(&[5, 3]))?.output_elements()
        }),
        Box::new(|| MatMulSpec::new(&shape(&[2, 3, 4]), &shape(&[4, 5]))?.output_elements()),
        Box::new(|| {
            ReductionSpec::over_axes(&shape(&[2, 3, 4]), [1], true, ReduceOp::Sum)?
                .output_elements()
        }),
        Box::new(|| {
            Conv2dSpec::new(
                &shape(&[8, 3, 32, 32]),
                16,
                [3, 3],
                [1, 1],
                [1, 1],
                [1, 1],
                1,
            )?
            .output_elements()
        }),
    ];
    for (i, case) in cases.iter().enumerate() {
        assert!(case().is_ok(), "descriptor {i} reported an unusable output");
    }
}

// --- two-dimensional convolution --------------------------------------------

#[test]
fn conv2d_resolves_spatial_geometry() {
    // 3x3 kernel, stride 1, padding 1: the classic same-size convolution.
    let spec = Conv2dSpec::new(
        &shape(&[8, 3, 32, 32]),
        16,
        [3, 3],
        [1, 1],
        [1, 1],
        [1, 1],
        1,
    )
    .unwrap();

    assert_eq!(spec.output.dims(), &[8, 16, 32, 32]);
    assert_eq!((spec.n, spec.c_in, spec.c_out), (8, 3, 16));
    assert_eq!((spec.h_in, spec.w_in), (32, 32));
    assert_eq!((spec.h_out, spec.w_out), (32, 32));
}

#[test]
fn conv2d_honors_stride_padding_and_dilation_per_axis() {
    // Asymmetric on purpose: a formula that swapped height and width would
    // still pass a square case.
    let spec = Conv2dSpec::new(
        &shape(&[1, 4, 10, 20]),
        8,
        [3, 5],
        [2, 3],
        [1, 0],
        [2, 1],
        4,
    )
    .unwrap();

    // h: (10 + 2 - 2*(3-1) - 1)/2 + 1 = 4;  w: (20 + 0 - 1*(5-1) - 1)/3 + 1 = 6
    assert_eq!((spec.h_out, spec.w_out), (4, 6));
    assert_eq!(spec.output.dims(), &[1, 8, 4, 6]);
    assert_eq!(spec.groups, 4);
}

#[test]
fn conv2d_rejects_channels_that_groups_does_not_divide() {
    let bad_in =
        Conv2dSpec::new(&shape(&[1, 6, 8, 8]), 8, [1, 1], [1, 1], [0, 0], [1, 1], 4).unwrap_err();
    assert_eq!(
        bad_in,
        ShapeError::DimensionMismatch {
            operation: OperationKind::Conv2d,
            axis: Axis::Named("in_channels"),
            lhs: 6,
            rhs: 4,
            constraint: DimensionConstraint::DivisibleBy,
        }
    );

    let bad_out =
        Conv2dSpec::new(&shape(&[1, 8, 8, 8]), 6, [1, 1], [1, 1], [0, 0], [1, 1], 4).unwrap_err();
    assert_eq!(bad_out.axis(), Some(Axis::Named("out_channels")));
}

#[test]
fn conv2d_rejects_zero_valued_parameters() {
    let zero_groups =
        Conv2dSpec::new(&shape(&[1, 4, 8, 8]), 4, [1, 1], [1, 1], [0, 0], [1, 1], 0).unwrap_err();
    assert!(matches!(
        zero_groups,
        ShapeError::InvalidParameter {
            operation: OperationKind::Conv2d,
            parameter: "groups",
            value: 0,
        }
    ));

    // A zero stride would divide by zero inside the output-size formula.
    let zero_stride =
        Conv2dSpec::new(&shape(&[1, 4, 8, 8]), 4, [1, 1], [0, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        zero_stride,
        ShapeError::InvalidParameter {
            parameter: "stride",
            value: 0,
            ..
        }
    ));
}

#[test]
fn conv2d_reports_a_kernel_that_does_not_fit_rather_than_underflowing() {
    let error =
        Conv2dSpec::new(&shape(&[1, 1, 4, 4]), 1, [9, 1], [1, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert_eq!(
        error,
        ShapeError::EmptyOutput {
            operation: OperationKind::Conv2d,
            axis: Axis::Named("height"),
        }
    );
}

#[test]
fn conv2d_requires_an_nchw_input() {
    let error =
        Conv2dSpec::new(&shape(&[3, 8, 8]), 4, [1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        error,
        ShapeError::RankMismatch {
            operation: OperationKind::Conv2d,
            expected: RankExpectation::Exactly(4),
            actual: 3,
        }
    ));
}

// --- shared descriptor properties -------------------------------------------

#[test]
fn descriptors_are_shareable_and_comparable() {
    fn assert_usable<T: OperationSpec + PartialEq + Send + Sync>(spec: &T) {
        // A descriptor is resolved once and read from every worker that
        // executes the operation, so it must cross threads and compare equal
        // to an identically resolved copy.
        assert_eq!(spec, &spec.clone());
        assert!(!format!("{spec:?}").is_empty());
    }

    assert_usable(&BroadcastSpec::contiguous(&shape(&[2, 3]), &shape(&[3])).unwrap());
    assert_usable(&MatMulSpec::new(&shape(&[2, 3]), &shape(&[3, 4])).unwrap());
    assert_usable(&ReductionSpec::over_axes(&shape(&[2, 3]), [1], false, ReduceOp::Sum).unwrap());
    assert_usable(
        &Conv2dSpec::new(&shape(&[1, 1, 4, 4]), 1, [3, 3], [1, 1], [1, 1], [1, 1], 1).unwrap(),
    );
}

#[test]
fn resolving_the_same_operation_twice_gives_the_same_descriptor() {
    // The property that makes a descriptor usable as a cache key: resolution
    // is a pure function of the operand shapes.
    let once = MatMulSpec::new(&shape(&[2, 3, 4]), &shape(&[4, 5])).unwrap();
    let twice = MatMulSpec::new(&shape(&[2, 3, 4]), &shape(&[4, 5])).unwrap();
    assert_eq!(once, twice);

    let broadcast_once = BroadcastSpec::contiguous(&shape(&[4, 1]), &shape(&[3])).unwrap();
    let broadcast_twice = BroadcastSpec::contiguous(&shape(&[4, 1]), &shape(&[3])).unwrap();
    assert_eq!(broadcast_once, broadcast_twice);
}
