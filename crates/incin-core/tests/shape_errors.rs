//! Rendering tests for `ShapeError` (`SHP-002`).
//!
//! One test per variant, asserting the exact rendered string. A shape
//! diagnostic is a user-facing artifact: if its wording changes, that is an API
//! change and this file should be the thing that notices.
//!
//! These run as an *integration* test on purpose. It proves the type is
//! reachable from outside the crate through `incin_core::prelude`, which is the
//! only path a downstream caller has — `shapes` itself is `pub(crate)`.

use incin_core::prelude::{
    Axis, DimensionConstraint, Error, OperationKind, RankExpectation, ShapeError,
};

#[test]
fn rank_mismatch_renders() {
    let err = ShapeError::RankMismatch {
        operation: OperationKind::Reshape,
        expected: RankExpectation::Exactly(4),
        actual: 3,
    };
    assert_eq!(err.to_string(), "reshape: expected rank exactly 4, got 3");
}

#[test]
fn dimension_mismatch_renders() {
    let err = ShapeError::DimensionMismatch {
        operation: OperationKind::Broadcast,
        axis: Axis::Index(1),
        lhs: 3,
        rhs: 4,
        constraint: DimensionConstraint::Broadcastable,
    };
    assert_eq!(
        err.to_string(),
        "broadcast: axis 1 mismatch: 3 vs 4, which must be equal, or one of them 1"
    );
}

#[test]
fn invalid_axis_range_renders() {
    let err = ShapeError::InvalidAxisRange {
        operation: OperationKind::Flatten,
        start: 3,
        end: 2,
        rank: 4,
    };
    assert_eq!(
        err.to_string(),
        "flatten: axis range 3..2 is invalid for rank 4"
    );
}

#[test]
fn invalid_parameter_renders() {
    let err = ShapeError::InvalidParameter {
        operation: OperationKind::Pool2d,
        parameter: "stride",
        value: 0,
    };
    assert_eq!(
        err.to_string(),
        "pool2d: parameter 'stride' has invalid value 0"
    );
}

#[test]
fn arithmetic_overflow_renders() {
    let err = ShapeError::ArithmeticOverflow {
        operation: OperationKind::Conv2d,
        expression: "dilation * (kernel - 1)",
    };
    assert_eq!(
        err.to_string(),
        "conv2d: arithmetic overflow evaluating 'dilation * (kernel - 1)'"
    );
}

#[test]
fn empty_output_renders() {
    let err = ShapeError::EmptyOutput {
        operation: OperationKind::Pool2d,
        axis: Axis::Named("height"),
    };
    assert_eq!(err.to_string(), "pool2d: axis 'height' would have length 0");
}

// --- component rendering ------------------------------------------------
//
// The variants above pin one rendering of each component. These pin the rest,
// so a change to a shared component cannot slip through by only being exercised
// in the one combination a variant test happens to use.

#[test]
fn every_rank_expectation_renders() {
    let cases = [
        (RankExpectation::Exactly(4), "exactly 4"),
        (RankExpectation::AtLeast(2), "at least 2"),
        (RankExpectation::AtMost(8), "at most 8"),
        (RankExpectation::Between { min: 2, max: 4 }, "between 2 and 4"),
        (
            RankExpectation::SameAs {
                operand: "lhs",
                rank: 3,
            },
            "the same rank as lhs (3)",
        ),
    ];
    for (expectation, rendered) in cases {
        assert_eq!(expectation.to_string(), rendered);
    }
}

#[test]
fn every_dimension_constraint_renders() {
    let cases = [
        (DimensionConstraint::Equal, "equal"),
        (DimensionConstraint::Broadcastable, "equal, or one of them 1"),
        (DimensionConstraint::DivisibleBy, "an exact multiple"),
        (DimensionConstraint::AtLeast, "greater than or equal"),
    ];
    for (constraint, rendered) in cases {
        assert_eq!(constraint.to_string(), rendered);
        assert_eq!(constraint.describe(), rendered);
    }
}

#[test]
fn every_axis_renders() {
    assert_eq!(Axis::Index(2).to_string(), "axis 2");
    assert_eq!(Axis::Named("channels").to_string(), "axis 'channels'");
    assert_eq!(Axis::Whole.to_string(), "the shape as a whole");
}

#[test]
fn operation_names_are_unique_and_lowercase() {
    // `name()` is the stable diagnostic spelling. Two operations sharing one
    // name would make a message ambiguous about which rule failed.
    const KINDS: [OperationKind; 23] = [
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

    let mut names: Vec<&'static str> = KINDS.iter().map(|k| k.name()).collect();
    names.sort_unstable();
    let unique = names.len();
    names.dedup();
    assert_eq!(names.len(), unique, "two operations render the same name");

    for kind in KINDS {
        let name = kind.name();
        assert!(!name.is_empty(), "{kind:?} has an empty name");
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{kind:?} renders as {name:?}, which is not snake_case"
        );
        assert_eq!(kind.to_string(), name, "Display and name() disagree");
    }
}

// --- accessors and integration -----------------------------------------

#[test]
fn operation_accessor_covers_every_variant() {
    let errors = [
        ShapeError::RankMismatch {
            operation: OperationKind::Reshape,
            expected: RankExpectation::Exactly(4),
            actual: 3,
        },
        ShapeError::DimensionMismatch {
            operation: OperationKind::Concat,
            axis: Axis::Index(0),
            lhs: 1,
            rhs: 2,
            constraint: DimensionConstraint::Equal,
        },
        ShapeError::InvalidAxisRange {
            operation: OperationKind::Slice,
            start: 0,
            end: 9,
            rank: 4,
        },
        ShapeError::InvalidParameter {
            operation: OperationKind::Conv1d,
            parameter: "dilation",
            value: 0,
        },
        ShapeError::ArithmeticOverflow {
            operation: OperationKind::MatMul,
            expression: "rows * cols",
        },
        ShapeError::EmptyOutput {
            operation: OperationKind::Pool1d,
            axis: Axis::Index(2),
        },
    ];

    let expected = [
        OperationKind::Reshape,
        OperationKind::Concat,
        OperationKind::Slice,
        OperationKind::Conv1d,
        OperationKind::MatMul,
        OperationKind::Pool1d,
    ];

    for (err, kind) in errors.into_iter().zip(expected) {
        assert_eq!(err.operation(), kind);
    }
}

#[test]
fn axis_accessor_is_some_only_for_axis_variants() {
    let with_axis = ShapeError::EmptyOutput {
        operation: OperationKind::Pool2d,
        axis: Axis::Named("width"),
    };
    let without_axis = ShapeError::ArithmeticOverflow {
        operation: OperationKind::Pool2d,
        expression: "numel",
    };
    assert_eq!(with_axis.axis(), Some(Axis::Named("width")));
    assert_eq!(without_axis.axis(), None);
}

#[test]
fn shape_error_converts_into_the_crate_error() {
    let shape = ShapeError::InvalidParameter {
        operation: OperationKind::Pool2d,
        parameter: "stride",
        value: 0,
    };

    // The `?` path a caller actually takes: a shape rule fails inside a
    // function returning the crate-wide `Error`.
    fn fallible(err: ShapeError) -> Result<(), Error> {
        Err(err)?;
        unreachable!()
    }

    let err = fallible(shape).unwrap_err();
    assert!(matches!(err, Error::Shape(inner) if inner == shape));
    // Wrapping is transparent: no prefix is added on the way out.
    assert_eq!(err.to_string(), shape.to_string());
}

#[test]
fn shape_errors_are_copy_and_allocation_free() {
    // Every field is `usize` or `&'static str`, so a shape rule in `no_std` or
    // an allocation-free context can report a precise diagnostic. `Copy` is the
    // observable consequence and the thing a refactor would break first.
    fn assert_copy<T: Copy>() {}
    assert_copy::<ShapeError>();
    assert_copy::<OperationKind>();
    assert_copy::<Axis>();
    assert_copy::<RankExpectation>();
    assert_copy::<DimensionConstraint>();

    let err = ShapeError::RankMismatch {
        operation: OperationKind::MatMul,
        expected: RankExpectation::AtLeast(2),
        actual: 1,
    };
    let copied = err;
    assert_eq!(err, copied);
}
