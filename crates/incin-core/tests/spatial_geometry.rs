//! Conv and pool geometry (`SHP-005`).
//!
//! The regression this file exists for is a *wrong answer*, not a panic. The
//! static conv/pool rules filled their output spatial dimensions with
//! `Default::default()`, which is the correct value for a `typenum` extent (the
//! type carries it) and **0** for a `usize` or symbolic one. A pooled tensor
//! with runtime spatial dims therefore claimed a zero-sized output and
//! propagated it. `SHP-001` confirmed the case by execution; it is
//! `pool2d_computes_runtime_spatial_dims` below.

use incin_core::prelude::{
    AdaptiveAvgPool2dShape, Axis, Dim, Dyn, OperationKind, Pool2dShape, Shape, ShapeError,
    SpatialConv1d, SpatialConv2d, spatial_out_size,
};
use incin_core::typenum::{U0, U1, U2, U3, U4, U8, U16};

// --- the confirmed regression -------------------------------------------

#[test]
fn pool2d_computes_runtime_spatial_dims() {
    // (B, C, H, W) with H and W runtime. 8x8, kernel 2, stride 2, no padding,
    // dilation 1 pools to 4x4. Before SHP-005 this returned (0, 0).
    type S = (U1, U1, usize, usize);
    let field: <S as Shape>::Field = (U1::default(), U1::default(), 8, 8);

    let out = <S as Pool2dShape<U2, U2, U0, U1>>::compute_output_shape(&field).unwrap();

    assert_eq!((out.2, out.3), (4, 4), "spatial dims were zeroed");
    assert_eq!(out.0.size(), 1);
    assert_eq!(out.1.size(), 1);
}

#[test]
fn conv2d_computes_runtime_spatial_dims_and_keeps_out_channels() {
    // Kernel 3, stride 1, padding 1, dilation 1 is size-preserving.
    type S = (U1, U1, usize, usize);
    let field: <S as Shape>::Field = (U1::default(), U1::default(), 8, 8);

    let out =
        <S as SpatialConv2d<usize, U3, U1, U1, U1>>::compute_output_shape(&field, 16).unwrap();

    assert_eq!(out.1, 16, "out_channels was lost");
    assert_eq!((out.2, out.3), (8, 8), "spatial dims were zeroed");
}

#[test]
fn conv1d_computes_the_runtime_length_dim() {
    type S = (U1, U1, usize);
    let field: <S as Shape>::Field = (U1::default(), U1::default(), 10);

    let out = <S as SpatialConv1d<usize, U3, U1, U0, U1>>::compute_output_shape(&field, 4).unwrap();

    assert_eq!(out.1, 4, "out_channels was lost");
    assert_eq!(out.2, 8, "length dim was zeroed");
}

#[test]
fn a_fully_static_shape_still_agrees_with_its_type_level_output() {
    // The typed path was already correct; this pins that the rewrite did not
    // change it. 16x16, kernel 2, stride 2 pools to 8x8, and the output type
    // says so independently of the runtime computation.
    type S = (U1, U1, U16, U16);
    let field: <S as Shape>::Field = Default::default();

    let out = <S as Pool2dShape<U2, U2, U0, U1>>::compute_output_shape(&field).unwrap();

    assert_eq!(out.2.size(), 8);
    assert_eq!(out.3.size(), 8);
    // `U8` is what the type system independently derived for this axis.
    let _: (U1, U1, U8, U8) = out;
}

// --- the named checked sequence -----------------------------------------

const OP: OperationKind = OperationKind::Pool2d;
const H: Axis = Axis::Named("height");

#[test]
fn out_size_matches_the_reference_formula() {
    // (in + 2p - d*(k-1) - 1) / s + 1, evaluated in u128 so the reference
    // cannot wrap the same way the code under test might.
    fn reference(input: usize, k: usize, s: usize, p: usize, d: usize) -> Option<usize> {
        let padded = input as u128 + 2 * p as u128;
        let extent = d as u128 * (k as u128 - 1) + 1;
        if extent > padded {
            return None;
        }
        Some(((padded - extent) / s as u128 + 1) as usize)
    }

    for input in [1usize, 2, 3, 5, 8, 9, 16, 32, 224] {
        for k in [1usize, 2, 3, 5, 7] {
            for s in [1usize, 2, 3] {
                for p in [0usize, 1, 2] {
                    for d in [1usize, 2, 3] {
                        let got = spatial_out_size(OP, H, input, k, s, p, d);
                        match reference(input, k, s, p, d) {
                            Some(want) => {
                                assert_eq!(got.unwrap(), want, "in={input} k={k} s={s} p={p} d={d}")
                            }
                            None => assert_eq!(
                                got.unwrap_err(),
                                ShapeError::EmptyOutput {
                                    operation: OP,
                                    axis: H
                                },
                                "in={input} k={k} s={s} p={p} d={d} should not fit"
                            ),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_valued_parameters_are_rejected_by_name() {
    // Each of these is a distinct failure with a distinct fix, so each names
    // its own parameter. Unchecked, a zero stride divides by zero and a zero
    // kernel underflows `kernel - 1`.
    for (parameter, kernel, stride, dilation) in [
        ("stride", 2usize, 0usize, 1usize),
        ("kernel", 0, 1, 1),
        ("dilation", 2, 1, 0),
    ] {
        let err = spatial_out_size(OP, H, 8, kernel, stride, 0, dilation).unwrap_err();
        assert_eq!(
            err,
            ShapeError::InvalidParameter {
                operation: OP,
                parameter,
                value: 0
            }
        );
    }
}

#[test]
fn a_kernel_that_does_not_fit_reports_an_empty_output() {
    // 3x3 input, 5x5 kernel, no padding. The old single-expression form
    // computed `3 + 0 - 4 - 1`, which underflows; in a release build that
    // wraps to a huge value and then divides down to a plausible extent.
    let err = spatial_out_size(OP, H, 3, 5, 1, 0, 1).unwrap_err();
    assert_eq!(
        err,
        ShapeError::EmptyOutput {
            operation: OP,
            axis: H
        }
    );
    assert_eq!(err.to_string(), "pool2d: axis 'height' would have length 0");

    // Enough padding and it fits again.
    assert_eq!(spatial_out_size(OP, H, 3, 5, 1, 1, 1).unwrap(), 1);
}

#[test]
fn a_zero_sized_input_axis_cannot_be_pooled() {
    let err = spatial_out_size(OP, H, 0, 1, 1, 0, 1).unwrap_err();
    assert_eq!(
        err,
        ShapeError::EmptyOutput {
            operation: OP,
            axis: H
        }
    );
}

#[test]
fn overflowing_terms_are_named_individually() {
    // padding
    assert_eq!(
        spatial_out_size(OP, H, 1, 2, 1, usize::MAX, 1)
            .unwrap_err()
            .to_string(),
        "pool2d: arithmetic overflow evaluating '2 * padding'"
    );
    // input + 2*padding
    assert_eq!(
        spatial_out_size(OP, H, usize::MAX, 2, 1, 1, 1)
            .unwrap_err()
            .to_string(),
        "pool2d: arithmetic overflow evaluating 'input + 2 * padding'"
    );
    // dilation * (kernel - 1)
    assert_eq!(
        spatial_out_size(OP, H, 8, usize::MAX, 1, 0, usize::MAX)
            .unwrap_err()
            .to_string(),
        "pool2d: arithmetic overflow evaluating 'dilation * (kernel - 1)'"
    );
}

#[test]
fn a_unit_kernel_with_unit_stride_is_the_identity() {
    for input in [1usize, 7, 64, 1000] {
        assert_eq!(spatial_out_size(OP, H, input, 1, 1, 0, 1).unwrap(), input);
    }
}

// --- the dynamic path ---------------------------------------------------

#[test]
fn dyn_pool2d_handles_both_accepted_ranks() {
    // Rank 4 is (B, C, H, W).
    let out =
        <Dyn as Pool2dShape<U2, U2, U0, U1>>::compute_output_shape(&vec![1, 3, 8, 8]).unwrap();
    assert_eq!(out, vec![1, 3, 4, 4]);

    // Rank 3 is (C, H, W). This used to fall through the `len() == 4` test and
    // return the input shape unpooled.
    let out = <Dyn as Pool2dShape<U2, U2, U0, U1>>::compute_output_shape(&vec![3, 8, 8]).unwrap();
    assert_eq!(out, vec![3, 4, 4], "rank 3 was returned unpooled");
}

#[test]
fn dyn_conv2d_handles_both_accepted_ranks() {
    let out =
        <Dyn as SpatialConv2d<usize, U3, U1, U1, U1>>::compute_output_shape(&vec![1, 3, 8, 8], 16)
            .unwrap();
    assert_eq!(out, vec![1, 16, 8, 8]);

    let out =
        <Dyn as SpatialConv2d<usize, U3, U1, U1, U1>>::compute_output_shape(&vec![3, 8, 8], 16)
            .unwrap();
    assert_eq!(out, vec![16, 8, 8], "rank 3 was returned unconvolved");
}

#[test]
fn dyn_conv1d_handles_both_accepted_ranks() {
    let out =
        <Dyn as SpatialConv1d<usize, U3, U1, U0, U1>>::compute_output_shape(&vec![1, 3, 10], 4)
            .unwrap();
    assert_eq!(out, vec![1, 4, 8]);

    let out = <Dyn as SpatialConv1d<usize, U3, U1, U0, U1>>::compute_output_shape(&vec![3, 10], 4)
        .unwrap();
    assert_eq!(out, vec![4, 8], "rank 2 was returned unconvolved");
}

#[test]
fn dyn_rules_reject_a_rank_they_cannot_handle() {
    // Previously every unhandled rank silently returned the input unchanged.
    let err = <Dyn as Pool2dShape<U2, U2, U0, U1>>::compute_output_shape(&vec![8, 8]).unwrap_err();
    assert_eq!(err.operation(), OperationKind::Pool2d);
    assert!(
        matches!(err, ShapeError::RankMismatch { actual: 2, .. }),
        "unexpected error {err}"
    );
    assert_eq!(
        err.to_string(),
        "pool2d: expected rank between 3 and 4, got 2"
    );

    let err = <Dyn as SpatialConv2d<usize, U3, U1, U1, U1>>::compute_output_shape(&vec![8], 4)
        .unwrap_err();
    assert!(matches!(err, ShapeError::RankMismatch { actual: 1, .. }));
}

#[test]
fn dyn_pool2d_propagates_a_kernel_that_does_not_fit() {
    let err =
        <Dyn as Pool2dShape<U8, U1, U0, U1>>::compute_output_shape(&vec![1, 3, 4, 4]).unwrap_err();
    assert_eq!(
        err,
        ShapeError::EmptyOutput {
            operation: OperationKind::Pool2d,
            axis: Axis::Named("height"),
        }
    );
}

// --- adaptive pooling ---------------------------------------------------

#[test]
fn adaptive_pool_fixes_the_output_extent() {
    type S = (U1, U3, usize, usize);
    let field: <S as Shape>::Field = (U1::default(), U3::default(), 37, 41);

    let out = <S as AdaptiveAvgPool2dShape<U4, U4>>::compute_output_shape(&field).unwrap();
    assert_eq!(out.2.size(), 4);
    assert_eq!(out.3.size(), 4);

    let out =
        <Dyn as AdaptiveAvgPool2dShape<U4, U4>>::compute_output_shape(&vec![1, 3, 37, 41]).unwrap();
    assert_eq!(out, vec![1, 3, 4, 4]);

    // Rank 3 used to be returned unchanged.
    let out =
        <Dyn as AdaptiveAvgPool2dShape<U4, U4>>::compute_output_shape(&vec![3, 37, 41]).unwrap();
    assert_eq!(out, vec![3, 4, 4]);
}

#[test]
fn adaptive_pool_rejects_a_zero_output_extent() {
    type S = (U1, U3, usize, usize);
    let field: <S as Shape>::Field = (U1::default(), U3::default(), 37, 41);

    let err = <S as AdaptiveAvgPool2dShape<U0, U4>>::compute_output_shape(&field).unwrap_err();
    assert_eq!(
        err,
        ShapeError::EmptyOutput {
            operation: OperationKind::AdaptiveAvgPool2d,
            axis: Axis::Named("height"),
        }
    );

    let err = <S as AdaptiveAvgPool2dShape<U4, U0>>::compute_output_shape(&field).unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("width")));
}
