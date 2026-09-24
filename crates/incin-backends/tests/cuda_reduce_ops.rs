//! CUDA reduce operations: axis reductions, argmax/argmin, cumsum, topk,
//! Welford var/std, argsort, and the composed norm rows.
//!
//! The previous version of this file was named for topk, cumsum, argmax and
//! argmin, and asserted that `DTypeDescriptor::size_bytes` returns `Ok` for
//! three dtypes. It launched nothing. Its one green result said nothing about
//! any operation this file is named for, while counting as coverage.
//!
//! The rewrite of the sibling optimizer suite, which was vacuous in the same
//! way, uncovered that no CUDA optimizer kernel was ever launched -- so these
//! are written on the assumption that a suite asserting nothing was hiding
//! something.
//!
//! Issue #87 adds three more claims this file pins on hardware:
//!
//! - axis-0 argmax/argmin/var/`std` must read the right rows (the
//!   non-last-axis addressing fix), not a stride-1 mis-slice;
//! - the six Welford `var`/`std` rows must record a tape entry and their
//!   ones-seeded backward must match the closed-form gradient;
//! - `Argsort` must return a permutation that actually addresses the input,
//!   and `Norm` must match the host L1/L2 magnitudes while recording.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_reduce_ops -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{
        argmax_argmin, cumsum, download_f32, download_i64, reduce, require_cuda, topk,
        upload_f32_shaped, var_std,
    },
};
use incin_core::backend_authoring::{AutogradBackend, Execute, StorageBackend};
use incin_core::exec::catalog::{
    ArgsortAttributes, AxisVarianceAttributes, NormAttributes, VarianceAttributes,
};
use incin_core::exec::{
    CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle, dispatch, op,
};
use incin_core::prelude::{CudaN, DTypeId};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-5 * left.abs().max(right.abs()).max(1.0)
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    download_f32(storage)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

/// Dispatch a single-input op under a fresh default context, returning the
/// output and how many tape entries the forward pushed.
fn run1<O, A>(input: &TestStorage, attributes: A) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = tape_depth();
    let out = dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised reduction must execute");
    (out, tape_depth() - before)
}

/// A 2x4 with a distinct maximum and minimum per row, so a reduction that
/// silently returned the first or last element would be caught.
fn matrix() -> (Vec<usize>, Vec<f32>) {
    (vec![2, 4], vec![3.0, 1.0, 4.0, 1.5, -2.0, 5.0, 0.5, -3.0])
}

#[test]
#[ignore = "requires CUDA hardware"]
fn axis_reductions_match_their_definitions() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);
    let rows: Vec<&[f32]> = values.chunks(4).collect();

    for (op, expect) in [
        ("sum", vec![9.5_f64, 0.5]),
        ("mean", vec![2.375, 0.125]),
        ("max", vec![4.0, 5.0]),
        ("min", vec![1.0, -3.0]),
        ("prod", vec![18.0, 15.0]),
    ] {
        let out = reduce(op, &input, 1, false).unwrap_or_else(|e| panic!("{op} must launch: {e}"));
        let got = download_f32(&out);
        assert_eq!(got.len(), rows.len(), "{op} should reduce axis 1 away");
        for (row, expected) in expect.iter().enumerate() {
            assert!(
                close(f64::from(got[row]), *expected),
                "{op} row {row}: kernel gave {}, definition gives {expected}",
                got[row]
            );
        }
    }
}

/// `keepdim` must keep the axis with extent one rather than drop it.
#[test]
#[ignore = "requires CUDA hardware"]
fn keepdim_retains_the_reduced_axis() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let dropped = reduce("sum", &input, 1, false).unwrap();
    let kept = reduce("sum", &input, 1, true).unwrap();

    assert_eq!(&dropped.shape[..], &[2]);
    assert_eq!(&kept.shape[..], &[2, 1]);
    assert_eq!(download_f32(&dropped), download_f32(&kept));
}

/// argmax and argmin must return *positions*, not values.
///
/// The rows are chosen so the extreme is neither first nor last, which a
/// kernel that returned a fixed index would otherwise pass.
#[test]
#[ignore = "requires CUDA hardware"]
fn argmax_and_argmin_return_positions() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    // row 0: [3, 1, 4, 1.5] -> max at 2, min at 1
    // row 1: [-2, 5, 0.5, -3] -> max at 1, min at 3
    let max_idx = download_i64(&argmax_argmin("argmax", &input, Some(1)).expect("argmax launches"));
    let min_idx = download_i64(&argmax_argmin("argmin", &input, Some(1)).expect("argmin launches"));

    assert_eq!(max_idx, vec![2, 1], "argmax positions");
    assert_eq!(min_idx, vec![1, 3], "argmin positions");
}

/// A prefix sum must be inclusive and per-row.
#[test]
#[ignore = "requires CUDA hardware"]
fn cumsum_accumulates_along_the_axis() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let got = download_f32(&cumsum(&input, 1).expect("cumsum launches"));

    let mut expected = Vec::new();
    for row in values.chunks(4) {
        let mut running = 0.0_f64;
        for value in row {
            running += f64::from(*value);
            expected.push(running);
        }
    }
    assert_eq!(got.len(), expected.len());
    for (index, want) in expected.iter().enumerate() {
        assert!(
            close(f64::from(got[index]), *want),
            "cumsum at {index}: kernel gave {}, definition gives {want}",
            got[index]
        );
    }
}

/// `topk` must return the k largest in descending order, with their indices.
#[test]
#[ignore = "requires CUDA hardware"]
fn topk_returns_ordered_values_and_their_indices() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let (vals, idx) = topk(&input, 2, 1, true).expect("topk launches");
    let (vals, idx) = (download_f32(&vals), download_i64(&idx));

    // row 0: [3, 1, 4, 1.5] -> 4 at 2, then 3 at 0
    // row 1: [-2, 5, 0.5, -3] -> 5 at 1, then 0.5 at 2
    assert!(close(f64::from(vals[0]), 4.0) && close(f64::from(vals[1]), 3.0));
    assert!(close(f64::from(vals[2]), 5.0) && close(f64::from(vals[3]), 0.5));
    assert_eq!(idx, vec![2, 0, 1, 2], "topk indices");

    // The indices must actually address the values returned beside them.
    for (position, index) in idx.iter().enumerate() {
        let row = position / 2;
        let source = values[row * 4 + usize::try_from(*index).unwrap()];
        assert!(
            close(f64::from(vals[position]), f64::from(source)),
            "topk index {index} does not point at the value it was returned with"
        );
    }
}

/// Issue #87: axis-0 argmax/argmin must index *rows*, not a flat offset.
#[test]
#[ignore = "requires CUDA hardware"]
fn reductions_along_axis_zero_read_the_right_rows() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    // Columns of [[3, 1, 4, 1.5], [-2, 5, 0.5, -3]].
    let max_idx = download_i64(&argmax_argmin("argmax", &input, Some(0)).expect("argmax axis0"));
    let min_idx = download_i64(&argmax_argmin("argmin", &input, Some(0)).expect("argmin axis0"));
    assert_eq!(max_idx, vec![0, 1, 0, 0], "argmax axis-0 positions");
    assert_eq!(min_idx, vec![1, 0, 1, 1], "argmin axis-0 positions");

    // Column variances, biased: [6.25, 4, 3.0625, 5.0625].
    let var = var_std(&input, Some(0), false, false, false).expect("var axis0 launches");
    assert_close(
        &read_f32(&var),
        &[6.25, 4.0, 3.0625, 5.0625],
        1e-5,
        "var axis0",
    );
    // Unbiased halves the divisor (n = 2): [12.5, 8, 6.125, 10.125].
    let var_u = var_std(&input, Some(0), false, true, false).expect("var axis0 unbiased");
    assert_close(
        &read_f32(&var_u),
        &[12.5, 8.0, 6.125, 10.125],
        1e-5,
        "var axis0 unbiased",
    );
    // Biased std of the same columns: [2.5, 2, 1.75, 2.25].
    let std = var_std(&input, Some(0), false, false, true).expect("std axis0 launches");
    assert_close(&read_f32(&std), &[2.5, 2.0, 1.75, 2.25], 1e-5, "std axis0");
}

/// Issue #87: the six Welford rows match the host formulas and record.
#[test]
#[ignore = "requires CUDA hardware"]
fn variance_and_std_match_the_host_reference_and_record() {
    require_cuda();
    // Population variance of 1..=6 = 17.5/6 = 35/12; sample = 3.5.
    let flat = upload_f32_shaped(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (var, recorded) = run1::<op::VarianceAll, _>(&flat, VarianceAttributes { unbiased: false });
    assert!(recorded >= 1, "var_all must record a tape entry");
    assert_close(&read_f32(&var), &[35.0 / 12.0], 1e-5, "var_all biased");
    let (var_u, _) = run1::<op::VarianceAll, _>(&flat, VarianceAttributes { unbiased: true });
    assert_close(&read_f32(&var_u), &[3.5], 1e-5, "var_all unbiased");
    let (std, recorded) = run1::<op::StdAll, _>(&flat, VarianceAttributes { unbiased: false });
    assert!(recorded >= 1, "std_all must record a tape entry");
    assert_close(
        &read_f32(&std),
        &[(35.0f64 / 12.0).sqrt()],
        1e-5,
        "std_all biased",
    );

    // matrix() grand mean 1.25, sum of squared deviations 54.
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);
    let (var, _) = run1::<op::VarianceAll, _>(&input, VarianceAttributes { unbiased: false });
    assert_close(&read_f32(&var), &[6.75], 1e-5, "matrix var_all biased");
    let (var_u, _) = run1::<op::VarianceAll, _>(&input, VarianceAttributes { unbiased: true });
    assert_close(
        &read_f32(&var_u),
        &[54.0 / 7.0],
        1e-5,
        "matrix var_all unbiased",
    );
    let (std, _) = run1::<op::StdAll, _>(&input, VarianceAttributes { unbiased: false });
    assert_close(
        &read_f32(&std),
        &[6.75f64.sqrt()],
        1e-5,
        "matrix std_all biased",
    );

    // Per-row (axis 1) biased variance: [1.421875, 9.546875].
    let (var_dim, recorded) = run1::<op::VarianceDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert!(recorded >= 1, "var_dim must record a tape entry");
    assert_close(
        &read_f32(&var_dim),
        &[1.421875, 9.546875],
        1e-5,
        "var_dim axis1 biased",
    );
    let (std_dim, recorded) = run1::<op::StdDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert!(recorded >= 1, "std_dim must record a tape entry");
    assert_close(
        &read_f32(&std_dim),
        &[1.421875f64.sqrt(), 9.546875f64.sqrt()],
        1e-5,
        "std_dim axis1 biased",
    );

    // keepdim keeps the reduced axis with extent one.
    let (kept, _) = run1::<op::VarianceKeepDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert_eq!(kept.shape.to_vec(), vec![2, 1], "var_keepdim shape");
    assert_close(
        &read_f32(&kept),
        &[1.421875, 9.546875],
        1e-5,
        "var_keepdim values",
    );
    let (kept_std, recorded) = run1::<op::StdKeepDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert!(recorded >= 1, "std_keepdim must record a tape entry");
    assert_eq!(kept_std.shape.to_vec(), vec![2, 1], "std_keepdim shape");
    assert_close(
        &read_f32(&kept_std),
        &[1.421875f64.sqrt(), 9.546875f64.sqrt()],
        1e-5,
        "std_keepdim values",
    );
}

/// Issue #87: the recorded backward recipes land the closed-form gradient.
///
/// Seeds are the ones vector `AutogradBackend::backward` plants on a
/// scalar-seeded walk, so each expectation is `d var / d x` (or `d std / d x`)
/// with `grad_out = 1`.
#[test]
#[ignore = "requires CUDA hardware"]
fn variance_and_std_backward_match_the_host_formulas() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);
    let input_id = TapeStorage::id(&input);

    // std_all: d std / d x_i = (x_i - mean) / (n * std), n = 8, mean = 1.25.
    let (out, recorded) = run1::<op::StdAll, _>(&input, VarianceAttributes { unbiased: false });
    assert!(recorded >= 1, "std_all must record before backward");
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("std_all backward on CUDA");
    let gx = grads.get(input_id).expect("std_all grads reach the input");
    let std = 6.75f64.sqrt();
    let want: Vec<f64> = values
        .iter()
        .map(|&v| (f64::from(v) - 1.25) / (8.0 * std))
        .collect();
    assert_close(&read_f32(gx), &want, 1e-4, "std_all dx");

    // var_dim axis 1: d var / d x = 2 * (x - mean_row) / n, n = 4.
    let (out, recorded) = run1::<op::VarianceDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert!(recorded >= 1, "var_dim must record before backward");
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("var_dim backward on CUDA");
    let gx = grads.get(input_id).expect("var_dim grads reach the input");
    let rows: Vec<&[f32]> = values.chunks(4).collect();
    let want: Vec<f64> = rows
        .iter()
        .flat_map(|row| {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / 4.0;
            row.iter()
                .map(|&v| 2.0 * (f64::from(v) - mean) / 4.0)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_close(&read_f32(gx), &want, 1e-5, "var_dim dx");

    // var_all: d var / d x_i = 2 * (x_i - mean) / n, n = 8.
    let (out, recorded) =
        run1::<op::VarianceAll, _>(&input, VarianceAttributes { unbiased: false });
    assert!(recorded >= 1, "var_all must record before backward");
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("var_all backward on CUDA");
    let gx = grads.get(input_id).expect("var_all grads reach the input");
    let want: Vec<f64> = values
        .iter()
        .map(|&v| 2.0 * (f64::from(v) - 1.25) / 8.0)
        .collect();
    assert_close(&read_f32(gx), &want, 1e-5, "var_all dx");
}

/// Issue #87: `Argsort` returns a permutation that addresses the input.
#[test]
#[ignore = "requires CUDA hardware"]
fn argsort_returns_a_permutation_that_addresses_the_input() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let ascending = ArgsortAttributes {
        axis: 1,
        descending: false,
        index_dtype: DTypeId::U32.descriptor(),
    };
    let (idx, recorded) = run1::<op::Argsort, _>(&input, ascending);
    // Argsort is training = false: no tape entry is owed.
    assert_eq!(recorded, 0, "argsort must not record a tape entry");
    let idx = download_i64(&idx);
    // row 0: [3, 1, 4, 1.5] -> 1, 1.5, 3, 4 at [1, 3, 0, 2]
    // row 1: [-2, 5, 0.5, -3] -> -3, -2, 0.5, 5 at [3, 0, 2, 1]
    assert_eq!(idx, vec![1, 3, 0, 2, 3, 0, 2, 1], "argsort ascending");
    for (flat, &position) in idx.iter().enumerate() {
        let row = flat / 4;
        let source = values[row * 4 + usize::try_from(position).unwrap()];
        let below = if flat % 4 == 0 {
            None
        } else {
            Some(values[row * 4 + usize::try_from(idx[flat - 1]).unwrap()])
        };
        if let Some(below) = below {
            assert!(
                f64::from(below) <= f64::from(source) + 1e-6,
                "ascending order broken at {flat}: {below} !<= {source}"
            );
        }
    }

    let descending = ArgsortAttributes {
        axis: 1,
        descending: true,
        index_dtype: DTypeId::U32.descriptor(),
    };
    let (idx, _) = run1::<op::Argsort, _>(&input, descending);
    let idx = download_i64(&idx);
    // row 0: [4, 3, 1.5, 1] at [2, 0, 3, 1]; row 1: [5, 0.5, -2, -3] at [1, 2, 0, 3].
    assert_eq!(idx, vec![2, 0, 3, 1, 1, 2, 0, 3], "argsort descending");
}

/// Issue #87: the composed `Norm` rows match L1/L2 and record.
#[test]
#[ignore = "requires CUDA hardware"]
fn norm_matches_the_host_magnitude_and_records() {
    require_cuda();
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let (l1, recorded) = run1::<op::Norm, _>(&input, NormAttributes { order: 1.0 });
    assert!(recorded >= 1, "norm order 1 must record a tape entry");
    // L1 of matrix(): 3+1+4+1.5+2+5+0.5+3 = 20.
    assert_close(&read_f32(&l1), &[20.0], 1e-5, "norm order 1");

    let (l2, recorded) = run1::<op::Norm, _>(&input, NormAttributes { order: 2.0 });
    assert!(recorded >= 1, "norm order 2 must record a tape entry");
    // L2: sqrt(9+1+16+2.25+4+25+0.25+9) = sqrt(66.5).
    assert_close(&read_f32(&l2), &[66.5f64.sqrt()], 1e-5, "norm order 2");
}
