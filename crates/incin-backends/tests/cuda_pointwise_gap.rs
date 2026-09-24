//! Issue #86 pointwise gap closure on CUDA: the twenty-three unary maths,
//! five activations, and eleven scalar/binary elementwise rows the issue
//! listed.
//!
//! The issue body predates the parity commit: all 39 rows are already
//! advertised and have an `Execute` impl, which the compile-time
//! `assert_every_advertised_cuda_row_executes` gate in `cuda/executor.rs`
//! holds. This file is the runtime half, every test `#[ignore]`d until a
//! CUDA runner picks it up:
//!
//! - forward values against a host reference computed with the same
//!   formulas the CPU kernels use (including the A&S erf approximation and
//!   half-away-from-zero rounding);
//! - tape depth for every row that claims `training = true`, so a forward
//!   that returns the right numbers but records nothing cannot pass — the
//!   hole that `frac`, `atan2`, `fmod` and `remainder` each had before the
//!   recording fixes in `cuda/backend/elementwise.rs`;
//! - real backwards through `sin`, `clamp`, `frac` and `atan2`, because a
//!   tape entry with a wrong recipe still produces a finite gradient of the
//!   right shape.
#![cfg(feature = "cuda")]

use incin_backends::cuda::{CudaBackendImpl, tape_depth};
use incin_core::backend_authoring::{AutogradBackend, Execute, HostInterop, StorageBackend};
use incin_core::exec::catalog::{ClampAttributes, LerpAttributes, NoAttributes, ScalarAttributes};
use incin_core::exec::{CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle, op};
use incin_core::prelude::{CudaN, DTypeId, DeviceId, Local};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Aborts unless a CUDA device is present.
///
/// Every caller is `#[ignore]`d, so reaching one is a deliberate request
/// for the hardware run; skipping here would report `ok` for a test that
/// launched nothing.
fn require_cuda() {
    assert!(
        TestBackend::from_bytes::<f32>(
            bytemuck::cast_slice(&[1.0f32]),
            &[1],
            DTypeId::F32.into(),
            &DeviceId::cuda(0),
        )
        .is_ok(),
        "no CUDA device, but this test is #[ignore]d -- running it is an explicit request for hardware"
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    let bytes = TestBackend::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice::<u8, f32>(&bytes)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

/// A&S 7.1.26, the same approximation CPU's `erf_approx_f64` evaluates.
fn erf_ref(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let v = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * v);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-v * v).exp())
}

/// Half-away-from-zero, matching Rust `f32::round`.
fn round_ref(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol || (a.is_nan() && e.is_nan()),
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

/// Runs one no-attribute unary and reports the tape entries it added.
fn run_unary<O>(input: &TestStorage) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = NoAttributes>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, NoAttributes, &inputs)
        .expect("an advertised unary must execute");
    (out, tape_depth() - before)
}

/// Runs one two-operand no-attribute binary and reports tape entries.
fn run_binary<O>(lhs: &TestStorage, rhs: &TestStorage) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = NoAttributes>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(rhs),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, NoAttributes, &inputs)
        .expect("an advertised binary must execute");
    (out, tape_depth() - before)
}

/// Runs one scalar-attribute unary and reports tape entries.
fn run_scalar<O>(input: &TestStorage, value: f64) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = ScalarAttributes>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    let before = tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<O, _>(&context, ScalarAttributes { value }, &inputs)
            .expect("an advertised scalar operation must execute");
    (out, tape_depth() - before)
}

/// Inputs that exercise every branch: positives, negatives, zero, halves
/// (for round), values outside [0, 1] (for erf/atanh), and non-integers
/// (for frac).
const UNARY_IN: [f32; 8] = [0.5, -0.5, 1.5, -1.5, 2.5, 0.0, -2.5, 0.75];

/// Values inside the domain shared by asin/atanh and strictly above 1 for
/// acosh.
const UNIT_IN: [f32; 6] = [0.0, 0.5, -0.5, 0.9, -0.9, 0.25];
const ACOSH_IN: [f32; 4] = [1.0, 1.5, 2.0, 3.0];
/// Positive inputs for log2/log10/rsqrt (and non-zero for rsqrt).
const LOG_IN: [f32; 5] = [1.0, 2.0, 4.0, 10.0, 0.25];

macro_rules! unary_matches {
    ($(($op:ident, $name:literal, $fn_name:ident, $f:expr, $input:expr, $tol:expr)),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires CUDA hardware"]
            fn $fn_name() {
                require_cuda();
                let input = upload($input, &[$input.len()]);
                let (out, recorded) = run_unary::<op::$op>(&input);
                let actual = read(&out);
                let expected: Vec<f64> =
                    $input.iter().map(|&x| $f(x)).collect();
                assert_close(&actual, &expected, $tol, $name);
                // Every row in this macro claims training = true.
                assert!(
                    recorded >= 1,
                    "{} advertises training = true, so it must leave at \
                     least one tape entry",
                    $name
                );
            }
        )*
    };
}

unary_matches!(
    (
        Abs,
        "abs",
        abs_matches,
        |x: f32| f64::from(x.abs()),
        &UNARY_IN,
        0.0
    ),
    (
        Neg,
        "neg",
        neg_matches,
        |x: f32| f64::from(-x),
        &UNARY_IN,
        0.0
    ),
    (
        Sin,
        "sin",
        sin_matches,
        |x: f32| f64::from(x.sin()),
        &UNARY_IN,
        1e-6
    ),
    (
        Cos,
        "cos",
        cos_matches,
        |x: f32| f64::from(x.cos()),
        &UNARY_IN,
        1e-6
    ),
    // tan(1.5) is about 14.1, where one f32 ulp is already ~1e-6; widen a
    // little without getting anywhere near a wrong-mode error (order 0.1+).
    (
        Tan,
        "tan",
        tan_matches,
        |x: f32| f64::from(x.tan()),
        &UNARY_IN,
        1e-5
    ),
    (
        Asin,
        "asin",
        asin_matches,
        |x: f32| f64::from(x.asin()),
        &UNIT_IN,
        1e-6
    ),
    (
        Acos,
        "acos",
        acos_matches,
        |x: f32| f64::from(x.acos()),
        &UNIT_IN,
        1e-6
    ),
    (
        Atan,
        "atan",
        atan_matches,
        |x: f32| f64::from(x.atan()),
        &UNARY_IN,
        1e-6
    ),
    // sinh/cosh reach ~6 at the end of UNARY_IN, where two one-ulp f32
    // evaluations can sit ~1e-6 apart; 1e-5 absorbs that and still catches
    // a wrong hyperbolic mode.
    (
        Sinh,
        "sinh",
        sinh_matches,
        |x: f32| f64::from(x.sinh()),
        &UNARY_IN,
        1e-5
    ),
    (
        Cosh,
        "cosh",
        cosh_matches,
        |x: f32| f64::from(x.cosh()),
        &UNARY_IN,
        1e-5
    ),
    (
        Asinh,
        "asinh",
        asinh_matches,
        |x: f32| f64::from(x.asinh()),
        &UNIT_IN,
        1e-6
    ),
    (
        Acosh,
        "acosh",
        acosh_matches,
        |x: f32| f64::from(x.acosh()),
        &ACOSH_IN,
        1e-6
    ),
    (
        Atanh,
        "atanh",
        atanh_matches,
        |x: f32| f64::from(x.atanh()),
        &UNIT_IN,
        1e-6
    ),
    // CUDA evaluates `erff`, the device's true erf, while this reference is
    // the A&S rational approximation CPU runs; the two sit within ~2e-7 of
    // each other, which 1e-6 absorbs.
    (
        Erf,
        "erf",
        erf_matches,
        |x: f32| erf_ref(f64::from(x)),
        &UNARY_IN,
        1e-6
    ),
    (
        Log2,
        "log2",
        log2_matches,
        |x: f32| f64::from(x.log2()),
        &LOG_IN,
        1e-6
    ),
    (
        Log10,
        "log10",
        log10_matches,
        |x: f32| f64::from(x.log10()),
        &LOG_IN,
        1e-6
    ),
    (
        Rsqrt,
        "rsqrt",
        rsqrt_matches,
        |x: f32| 1.0 / f64::from(x.sqrt()),
        &LOG_IN,
        1e-6
    ),
    // Every value in UNARY_IN has an exact fractional part in f32, so the
    // kernel's `x - truncf(x)` and Rust's `fract` agree bit for bit.
    (
        Frac,
        "frac",
        frac_matches,
        |x: f32| f64::from(x.fract()),
        &UNARY_IN,
        0.0
    ),
    (
        Elu,
        "elu",
        elu_matches,
        |x: f32| f64::from(if x >= 0.0 { x } else { x.exp() - 1.0 }),
        &UNARY_IN,
        1e-6
    ),
    // The kernel spells the first constant 0.7978845608f; clippy's
    // 0.797_884_6_f32 (seven significant digits) parses to the identical
    // f32, so both sides still evaluate the same coefficients and only
    // rounding order differs. 0.044715f32 is the kernel's literal as-is.
    (
        Gelu,
        "gelu",
        gelu_matches,
        |x: f32| {
            f64::from(0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715f32 * x * x * x)).tanh()))
        },
        &UNARY_IN,
        1e-5
    ),
    (
        Mish,
        "mish",
        mish_matches,
        |x: f32| f64::from(x * x.exp().ln_1p().tanh()),
        &UNARY_IN,
        1e-5
    ),
    (
        Swish,
        "swish",
        swish_matches,
        |x: f32| f64::from(x / (1.0 + (-x).exp())),
        &UNARY_IN,
        1e-6
    ),
);

/// `step`, `sign`, `floor`, `ceil` and `round` claim `training = false`,
/// and `trunc` claims `training = true` but is one of the conformance
/// harness's `carries_no_gradient` rows (piecewise constant, derivative
/// zero), so none of them is required to push a tape entry. Their forwards
/// still have to match, including on the half-integers where a
/// banker's-rounding device would disagree.
#[test]
#[ignore = "requires CUDA hardware"]
fn flat_unaries_match_the_host_reference() {
    require_cuda();
    let input = upload(&UNARY_IN, &[UNARY_IN.len()]);

    let (out, _) = run_unary::<op::Step>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        0.0,
        "step",
    );

    let (out, _) = run_unary::<op::Sign>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
        0.0,
        "sign",
    );

    let (out, _) = run_unary::<op::Floor>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(x.floor()))
            .collect::<Vec<_>>(),
        0.0,
        "floor",
    );

    let (out, _) = run_unary::<op::Ceil>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(x.ceil()))
            .collect::<Vec<_>>(),
        0.0,
        "ceil",
    );

    let (out, _) = run_unary::<op::Round>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(round_ref(x)))
            .collect::<Vec<_>>(),
        0.0,
        "round",
    );

    let (out, _) = run_unary::<op::Trunc>(&input);
    assert_close(
        &read(&out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(x.trunc()))
            .collect::<Vec<_>>(),
        0.0,
        "trunc",
    );
}

const SCALAR_IN: [f32; 5] = [1.0, -2.0, 0.5, 3.25, 0.0];

#[test]
#[ignore = "requires CUDA hardware"]
fn scalar_forms_match_the_host_reference_and_record() {
    require_cuda();
    let input = upload(&SCALAR_IN, &[SCALAR_IN.len()]);
    let c = 2.5f64;
    let c32 = c as f32;

    let cases: [(&str, TestStorage, Vec<f64>); 3] = [
        {
            let (out, recorded) = run_scalar::<op::AddScalar>(&input, c);
            assert!(recorded >= 1, "add_scalar must record");
            (
                "add_scalar",
                out,
                SCALAR_IN.iter().map(|&x| f64::from(x + c32)).collect(),
            )
        },
        {
            let (out, recorded) = run_scalar::<op::MulScalar>(&input, c);
            assert!(recorded >= 1, "mul_scalar must record");
            (
                "mul_scalar",
                out,
                SCALAR_IN.iter().map(|&x| f64::from(x * c32)).collect(),
            )
        },
        {
            // Exponent 2.0 keeps every base's `powf` well-defined on the
            // host reference, including the negatives.
            let (out, recorded) = run_scalar::<op::Powf>(&input, 2.0);
            assert!(recorded >= 1, "powf must record");
            (
                "powf",
                out,
                SCALAR_IN.iter().map(|&x| f64::from(x).powf(2.0)).collect(),
            )
        },
    ];

    for (label, out, expected) in cases {
        assert_close(&read(&out), &expected, 1e-5, label);
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn clamp_matches_the_host_reference_and_records() {
    require_cuda();
    let input = upload(&UNARY_IN, &[UNARY_IN.len()]);
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
        &input,
    )];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Clamp, _>(
        &context,
        ClampAttributes {
            min: -1.0,
            max: 1.0,
        },
        &inputs,
    )
    .expect("clamp must execute");
    let recorded = tape_depth() - before;
    assert!(recorded >= 1, "clamp advertises training = true");

    let expected: Vec<f64> = UNARY_IN
        .iter()
        .map(|&x| f64::from(x.clamp(-1.0, 1.0)))
        .collect();
    assert_close(&read(&out), &expected, 0.0, "clamp");
}

const PAIR_LHS: [f32; 6] = [1.0, -1.0, 2.0, -2.0, 0.5, 3.0];
const PAIR_RHS: [f32; 6] = [2.0, 2.0, -1.0, -0.5, -1.5, 0.5];

#[test]
#[ignore = "requires CUDA hardware"]
fn atan2_matches_the_host_reference_and_records() {
    require_cuda();
    let lhs = upload(&PAIR_LHS, &[6]);
    let rhs = upload(&PAIR_RHS, &[6]);
    let (out, recorded) = run_binary::<op::Atan2>(&lhs, &rhs);
    assert!(recorded >= 1, "atan2 advertises training = true");
    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&y, &x)| f64::from(y.atan2(x)))
        .collect();
    // f32 atan2 on the GPU vs f64 on the host: 1e-5, the same headroom the
    // WGPU gap suite allows for values near 0.46.
    assert_close(&read(&out), &expected, 1e-5, "atan2");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fmod_and_remainder_match_the_host_reference_and_record() {
    require_cuda();
    let lhs = upload(&PAIR_LHS, &[6]);
    let rhs = upload(&PAIR_RHS, &[6]);

    let (fmod_out, fmod_recorded) = run_binary::<op::Fmod>(&lhs, &rhs);
    assert!(fmod_recorded >= 1, "fmod advertises training = true");
    // Rust `f64::%` for floats is the truncated-division remainder, which
    // is what CPU's `canonical_fmod` applies.
    let expected_fmod: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from(a) % f64::from(b))
        .collect();
    assert_close(&read(&fmod_out), &expected_fmod, 1e-6, "fmod");

    let (rem_out, rem_recorded) = run_binary::<op::Remainder>(&lhs, &rhs);
    assert!(rem_recorded >= 1, "remainder advertises training = true");
    // The least non-negative residue, CPU's `canonical_remainder`. The pair
    // `(-1, 2)` is where this diverges from an IEEE `remainderf` (which
    // would answer `-1`), so it is load-bearing for the kernel to be the
    // fmod-adjusted spelling rather than the libdevice one.
    let expected_rem: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from(a).rem_euclid(f64::from(b)))
        .collect();
    assert_close(&read(&rem_out), &expected_rem, 1e-6, "remainder");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn maximum_minimum_and_abs_diff_match_the_host_reference_and_record() {
    require_cuda();
    let lhs = upload(&PAIR_LHS, &[6]);
    let rhs = upload(&PAIR_RHS, &[6]);

    let (out, recorded) = run_binary::<op::Maximum>(&lhs, &rhs);
    assert!(recorded >= 1, "maximum advertises training = true");
    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from(a.max(b)))
        .collect();
    assert_close(&read(&out), &expected, 0.0, "maximum");

    let (out, recorded) = run_binary::<op::Minimum>(&lhs, &rhs);
    assert!(recorded >= 1, "minimum advertises training = true");
    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from(a.min(b)))
        .collect();
    assert_close(&read(&out), &expected, 0.0, "minimum");

    let (out, recorded) = run_binary::<op::AbsDiff>(&lhs, &rhs);
    assert!(recorded >= 1, "abs_diff advertises training = true");
    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from((a - b).abs()))
        .collect();
    assert_close(&read(&out), &expected, 0.0, "abs_diff");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn lerp_matches_the_host_reference_and_records() {
    require_cuda();
    let start = upload(&PAIR_LHS, &[6]);
    let end = upload(&PAIR_RHS, &[6]);
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&start),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&end),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Lerp, _>(
        &context,
        LerpAttributes { weight: 0.25 },
        &inputs,
    )
    .expect("lerp must execute");
    let recorded = tape_depth() - before;
    assert!(recorded >= 1, "lerp advertises training = true");

    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&s, &e)| f64::from(s + (e - s) * 0.25))
        .collect();
    assert_close(&read(&out), &expected, 1e-6, "lerp");
}

/// A real backward through `sin`: d/dx sin(x) = cos(x), driven by a mean
/// reduction so the seed gradient is uniform and easy to check.
#[test]
#[ignore = "requires CUDA hardware"]
fn sin_trains_with_the_cosine_gradient() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let x = upload(&[0.0, 0.5, 1.0, -0.5], &[4]);
    let x_id = TapeStorage::id(&x);

    let sin_x = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&x)];
        incin_core::exec::dispatch::execute::<op::Sin, _>(&context, NoAttributes, &inputs)
            .expect("sin executes")
    };
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
            &sin_x,
        )];
        incin_core::exec::dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &inputs)
            .expect("mean_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(x_id).expect("x has a gradient");
    // d mean(sin(x)) / dx_i = cos(x_i) / 4
    let expected: Vec<f64> = [0.0f32, 0.5, 1.0, -0.5]
        .iter()
        .map(|&xi| f64::from(xi.cos()) / 4.0)
        .collect();
    assert_close(&read(grad), &expected, 1e-5, "sin backward");
}

/// A real backward through `clamp`: the cotangent passes through the
/// interior and stops at both clamped regions.
#[test]
#[ignore = "requires CUDA hardware"]
fn clamp_trains_with_a_masked_gradient() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    // Element 0 clamped low, element 1 interior, element 2 clamped high.
    let x = upload(&[-2.0, 0.5, 3.0], &[3]);
    let x_id = TapeStorage::id(&x);

    let clamped = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&x)];
        incin_core::exec::dispatch::execute::<op::Clamp, _>(
            &context,
            ClampAttributes {
                min: -1.0,
                max: 1.0,
            },
            &inputs,
        )
        .expect("clamp executes")
    };
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
            &clamped,
        )];
        incin_core::exec::dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &inputs)
            .expect("sum_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(x_id).expect("x has a gradient");
    assert_close(&read(grad), &[0.0, 1.0, 0.0], 1e-6, "clamp backward");
}

/// A real backward through `frac`: the derivative is 1 wherever it exists,
/// so the mean's uniform cotangent passes straight through to every input.
/// This is the recipe `cuda_frac_storage`'s tape entry carries; before the
/// recording fix the entry did not exist at all and this walk would find
/// no gradient for `x`.
#[test]
#[ignore = "requires CUDA hardware"]
fn frac_trains_with_an_identity_gradient() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let x = upload(&[1.25, -0.75, 3.5], &[3]);
    let x_id = TapeStorage::id(&x);

    let frac_x = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&x)];
        incin_core::exec::dispatch::execute::<op::Frac, _>(&context, NoAttributes, &inputs)
            .expect("frac executes")
    };
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
            &frac_x,
        )];
        incin_core::exec::dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &inputs)
            .expect("mean_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(x_id).expect("x has a gradient");
    assert_close(
        &read(grad),
        &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        1e-6,
        "frac backward",
    );
}

/// A real backward through `atan2`: the quotient rule CPU's
/// `canonical_atan2` uses, `d/dy = g * x / (x^2 + y^2)` and
/// `d/dx = g * (-y) / (x^2 + y^2)`, seeded by a sum so `g` is 1.
#[test]
#[ignore = "requires CUDA hardware"]
fn atan2_trains_with_the_quotient_rule() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let y = upload(&[1.0, 2.0], &[2]);
    let x = upload(&[3.0, 4.0], &[2]);
    let y_id = TapeStorage::id(&y);
    let x_id = TapeStorage::id(&x);

    let angles = {
        let inputs = [
            TensorHandle::from_storage::<TestBackend, f32, Local>(&y),
            TensorHandle::from_storage::<TestBackend, f32, Local>(&x),
        ];
        incin_core::exec::dispatch::execute::<op::Atan2, _>(&context, NoAttributes, &inputs)
            .expect("atan2 executes")
    };
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
            &angles,
        )];
        incin_core::exec::dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &inputs)
            .expect("sum_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");

    let grad_y = grads.get(y_id).expect("y has a gradient");
    // dy_i = x_i / (x_i^2 + y_i^2)
    assert_close(
        &read(grad_y),
        &[3.0 / 10.0, 4.0 / 20.0],
        1e-5,
        "atan2 backward wrt y",
    );

    let grad_x = grads.get(x_id).expect("x has a gradient");
    // dx_i = -y_i / (x_i^2 + y_i^2)
    assert_close(
        &read(grad_x),
        &[-1.0 / 10.0, -2.0 / 20.0],
        1e-5,
        "atan2 backward wrt x",
    );
}
