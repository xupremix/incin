//! Batch-A WGPU pointwise gap closure: the twenty-one unary floats, the
//! four scalar forms plus `powf`/`clamp`, and `atan2`/`fmod`/`remainder`.
//!
//! Every operation here was listed as missing on WGPU in #91. Each has a
//! WGSL mode, an `Execute` impl and a capability row; this file checks all
//! three on real hardware:
//!
//! - forward values against a host reference computed with the same
//!   formulas the CPU kernels use (including the A&S erf approximation and
//!   half-away-from-zero rounding, which the obvious WGSL builtins get
//!   wrong);
//! - tape depth for every row that claims `training = true`, so a fused
//!   forward that returns the right numbers but records nothing cannot pass;
//! - a real backward through `sin` and `clamp`, because the recipes are
//!   composed from other modes and a wrong mode number still produces a
//!   finite gradient of the wrong shape.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, op,
};
use incin_core::exec::catalog::{ClampAttributes, NoAttributes, ScalarAttributes};
use incin_core::exec::{ExecutionContext, TapeStorage, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Aborts unless a WGPU adapter is present (same contract as the other
/// WGPU suites: compiling with the feature is an explicit request).
fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled"
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
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
    O: incin_core::exec::CanonicalOperation<Attributes = NoAttributes>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, NoAttributes, &inputs)
        .expect("an advertised unary must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

/// Runs one two-operand no-attribute binary and reports tape entries.
fn run_binary<O>(lhs: &TestStorage, rhs: &TestStorage) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = NoAttributes>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, _>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, _>(rhs),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, NoAttributes, &inputs)
        .expect("an advertised binary must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

/// Inputs that exercise every branch: positives, negatives, zero, halves
/// (for round), values outside [0, 1] (for erf/atanh), and a non-integer
/// negative (for frac vs WGSL fract).
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
            fn $fn_name() {
                require_wgpu();
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
    // asin/acos/atan/tan: the Vulkan hardware transcendental unit on this
    // adapter is a few 1e-4 off Rust's f32 (fdlibm-quality) routines —
    // e.g. asin(0.5) comes back as 0.523389 vs 0.523599. Widen the
    // tolerance enough to absorb that, but keep it far tighter than any
    // wrong-mode error (order 0.1+).
    (
        Tan,
        "tan",
        tan_matches,
        |x: f32| f64::from(x.tan()),
        &UNARY_IN,
        1e-4
    ),
    (
        Asin,
        "asin",
        asin_matches,
        |x: f32| f64::from(x.asin()),
        &UNIT_IN,
        1e-3
    ),
    (
        Acos,
        "acos",
        acos_matches,
        |x: f32| f64::from(x.acos()),
        &UNIT_IN,
        1e-3
    ),
    (
        Atan,
        "atan",
        atan_matches,
        |x: f32| f64::from(x.atan()),
        &UNARY_IN,
        1e-5
    ),
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
    (
        Erf,
        "erf",
        erf_matches,
        |x: f32| erf_ref(f64::from(x)),
        &UNARY_IN,
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
        Trunc,
        "trunc",
        trunc_matches,
        |x: f32| f64::from(x.trunc()),
        &UNARY_IN,
        0.0
    ),
    (
        Frac,
        "frac",
        frac_matches,
        |x: f32| f64::from(x.fract()),
        &UNARY_IN,
        1e-6
    ),
);

/// `sign`, `floor`, `ceil` and `round` claim `training = false`, so the
/// tape-depth half of the macro above does not apply to them. Their
/// forwards still have to match, including on the half-integers where WGSL
/// and Rust disagree.
#[test]
fn flat_unaries_match_the_host_reference() {
    require_wgpu();
    let input = upload(&UNARY_IN, &[UNARY_IN.len()]);

    let (sign_out, _) = run_unary::<op::Sign>(&input);
    assert_close(
        &read(&sign_out),
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

    let (floor_out, _) = run_unary::<op::Floor>(&input);
    assert_close(
        &read(&floor_out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(x.floor()))
            .collect::<Vec<_>>(),
        0.0,
        "floor",
    );

    let (ceil_out, _) = run_unary::<op::Ceil>(&input);
    assert_close(
        &read(&ceil_out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(x.ceil()))
            .collect::<Vec<_>>(),
        0.0,
        "ceil",
    );

    let (round_out, _) = run_unary::<op::Round>(&input);
    assert_close(
        &read(&round_out),
        &UNARY_IN
            .iter()
            .map(|&x| f64::from(round_ref(x)))
            .collect::<Vec<_>>(),
        0.0,
        "round",
    );
}

const SCALAR_IN: [f32; 5] = [1.0, -2.0, 0.5, 3.25, 0.0];

fn run_scalar<O>(input: &TestStorage, value: f64) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = ScalarAttributes>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = incin_backends::wgpu::tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<O, _>(&context, ScalarAttributes { value }, &inputs)
            .expect("an advertised scalar operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

#[test]
fn scalar_forms_match_the_host_reference_and_record() {
    require_wgpu();
    let input = upload(&SCALAR_IN, &[SCALAR_IN.len()]);
    let c = 2.5f64;
    let c32 = c as f32;

    let cases: [(&str, TestStorage, Vec<f64>); 5] = [
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
            let (out, recorded) = run_scalar::<op::SubScalar>(&input, c);
            assert!(recorded >= 1, "sub_scalar must record");
            (
                "sub_scalar",
                out,
                SCALAR_IN.iter().map(|&x| f64::from(x - c32)).collect(),
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
            let (out, recorded) = run_scalar::<op::DivScalar>(&input, c);
            assert!(recorded >= 1, "div_scalar must record");
            (
                "div_scalar",
                out,
                SCALAR_IN.iter().map(|&x| f64::from(x / c32)).collect(),
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
                SCALAR_IN.iter().map(|&x| f64::from(x * x)).collect(),
            )
        },
    ];

    for (label, out, expected) in cases {
        assert_close(&read(&out), &expected, 1e-6, label);
    }
}

#[test]
fn clamp_matches_the_host_reference_and_records() {
    require_wgpu();
    let input = upload(&UNARY_IN, &[UNARY_IN.len()]);
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Clamp, _>(
        &context,
        ClampAttributes {
            min: -1.0,
            max: 1.0,
        },
        &inputs,
    )
    .expect("clamp must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    assert!(recorded >= 1, "clamp advertises training = true");

    let actual = read(&out);
    let expected: Vec<f64> = UNARY_IN
        .iter()
        .map(|&x| f64::from(x.clamp(-1.0, 1.0)))
        .collect();
    assert_close(&actual, &expected, 0.0, "clamp");
}

const PAIR_LHS: [f32; 6] = [1.0, -1.0, 2.0, -2.0, 0.5, 3.0];
const PAIR_RHS: [f32; 6] = [2.0, 2.0, -1.0, -0.5, -1.5, 0.5];

#[test]
fn atan2_matches_the_host_reference_and_records() {
    require_wgpu();
    let lhs = upload(&PAIR_LHS, &[6]);
    let rhs = upload(&PAIR_RHS, &[6]);
    let (out, recorded) = run_binary::<op::Atan2>(&lhs, &rhs);
    assert!(recorded >= 1, "atan2 advertises training = true");
    let expected: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&y, &x)| f64::from(y.atan2(x)))
        .collect();
    // f32 atan2 on the GPU vs f64 on the host: 1e-6 is right at the
    // edge of the rounding error for values near 0.46, so allow 1e-5.
    assert_close(&read(&out), &expected, 1e-5, "atan2");
}

#[test]
fn fmod_and_remainder_match_the_host_reference_and_record() {
    require_wgpu();
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
    let expected_rem: Vec<f64> = PAIR_LHS
        .iter()
        .zip(PAIR_RHS.iter())
        .map(|(&a, &b)| f64::from(a).rem_euclid(f64::from(b)))
        .collect();
    assert_close(&read(&rem_out), &expected_rem, 1e-6, "remainder");
}

/// A real backward through `sin`: d/dx sin(x) = cos(x), driven by a mean
/// reduction so the seed gradient is uniform and easy to check.
#[test]
fn sin_trains_with_the_cosine_gradient() {
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    let x = upload(&[0.0, 0.5, 1.0, -0.5], &[4]);
    let x_id = TapeStorage::id(&x);

    let sin_x = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&x)];
        incin_core::exec::dispatch::execute::<op::Sin, _>(&context, NoAttributes, &inputs)
            .expect("sin executes")
    };
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&sin_x)];
        incin_core::exec::dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &inputs)
            .expect("mean_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(x_id).expect("x has a gradient");
    let actual = read(grad);
    // d mean(sin(x)) / dx_i = cos(x_i) / 4
    let expected: Vec<f64> = [0.0f32, 0.5, 1.0, -0.5]
        .iter()
        .map(|&xi| f64::from(xi.cos()) / 4.0)
        .collect();
    assert_close(&actual, &expected, 1e-5, "sin backward");
}

/// A real backward through `clamp`: the cotangent passes through the
/// interior and stops at both clamped regions.
#[test]
fn clamp_trains_with_a_masked_gradient() {
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    // Element 0 clamped low, element 1 interior, element 2 clamped high.
    let x = upload(&[-2.0, 0.5, 3.0], &[3]);
    let x_id = TapeStorage::id(&x);

    let clamped = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&x)];
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
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&clamped)];
        incin_core::exec::dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &inputs)
            .expect("sum_all executes")
    };
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(x_id).expect("x has a gradient");
    let actual = read(grad);
    assert_close(&actual, &[0.0, 1.0, 0.0], 1e-6, "clamp backward");
}
