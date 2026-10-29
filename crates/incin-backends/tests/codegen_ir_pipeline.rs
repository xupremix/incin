//! Integration tests for Kernel IR, Algebraic Optimization, Symbolic Differentiation, and Codegen DSL.

use incin_backends::codegen::{
    IrBinaryOp, IrExpr, IrTernaryOp, IrUnaryOp, define_binary_custom_op, define_unary_custom_op,
    exp, lower_scalar, relu, sigmoid,
};
use incin_core::tensor::dtype::DTypeId;

/// Aborts unless a CUDA device is present.
///
/// Replaces a `catch_unwind` probe whose result was bound and then used to
/// `return` early, reporting `ok` for a test that compiled and launched
/// nothing. The JIT test below is the one place this file exercises real
/// hardware, so a missing device is a broken run rather than a lighter one.
///
/// # Panics
///
/// If no CUDA device can be opened on ordinal 0.
#[cfg(feature = "cuda")]
fn require_cuda() {
    assert!(
        cudarc::driver::CudaContext::new(0).is_ok(),
        "no CUDA device, but the `cuda` feature is enabled -- that is an explicit request \
         for this backend. Skipping here would report `ok` for a test that ran nothing."
    );
}

#[test]
fn test_ir_expression_dsl_and_eval() {
    // f(x, y) = x^2 + 2*x*y + y^2 = (x + y)^2
    let x = IrExpr::arg(0);
    let y = IrExpr::arg(1);

    let expr = x.clone() * x + 2.0 * y.clone() * IrExpr::arg(0) + y.clone() * y;
    let result = expr.eval(&[3.0, 4.0]); // 3^2 + 2*4*3 + 4^2 = 9 + 24 + 16 = 49
    assert!((result - 49.0).abs() < 1e-6);
}

#[test]
fn test_symbolic_automatic_differentiation_abs_sign_at_zero() {
    // d/dx |x| is sign(x) with sign(0) = 0: the hand-written CUDA derivative,
    // CPU `Sign`, and PyTorch all agree, and the fused path must agree with
    // them rather than with the one-sided (x > 0 ? 1 : -1) form, which
    // silently diverges exactly where l1-style losses most often evaluate.
    let x = IrExpr::arg(0);
    let forward = IrExpr::unary(IrUnaryOp::Abs, x);
    let diff = forward.diff(0);

    for (val, expected) in [(-2.0, -1.0), (0.0, 0.0), (3.0, 1.0)] {
        let analytical = diff.eval(&[val]);
        assert_eq!(
            analytical, expected,
            "symbolic d|x|/dx at {val}: got {analytical}, want {expected}"
        );
    }

    // Central differences agree, including at the kink.
    let h = 1e-5;
    for &val in &[-2.0, 0.0, 3.0] {
        let fwd = |v: f64| v.abs();
        let numerical = (fwd(val + h) - fwd(val - h)) / (2.0 * h);
        let analytical = diff.eval(&[val]);
        assert!(
            (analytical - numerical).abs() < 1e-4,
            "val {val}: analytical {analytical} vs numerical {numerical}"
        );
    }
}

#[test]
fn test_ir_algebraic_simplification_and_constant_folding() {
    let x = IrExpr::arg(0);

    // x + 0 -> x
    let add_zero = (x.clone() + 0.0).optimize();
    assert_eq!(add_zero, IrExpr::arg(0));

    // x * 1 -> x
    let mul_one = (x.clone() * 1.0).optimize();
    assert_eq!(mul_one, IrExpr::arg(0));

    // x * 0 -> 0
    let mul_zero = (x.clone() * 0.0).optimize();
    assert_eq!(mul_zero, IrExpr::constant(0.0));

    // x - x -> 0
    let sub_self = (x.clone() - x).optimize();
    assert_eq!(sub_self, IrExpr::constant(0.0));

    // Constant folding: 2 + 3 * 4 -> 14
    let const_expr = (IrExpr::constant(2.0) + IrExpr::constant(3.0) * 4.0).optimize();
    assert_eq!(const_expr, IrExpr::constant(14.0));
}

#[test]
fn test_ir_fma_fusion() {
    let a = IrExpr::arg(0);
    let b = IrExpr::arg(1);
    let c = IrExpr::arg(2);

    // a * b + c -> fma(a, b, c)
    let fused = (a * b + c).optimize();
    match fused {
        IrExpr::Ternary(IrTernaryOp::Fma, ref m1, ref m2, ref acc) => {
            assert_eq!(**m1, IrExpr::arg(0));
            assert_eq!(**m2, IrExpr::arg(1));
            assert_eq!(**acc, IrExpr::arg(2));
        }
        other => panic!("expected FMA fusion, got {other:?}"),
    }
}

#[test]
fn test_symbolic_automatic_differentiation_sigmoid() {
    // f(x) = sigmoid(x) -> f'(x) = sigmoid(x) * (1 - sigmoid(x))
    let x = IrExpr::arg(0);
    let forward = sigmoid(x);
    let diff = forward.diff(0);

    for &val in &[-2.0, -0.5, 0.0, 1.0, 2.5] {
        let analytical = diff.eval(&[val]);

        // Numerical finite difference: (f(x + h) - f(x - h)) / (2h)
        let h = 1e-5;
        let f_plus = forward.eval(&[val + h]);
        let f_minus = forward.eval(&[val - h]);
        let numerical = (f_plus - f_minus) / (2.0 * h);

        assert!(
            (analytical - numerical).abs() < 1e-4,
            "val {val}: analytical {analytical} vs numerical {numerical}"
        );
    }
}

#[test]
fn test_symbolic_automatic_differentiation_swish() {
    // Swish: f(x) = x * sigmoid(x)
    let x = IrExpr::arg(0);
    let forward = x.clone() * sigmoid(x);
    let diff = forward.diff(0);

    for &val in &[-3.0, -1.0, 0.0, 0.5, 2.0] {
        let analytical = diff.eval(&[val]);

        let h = 1e-5;
        let f_plus = forward.eval(&[val + h]);
        let f_minus = forward.eval(&[val - h]);
        let numerical = (f_plus - f_minus) / (2.0 * h);

        assert!(
            (analytical - numerical).abs() < 1e-4,
            "val {val}: analytical {analytical} vs numerical {numerical}"
        );
    }
}

#[test]
fn test_symbolic_automatic_differentiation_complex_binary() {
    // f(x, y) = x^2 * y + exp(x * y)
    let x = IrExpr::arg(0);
    let y = IrExpr::arg(1);
    let forward = x.clone() * x.clone() * y.clone() + exp(x * y);

    // Partial derivative wrt x: 2*x*y + y*exp(x*y)
    let df_dx = forward.diff(0);
    // Partial derivative wrt y: x^2 + x*exp(x*y)
    let df_dy = forward.diff(1);

    let (x_val, y_val) = (1.5, 2.0);
    let h = 1e-5;

    // Check df/dx
    let num_df_dx =
        (forward.eval(&[x_val + h, y_val]) - forward.eval(&[x_val - h, y_val])) / (2.0 * h);
    assert!((df_dx.eval(&[x_val, y_val]) - num_df_dx).abs() < 1e-4);

    // Check df/dy
    let num_df_dy =
        (forward.eval(&[x_val, y_val + h]) - forward.eval(&[x_val, y_val - h])) / (2.0 * h);
    assert!((df_dy.eval(&[x_val, y_val]) - num_df_dy).abs() < 1e-4);
}

#[test]
fn test_kernel_definition_cuda_generation() {
    // Custom op: GeLU-like scaled polynomial: f(x) = x * relu(x) + 0.1 * x
    let op = define_unary_custom_op("scaled_relu_plus", DTypeId::F32, |x| {
        x.clone() * relu(x.clone()) + 0.1 * x
    });

    let forward_cuda = op.render_forward_cuda();
    assert!(forward_cuda.contains("extern \"C\" __global__ void scaled_relu_plus_forward"));
    assert!(forward_cuda.contains("const float* __restrict__ in0"));
    assert!(forward_cuda.contains("float* __restrict__ out"));

    let backward_cuda = op
        .render_backward_cuda(0)
        .expect("backward kernel for arg 0");
    assert!(backward_cuda.contains("extern \"C\" __global__ void scaled_relu_plus_backward_0"));
    assert!(backward_cuda.contains("const float* __restrict__ grad_out"));
    assert!(backward_cuda.contains("float* __restrict__ grad_in0"));
}

#[test]
fn test_binary_custom_op_backward_generation() {
    // Custom binary op: f(a, b) = a * sigmoid(b)
    let op = define_binary_custom_op("gated_linear_unit", DTypeId::F32, |a, b| a * sigmoid(b));

    assert_eq!(op.input_arity, 2);
    assert_eq!(op.backward_derivatives.len(), 2);

    let bwd_a = op.render_backward_cuda(0).expect("grad for a");
    assert!(bwd_a.contains("gated_linear_unit_backward_0"));
    assert!(bwd_a.contains("grad_in0"));

    let bwd_b = op.render_backward_cuda(1).expect("grad for b");
    assert!(bwd_b.contains("gated_linear_unit_backward_1"));
    assert!(bwd_b.contains("grad_in1"));
}

#[test]
fn test_cpu_jit_kernel_forward_and_backward() {
    use incin_backends::codegen::CpuJitKernel;

    // Custom op: Swish: f(x) = x * sigmoid(x)
    let op = define_unary_custom_op("swish_jit_test", DTypeId::F32, |x| x.clone() * sigmoid(x));
    let cpu_kernel = CpuJitKernel::new(op);

    let input = [1.0f32, 2.0, -1.0, 0.0];
    let mut output = [0.0f32; 4];
    cpu_kernel
        .eval_f32(&[&input], &mut output)
        .expect("CPU JIT eval");

    for (i, &val) in input.iter().enumerate() {
        let expected = val / (1.0 + (-val).exp());
        assert!((output[i] - expected).abs() < 1e-5);
    }

    let grad_out = [1.0f32, 1.0, 1.0, 1.0];
    let mut grad_in = [0.0f32; 4];
    cpu_kernel
        .eval_backward_f32(&grad_out, &[&input], 0, &mut grad_in)
        .expect("CPU JIT backward eval");

    // Check at x = 0.0: d/dx (x * sigmoid(x)) = sigmoid(0) + 0 * ... = 0.5
    assert!((grad_in[3] - 0.5).abs() < 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
// Launches a kernel, so it belongs to the hardware suite rather than the one
// that passes anywhere. `hardware.yml` runs both, and `require_cuda` below
// documents the rule this was the exception to: every hardware test is
// ignored, so reaching one means the caller asked for the device.
#[ignore = "requires CUDA hardware"]
fn test_cuda_jit_kernel_forward_and_backward() {
    use incin_backends::codegen::CudaJitKernel;
    use incin_backends::cuda::CudaBackendImpl;
    use incin_core::backend_authoring::HostInterop;
    use incin_core::tensor::device::Cuda;
    use incin_core::tensor::device::DeviceId;

    require_cuda();

    let op = define_unary_custom_op("swish_cuda_jit", DTypeId::F32, |x| x.clone() * sigmoid(x));
    // A compile failure used to `return`, which reported `ok`. That is the
    // defect this whole file is meant to catch: the CUDA embedding module
    // failed to compile for want of a `--gpu-architecture` flag and nothing
    // noticed, because the only test that would have seen it swallowed the
    // error and passed.
    let jit_kernel = CudaJitKernel::compile(op, 0)
        .expect("the JIT must compile swish; a failure here is the bug, not a reason to skip");

    let values = [1.0f32, 2.0, -1.0, 0.0];
    let in_storage = CudaBackendImpl::<Cuda>::from_bytes::<f32>(
        bytemuck::cast_slice(&values),
        &[4],
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
    )
    .expect("create input storage");

    let out_storage = jit_kernel
        .launch_forward(&[&in_storage])
        .expect("launch forward JIT");
    assert_eq!(out_storage.shape.dims(), &[4]);

    let bytes = CudaBackendImpl::<Cuda>::to_bytes::<f32>(&out_storage).expect("readback");
    let out_host: &[f32] = bytemuck::cast_slice(&bytes);
    for (i, &val) in values.iter().enumerate() {
        let expected = val / (1.0 + (-val).exp());
        assert!((out_host[i] - expected).abs() < 1e-4);
    }
}

// ---------------------------------------------------------------------------
// The transcendental and rounding vocabulary
//
// These operators exist so `codegen::catalog` can eventually express the
// pointwise operations it currently declines (`tan`, `asin`, `erf`, the
// rounding family and the rest), each of which ships a hand-written CUDA
// literal and a hand-written derivative literal that nothing checks. The
// checks below are the host half of that: they pin the derivative rules and
// the emitted call text without needing a device. The GPU half, comparing the
// emitted kernel against the shipped literal, lives in
// `cuda::ops::ir_conformance_tests` and runs only with hardware.
// ---------------------------------------------------------------------------

/// Central finite difference of `expr` in argument 0.
fn numerical_derivative(expr: &IrExpr, at: f64, h: f64) -> f64 {
    (expr.eval(&[at + h]) - expr.eval(&[at - h])) / (2.0 * h)
}

#[test]
fn the_new_unary_derivatives_match_a_numerical_reference() {
    // Sample points are chosen inside each operator's domain, and far enough
    // from its singularities that a central difference is meaningful: `asin`
    // and `acos` blow up at +-1, `acosh` at 1, `atanh` at +-1, and `tan` at
    // +-pi/2.
    let cases: &[(IrUnaryOp, &[f64])] = &[
        (IrUnaryOp::Tan, &[-1.0, -0.3, 0.0, 0.5, 1.2]),
        (IrUnaryOp::Asin, &[-0.8, -0.3, 0.0, 0.4, 0.85]),
        (IrUnaryOp::Acos, &[-0.8, -0.3, 0.0, 0.4, 0.85]),
        (IrUnaryOp::Atan, &[-3.0, -0.5, 0.0, 1.0, 4.0]),
        (IrUnaryOp::Sinh, &[-2.0, -0.5, 0.0, 1.0, 2.0]),
        (IrUnaryOp::Cosh, &[-2.0, -0.5, 0.0, 1.0, 2.0]),
        (IrUnaryOp::Asinh, &[-2.0, -0.5, 0.0, 1.0, 3.0]),
        (IrUnaryOp::Acosh, &[1.2, 1.8, 3.0, 6.0]),
        (IrUnaryOp::Atanh, &[-0.8, -0.3, 0.0, 0.4, 0.85]),
    ];

    for &(op, samples) in cases {
        let forward = IrExpr::unary(op, IrExpr::arg(0));
        let derivative = forward.diff(0);
        for &x in samples {
            let symbolic = derivative.eval(&[x]);
            let numerical = numerical_derivative(&forward, x, 1e-5);
            let tolerance = 1e-4 * symbolic.abs().max(numerical.abs()).max(1.0);
            assert!(
                (symbolic - numerical).abs() <= tolerance,
                "{op:?} at {x}: symbolic {symbolic} vs numerical {numerical}"
            );
        }
    }
}

#[test]
fn the_erf_derivative_matches_a_numerical_reference() {
    // `erf` is separated out because the host evaluator is a rational
    // approximation accurate to about 1.5e-7, not the device function. A
    // central difference at h = 1e-5 would divide that error by 1e-5 and
    // report 1e-2 of noise, so the step is widened until the approximation
    // error is the smaller term.
    let forward = IrExpr::unary(IrUnaryOp::Erf, IrExpr::arg(0));
    let derivative = forward.diff(0);
    for &x in &[-1.5, -0.5, 0.0, 0.7, 1.8] {
        let symbolic = derivative.eval(&[x]);
        let numerical = numerical_derivative(&forward, x, 1e-2);
        assert!(
            (symbolic - numerical).abs() <= 1e-3,
            "erf at {x}: symbolic {symbolic} vs numerical {numerical}"
        );
    }
}

#[test]
fn the_host_erf_approximation_tracks_the_real_function() {
    // The values the rational approximation is judged against, and the bound
    // it is documented to hold to. A change that swaps the approximation for
    // another has to keep this true or it is not the same function.
    let forward = IrExpr::unary(IrUnaryOp::Erf, IrExpr::arg(0));
    let known: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (-1.0, -0.842_700_792_949_714_9),
    ];
    for &(x, expected) in known {
        let got = forward.eval(&[x]);
        assert!(
            (got - expected).abs() <= 2e-7,
            "erf({x}): got {got}, expected {expected}"
        );
    }
}

#[test]
fn the_rounding_family_differentiates_to_zero() {
    // Piecewise constant, so the derivative is zero everywhere it exists and
    // the jumps are a measure-zero set. This is the same answer the CPU
    // backend records for these operations.
    for op in [
        IrUnaryOp::Floor,
        IrUnaryOp::Ceil,
        IrUnaryOp::Round,
        IrUnaryOp::Trunc,
    ] {
        let derivative = IrExpr::unary(op, IrExpr::arg(0)).diff(0);
        for &x in &[-2.5, -0.5, 0.0, 0.5, 1.5, 2.5] {
            assert!(
                derivative.eval(&[x]).abs() < f64::EPSILON,
                "{op:?} at {x} should have a zero derivative"
            );
        }
    }
}

#[test]
fn the_rounding_family_evaluates_the_way_rust_does() {
    // `round` is the one worth pinning: it breaks ties away from zero, which
    // is what `f64::round` and CUDA's `round` do, and not what `rint` does.
    let round = IrExpr::unary(IrUnaryOp::Round, IrExpr::arg(0));
    let trunc = IrExpr::unary(IrUnaryOp::Trunc, IrExpr::arg(0));
    for &x in &[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
        assert!(
            (round.eval(&[x]) - x.round()).abs() < f64::EPSILON,
            "round({x})"
        );
        assert!(
            (trunc.eval(&[x]) - x.trunc()).abs() < f64::EPSILON,
            "trunc({x})"
        );
    }
    // Ties to even would give 2.0 here, away from zero gives 3.0.
    assert!((round.eval(&[2.5]) - 3.0).abs() < f64::EPSILON);
}

#[test]
fn atan2_differentiates_in_both_arguments() {
    // d/da atan2(a, b) = b / (a^2 + b^2), d/db = -a / (a^2 + b^2).
    let forward = IrExpr::binary(IrBinaryOp::Atan2, IrExpr::arg(0), IrExpr::arg(1));
    let d_da = forward.diff(0);
    let d_db = forward.diff(1);

    for &(a, b) in &[(1.0, 2.0), (-1.5, 0.5), (0.3, -2.0), (-2.0, -3.0)] {
        let denominator = a * a + b * b;
        let expected_da = b / denominator;
        let expected_db = -a / denominator;
        assert!(
            (d_da.eval(&[a, b]) - expected_da).abs() < 1e-9,
            "d/da atan2({a}, {b})"
        );
        assert!(
            (d_db.eval(&[a, b]) - expected_db).abs() < 1e-9,
            "d/db atan2({a}, {b})"
        );
        // And the forward agrees with the host function it is named after.
        assert!((forward.eval(&[a, b]) - a.atan2(b)).abs() < 1e-12);
    }
}

#[test]
fn the_new_operators_render_the_device_call_in_both_precisions() {
    // Both CUDA emitters have to name the same function for one operator.
    // `render_cuda_expr` writes against the IR's own kernel signature and
    // `lower_scalar` against caller-named operands, which is the only
    // difference the text below should show.
    let expectations: &[(IrUnaryOp, &str)] = &[
        (IrUnaryOp::Tan, "tan"),
        (IrUnaryOp::Asin, "asin"),
        (IrUnaryOp::Acos, "acos"),
        (IrUnaryOp::Atan, "atan"),
        (IrUnaryOp::Sinh, "sinh"),
        (IrUnaryOp::Cosh, "cosh"),
        (IrUnaryOp::Asinh, "asinh"),
        (IrUnaryOp::Acosh, "acosh"),
        (IrUnaryOp::Atanh, "atanh"),
        (IrUnaryOp::Erf, "erf"),
        (IrUnaryOp::Floor, "floor"),
        (IrUnaryOp::Ceil, "ceil"),
        (IrUnaryOp::Round, "round"),
        (IrUnaryOp::Trunc, "trunc"),
    ];

    for &(op, name) in expectations {
        let expr = IrExpr::unary(op, IrExpr::arg(0));

        let f32_text = expr.render_cuda_expr(DTypeId::F32);
        assert_eq!(f32_text, format!("{name}f(in0[idx])"), "{op:?} in f32");
        let f64_text = expr.render_cuda_expr(DTypeId::F64);
        assert_eq!(f64_text, format!("{name}(in0[idx])"), "{op:?} in f64");

        let fragment = lower_scalar(&expr, &["x"], DTypeId::F32).expect("lowers");
        assert!(
            fragment.value.contains(&format!("{name}f(x)"))
                || fragment
                    .prologue
                    .iter()
                    .any(|line| line.contains(&format!("{name}f(x)"))),
            "{op:?} did not emit {name}f(x); got {fragment:?}"
        );
    }

    let atan2 = IrExpr::binary(IrBinaryOp::Atan2, IrExpr::arg(0), IrExpr::arg(1));
    assert_eq!(
        atan2.render_cuda_expr(DTypeId::F32),
        "atan2f(in0[idx], in1[idx])"
    );
    assert_eq!(
        atan2.render_cuda_expr(DTypeId::F64),
        "atan2(in0[idx], in1[idx])"
    );
}
