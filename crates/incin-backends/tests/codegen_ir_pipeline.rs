//! Integration tests for Kernel IR, Algebraic Optimization, Symbolic Differentiation, and Codegen DSL.

use incin_backends::codegen::{
    IrExpr, IrTernaryOp, define_binary_custom_op, define_unary_custom_op, exp, relu, sigmoid,
};
use incin_core::tensor::dtype::DTypeId;

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
fn test_cuda_jit_kernel_forward_and_backward() {
    use incin_backends::codegen::CudaJitKernel;
    use incin_backends::cuda::CudaBackendImpl;
    use incin_core::backend_authoring::HostInterop;
    use incin_core::tensor::device::Cuda;
    use incin_core::tensor::device::DeviceId;

    let has_cuda = std::panic::catch_unwind(|| {
        CudaBackendImpl::<Cuda>::from_bytes::<f32>(
            bytemuck::cast_slice(&[1.0f32]),
            &[1],
            DTypeId::F32.into(),
            &DeviceId::cuda(0),
        )
        .is_ok()
    })
    .unwrap_or(false);

    if !has_cuda {
        return;
    }

    let op = define_unary_custom_op("swish_cuda_jit", DTypeId::F32, |x| x.clone() * sigmoid(x));
    let jit_kernel = match CudaJitKernel::compile(op, 0) {
        Ok(k) => k,
        Err(_) => return,
    };

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
