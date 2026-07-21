//! Cross-backend parity: `NativeBackend<f32, Cpu>` vs `CandleBackend<f32,
//! Cpu>`, driven purely through the `Backend` trait surface (D-03), one
//! `#[test]` per individual op for failure localization.
//!
//! Every forward+backward test uses the SAME generic `run_and_grad<B:
//! Backend>` helper for both backends (Pitfall 3) — never two separately
//! coded per-backend branches — following the exact pattern already proven
//! by `crates/kindle-native/tests/linear_regression_parity.rs` (Phase 1).
//!
//! Tolerance: `1e-2` relative error with the same absolute-tolerance-floor
//! escape hatch as `kindle-native`'s own `testutil::gradcheck` (near-zero
//! true values produce pure finite-precision noise, not a real divergence).
//!
//! Requires `--features native,candle`.

use kindle_backends::candle::CandleBackend;
use kindle_core::prelude::Reduction;
use kindle_core::prelude::*;
use kindle_native::NativeBackend;

// ── Type aliases ─────────────────────────────────────────────────────────

/// Auto-generated documentation for NB.
type NB = NativeBackend<f32, Cpu>;
/// Auto-generated documentation for CB.
type CB = CandleBackend<f32, Cpu>;

// ── Shared helpers (mirrors linear_regression_parity.rs) ───────────────────

/// Auto-generated documentation for as_bytes.
fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Auto-generated documentation for make_storage.
fn make_storage<B: Backend>(data: &[f32], shape: &[usize]) -> B::Storage<f32> {
    B::from_bytes::<f32>(
        &as_bytes(data),
        shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .expect("from_bytes")
}

/// Reshape `t` to 1D before reading it out via `float_to_vec1` (Pitfall 5) —
/// `NativeBackend::float_to_vec1` errors on any rank != 1 input, so every
/// non-scalar, non-1D output/gradient must be flattened first. Scalar (rank
/// 0) outputs are read via `float_to_scalar` instead.
fn read_flat<B: Backend>(t: &B::Storage<f32>) -> Vec<f64> {
    let shape = B::shape::<f32>(t);
    let total: usize = shape.iter().product::<usize>().max(1);
    if shape.is_empty() {
        return vec![B::float_to_scalar::<f32>(t).expect("float_to_scalar")];
    }
    if shape.len() == 1 {
        return B::float_to_vec1::<f32>(t).expect("float_to_vec1");
    }
    let flat = B::reshape::<f32>(t, &[total]).expect("reshape to 1D for read_flat");
    B::float_to_vec1::<f32>(&flat).expect("float_to_vec1 after reshape")
}

/// Single-input generic op-under-test: builds `input_data` as a `Var` on
/// backend `B`, runs `op`, reduces to a scalar loss via `sum_all` if the
/// output isn't already scalar, drives `B::backward`, and returns
/// (forward_values, gradient_values) — both read back flattened to 1D
/// (Pitfall 5).
fn run_and_grad<B: Backend>(
    op: impl Fn(&B::Storage<f32>) -> B::Storage<f32>,
    input_data: &[f32],
    input_shape: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let x_stor = make_storage::<B>(input_data, input_shape);
    let x_var = B::var_from_tensor::<f32>(&x_stor).expect("var_from_tensor");
    let x = B::var_as_tensor::<f32>(&x_var).expect("var_as_tensor");

    let out = op(&x);
    let forward_vals = read_flat::<B>(&out);

    let out_shape = B::shape::<f32>(&out);
    let loss = if out_shape.iter().product::<usize>() == 1 {
        out
    } else {
        B::sum_all::<f32>(&out).expect("sum_all")
    };
    let grads = B::backward::<f32>(&loss).expect("backward");
    let x_now = B::var_as_tensor::<f32>(&x_var).expect("var_as_tensor (post-backward)");
    let grad_vals = match B::get_grad::<f32>(&x_now, &grads).expect("get_grad") {
        Some(g) => read_flat::<B>(&g),
        None => Vec::new(),
    };

    (forward_vals, grad_vals)
}

/// Two-input generic op-under-test variant for add/sub/mul/div/matmul.
/// Returns (forward_values, lhs_gradient_values, rhs_gradient_values).
fn run_and_grad2<B: Backend>(
    op: impl Fn(&B::Storage<f32>, &B::Storage<f32>) -> B::Storage<f32>,
    lhs_data: &[f32],
    lhs_shape: &[usize],
    rhs_data: &[f32],
    rhs_shape: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let lhs_stor = make_storage::<B>(lhs_data, lhs_shape);
    let rhs_stor = make_storage::<B>(rhs_data, rhs_shape);
    let lhs_var = B::var_from_tensor::<f32>(&lhs_stor).expect("var_from_tensor lhs");
    let rhs_var = B::var_from_tensor::<f32>(&rhs_stor).expect("var_from_tensor rhs");
    let lhs = B::var_as_tensor::<f32>(&lhs_var).expect("var_as_tensor lhs");
    let rhs = B::var_as_tensor::<f32>(&rhs_var).expect("var_as_tensor rhs");

    let out = op(&lhs, &rhs);
    let forward_vals = read_flat::<B>(&out);

    let out_shape = B::shape::<f32>(&out);
    let loss = if out_shape.iter().product::<usize>() == 1 {
        out
    } else {
        B::sum_all::<f32>(&out).expect("sum_all")
    };
    let grads = B::backward::<f32>(&loss).expect("backward");

    let lhs_now = B::var_as_tensor::<f32>(&lhs_var).expect("var_as_tensor lhs (post-backward)");
    let rhs_now = B::var_as_tensor::<f32>(&rhs_var).expect("var_as_tensor rhs (post-backward)");
    let lhs_grad = match B::get_grad::<f32>(&lhs_now, &grads).expect("get_grad lhs") {
        Some(g) => read_flat::<B>(&g),
        None => Vec::new(),
    };
    let rhs_grad = match B::get_grad::<f32>(&rhs_now, &grads).expect("get_grad rhs") {
        Some(g) => read_flat::<B>(&g),
        None => Vec::new(),
    };

    (forward_vals, lhs_grad, rhs_grad)
}

/// Elementwise compare two `Vec<f64>` at `tol` relative error, with the same
/// absolute-tolerance-floor escape hatch as `testutil::gradcheck` (a
/// near-zero true value makes finite-precision noise dominate the ratio).
fn assert_close(a: &[f64], b: &[f64], tol: f64, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let abs_diff = (av - bv).abs();
        if abs_diff < 1e-3 {
            continue;
        }
        let denom = av.abs().max(bv.abs()).max(1e-6);
        let rel_err = abs_diff / denom;
        assert!(
            rel_err < tol,
            "{label}[{i}]: native/candle mismatch — a={av}, b={bv}, rel_err={rel_err} (tol={tol})"
        );
    }
}

// ── CreationOps (forward-only: no differentiable input) ────────────────────

#[test]
/// Auto-generated documentation for zeros_parity.
fn zeros_parity() {
    let n = NB::zeros::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let c = CB::zeros::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    assert_eq!(NB::shape::<f32>(&n), CB::shape::<f32>(&c));
    assert_close(&read_flat::<NB>(&n), &read_flat::<CB>(&c), 1e-2, "zeros");
}

#[test]
/// Auto-generated documentation for ones_parity.
fn ones_parity() {
    let n = NB::ones::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let c = CB::ones::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    assert_eq!(NB::shape::<f32>(&n), CB::shape::<f32>(&c));
    assert_close(&read_flat::<NB>(&n), &read_flat::<CB>(&c), 1e-2, "ones");
}

#[test]
/// Auto-generated documentation for tensor_to_device_parity.
fn tensor_to_device_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let n0 = make_storage::<NB>(&data, &[4]);
    let c0 = make_storage::<CB>(&data, &[4]);
    let n = NB::tensor_to_device::<f32>(&n0, &KindleDevice::cpu()).unwrap();
    let c = CB::tensor_to_device::<f32>(&c0, &KindleDevice::cpu()).unwrap();
    assert_eq!(NB::shape::<f32>(&n), CB::shape::<f32>(&c));
    assert_close(
        &read_flat::<NB>(&n),
        &read_flat::<CB>(&c),
        1e-2,
        "tensor_to_device",
    );
}

// ── NumericOps ───────────────────────────────────────────────────────────

#[test]
/// Auto-generated documentation for add_forward_and_backward_parity.
fn add_forward_and_backward_parity() {
    let a = vec![1.0, -2.0, 3.0, 0.5];
    let b = vec![4.0, 1.0, -3.0, 2.0];
    let (fwd_n, ga_n, gb_n) =
        run_and_grad2::<NB>(|x, y| NB::add::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    let (fwd_c, ga_c, gb_c) =
        run_and_grad2::<CB>(|x, y| CB::add::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "add forward");
    assert_close(&ga_n, &ga_c, 1e-2, "add backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "add backward rhs");
}

#[test]
/// Auto-generated documentation for sub_forward_and_backward_parity.
fn sub_forward_and_backward_parity() {
    let a = vec![5.0, -1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, -2.0, 0.5];
    let (fwd_n, ga_n, gb_n) =
        run_and_grad2::<NB>(|x, y| NB::sub::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    let (fwd_c, ga_c, gb_c) =
        run_and_grad2::<CB>(|x, y| CB::sub::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sub forward");
    assert_close(&ga_n, &ga_c, 1e-2, "sub backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "sub backward rhs");
}

#[test]
/// Auto-generated documentation for mul_forward_and_backward_parity.
fn mul_forward_and_backward_parity() {
    let a = vec![2.0, -1.0, 0.5, 3.0];
    let b = vec![3.0, 4.0, -2.0, 1.5];
    let (fwd_n, ga_n, gb_n) =
        run_and_grad2::<NB>(|x, y| NB::mul::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    let (fwd_c, ga_c, gb_c) =
        run_and_grad2::<CB>(|x, y| CB::mul::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "mul forward");
    assert_close(&ga_n, &ga_c, 1e-2, "mul backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "mul backward rhs");
}

#[test]
/// Auto-generated documentation for div_forward_and_backward_parity.
fn div_forward_and_backward_parity() {
    let a = vec![4.0, -6.0, 9.0, 2.0];
    let b = vec![2.0, 3.0, -3.0, 0.5];
    let (fwd_n, ga_n, gb_n) =
        run_and_grad2::<NB>(|x, y| NB::div::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    let (fwd_c, ga_c, gb_c) =
        run_and_grad2::<CB>(|x, y| CB::div::<f32>(x, y).unwrap(), &a, &[4], &b, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "div forward");
    assert_close(&ga_n, &ga_c, 1e-2, "div backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "div backward rhs");
}

// ── FloatOps (13 unary/scalar kernels) ──────────────────────────────────────

#[test]
/// Auto-generated documentation for relu_forward_and_backward_parity.
fn relu_forward_and_backward_parity() {
    // Avoids x == 0.0 exactly: relu's subgradient at the kink is convention-
    // dependent (NativeBackend and Candle may legitimately pick different
    // sides of {0, 1}), which is not a real numerical divergence to test for.
    let data = vec![-2.0, -0.5, 0.1, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::relu::<f32>(x).unwrap(), &data, &[5]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::relu::<f32>(x).unwrap(), &data, &[5]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "relu forward");
    assert_close(&grad_n, &grad_c, 1e-2, "relu backward");
}

#[test]
/// Auto-generated documentation for gelu_forward_and_backward_parity.
fn gelu_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.3, 1.0, 2.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::gelu::<f32>(x).unwrap(), &data, &[5]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::gelu::<f32>(x).unwrap(), &data, &[5]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "gelu forward");
    assert_close(&grad_n, &grad_c, 1e-2, "gelu backward");
}

#[test]
/// Auto-generated documentation for abs_forward_and_backward_parity.
fn abs_forward_and_backward_parity() {
    let data = vec![-3.0, -1.0, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::abs::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::abs::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "abs forward");
    assert_close(&grad_n, &grad_c, 1e-2, "abs backward");
}

#[test]
/// Auto-generated documentation for exp_forward_and_backward_parity.
fn exp_forward_and_backward_parity() {
    let data = vec![-1.0, 0.0, 0.5, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::exp::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::exp::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "exp forward");
    assert_close(&grad_n, &grad_c, 1e-2, "exp backward");
}

#[test]
/// Auto-generated documentation for neg_forward_and_backward_parity.
fn neg_forward_and_backward_parity() {
    let data = vec![-2.0, 3.0, 0.0, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::neg::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::neg::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "neg forward");
    assert_close(&grad_n, &grad_c, 1e-2, "neg backward");
}

#[test]
/// Auto-generated documentation for sqrt_forward_and_backward_parity.
fn sqrt_forward_and_backward_parity() {
    let data = vec![1.0, 4.0, 9.0, 2.25];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sqrt::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sqrt::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sqrt forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sqrt backward");
}

#[test]
/// Auto-generated documentation for log_forward_and_backward_parity.
fn log_forward_and_backward_parity() {
    let data = vec![0.5, 1.0, 2.0, 4.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::log::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::log::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "log forward");
    assert_close(&grad_n, &grad_c, 1e-2, "log backward");
}

#[test]
/// Auto-generated documentation for tanh_forward_and_backward_parity.
fn tanh_forward_and_backward_parity() {
    let data = vec![-1.5, -0.3, 0.3, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::tanh::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::tanh::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "tanh forward");
    assert_close(&grad_n, &grad_c, 1e-2, "tanh backward");
}

#[test]
/// Auto-generated documentation for sigmoid_forward_and_backward_parity.
fn sigmoid_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sigmoid::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sigmoid::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sigmoid forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sigmoid backward");
}

#[test]
/// Auto-generated documentation for swish_forward_and_backward_parity.
fn swish_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::swish::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::swish::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "swish forward");
    assert_close(&grad_n, &grad_c, 1e-2, "swish backward");
}

#[test]
/// Auto-generated documentation for softmax_forward_and_backward_parity.
fn softmax_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::softmax::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::softmax::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "softmax forward");
    assert_close(&grad_n, &grad_c, 1e-2, "softmax backward");
}

#[test]
/// Auto-generated documentation for add_scalar_float_forward_and_backward_parity.
fn add_scalar_float_forward_and_backward_parity() {
    let data = vec![1.0, -2.0, 3.0, 0.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::add_scalar_float::<f32>(x, 2.5).unwrap(),
        &data,
        &[4],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::add_scalar_float::<f32>(x, 2.5).unwrap(),
        &data,
        &[4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "add_scalar_float forward");
    assert_close(&grad_n, &grad_c, 1e-2, "add_scalar_float backward");
}

#[test]
/// Auto-generated documentation for mul_scalar_float_forward_and_backward_parity.
fn mul_scalar_float_forward_and_backward_parity() {
    let data = vec![1.0, -2.0, 3.0, 0.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::mul_scalar_float::<f32>(x, 1.5).unwrap(),
        &data,
        &[4],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::mul_scalar_float::<f32>(x, 1.5).unwrap(),
        &data,
        &[4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "mul_scalar_float forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mul_scalar_float backward");
}

// ── ReductionOps (12 float-output reductions; argmax/argmin in Task 2) ─────

#[test]
/// Auto-generated documentation for sum_all_forward_and_backward_parity.
fn sum_all_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sum_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sum_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sum_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sum_all backward");
}

#[test]
/// Auto-generated documentation for mean_all_forward_and_backward_parity.
fn mean_all_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::mean_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::mean_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "mean_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mean_all backward");
}

#[test]
/// Auto-generated documentation for max_all_forward_and_backward_parity.
fn max_all_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::max_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::max_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "max_all backward");
}

#[test]
/// Auto-generated documentation for min_all_forward_and_backward_parity.
fn min_all_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::min_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::min_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "min_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "min_all backward");
}

#[test]
/// Auto-generated documentation for sum_dim_forward_and_backward_parity.
fn sum_dim_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sum_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sum_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sum_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sum_dim backward");
}

#[test]
/// Auto-generated documentation for sum_keepdim_forward_and_backward_parity.
fn sum_keepdim_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::sum_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::sum_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sum_keepdim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sum_keepdim backward");
}

#[test]
/// Auto-generated documentation for mean_dim_forward_and_backward_parity.
fn mean_dim_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::mean_dim::<f32>(x, 0).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::mean_dim::<f32>(x, 0).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "mean_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mean_dim backward");
}

#[test]
/// Auto-generated documentation for mean_keepdim_forward_and_backward_parity.
fn mean_keepdim_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::mean_keepdim::<f32>(x, 0).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::mean_keepdim::<f32>(x, 0).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "mean_keepdim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mean_keepdim backward");
}

#[test]
/// Auto-generated documentation for max_dim_forward_and_backward_parity.
fn max_dim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::max_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::max_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "max_dim backward");
}

#[test]
/// Auto-generated documentation for max_keepdim_forward_and_backward_parity.
fn max_keepdim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::max_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::max_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_keepdim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "max_keepdim backward");
}

#[test]
/// Auto-generated documentation for min_dim_forward_and_backward_parity.
fn min_dim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::min_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::min_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "min_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "min_dim backward");
}

#[test]
/// Auto-generated documentation for min_keepdim_forward_and_backward_parity.
fn min_keepdim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::min_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::min_keepdim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "min_keepdim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "min_keepdim backward");
}

// ── TensorOps shape methods ─────────────────────────────────────────────

#[test]
/// Auto-generated documentation for reshape_forward_and_backward_parity.
fn reshape_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::reshape::<f32>(x, &[3, 2]).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::reshape::<f32>(x, &[3, 2]).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "reshape forward");
    assert_close(&grad_n, &grad_c, 1e-2, "reshape backward");
}

#[test]
/// Auto-generated documentation for transpose_forward_and_backward_parity.
fn transpose_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::transpose::<f32>(x, 0, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::transpose::<f32>(x, 0, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "transpose forward");
    assert_close(&grad_n, &grad_c, 1e-2, "transpose backward");
}

#[test]
/// Auto-generated documentation for matmul_forward_and_backward_parity.
fn matmul_forward_and_backward_parity() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2]
    let (fwd_n, ga_n, gb_n) = run_and_grad2::<NB>(
        |x, y| NB::matmul::<f32>(x, y).unwrap(),
        &a,
        &[2, 3],
        &b,
        &[3, 2],
    );
    let (fwd_c, ga_c, gb_c) = run_and_grad2::<CB>(
        |x, y| CB::matmul::<f32>(x, y).unwrap(),
        &a,
        &[2, 3],
        &b,
        &[3, 2],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "matmul forward");
    assert_close(&ga_n, &ga_c, 1e-2, "matmul backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "matmul backward rhs");
}

#[test]
/// Auto-generated documentation for broadcast_as_forward_and_backward_parity.
fn broadcast_as_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::broadcast_as::<f32>(x, &[2, 3]).unwrap(),
        &data,
        &[3],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::broadcast_as::<f32>(x, &[2, 3]).unwrap(),
        &data,
        &[3],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "broadcast_as forward");
    assert_close(&grad_n, &grad_c, 1e-2, "broadcast_as backward");
}

#[test]
/// Auto-generated documentation for narrow_forward_and_backward_parity.
fn narrow_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::narrow::<f32>(x, 1, 1, 2).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::narrow::<f32>(x, 1, 1, 2).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "narrow forward");
    assert_close(&grad_n, &grad_c, 1e-2, "narrow backward");
}

#[test]
/// Auto-generated documentation for squeeze_forward_and_backward_parity.
fn squeeze_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::squeeze::<f32>(x, 0).unwrap(), &data, &[1, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::squeeze::<f32>(x, 0).unwrap(), &data, &[1, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "squeeze forward");
    assert_close(&grad_n, &grad_c, 1e-2, "squeeze backward");
}

#[test]
/// Auto-generated documentation for stack_forward_and_backward_parity.
fn stack_forward_and_backward_parity() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let (fwd_n, ga_n, gb_n) = run_and_grad2::<NB>(
        |x, y| NB::stack::<f32>(&[x, y], 0).unwrap(),
        &a,
        &[3],
        &b,
        &[3],
    );
    let (fwd_c, ga_c, gb_c) = run_and_grad2::<CB>(
        |x, y| CB::stack::<f32>(&[x, y], 0).unwrap(),
        &a,
        &[3],
        &b,
        &[3],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "stack forward");
    assert_close(&ga_n, &ga_c, 1e-2, "stack backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "stack backward rhs");
}

#[test]
/// Auto-generated documentation for concat_forward_and_backward_parity.
fn concat_forward_and_backward_parity() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let (fwd_n, ga_n, gb_n) = run_and_grad2::<NB>(
        |x, y| NB::concat::<f32>(&[x, y], 0).unwrap(),
        &a,
        &[3],
        &b,
        &[3],
    );
    let (fwd_c, ga_c, gb_c) = run_and_grad2::<CB>(
        |x, y| CB::concat::<f32>(&[x, y], 0).unwrap(),
        &a,
        &[3],
        &b,
        &[3],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "concat forward");
    assert_close(&ga_n, &ga_c, 1e-2, "concat backward lhs");
    assert_close(&gb_n, &gb_c, 1e-2, "concat backward rhs");
}

#[test]
/// Auto-generated documentation for slice_forward_and_backward_parity.
fn slice_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::slice::<f32>(x, &[(0, 2), (1, 3)]).unwrap(),
        &data,
        &[2, 4],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::slice::<f32>(x, &[(0, 2), (1, 3)]).unwrap(),
        &data,
        &[2, 4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "slice forward");
    assert_close(&grad_n, &grad_c, 1e-2, "slice backward");
}

#[test]
/// Auto-generated documentation for flatten_forward_and_backward_parity.
fn flatten_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (fwd_n, grad_n) =
        run_and_grad::<NB>(|x| NB::flatten::<f32>(x, 1, 2).unwrap(), &data, &[2, 2, 2]);
    let (fwd_c, grad_c) =
        run_and_grad::<CB>(|x| CB::flatten::<f32>(x, 1, 2).unwrap(), &data, &[2, 2, 2]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "flatten forward");
    assert_close(&grad_n, &grad_c, 1e-2, "flatten backward");
}

#[test]
/// Auto-generated documentation for broadcast_left_forward_and_backward_parity.
fn broadcast_left_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::broadcast_left::<f32>(x, &[2, 3]).unwrap(),
        &data,
        &[3],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::broadcast_left::<f32>(x, &[2, 3]).unwrap(),
        &data,
        &[3],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "broadcast_left forward");
    assert_close(&grad_n, &grad_c, 1e-2, "broadcast_left backward");
}

// ── ReductionOps: argmax/argmin (forward-only, no backward — non-
//    differentiable integer-index output, confirmed by 05-AUDIT.md's
//    backward-rule audit). `float_to_scalar`/`float_to_vec1` are used to
//    read the index values back out purely through the `Backend` trait: on
//    `NativeBackend` the `K` type parameter is a compile-time-only marker
//    (the real read goes through `NativeStorage::get`'s `get_f64`, which is
//    buffer-dtype-agnostic), and on `CandleBackend` `float_to_vec1`
//    internally converts to F32 via `to_dtype` regardless of the tensor's
//    actual dtype — so calling the `f32` read accessors on an `i64`-backed
//    argmax/argmin output is well-defined on both backends without needing
//    `int_to_scalar`/`int_to_vec1` (which are formally descoped/stubbed on
//    `NativeBackend` per 05-AUDIT.md's Descope Decision). ─────────────────

#[test]
/// Auto-generated documentation for argmax_forward_parity.
fn argmax_forward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let n_stor = make_storage::<NB>(&data, &[2, 3]);
    let c_stor = make_storage::<CB>(&data, &[2, 3]);

    let n_dim = NB::argmax::<f32, i64>(&n_stor, Some(1)).unwrap();
    let c_dim = CB::argmax::<f32, i64>(&c_stor, Some(1)).unwrap();
    assert_eq!(NB::shape::<i64>(&n_dim), CB::shape::<i64>(&c_dim));
    assert_close(
        &read_flat::<NB>(&n_dim),
        &read_flat::<CB>(&c_dim),
        1e-2,
        "argmax dim",
    );

    let n_all = NB::argmax::<f32, i64>(&n_stor, None).unwrap();
    let c_all = CB::argmax::<f32, i64>(&c_stor, None).unwrap();
    assert_close(
        &read_flat::<NB>(&n_all),
        &read_flat::<CB>(&c_all),
        1e-2,
        "argmax all",
    );
}

#[test]
/// Auto-generated documentation for argmin_forward_parity.
fn argmin_forward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let n_stor = make_storage::<NB>(&data, &[2, 3]);
    let c_stor = make_storage::<CB>(&data, &[2, 3]);

    let n_dim = NB::argmin::<f32, i64>(&n_stor, Some(1)).unwrap();
    let c_dim = CB::argmin::<f32, i64>(&c_stor, Some(1)).unwrap();
    assert_eq!(NB::shape::<i64>(&n_dim), CB::shape::<i64>(&c_dim));
    assert_close(
        &read_flat::<NB>(&n_dim),
        &read_flat::<CB>(&c_dim),
        1e-2,
        "argmin dim",
    );

    let n_all = NB::argmin::<f32, i64>(&n_stor, None).unwrap();
    let c_all = CB::argmin::<f32, i64>(&c_stor, None).unwrap();
    assert_close(
        &read_flat::<NB>(&n_all),
        &read_flat::<CB>(&c_all),
        1e-2,
        "argmin all",
    );
}

// ── ModuleOps (9 methods, forward+backward) ────────────────────────────────

/// `layer_norm` forward-only parity — backward is NOT compared. Confirmed by
/// direct source read of `candle-nn-0.9.1/src/ops.rs::layer_norm`: it calls
/// `xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })`, a fused kernel
/// with an explicit `_no_bwd` (no-backward) contract — `CandleBackend::
/// layer_norm`'s `B::backward` always returns a `GradStore` with no entry
/// for the input (confirmed empirically: `get_grad` returns `None`,
/// independent of test data). This is a genuine, permanent Candle-side
/// limitation (not test-input-dependent, not floating-point noise, not a
/// `NativeBackend` gap) — `NativeBackend::layer_norm` DOES have a real,
/// composed backward (05-AUDIT.md's Section 2, `layer_norm` = Composed).
/// Backward parity for this op is therefore untestable against Candle by
/// construction; forward-only comparison is the correct, honest test here.
#[test]
fn layer_norm_forward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0];

    let n_x = make_storage::<NB>(&data, &[2, 3]);
    let n_w = make_storage::<NB>(&weight, &[3]);
    let n_b = make_storage::<NB>(&bias, &[3]);
    let n_out = NB::layer_norm::<f32>(&n_x, &n_w, Some(&n_b), 1e-5).unwrap();

    let c_x = make_storage::<CB>(&data, &[2, 3]);
    let c_w = make_storage::<CB>(&weight, &[3]);
    let c_b = make_storage::<CB>(&bias, &[3]);
    let c_out = CB::layer_norm::<f32>(&c_x, &c_w, Some(&c_b), 1e-5).unwrap();

    assert_close(
        &read_flat::<NB>(&n_out),
        &read_flat::<CB>(&c_out),
        1e-2,
        "layer_norm forward",
    );
}

#[test]
/// Auto-generated documentation for batch_norm_forward_and_backward_parity.
fn batch_norm_forward_and_backward_parity() {
    // [1, 3, 2, 2] — inference-mode-only (Phase 4 carried decision): running
    // stats are consumed as fixed constants, never updated.
    let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0];
    let running_mean = vec![0.0, 0.0, 0.0];
    let running_var = vec![1.0, 1.0, 1.0];

    let n_w = make_storage::<NB>(&weight, &[3]);
    let n_b = make_storage::<NB>(&bias, &[3]);
    let n_rm = make_storage::<NB>(&running_mean, &[3]);
    let n_rv = make_storage::<NB>(&running_var, &[3]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| {
            NB::batch_norm::<f32>(
                x,
                Some(&n_w),
                Some(&n_b),
                Some(&n_rm),
                Some(&n_rv),
                1e-5,
                0.1,
            )
            .unwrap()
        },
        &data,
        &[1, 3, 2, 2],
    );

    let c_w = make_storage::<CB>(&weight, &[3]);
    let c_b = make_storage::<CB>(&bias, &[3]);
    let c_rm = make_storage::<CB>(&running_mean, &[3]);
    let c_rv = make_storage::<CB>(&running_var, &[3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| {
            CB::batch_norm::<f32>(
                x,
                Some(&c_w),
                Some(&c_b),
                Some(&c_rm),
                Some(&c_rv),
                1e-5,
                0.1,
            )
            .unwrap()
        },
        &data,
        &[1, 3, 2, 2],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "batch_norm forward");
    assert_close(&grad_n, &grad_c, 1e-2, "batch_norm backward");
}

#[test]
/// Auto-generated documentation for embedding_forward_and_backward_parity.
fn embedding_forward_and_backward_parity() {
    // Weight [4, 3] (vocab=4, embed_dim=3); indices [1, 3] -> rows 1 and 3.
    let weight = vec![
        0.1, 0.2, 0.3, // row 0
        0.4, 0.5, 0.6, // row 1
        0.7, 0.8, 0.9, // row 2
        1.0, 1.1, 1.2, // row 3
    ];
    let indices_f: Vec<f32> = vec![1.0, 3.0];

    // Embedding's index input has no gradient (KInt, non-differentiable);
    // only the weight table is differentiable, so run_and_grad is applied to
    // the WEIGHT, with indices captured by closure as a fixed constant.
    // NativeBackend::from_bytes only supports F32 (Pattern 1's audit table),
    // so the integer index storage is built directly via NativeStorage's own
    // I64 buffer constructor instead — still purely-Backend-trait on the
    // WEIGHT side, which is the side under gradient test.
    let n_idx = kindle_native::storage::NativeStorage::from_contiguous(
        kindle_native::storage::NativeBuffer::I64(vec![1, 3]),
        vec![2],
    );

    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |w| NB::embedding::<f32, i64>(&n_idx, w).unwrap(),
        &weight,
        &[4, 3],
    );

    // CandleBackend::from_bytes always reads the input bytes as f32 first
    // (regardless of `dtype`), then converts via `to_dtype` — so the index
    // bytes must be f32-encoded (`as_bytes`), not raw i64 bytes.
    let c_idx = CB::from_bytes::<i64>(
        &as_bytes(&indices_f),
        &[2],
        KindleDType::I64,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |w| CB::embedding::<f32, i64>(&c_idx, w).unwrap(),
        &weight,
        &[4, 3],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "embedding forward");
    assert_close(&grad_n, &grad_c, 1e-2, "embedding backward");
}

#[test]
/// Auto-generated documentation for conv1d_forward_and_backward_parity.
fn conv1d_forward_and_backward_parity() {
    // input [1, 2, 6], weight [3, 2, 3] (Cout=3, Cin=2, K=3), stride=1, pad=1.
    let data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
    let weight: Vec<f32> = (1..=18).map(|v| v as f32 * 0.05).collect();

    let n_w = make_storage::<NB>(&weight, &[3, 2, 3]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::conv1d::<f32>(x, &n_w, None, 1, 1, 1, 1).unwrap(),
        &data,
        &[1, 2, 6],
    );

    let c_w = make_storage::<CB>(&weight, &[3, 2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::conv1d::<f32>(x, &c_w, None, 1, 1, 1, 1).unwrap(),
        &data,
        &[1, 2, 6],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "conv1d forward");
    assert_close(&grad_n, &grad_c, 1e-2, "conv1d backward");
}

/// conv2d/conv_transpose2d tolerance investigation (Pitfall 4): NativeBackend's
/// im2col+matmul accumulation order differs from Candle's native conv kernel,
/// so a wider-than-default tolerance MAY be needed. Both are first attempted
/// at the shared 1e-2 default; empirically confirmed to pass at 1e-2 (no
/// widening applied — see the exact measured magnitudes below, captured via
/// a temporary max-relative-error instrumentation pass run against this
/// exact test body/input data before this comment was written, then
/// reverted):
///
///   conv2d forward:            max observed relative error ≈ 2.01e-3
///   conv2d backward:           max observed relative error ≈ 1.49e-2
///                               (the single element driving this max has
///                               abs_diff < 1e-3 — i.e. it is caught by
///                               `assert_close`'s absolute-tolerance floor,
///                               the same near-zero-true-value escape hatch
///                               `testutil::gradcheck` already uses — not a
///                               genuine >1e-2 divergence on a
///                               non-negligible gradient value)
///   conv_transpose2d forward:  max observed relative error ≈ 6.52e-3
///   conv_transpose2d backward: max observed relative error ≈ 7.45e-3
///
/// All four are consistent with accumulated-term-count floating-point
/// rounding noise (RESEARCH.md's own prediction), not a systematic bias or
/// implementation bug — no widening of the shared 1e-2 default was
/// necessary for either op at this [1,3,8,8]/[1,3,4,4]-scale input.
#[test]
fn conv2d_forward_and_backward_parity() {
    // input [1, 3, 8, 8], weight [4, 3, 3, 3] (Cout=4, Cin=3, K=3x3).
    let data: Vec<f32> = (0..192).map(|v| (v as f32 % 7.0 - 3.0) * 0.1).collect();
    let weight: Vec<f32> = (0..108).map(|v| (v as f32 % 5.0 - 2.0) * 0.05).collect();

    let n_w = make_storage::<NB>(&weight, &[4, 3, 3, 3]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::conv2d::<f32>(x, &n_w, None, 1, 1, 1, 1).unwrap(),
        &data,
        &[1, 3, 8, 8],
    );

    let c_w = make_storage::<CB>(&weight, &[4, 3, 3, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::conv2d::<f32>(x, &c_w, None, 1, 1, 1, 1).unwrap(),
        &data,
        &[1, 3, 8, 8],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "conv2d forward");
    assert_close(&grad_n, &grad_c, 1e-2, "conv2d backward");
}

#[test]
/// Auto-generated documentation for conv_transpose2d_forward_and_backward_parity.
fn conv_transpose2d_forward_and_backward_parity() {
    // input [1, 3, 4, 4], weight [3, 4, 3, 3] (Cin=3, Cout=4, K=3x3) per
    // conv_transpose2d's (Cin, Cout, Kh, Kw) weight-layout convention.
    let data: Vec<f32> = (0..48).map(|v| (v as f32 % 6.0 - 2.5) * 0.1).collect();
    let weight: Vec<f32> = (0..108).map(|v| (v as f32 % 5.0 - 2.0) * 0.05).collect();

    let n_w = make_storage::<NB>(&weight, &[3, 4, 3, 3]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::conv_transpose2d::<f32>(x, &n_w, None, 1, 1, 0, 1, 1).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );

    let c_w = make_storage::<CB>(&weight, &[3, 4, 3, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::conv_transpose2d::<f32>(x, &c_w, None, 1, 1, 0, 1, 1).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "conv_transpose2d forward");
    assert_close(&grad_n, &grad_c, 1e-2, "conv_transpose2d backward");
}

#[test]
/// Auto-generated documentation for max_pool2d_forward_and_backward_parity.
fn max_pool2d_forward_and_backward_parity() {
    let data: Vec<f32> = (0..48).map(|v| v as f32 % 9.0).collect(); // [1,3,4,4]
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::max_pool2d::<f32>(x, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::max_pool2d::<f32>(x, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_pool2d forward");

    // Backward scale note (confirmed by reading candle-core 0.9.1's
    // backprop.rs Op::MaxPool2D arm, not assumed): Candle scales the winning
    // position's gradient by `1/(kernel_h*kernel_w)` via an
    // avg_pool2d-over-the-winner-mask step BEFORE upsampling back to the
    // full window, even when a window has exactly one unique maximum (i.e.
    // the winner receives `grad_out * 1/kernel_area`, not the full
    // `grad_out`). `NativeBackend::max_pool2d` gives the winner the FULL
    // upstream gradient (the standard/PyTorch-matching convention — see
    // `pool.rs`'s own `scatter_pool_grad_2d`). This is a confirmed, semantic
    // backward-definition difference between the two backends' `max_pool2d`
    // (not floating-point rounding noise, so it is not a Pitfall-4-style
    // tolerance-widening case) — NativeBackend's gradient at each winning
    // position is exactly `kernel_area` times Candle's. Normalize Candle's
    // gradient by that fixed, known factor before comparing at the shared
    // 1e-2 tolerance, rather than silently loosening the comparison.
    let kernel_area = 4.0; // 2*2
    let grad_c_scaled: Vec<f64> = grad_c.iter().map(|&g| g * kernel_area).collect();
    assert_close(
        &grad_n,
        &grad_c_scaled,
        1e-2,
        "max_pool2d backward (Candle scaled by kernel_area)",
    );
}

#[test]
/// Auto-generated documentation for avg_pool2d_forward_and_backward_parity.
fn avg_pool2d_forward_and_backward_parity() {
    let data: Vec<f32> = (0..48).map(|v| v as f32 % 9.0).collect(); // [1,3,4,4]
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |x| NB::avg_pool2d::<f32>(x, (2, 2), (2, 2), (0, 0)).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |x| CB::avg_pool2d::<f32>(x, (2, 2), (2, 2), (0, 0)).unwrap(),
        &data,
        &[1, 3, 4, 4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "avg_pool2d forward");
    assert_close(&grad_n, &grad_c, 1e-2, "avg_pool2d backward");
}

/// `adaptive_avg_pool2d` has no `CandleBackend` implementation to compare
/// against at all: `CandleBackend::adaptive_avg_pool2d` is `unimplemented!()`
/// unconditionally (confirmed at `crates/kindle-backends/src/lib.rs`, and
/// documented as an INTENTIONAL, permanent gap by REQUIREMENTS.md's own
/// NATBACK-03 wording: "`l1_loss`/`bce_with_logits_loss`/
/// `adaptive_avg_pool2d`... unlike `CandleBackend`, which currently stubs
/// these three" — `NativeBackend` is REQUIRED to exceed `CandleBackend`'s
/// coverage here, not match it). Since no reference implementation exists on
/// the other backend, this test verifies `NativeBackend::adaptive_avg_pool2d`
/// forward+backward self-consistency instead of cross-backend parity: (a)
/// forward output shape matches the requested `output_size`, and (b) the
/// backward gradient of `sum_all(out)` w.r.t. the input sums to exactly
/// `out_h * out_w * channels` (each output position's gradient of 1.0,
/// distributed uniformly across its own variable-size window per Phase 4's
/// own `adaptive_avg_pool2d_impl` design, must still sum back to the same
/// total as the number of output positions — the pooling is a weighted
/// average, so gradient mass is conserved exactly, a property provable
/// without a second backend).
#[test]
fn adaptive_avg_pool2d_native_self_consistency() {
    // input=5 -> output=3 (Phase 4's own non-uniform-window precedent).
    let data: Vec<f32> = (0..75).map(|v| v as f32 % 9.0).collect(); // [1,3,5,5]
    let (_, grad_n) = run_and_grad::<NB>(
        |x| NB::adaptive_avg_pool2d::<f32>(x, (3, 3)).unwrap(),
        &data,
        &[1, 3, 5, 5],
    );

    let x_stor = make_storage::<NB>(&data, &[1, 3, 5, 5]);
    let out = NB::adaptive_avg_pool2d::<f32>(&x_stor, (3, 3)).unwrap();
    assert_eq!(
        NB::shape::<f32>(&out),
        vec![1, 3, 3, 3],
        "adaptive_avg_pool2d output shape"
    );

    let grad_sum: f64 = grad_n.iter().sum();
    let expected_sum = 3.0 * 3.0 * 3.0; // out_h * out_w * channels, each output grad = 1.0
    assert!(
        (grad_sum - expected_sum).abs() < 1e-2,
        "adaptive_avg_pool2d backward: gradient mass not conserved — got {grad_sum}, expected {expected_sum}"
    );
}

// ── LossOps (4 methods, forward+backward, Reduction::Mean primarily) ───────
//
// mse_loss's Reduction::Sum is NOT spot-checked here: CandleBackend::
// mse_loss's `_reduction` parameter is unused (confirmed by direct source
// read — it always calls `candle_nn::loss::mse`, which is Mean-only), so a
// Sum-vs-Mean comparison against Candle would fail by construction, not due
// to a real numerical divergence. Reduction::Mean is fully exercised below.

#[test]
/// Auto-generated documentation for mse_loss_forward_and_backward_parity_mean.
fn mse_loss_forward_and_backward_parity_mean() {
    let pred = vec![1.0, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 2.5, 4.5];
    let n_t = make_storage::<NB>(&target, &[4]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |p| NB::mse_loss::<f32>(p, &n_t, Reduction::Mean).unwrap(),
        &pred,
        &[4],
    );
    let c_t = make_storage::<CB>(&target, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |p| CB::mse_loss::<f32>(p, &c_t, Reduction::Mean).unwrap(),
        &pred,
        &[4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "mse_loss(mean) forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mse_loss(mean) backward");
}

/// `l1_loss` has no `CandleBackend` implementation (`unimplemented!()`
/// unconditionally — same documented, intentional gap as
/// `adaptive_avg_pool2d`, per REQUIREMENTS.md's NATBACK-03 wording).
/// `NativeBackend::l1_loss` is composed entirely from already-real,
/// already-parity-tested primitives (`sub` -> `abs` -> `mean_all`, per
/// `ops/loss.rs`'s own doc comment), so this test builds the mathematically
/// equivalent composition on `CandleBackend` FROM THOSE SAME real, already-
/// covered `Backend` trait methods (not a hand-rolled candle-core call) as
/// the comparison reference — this still exercises `NativeBackend::l1_loss`
/// itself (the actual audit target) against a Candle-side value produced
/// purely through the shared `Backend` trait surface.
#[test]
fn l1_loss_forward_and_backward_parity() {
    let pred = vec![1.0, -2.0, 3.5, 0.5];
    let target = vec![1.5, -1.0, 2.5, 1.5];
    let n_t = make_storage::<NB>(&target, &[4]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |p| NB::l1_loss::<f32>(p, &n_t, Reduction::Mean).unwrap(),
        &pred,
        &[4],
    );
    let c_t = make_storage::<CB>(&target, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |p| {
            let diff = CB::sub::<f32>(p, &c_t).unwrap();
            let abs_diff = CB::abs::<f32>(&diff).unwrap();
            CB::mean_all::<f32>(&abs_diff).unwrap()
        },
        &pred,
        &[4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "l1_loss forward");
    assert_close(&grad_n, &grad_c, 1e-2, "l1_loss backward");
}

/// `bce_with_logits_loss` has no `CandleBackend` implementation
/// (`unimplemented!()` unconditionally — same documented, intentional gap).
/// `NativeBackend::bce_with_logits_loss` implements the numerically-stable
/// formula `max(x,0) - x*z + log(1+exp(-|x|))`, composed entirely from
/// already-real, already-parity-tested primitives (per `ops/loss.rs`'s own
/// doc comment: `relu`/`mul`/`sub`/`abs`/`neg`/`exp`/`add_scalar_float`/
/// `log`/`add`). This test builds the identical composition on
/// `CandleBackend` from those same real trait methods as the comparison
/// reference, exercising `NativeBackend::bce_with_logits_loss` itself
/// against a Candle-side value produced purely through the shared `Backend`
/// trait surface.
#[test]
fn bce_with_logits_loss_forward_and_backward_parity() {
    let pred = vec![0.5, -1.0, 2.0, -0.5];
    let target = vec![1.0, 0.0, 1.0, 0.0];
    let n_t = make_storage::<NB>(&target, &[4]);
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |p| NB::bce_with_logits_loss::<f32>(p, &n_t, Reduction::Mean).unwrap(),
        &pred,
        &[4],
    );
    let c_t = make_storage::<CB>(&target, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |p| {
            // max(x,0) - x*z + log(1+exp(-|x|)), the same stable formula
            // NativeBackend::bce_with_logits_loss composes.
            let relu_p = CB::relu::<f32>(p).unwrap();
            let p_mul_z = CB::mul::<f32>(p, &c_t).unwrap();
            let term1 = CB::sub::<f32>(&relu_p, &p_mul_z).unwrap();
            let abs_p = CB::abs::<f32>(p).unwrap();
            let neg_abs_p = CB::neg::<f32>(&abs_p).unwrap();
            let exp_term = CB::exp::<f32>(&neg_abs_p).unwrap();
            let one_plus_exp = CB::add_scalar_float::<f32>(&exp_term, 1.0).unwrap();
            let log_term = CB::log::<f32>(&one_plus_exp).unwrap();
            let elementwise = CB::add::<f32>(&term1, &log_term).unwrap();
            CB::mean_all::<f32>(&elementwise).unwrap()
        },
        &pred,
        &[4],
    );
    assert_close(&fwd_n, &fwd_c, 1e-2, "bce_with_logits_loss forward");
    assert_close(&grad_n, &grad_c, 1e-2, "bce_with_logits_loss backward");
}

#[test]
/// Auto-generated documentation for cross_entropy_loss_forward_and_backward_parity.
fn cross_entropy_loss_forward_and_backward_parity() {
    // pred [2, 3] logits, target [2] class indices.
    let pred = vec![1.0, 2.0, 0.5, 0.2, 1.5, 3.0];
    let n_target = kindle_native::storage::NativeStorage::from_contiguous(
        kindle_native::storage::NativeBuffer::I64(vec![1, 2]),
        vec![2],
    );
    let (fwd_n, grad_n) = run_and_grad::<NB>(
        |p| NB::cross_entropy_loss::<f32, i64>(p, &n_target, Reduction::Mean).unwrap(),
        &pred,
        &[2, 3],
    );

    // See embedding_forward_and_backward_parity's comment: CandleBackend::
    // from_bytes always reads as f32 first, so target values must be f32-
    // encoded bytes, not raw i64 bytes.
    let c_target = CB::from_bytes::<i64>(
        &as_bytes(&[1.0, 2.0]),
        &[2],
        KindleDType::I64,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let (fwd_c, grad_c) = run_and_grad::<CB>(
        |p| CB::cross_entropy_loss::<f32, i64>(p, &c_target, Reduction::Mean).unwrap(),
        &pred,
        &[2, 3],
    );

    assert_close(&fwd_n, &fwd_c, 1e-2, "cross_entropy_loss forward");
    assert_close(&grad_n, &grad_c, 1e-2, "cross_entropy_loss backward");
}
