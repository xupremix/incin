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
use kindle_core::prelude::*;
use kindle_native::NativeBackend;

// ── Type aliases ─────────────────────────────────────────────────────────

type NB = NativeBackend<f32, Cpu>;
type CB = CandleBackend<f32, Cpu>;

// ── Shared helpers (mirrors linear_regression_parity.rs) ───────────────────

fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

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
fn zeros_parity() {
    let n = NB::zeros::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let c = CB::zeros::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    assert_eq!(NB::shape::<f32>(&n), CB::shape::<f32>(&c));
    assert_close(&read_flat::<NB>(&n), &read_flat::<CB>(&c), 1e-2, "zeros");
}

#[test]
fn ones_parity() {
    let n = NB::ones::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let c = CB::ones::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    assert_eq!(NB::shape::<f32>(&n), CB::shape::<f32>(&c));
    assert_close(&read_flat::<NB>(&n), &read_flat::<CB>(&c), 1e-2, "ones");
}

#[test]
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
fn gelu_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.3, 1.0, 2.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::gelu::<f32>(x).unwrap(), &data, &[5]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::gelu::<f32>(x).unwrap(), &data, &[5]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "gelu forward");
    assert_close(&grad_n, &grad_c, 1e-2, "gelu backward");
}

#[test]
fn abs_forward_and_backward_parity() {
    let data = vec![-3.0, -1.0, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::abs::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::abs::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "abs forward");
    assert_close(&grad_n, &grad_c, 1e-2, "abs backward");
}

#[test]
fn exp_forward_and_backward_parity() {
    let data = vec![-1.0, 0.0, 0.5, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::exp::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::exp::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "exp forward");
    assert_close(&grad_n, &grad_c, 1e-2, "exp backward");
}

#[test]
fn neg_forward_and_backward_parity() {
    let data = vec![-2.0, 3.0, 0.0, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::neg::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::neg::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "neg forward");
    assert_close(&grad_n, &grad_c, 1e-2, "neg backward");
}

#[test]
fn sqrt_forward_and_backward_parity() {
    let data = vec![1.0, 4.0, 9.0, 2.25];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sqrt::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sqrt::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sqrt forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sqrt backward");
}

#[test]
fn log_forward_and_backward_parity() {
    let data = vec![0.5, 1.0, 2.0, 4.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::log::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::log::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "log forward");
    assert_close(&grad_n, &grad_c, 1e-2, "log backward");
}

#[test]
fn tanh_forward_and_backward_parity() {
    let data = vec![-1.5, -0.3, 0.3, 1.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::tanh::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::tanh::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "tanh forward");
    assert_close(&grad_n, &grad_c, 1e-2, "tanh backward");
}

#[test]
fn sigmoid_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sigmoid::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sigmoid::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sigmoid forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sigmoid backward");
}

#[test]
fn swish_forward_and_backward_parity() {
    let data = vec![-2.0, -0.5, 0.5, 2.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::swish::<f32>(x).unwrap(), &data, &[4]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::swish::<f32>(x).unwrap(), &data, &[4]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "swish forward");
    assert_close(&grad_n, &grad_c, 1e-2, "swish backward");
}

#[test]
fn softmax_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.5];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::softmax::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::softmax::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "softmax forward");
    assert_close(&grad_n, &grad_c, 1e-2, "softmax backward");
}

#[test]
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
fn sum_all_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sum_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sum_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sum_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sum_all backward");
}

#[test]
fn mean_all_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::mean_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::mean_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "mean_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "mean_all backward");
}

#[test]
fn max_all_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::max_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::max_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "max_all backward");
}

#[test]
fn min_all_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::min_all::<f32>(x).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::min_all::<f32>(x).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "min_all forward");
    assert_close(&grad_n, &grad_c, 1e-2, "min_all backward");
}

#[test]
fn sum_dim_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::sum_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::sum_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "sum_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "sum_dim backward");
}

#[test]
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
fn max_dim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::max_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::max_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "max_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "max_dim backward");
}

#[test]
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
fn min_dim_forward_and_backward_parity() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::min_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::min_dim::<f32>(x, 1).unwrap(), &data, &[2, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "min_dim forward");
    assert_close(&grad_n, &grad_c, 1e-2, "min_dim backward");
}

#[test]
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
fn squeeze_forward_and_backward_parity() {
    let data = vec![1.0, 2.0, 3.0];
    let (fwd_n, grad_n) = run_and_grad::<NB>(|x| NB::squeeze::<f32>(x, 0).unwrap(), &data, &[1, 3]);
    let (fwd_c, grad_c) = run_and_grad::<CB>(|x| CB::squeeze::<f32>(x, 0).unwrap(), &data, &[1, 3]);
    assert_close(&fwd_n, &fwd_c, 1e-2, "squeeze forward");
    assert_close(&grad_n, &grad_c, 1e-2, "squeeze backward");
}

#[test]
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
