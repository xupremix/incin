//! Integration test: full linear-regression training-loop parity between
//! `NativeBackend<f32, Cpu>` and `CandleBackend<f32, Cpu>`, plus an
//! end-to-end gradient-accumulation proof through the public Backend API.
//!
//! Both training loops use ONLY the `Backend` trait surface (no raw
//! `candle_core` or `NativeStorage` internals) — identical construction
//! via `from_bytes` / `var_from_tensor`, identical forward/backward/step
//! sequence, identical literal data.

use kindle_backends::candle::CandleBackend;
use kindle_core::nn::Reduction;
use kindle_core::prelude::*;
use kindle_native::NativeBackend;

// ── Type aliases ─────────────────────────────────────────────────────────────

type NB = NativeBackend<f32, Cpu>;
type CB = CandleBackend<f32, Cpu>;

// ── Data fixtures ─────────────────────────────────────────────────────────────

/// x: shape [4, 2] — 4 samples, 2 features
const X_DATA: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

/// target: shape [4, 1] — y = 2*x0 - x1 + 0.5
const TARGET_DATA: [f32; 4] = [0.5, 2.5, 4.5, 6.5];

/// Weight initial value: shape [1, 2]
const W_INIT: [f32; 2] = [0.5, 0.5];

/// Bias initial value: shape [1]
const B_INIT: [f32; 1] = [0.0];

/// Convert a `&[f32]` to the raw bytes needed by `Backend::from_bytes`.
fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect()
}

/// Build a `Backend::Storage<f32>` from literal float data using `from_bytes`
/// (the only backend-agnostic construction path that doesn't invoke the
/// backend's own RNG, per T-01-22 mitigation).
fn make_storage<B: Backend>(data: &[f32], shape: &[usize]) -> B::Storage<f32> {
    B::from_bytes::<f32>(
        &as_bytes(data),
        shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .expect("from_bytes")
}

/// Read the single f64 scalar from a 0-d or 1-d storage.
fn scalar<B: Backend>(t: &B::Storage<f32>) -> f64 {
    B::float_to_scalar::<f32>(t).expect("float_to_scalar")
}

// ── Generic training loop ─────────────────────────────────────────────────────

/// Hand-rolled linear regression: y_hat = x @ w.T + b (matmul + add).
/// Returns the per-epoch loss values (f64 scalars for easy comparison).
fn train<B: Backend>(n_epochs: usize, lr: f64) -> Vec<f64> {
    let x = make_storage::<B>(&X_DATA, &[4, 2]);
    let target = make_storage::<B>(&TARGET_DATA, &[4, 1]);

    let w_init_stor = make_storage::<B>(&W_INIT, &[1, 2]);
    let b_init_stor = make_storage::<B>(&B_INIT, &[1]);

    let mut w_var = B::var_from_tensor::<f32>(&w_init_stor).expect("w var");
    let mut b_var = B::var_from_tensor::<f32>(&b_init_stor).expect("b var");

    let mut losses = Vec::with_capacity(n_epochs);
    for _ in 0..n_epochs {
        // Forward: y_hat = x @ w.T + b  (tape-tracked)
        let w_t = B::var_as_tensor::<f32>(&w_var).unwrap();
        let b_t = B::var_as_tensor::<f32>(&b_var).unwrap();
        let w_tr = B::transpose::<f32>(&w_t, 0, 1).unwrap(); // [2, 1]
        let y_hat = B::matmul::<f32>(&x, &w_tr).unwrap();    // [4, 1]
        let y_hat = B::add::<f32>(&y_hat, &b_t).unwrap();    // broadcast bias

        // Loss = MSE Mean
        let loss = B::mse_loss::<f32>(&y_hat, &target, Reduction::Mean).unwrap();
        losses.push(scalar::<B>(&loss));

        // Backward
        let grads = B::backward::<f32>(&loss).unwrap();

        // Manual SGD update (avoids needing to fight with the params HashMap's
        // private-field constraint again — equivalent to SGD::step with a single
        // iteration over (w, b)).
        let w_now = B::var_as_tensor::<f32>(&w_var).unwrap();
        if let Some(gw) = B::get_grad::<f32>(&w_now, &grads).unwrap() {
            let updated_w = B::sub::<f32>(&w_now, &B::mul_scalar_float::<f32>(&gw, lr).unwrap()).unwrap();
            B::assign_var::<f32>(&mut w_var, &updated_w).unwrap();
        }
        let b_now = B::var_as_tensor::<f32>(&b_var).unwrap();
        if let Some(gb) = B::get_grad::<f32>(&b_now, &grads).unwrap() {
            let updated_b = B::sub::<f32>(&b_now, &B::mul_scalar_float::<f32>(&gb, lr).unwrap()).unwrap();
            B::assign_var::<f32>(&mut b_var, &updated_b).unwrap();
        }
    }
    losses
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn linear_regression_loss_curve_matches_candle() {
    let n_epochs = 8;
    let lr = 0.01;
    let rel_tol = 0.05; // 5 % relative tolerance

    let native_losses = train::<NB>(n_epochs, lr);
    let candle_losses = train::<CB>(n_epochs, lr);

    assert_eq!(
        native_losses.len(),
        candle_losses.len(),
        "both backends must produce the same number of epoch losses"
    );

    for (epoch, (&nl, &cl)) in native_losses.iter().zip(candle_losses.iter()).enumerate() {
        let rel_err = (nl - cl).abs() / cl.abs().max(1e-8);
        assert!(
            rel_err < rel_tol,
            "epoch {epoch}: native={nl:.6}, candle={cl:.6}, rel_err={rel_err:.4} > {rel_tol}"
        );
    }

    // Native loss must decrease overall.
    assert!(
        native_losses.last().unwrap() < native_losses.first().unwrap(),
        "native loss should decrease: first={:.6}, last={:.6}",
        native_losses.first().unwrap(),
        native_losses.last().unwrap()
    );
}

#[test]
fn gradient_accumulation_sums_on_reuse_end_to_end() {
    // Build a graph where x is used TWICE: loss = mean(x + x) = mean(2x).
    // d(mean(2x))/d(x_i) = 2/n.
    // Proves accumulation is summed, not overwritten, through the public API.
    use kindle_native::storage::{NativeBuffer, NativeStorage};

    let n = 4usize;
    let vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let x_stor = NativeStorage::from_contiguous(NativeBuffer::F32(vals), vec![n]);
    let x_id = x_stor.id;

    // Use x twice in the same graph.
    let y = NB::add::<f32>(&x_stor, &x_stor).unwrap(); // y = 2x, tape records two reads of x
    let loss = NB::mean_all::<f32>(&y).unwrap();        // scalar

    let grads = NB::backward::<f32>(&loss).unwrap();
    let g = grads.grads.get(&x_id).expect("x should have a gradient");

    // Each element's expected gradient: d(mean(x+x))/d(x_i) = 2/n
    let expected = 2.0f32 / n as f32;
    match &*g.buffer {
        NativeBuffer::F32(v) => {
            for (i, &gv) in v.iter().enumerate() {
                assert!(
                    (gv - expected).abs() < 1e-5,
                    "grad[{i}]: expected {expected:.6}, got {gv:.6} \
                     (must be 2/n from accumulation, not 1/n from overwrite)"
                );
            }
        }
        _ => panic!("expected F32 gradient buffer"),
    }
}
