//! Integration test: `SGD<NativeBackend<f32, Cpu>, f32>::step()` (unmodified
//! `kindle-core` source) runs correctly across multiple sequential steps,
//! updating parameters only via `assign_var`.
//!
//! Covers NATBACK-09 ("NativeVar/RawVar mutation happens only at the
//! assign_var optimizer-step boundary").
//!
//! ## Design note
//!
//! `SGD.params` is private, so the test cannot read back the updated var
//! through SGD's internals. Instead, each test keeps its own reference to the
//! *pre-step* weight storage (recorded before building the forward graph)
//! and computes the expected post-step value analytically, then verifies the
//! next forward pass's `var_as_tensor` read through a fresh `NativeVar` clone
//! sees the right value. Since `NativeVar` is `Rc<RefCell<NativeStorage>>`,
//! a clone shares the same `RefCell`, so `assign_var` through SGD's owned
//! copy is visible through the clone the test holds.

extern crate alloc;
use alloc::collections::BTreeMap;

use kindle_core::prelude::*;
use kindle_native::NativeBackend;

/// Auto-generated documentation for B.
type B = NativeBackend<f32, Cpu>;

/// Build a flat F32 `Storage` from a literal vec, shape `[n]`.
fn storage1d(v: Vec<f32>) -> <B as Backend>::Storage<f32> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    B::from_bytes::<f32>(&bytes, &[v.len()], KindleDType::F32, &KindleDevice::cpu()).unwrap()
}

/// Extract the raw F32 values from a `Storage`.
fn f32_vec(s: &<B as Backend>::Storage<f32>) -> Vec<f32> {
    let bytes = B::to_bytes::<f32>(s).unwrap();
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        vec.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
    }
    vec
}

// ---------------------------------------------------------------------------
// Test 1: SGD::step() moves parameters in the correct direction
// ---------------------------------------------------------------------------

/// A loss `L = 0.5 * mean(w * w)` (so `dL/dw_i = w_i / n`) with the full
/// analytic chain:
///   - `w_1 = w_0 - lr * (w_0 / n)` for each element
///
/// We verify after two SGD steps that the parameter values match the analytic
/// formula, confirming SGD::step() is correctly wired end-to-end.
#[test]
fn sgd_step_updates_native_params_in_correct_direction() {
    let lr = 0.1f64;
    let n = 3usize;
    let w_init = vec![2.0f32, 4.0, 6.0];

    // Keep a clone of the NativeVar so we can read back through it after SGD updates it
    // (NativeVar is Rc<RefCell<_>>, so clone shares the same inner storage).
    let w_storage = storage1d(w_init.clone());
    let w_var = B::var_from_tensor::<f32>(&w_storage).unwrap();
    let w_var_clone = w_var.clone(); // our read-back handle

    let mut params: BTreeMap<String, <B as Backend>::RawVar> = BTreeMap::new();
    params.insert(String::from("w"), w_var);
    let mut sgd = SGD::<B, f32>::new(params, lr);

    // --- Step 1 ---
    // Forward: w_t through a tape-tracked loss = mean(w * w) * 0.5
    let w_t = B::var_as_tensor::<f32>(&w_var_clone).unwrap();
    let sq = B::mul::<f32>(&w_t, &w_t).unwrap();
    let half_sq = B::mul_scalar_float::<f32>(&sq, 0.5).unwrap();
    let loss = B::mean_all::<f32>(&half_sq).unwrap();
    let grads = B::backward::<f32>(&loss).unwrap();

    // grad_w = w / n (d(mean(w^2)*0.5)/dw_i = w_i/n)
    let grad_w = B::get_grad::<f32>(&w_t, &grads)
        .unwrap()
        .expect("w should have gradient");
    for (i, (&gv, &wv)) in f32_vec(&grad_w).iter().zip(w_init.iter()).enumerate() {
        let expected_grad = wv / n as f32;
        assert!(
            (gv - expected_grad).abs() < 1e-4,
            "grad[{i}]: expected {expected_grad}, got {gv}"
        );
    }

    sgd.step(&Gradients(grads)).unwrap();

    // Read back through our clone: assign_var updated the shared RefCell.
    let w_after_1 = f32_vec(&B::var_as_tensor::<f32>(&w_var_clone).unwrap());
    let expected_1: Vec<f32> = w_init
        .iter()
        .map(|&x| x - lr as f32 * (x / n as f32))
        .collect();
    for (i, (got, exp)) in w_after_1.iter().zip(expected_1.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "after step 1: w[{i}] expected {exp}, got {got}"
        );
    }

    // --- Step 2 ---
    let w_t2 = B::var_as_tensor::<f32>(&w_var_clone).unwrap();
    let sq2 = B::mul::<f32>(&w_t2, &w_t2).unwrap();
    let half_sq2 = B::mul_scalar_float::<f32>(&sq2, 0.5).unwrap();
    let loss2 = B::mean_all::<f32>(&half_sq2).unwrap();
    let grads2 = B::backward::<f32>(&loss2).unwrap();
    sgd.step(&Gradients(grads2)).unwrap();

    let w_after_2 = f32_vec(&B::var_as_tensor::<f32>(&w_var_clone).unwrap());
    let expected_2: Vec<f32> = expected_1
        .iter()
        .map(|&x| x - lr as f32 * (x / n as f32))
        .collect();
    for (i, (got, exp)) in w_after_2.iter().zip(expected_2.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "after step 2: w[{i}] expected {exp}, got {got}"
        );
    }

    // Sanity: loss should be decreasing (||w||^2 shrinks each step).
    let l0: f32 = w_init.iter().map(|x| x * x).sum::<f32>() * 0.5 / n as f32;
    let l2: f32 = w_after_2.iter().map(|x| x * x).sum::<f32>() * 0.5 / n as f32;
    assert!(l2 < l0, "loss should decrease: l0={l0}, l2={l2}");
}

// ---------------------------------------------------------------------------
// Test 2: mutation is restricted to assign_var (NATBACK-09)
// ---------------------------------------------------------------------------

/// After multiple SGD steps, the next `var_as_tensor` read reflects only the
/// LAST `assign_var` call's value — not a stale or accumulated state.
/// `w_0 = [4.0]`, lr = 0.5, loss = 0.5 * w^2 → dL/dw = w.
/// After k steps: w_k = 4.0 * (1 - 0.5)^k.
#[test]
fn sgd_mutation_is_restricted_to_assign_var_boundary() {
    let lr = 0.5f64;
    let w_init = 4.0f32;

    let w_storage = storage1d(vec![w_init]);
    let w_var = B::var_from_tensor::<f32>(&w_storage).unwrap();
    let w_var_clone = w_var.clone();

    let mut params: BTreeMap<String, <B as Backend>::RawVar> = BTreeMap::new();
    params.insert(String::from("w"), w_var);
    let mut sgd = SGD::<B, f32>::new(params, lr);

    let n_steps = 5usize;
    for _ in 0..n_steps {
        let w_t = B::var_as_tensor::<f32>(&w_var_clone).unwrap();
        let sq = B::mul::<f32>(&w_t, &w_t).unwrap();
        let half_sq = B::mul_scalar_float::<f32>(&sq, 0.5).unwrap();
        let loss = B::mean_all::<f32>(&half_sq).unwrap();
        let grads = B::backward::<f32>(&loss).unwrap();
        sgd.step(&Gradients(grads)).unwrap();
    }

    let w_final = f32_vec(&B::var_as_tensor::<f32>(&w_var_clone).unwrap());
    // Expected: 4.0 * (1 - 0.5)^5 = 4.0 / 32 = 0.125
    let expected = w_init * (1.0 - lr as f32).powi(n_steps as i32);
    assert!(
        (w_final[0] - expected).abs() < 1e-4,
        "after {n_steps} steps: expected {expected}, got {}",
        w_final[0]
    );
}
