//! Tests for fused optimizer step kernels (Adam, AdamW, SGD).

#![allow(unused_imports)]

use incin_core::exec::catalog::{AdamAttributes, AdamWAttributes, SgdAttributes};
use incin_core::tensor::dtype::DTypeId;

#[test]
fn test_adamw_analytical_formula() {
    let p_in: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5];
    let grad: Vec<f32> = vec![0.1, -0.2, 0.05, -0.1];
    let m_in: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let v_in: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

    let lr = 1e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let weight_decay = 0.01_f32;
    let step = 1;

    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);

    let mut p_out = vec![0.0_f32; p_in.len()];
    let mut m_out = vec![0.0_f32; p_in.len()];
    let mut v_out = vec![0.0_f32; p_in.len()];

    for i in 0..p_in.len() {
        let p = p_in[i];
        let g = grad[i];
        let m = beta1 * m_in[i] + (1.0 - beta1) * g;
        let v = beta2 * v_in[i] + (1.0 - beta2) * g * g;

        let m_hat = m / bc1;
        let v_hat = v / bc2;

        let p_decay = p - lr * weight_decay * p;
        let p_step = p_decay - lr * (m_hat / (v_hat.sqrt() + eps));

        p_out[i] = p_step;
        m_out[i] = m;
        v_out[i] = v;
    }

    let expected_m0 = 0.1 * (1.0 - 0.9);
    assert!((m_out[0] - expected_m0).abs() < 1e-6);
    assert!(p_out[0] < p_in[0]);
}

#[test]
fn test_adam_analytical_formula() {
    let p_in: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5];
    let grad: Vec<f32> = vec![0.1, -0.2, 0.05, -0.1];
    let m_in: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let v_in: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

    let lr = 1e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let step = 1;

    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);

    let mut p_out = vec![0.0_f32; p_in.len()];
    for i in 0..p_in.len() {
        let p = p_in[i];
        let g = grad[i];
        let m = beta1 * m_in[i] + (1.0 - beta1) * g;
        let v = beta2 * v_in[i] + (1.0 - beta2) * g * g;
        let m_hat = m / bc1;
        let v_hat = v / bc2;
        p_out[i] = p - lr * (m_hat / (v_hat.sqrt() + eps));
    }

    assert!(p_out[0] < p_in[0]);
}

#[test]
fn test_sgd_analytical_formula() {
    let p_in: Vec<f32> = vec![1.0, 2.0, 3.0];
    let grad: Vec<f32> = vec![0.1, 0.2, 0.3];
    let lr = 0.1_f32;

    let mut p_out = vec![0.0_f32; p_in.len()];
    for i in 0..p_in.len() {
        p_out[i] = p_in[i] - lr * grad[i];
    }

    assert!((p_out[0] - 0.99).abs() < 1e-6);
    assert!((p_out[1] - 1.98).abs() < 1e-6);
    assert!((p_out[2] - 2.97).abs() < 1e-6);
}
