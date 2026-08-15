#![cfg(all(
    feature = "wgpu",
    feature = "cpu",
    feature = "legacy-operation-api-tests"
))]

//! Cross-backend numerical parity tests.
//!
//! Validates that `CpuBackendImpl` and `WgpuBackendImpl` produce outputs within
//! 1e-4 for all common ops. Guards against silent divergence between the
//! CPU reference implementation and the WGSL shaders.

use incin::backend_authoring::*;
use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;

type Native = CpuBackendImpl;
type Wgpu = WgpuBackendImpl<incin::WgpuN<incin::typenum::U0>>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn native_storage(
    data: &[f32],
    shape: &[usize],
) -> <Native as incin_core::backend_authoring::StorageBackend>::Storage<f32> {
    let bytes = bytemuck::cast_slice(data);
    Native::from_bytes::<f32>(bytes, shape, DTypeId::F32.descriptor(), &DeviceId::cpu()).unwrap()
}

fn wgpu_storage(
    data: &[f32],
    shape: &[usize],
) -> <Wgpu as incin_core::backend_authoring::StorageBackend>::Storage<f32> {
    let bytes = bytemuck::cast_slice(data);
    Wgpu::from_bytes::<f32>(bytes, shape, DTypeId::F32.descriptor(), &DeviceId::wgpu(0)).unwrap()
}

fn native_vec(
    s: &<Native as incin_core::backend_authoring::StorageBackend>::Storage<f32>,
) -> Vec<f64> {
    Native::float_to_vec1::<f32>(s).unwrap()
}

fn wgpu_vec(s: &<Wgpu as incin_core::backend_authoring::StorageBackend>::Storage<f32>) -> Vec<f64> {
    Wgpu::float_to_vec1::<f32>(s).unwrap()
}

/// Maximum absolute difference between two flat f64 vectors.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[track_caller]
fn assert_close(a: &[f64], b: &[f64], tol: f64, label: &str) {
    let diff = max_abs_diff(a, b);
    assert!(
        diff <= tol,
        "{label}: max abs diff {diff:.2e} > tolerance {tol:.2e}\n  native={a:.4?}\n  wgpu  ={b:.4?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parity_add() -> Result<()> {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];

    let n = native_vec(&Native::add::<f32>(
        &native_storage(&a, &[2, 2]),
        &native_storage(&b, &[2, 2]),
    )?);
    let w = wgpu_vec(&Wgpu::add::<f32>(
        &wgpu_storage(&a, &[2, 2]),
        &wgpu_storage(&b, &[2, 2]),
    )?);
    assert_close(&n, &w, 1e-5, "add");
    Ok(())
}

#[test]
fn test_parity_relu() -> Result<()> {
    let data = vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0];
    let n = native_vec(&Native::relu::<f32>(&native_storage(&data, &[6]))?);
    let w = wgpu_vec(&Wgpu::relu::<f32>(&wgpu_storage(&data, &[6]))?);
    assert_close(&n, &w, 1e-5, "relu");
    Ok(())
}

#[test]
fn test_parity_gelu() -> Result<()> {
    let data = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];
    let n = native_vec(&Native::gelu::<f32>(&native_storage(&data, &[6]))?);
    let w = wgpu_vec(&Wgpu::gelu::<f32>(&wgpu_storage(&data, &[6]))?);
    // GELU approximations differ slightly between CPU and GPU implementations
    assert_close(&n, &w, 2e-4, "gelu");
    Ok(())
}

#[test]
fn test_parity_sigmoid() -> Result<()> {
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let n = native_vec(&Native::sigmoid::<f32>(&native_storage(&data, &[5]))?);
    let w = wgpu_vec(&Wgpu::sigmoid::<f32>(&wgpu_storage(&data, &[5]))?);
    assert_close(&n, &w, 1e-5, "sigmoid");
    Ok(())
}

#[test]
fn test_parity_softmax() -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5];
    let n = native_vec(&Native::softmax::<f32>(&native_storage(&data, &[2, 3]), 1)?);
    let w = wgpu_vec(&Wgpu::softmax::<f32>(&wgpu_storage(&data, &[2, 3]), 1)?);
    assert_close(&n, &w, 1e-4, "softmax");
    Ok(())
}

#[test]
fn test_parity_matmul_2d() -> Result<()> {
    // [2×3] @ [3×2] = [2×2]
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let n = native_vec(&Native::matmul::<f32>(
        &native_storage(&a, &[2, 3]),
        &native_storage(&b, &[3, 2]),
    )?);
    let w = wgpu_vec(&Wgpu::matmul::<f32>(
        &wgpu_storage(&a, &[2, 3]),
        &wgpu_storage(&b, &[3, 2]),
    )?);
    assert_close(&n, &w, 1e-3, "matmul 2d");
    Ok(())
}

#[test]
fn test_parity_sum_dim() -> Result<()> {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let n = native_vec(&Native::sum_dim::<f32>(&native_storage(&data, &[3, 4]), 1)?);
    let w = wgpu_vec(&Wgpu::sum_dim::<f32>(&wgpu_storage(&data, &[3, 4]), 1)?);
    assert_close(&n, &w, 1e-4, "sum_dim");
    Ok(())
}

#[test]
fn test_parity_gradient_add() -> Result<()> {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, 1.5, 2.5, 3.5];

    let a_native = native_storage(&a_data, &[2, 2]);
    let b_native = native_storage(&b_data, &[2, 2]);
    let out_native = Native::add::<f32>(&a_native, &b_native)?;
    let grads_native = Native::backward::<f32>(&out_native)?;
    let grad_a_native = native_vec(&Native::get_grad::<f32>(&a_native, &grads_native)?.unwrap());

    let a_wgpu = wgpu_storage(&a_data, &[2, 2]);
    let b_wgpu = wgpu_storage(&b_data, &[2, 2]);
    let out_wgpu = Wgpu::add::<f32>(&a_wgpu, &b_wgpu)?;
    let grads_wgpu = Wgpu::backward::<f32>(&out_wgpu)?;
    let grad_a_wgpu = wgpu_vec(&Wgpu::get_grad::<f32>(&a_wgpu, &grads_wgpu)?.unwrap());

    assert_close(&grad_a_native, &grad_a_wgpu, 1e-4, "add gradient");
    Ok(())
}

#[test]
fn test_parity_gradient_mul() -> Result<()> {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![2.0f32, 3.0, 4.0, 5.0];

    let a_native = native_storage(&a_data, &[2, 2]);
    let b_native = native_storage(&b_data, &[2, 2]);
    let out_native = Native::mul::<f32>(&a_native, &b_native)?;
    let grads_native = Native::backward::<f32>(&out_native)?;
    let grad_a_native = native_vec(&Native::get_grad::<f32>(&a_native, &grads_native)?.unwrap());

    let a_wgpu = wgpu_storage(&a_data, &[2, 2]);
    let b_wgpu = wgpu_storage(&b_data, &[2, 2]);
    let out_wgpu = Wgpu::mul::<f32>(&a_wgpu, &b_wgpu)?;
    let grads_wgpu = Wgpu::backward::<f32>(&out_wgpu)?;
    let grad_a_wgpu = wgpu_vec(&Wgpu::get_grad::<f32>(&a_wgpu, &grads_wgpu)?.unwrap());

    assert_close(&grad_a_native, &grad_a_wgpu, 1e-4, "mul gradient");
    Ok(())
}

#[test]
fn test_parity_layer_norm() -> Result<()> {
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight_data = vec![1.0f32, 1.0, 1.0];
    let bias_data = vec![0.0f32, 0.0, 0.0];

    let n = native_vec(&Native::layer_norm::<f32>(
        &native_storage(&input_data, &[2, 3]),
        &native_storage(&weight_data, &[3]),
        Some(&native_storage(&bias_data, &[3])),
        1e-5,
    )?);
    let w = wgpu_vec(&Wgpu::layer_norm::<f32>(
        &wgpu_storage(&input_data, &[2, 3]),
        &wgpu_storage(&weight_data, &[3]),
        Some(&wgpu_storage(&bias_data, &[3])),
        1e-5,
    )?);
    assert_close(&n, &w, 1e-4, "layer_norm");
    Ok(())
}

#[test]
fn test_parity_mse_loss() -> Result<()> {
    let pred_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let target_data = vec![0.0f32, 1.0, 2.0, 3.0];

    let n = native_vec(&Native::mse_loss::<f32>(
        &native_storage(&pred_data, &[4]),
        &native_storage(&target_data, &[4]),
        Reduction::Mean,
    )?);
    let w = wgpu_vec(&Wgpu::mse_loss::<f32>(
        &wgpu_storage(&pred_data, &[4]),
        &wgpu_storage(&target_data, &[4]),
        Reduction::Mean,
    )?);
    assert_close(&n, &w, 1e-4, "mse_loss");
    Ok(())
}

#[test]
fn test_parity_cross_entropy_loss() -> Result<()> {
    let pred_data = vec![2.0f32, 1.0, 0.1, 0.5, 2.5, 0.3];
    let target_data = vec![0.0f32, 1.0];

    let n = native_vec(&Native::cross_entropy_loss::<f32, f32>(
        &native_storage(&pred_data, &[2, 3]),
        &native_storage(&target_data, &[2]),
        Reduction::Mean,
    )?);
    let w = wgpu_vec(&Wgpu::cross_entropy_loss::<f32, f32>(
        &wgpu_storage(&pred_data, &[2, 3]),
        &wgpu_storage(&target_data, &[2]),
        Reduction::Mean,
    )?);
    assert_close(&n, &w, 1e-4, "cross_entropy_loss");
    Ok(())
}
