//! Gradient parity.
//!
//! Two things live here and they answer different questions. The
//! cross-backend cases compare CPU against WGPU or CUDA and need that
//! hardware. The CPU cases below them compare the backward pass against
//! hand-computed calculus and against the invariants the shared walk is
//! responsible for, and need nothing.
//!
//! The file was `#![cfg(all(feature = "cpu", any(feature = "wgpu", feature =
//! "cuda")))]` until `GRD-003`, which is to say the row that names
//! `--features std,cpu --test gradient_parity` as its evidence compiled zero
//! tests under it.

#![cfg(all(feature = "cpu", feature = "legacy-operation-api-tests"))]
//! Historical backend-helper parity tests. Current CPU parity is covered by
//! the descriptor-based canonical suite.

use incin_backends::cpu::CpuBackendImpl;
#[cfg(feature = "cuda")]
use incin_backends::cuda::CudaBackendImpl;
#[cfg(feature = "wgpu")]
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::exec::{TapeStorage, check_gradients};
use incin_core::prelude::*;
#[cfg(any(feature = "wgpu", feature = "cuda"))]

type CpuB = CpuBackendImpl;
#[cfg(feature = "wgpu")]
type WgpuB = WgpuBackendImpl<incin_core::prelude::WgpuN<incin_core::typenum::U0>>;
#[cfg(feature = "cuda")]
type CudaB = CudaBackendImpl<incin_core::prelude::CudaN<incin_core::typenum::U0>>;

fn read_f32<B: Backend>(s: &B::Storage<f32>) -> Vec<f32> {
    let bytes = B::to_bytes::<f32>(s).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Only the cross-backend cases compare within a tolerance; the CPU ones below
/// assert exact values, because they are asserting arithmetic rather than
/// agreement between two devices.
#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > tol {
            return false;
        }
    }
    true
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_elementwise_add() {
    let shape = vec![2, 3];
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![10.0f32, 20.0, 30.0, 40.0, 5.0, 60.0];

    // CPU
    let cpu_a = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::add::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::add::<f32>(&wgpu_a, &wgpu_b).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    // Compare forward
    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "Forward mismatch: CPU {:?} vs WGPU {:?}",
        cpu_res,
        wgpu_res
    );

    // Compare backward
    let cpu_ga = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_a, &cpu_grads).unwrap().unwrap());
    let wgpu_ga = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_a, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_ga, &wgpu_ga, 1e-4),
        "Grad A mismatch: CPU {:?} vs WGPU {:?}",
        cpu_ga,
        wgpu_ga
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_activations_softmax() {
    let shape = vec![1, 4];
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];

    // CPU
    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::softmax::<f32>(&cpu_in, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::softmax::<f32>(&wgpu_in, 1).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "Softmax Forward mismatch"
    );

    let cpu_g = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_in, &cpu_grads).unwrap().unwrap());
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-4),
        "Softmax Grad mismatch"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_matmul() {
    let shape_a = vec![2, 3];
    let shape_b = vec![3, 2];
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    // CPU
    let cpu_a = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::matmul::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::matmul::<f32>(&wgpu_a, &wgpu_b).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-3),
        "Matmul Forward mismatch: CPU {:?} vs WGPU {:?}",
        cpu_res,
        wgpu_res
    );

    let cpu_ga = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_a, &cpu_grads).unwrap().unwrap());
    let wgpu_ga = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_a, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_ga, &wgpu_ga, 1e-3),
        "Matmul Grad A mismatch"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_layer_norm() {
    let shape_x = vec![1, 4];
    let shape_w = vec![4];
    let data_x = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_w = vec![1.0f32, 1.0, 1.0, 1.0];
    let data_b = vec![0.0f32, 0.0, 0.0, 0.0];

    // CPU
    let cpu_x = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::layer_norm::<f32>(&cpu_x, &cpu_w, Some(&cpu_b), 1e-5).unwrap();

    // WGPU
    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::layer_norm::<f32>(&wgpu_x, &wgpu_w, Some(&wgpu_b), 1e-5).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-3),
        "LayerNorm Forward mismatch: CPU {:?} vs WGPU {:?}",
        cpu_res,
        wgpu_res
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_max_pool2d() {
    // Exercises the newest, most complex WGPU backward in this file (host
    // readback + recomputed argmax scatter, wgpu/backend.rs) against CPU's
    // independently-implemented max_window_2d/scatter_pool_grad_2d
    // (cpu/ops/pool.rs) — this is exactly the kind of cross-backend check
    // that would have caught a subtle indexing mistake in either one.
    let shape = vec![1, 1, 4, 4];
    let data = vec![
        1.0f32, 8.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0, 10.0, 0.5, 11.0, 1.5, 12.0, 2.5, 13.0, 3.5,
    ];

    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::max_pool2d::<f32>(&cpu_in, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::max_pool2d::<f32>(&wgpu_in, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "max_pool2d forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_g = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_in, &cpu_grads).unwrap().unwrap());
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-4),
        "max_pool2d grad mismatch: CPU {cpu_g:?} vs WGPU {wgpu_g:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_cross_entropy_loss_nonzero_target() {
    // Regression coverage for C-9 (embedding/cross_entropy_loss bit-
    // reinterpreted F32-stored index bytes as u32) at the cross-backend
    // level: uses a non-zero target class specifically, which is exactly
    // what that bug corrupted.
    let shape_pred = vec![2, 3];
    let shape_tgt = vec![2];
    let data_pred = vec![2.0f32, 1.0, -0.5, 0.5, 3.0, 0.2];
    let data_tgt = vec![0.0f32, 2.0]; // class 0, class 2 (non-zero)

    let cpu_pred = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_tgt = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::cross_entropy_loss::<f32, f32>(&cpu_pred, &cpu_tgt, Reduction::Mean).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_pred = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_tgt = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out =
        WgpuB::cross_entropy_loss::<f32, f32>(&wgpu_pred, &wgpu_tgt, Reduction::Mean).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "cross_entropy_loss forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_g = read_f32::<CpuB>(
        &CpuB::get_grad::<f32>(&cpu_pred, &cpu_grads)
            .unwrap()
            .unwrap(),
    );
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_pred, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-4),
        "cross_entropy_loss grad mismatch: CPU {cpu_g:?} vs WGPU {wgpu_g:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_cross_entropy_rejects_unsupported_u32_targets() {
    let shape_pred = vec![2, 3];
    let shape_tgt = vec![2];
    let data_pred = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_tgt = vec![0u32, 2u32];

    // CPU (target is f32 / i64)
    let data_tgt_f32: Vec<f32> = data_tgt.iter().map(|&x| x as f32).collect();
    let cpu_pred = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_tgt = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt_f32),
        &shape_tgt,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::cross_entropy_loss::<f32, f32>(&cpu_pred, &cpu_tgt, Reduction::Mean).unwrap();
    assert_eq!(CpuB::shape::<f32>(&cpu_out), Vec::<usize>::new());

    // WGPU
    let _wgpu_pred = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let error = WgpuB::from_bytes::<u32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::U32.into(),
        &DeviceId::wgpu(0),
    )
    .err()
    .expect("WGPU must reject U32 storage");
    if let Error::UnsupportedDType { dtype, backend, op } = error {
        assert_eq!(dtype, DTypeId::U32.descriptor());
        assert_eq!(backend, "Wgpu");
        assert_eq!(op, "from_bytes");
    } else {
        panic!("expected UnsupportedDType");
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_activations_relu() {
    let shape = vec![2, 3];
    let data = vec![-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0];

    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::relu::<f32>(&cpu_in).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::relu::<f32>(&wgpu_in).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "ReLU forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_g = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_in, &cpu_grads).unwrap().unwrap());
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-4),
        "ReLU grad mismatch: CPU {cpu_g:?} vs WGPU {wgpu_g:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_activations_gelu() {
    let shape = vec![2, 3];
    let data = vec![-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0];

    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::gelu::<f32>(&cpu_in).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::gelu::<f32>(&wgpu_in).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-3),
        "GELU forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_g = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_in, &cpu_grads).unwrap().unwrap());
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-3),
        "GELU grad mismatch: CPU {cpu_g:?} vs WGPU {wgpu_g:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_batch_norm() {
    let shape_x = vec![2, 2, 2, 2];
    let shape_c = vec![2];
    let data_x = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let data_w = vec![1.0f32, 1.0];
    let data_b = vec![0.0f32, 0.0];

    let cpu_x = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::batch_norm::<f32>(&cpu_x, Some(&cpu_w), Some(&cpu_b), None, None, 1e-5, 0.1).unwrap();

    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out =
        WgpuB::batch_norm::<f32>(&wgpu_x, Some(&wgpu_w), Some(&wgpu_b), None, None, 1e-5, 0.1)
            .unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-3),
        "BatchNorm Forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_reductions_sum_and_mean() {
    let shape = vec![2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Sum dim 1
    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_sum = CpuB::sum_dim::<f32>(&cpu_in, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_sum).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_sum = WgpuB::sum_dim::<f32>(&wgpu_in, 1).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_sum).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_sum);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_sum);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "sum_dim forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_g = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_in, &cpu_grads).unwrap().unwrap());
    let wgpu_g = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_g, &wgpu_g, 1e-4),
        "sum_dim grad mismatch: CPU {cpu_g:?} vs WGPU {wgpu_g:?}"
    );

    // Mean dim 0
    let cpu_mean = CpuB::mean_dim::<f32>(&cpu_in, 0).unwrap();
    let cpu_mean_grads = CpuB::backward::<f32>(&cpu_mean).unwrap();

    let wgpu_mean = WgpuB::mean_dim::<f32>(&wgpu_in, 0).unwrap();
    let wgpu_mean_grads = WgpuB::backward::<f32>(&wgpu_mean).unwrap();

    let cpu_mean_res = read_f32::<CpuB>(&cpu_mean);
    let wgpu_mean_res = read_f32::<WgpuB>(&wgpu_mean);
    assert!(
        approx_eq_slice(&cpu_mean_res, &wgpu_mean_res, 1e-4),
        "mean_dim forward mismatch: CPU {cpu_mean_res:?} vs WGPU {wgpu_mean_res:?}"
    );

    let cpu_mg = read_f32::<CpuB>(
        &CpuB::get_grad::<f32>(&cpu_in, &cpu_mean_grads)
            .unwrap()
            .unwrap(),
    );
    let wgpu_mg = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_in, &wgpu_mean_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_mg, &wgpu_mg, 1e-4),
        "mean_dim grad mismatch: CPU {cpu_mg:?} vs WGPU {wgpu_mg:?}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn parity_conv2d() {
    let shape_x = vec![1, 1, 4, 4];
    let shape_w = vec![1, 1, 3, 3];
    let data_x: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let data_w = vec![1.0f32; 9];

    let cpu_x = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::conv2d::<f32>(&cpu_x, &cpu_w, None, 1, 0, 1, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_out = WgpuB::conv2d::<f32>(&wgpu_x, &wgpu_w, None, 1, 0, 1, 1).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-4),
        "Conv2d forward mismatch: CPU {cpu_res:?} vs WGPU {wgpu_res:?}"
    );

    let cpu_gx = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_x, &cpu_grads).unwrap().unwrap());
    let wgpu_gx = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_x, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_gx, &wgpu_gx, 1e-4),
        "Conv2d grad X mismatch: CPU {cpu_gx:?} vs WGPU {wgpu_gx:?}"
    );

    let cpu_gw = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_w, &cpu_grads).unwrap().unwrap());
    let wgpu_gw = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_w, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_gw, &wgpu_gw, 1e-4),
        "Conv2d grad W mismatch: CPU {cpu_gw:?} vs WGPU {wgpu_gw:?}"
    );
}

// =========================================================================
// CUDA Parity Tests (gated on feature = "cuda", #[ignore]d without hardware)
// =========================================================================

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_parity_elementwise_add() {
    let shape = vec![2, 3];
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![10.0f32, 20.0, 30.0, 40.0, 5.0, 60.0];

    let cpu_a = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::add::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_a = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_out = CudaB::add::<f32>(&cuda_a, &cuda_b).unwrap();
    let cuda_grads = CudaB::backward::<f32>(&cuda_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let cuda_res = read_f32::<CudaB>(&cuda_out);
    assert!(approx_eq_slice(&cpu_res, &cuda_res, 1e-4));

    let cpu_ga = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_a, &cpu_grads).unwrap().unwrap());
    let cuda_ga = read_f32::<CudaB>(
        &CudaB::get_grad::<f32>(&cuda_a, &cuda_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(approx_eq_slice(&cpu_ga, &cuda_ga, 1e-4));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_parity_matmul() {
    let shape_a = vec![2, 3];
    let shape_b = vec![3, 2];
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let cpu_a = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::matmul::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_a = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_out = CudaB::matmul::<f32>(&cuda_a, &cuda_b).unwrap();
    let cuda_grads = CudaB::backward::<f32>(&cuda_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let cuda_res = read_f32::<CudaB>(&cuda_out);
    assert!(approx_eq_slice(&cpu_res, &cuda_res, 1e-3));

    let cpu_ga = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_a, &cpu_grads).unwrap().unwrap());
    let cuda_ga = read_f32::<CudaB>(
        &CudaB::get_grad::<f32>(&cuda_a, &cuda_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(approx_eq_slice(&cpu_ga, &cuda_ga, 1e-3));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_parity_conv2d() {
    let shape_x = vec![1, 1, 4, 4];
    let shape_w = vec![1, 1, 3, 3];
    let data_x: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let data_w = vec![1.0f32; 9];

    let cpu_x = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::conv2d::<f32>(&cpu_x, &cpu_w, None, 1, 0, 1, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_x = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_w = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_out = CudaB::conv2d::<f32>(&cuda_x, &cuda_w, None, 1, 0, 1, 1).unwrap();
    let cuda_grads = CudaB::backward::<f32>(&cuda_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let cuda_res = read_f32::<CudaB>(&cuda_out);
    assert!(approx_eq_slice(&cpu_res, &cuda_res, 1e-4));

    let cpu_gx = read_f32::<CpuB>(&CpuB::get_grad::<f32>(&cpu_x, &cpu_grads).unwrap().unwrap());
    let cuda_gx = read_f32::<CudaB>(
        &CudaB::get_grad::<f32>(&cuda_x, &cuda_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(approx_eq_slice(&cpu_gx, &cuda_gx, 1e-4));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_parity_batch_norm() {
    let shape_x = vec![2, 2, 2, 2];
    let shape_c = vec![2];
    let data_x = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let data_w = vec![1.0f32, 1.0];
    let data_b = vec![0.0f32, 0.0];

    let cpu_x = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::batch_norm::<f32>(&cpu_x, Some(&cpu_w), Some(&cpu_b), None, None, 1e-5, 0.1).unwrap();

    let cuda_x = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_w = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_out =
        CudaB::batch_norm::<f32>(&cuda_x, Some(&cuda_w), Some(&cuda_b), None, None, 1e-5, 0.1)
            .unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let cuda_res = read_f32::<CudaB>(&cuda_out);
    assert!(approx_eq_slice(&cpu_res, &cuda_res, 1e-3));
}

// ── CPU (`GRD-003`) ──────────────────────────────────────────────────────────
//
// These need no accelerator, and they are the reason this file's crate-level
// cfg changed. They are not a second copy of the cross-backend cases above:
// those ask whether two devices agree, and these ask whether the shared
// backward walk in `incin_core::exec::tape` still computes calculus, still
// sums a reused tensor's contributions rather than overwriting them, and still
// drains before it invokes anything.

/// A CPU tensor from `data` with `shape`.
fn cpu(
    data: &[f32],
    shape: &[usize],
) -> <CpuB as incin_core::backend_authoring::StorageBackend>::Storage<f32> {
    CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(data),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .unwrap()
}

/// The gradient the backward pass accumulated for `t`.
fn grad_of(
    t: &<CpuB as incin_core::backend_authoring::StorageBackend>::Storage<f32>,
    grads: &<CpuB as Backend>::Grads,
) -> Option<Vec<f32>> {
    CpuB::get_grad::<f32>(t, grads)
        .unwrap()
        .map(|g| read_f32::<CpuB>(&g))
}

#[test]
fn a_product_differentiates_to_the_other_operand() {
    // d(a*b)/da = b and d(a*b)/db = a. Asserted against the arithmetic rather
    // than against a recording of what the old walk returned, so the test
    // still means something after the next migration.
    let a = cpu(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = cpu(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    let out = CpuB::mul::<f32>(&a, &b).unwrap();
    let grads = CpuB::backward::<f32>(&out).unwrap();

    assert_eq!(grad_of(&a, &grads).unwrap(), vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(grad_of(&b, &grads).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_tensor_used_twice_receives_the_sum_of_both_contributions() {
    // `x + x` differentiates to 2, not to 1. Writing the accumulation as an
    // insert returns 1 and loses one of the two contributions silently: that
    // is CPUBACK-05, and the walk it was found in now lives in the core, so
    // this is the test that follows it there.
    let x = cpu(&[1.0, 2.0, 3.0], &[3]);

    let out = CpuB::add::<f32>(&x, &x).unwrap();
    let grads = CpuB::backward::<f32>(&out).unwrap();

    assert_eq!(grad_of(&x, &grads).unwrap(), vec![2.0, 2.0, 2.0]);
}

#[test]
fn a_deeper_reuse_still_sums_every_path() {
    // y = x*x, so dy/dx = 2x through two separate paths of the product rule.
    let x = cpu(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let out = CpuB::mul::<f32>(&x, &x).unwrap();
    let grads = CpuB::backward::<f32>(&out).unwrap();

    assert_eq!(grad_of(&x, &grads).unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn the_loss_is_seeded_with_ones() {
    let x = cpu(&[5.0, -1.0], &[2]);
    let out = CpuB::relu::<f32>(&x).unwrap();

    let grads = CpuB::backward::<f32>(&out).unwrap();

    // Seeded with ones and then masked by relu's own derivative, which is the
    // only thing that could have turned the second entry into a zero.
    assert_eq!(grad_of(&x, &grads).unwrap(), vec![1.0, 0.0]);
}

#[test]
fn the_tape_is_drained_before_any_recipe_runs() {
    let x = cpu(&[1.0, 2.0], &[2]);
    let y = cpu(&[3.0, 4.0], &[2]);
    let out = CpuB::mul::<f32>(&x, &y).unwrap();

    assert!(incin_backends::cpu::tape_depth() > 0);
    let first = CpuB::backward::<f32>(&out).unwrap();
    assert_eq!(incin_backends::cpu::tape_depth(), 0);

    // A second pass over the same loss has nothing left to walk, so it reaches
    // the seed and nothing else. Draining afterwards instead of before would
    // make this return the first pass's gradients a second time, and a caller
    // looping over batches would silently double them.
    let second = CpuB::backward::<f32>(&out).unwrap();
    assert_eq!(grad_of(&x, &first).unwrap(), vec![3.0, 4.0]);
    assert!(grad_of(&x, &second).is_none());
}

#[test]
fn an_output_nothing_reached_is_skipped_rather_than_failed() {
    // `unrelated` is recorded on the same tape but is not upstream of the loss
    // the walk starts from. Its node must be passed over, not treated as a
    // missing gradient.
    let a = cpu(&[1.0, 2.0], &[2]);
    let b = cpu(&[3.0, 4.0], &[2]);
    let unrelated = CpuB::mul::<f32>(&a, &b).unwrap();

    let c = cpu(&[5.0, 6.0], &[2]);
    let loss = CpuB::add::<f32>(&c, &c).unwrap();

    let grads = CpuB::backward::<f32>(&loss).unwrap();

    assert_eq!(grad_of(&c, &grads).unwrap(), vec![2.0, 2.0]);
    assert!(grad_of(&unrelated, &grads).is_none());
    assert!(grad_of(&a, &grads).is_none());
}

#[test]
fn a_recipe_that_records_while_it_runs_does_not_deadlock_the_tape() {
    // Convolution backward is built out of other backend operations, each of
    // which records. A walk that still held the tape it was draining would
    // re-enter it — and with the tape behind a RefCell, that is a panic on the
    // second borrow rather than anything to do with gradients. This is the
    // case that made `tape::backward` take its nodes by value.
    let input = cpu(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    );
    let weight = cpu(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);

    let out = CpuB::conv2d::<f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
    let grads = CpuB::backward::<f32>(&out).unwrap();

    // The weight sees each 2x2 window it was applied to, summed over the four
    // output positions: the top-left taps are 1+2+4+5 = 12, and each step
    // right or down adds one to every tap.
    assert_eq!(
        grad_of(&weight, &grads).unwrap(),
        vec![12.0, 16.0, 24.0, 28.0]
    );
    // Every input element is credited once per window that covered it, and the
    // kernel's off-diagonal taps are zero, so the corners differ from the
    // middle.
    assert_eq!(
        grad_of(&input, &grads).unwrap(),
        vec![1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn identities_are_unique_across_allocations() {
    // One counter serves the whole workspace since `GRD-003`. Two allocations
    // that shared an id would have their gradients merged into one entry.
    let a = cpu(&[1.0], &[1]);
    let b = cpu(&[1.0], &[1]);
    assert_ne!(TapeStorage::id(&a), TapeStorage::id(&b));
}

#[test]
fn the_nan_check_returns_rather_than_aborts() {
    // `GRD-005`: this was `Backend::backward_with_nan_check`, which panicked.
    // The check is an execution-policy axis now, and its failure is a value.
    let x = cpu(&[0.0, 1.0], &[2]);
    let zero = cpu(&[0.0, 0.0], &[2]);
    let out = CpuB::div::<f32>(&x, &zero).unwrap();

    let Err(err) = check_gradients(|| CpuB::backward::<f32>(&out)) else {
        panic!("a NaN gradient was not reported under NanPolicy::Reject");
    };
    assert!(matches!(
        err,
        Error::Backward(BackwardError::NonFinite { .. })
    ));
}

#[test]
fn a_chain_propagates_through_every_layer() {
    // Two dependent operations, which is the shortest chain that can tell a
    // reverse walk from a forward one. Walking forward reaches the first
    // node before anything has credited its output, so it is skipped as an
    // unreached branch and `x` comes back with no gradient at all — the same
    // silence a correct walk produces for a genuinely unrelated tensor.
    let x = cpu(&[1.0, 2.0], &[2]);
    let a = cpu(&[3.0, 3.0], &[2]);
    let b = cpu(&[5.0, 5.0], &[2]);

    let first = CpuB::mul::<f32>(&x, &a).unwrap();
    let second = CpuB::mul::<f32>(&first, &b).unwrap();
    let grads = CpuB::backward::<f32>(&second).unwrap();

    // d(x*a*b)/dx = a*b = 15, and the intermediate carries b alone.
    assert_eq!(grad_of(&x, &grads).unwrap(), vec![15.0, 15.0]);
    assert_eq!(grad_of(&first, &grads).unwrap(), vec![5.0, 5.0]);
    assert_eq!(grad_of(&a, &grads).unwrap(), vec![5.0, 10.0]);
}
