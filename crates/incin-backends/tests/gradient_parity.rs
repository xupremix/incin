#![cfg(feature = "cpu")]

use incin_backends::cpu::CpuBackendImpl;
#[cfg(feature = "cuda")]
use incin_backends::cuda::CudaBackendImpl;
#[cfg(feature = "wgpu")]
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::prelude::*;

type CpuB = CpuBackendImpl;
#[cfg(feature = "wgpu")]
type WgpuB = WgpuBackendImpl<f32, incin_core::prelude::WgpuN<incin_core::typenum::U0>>;
#[cfg(feature = "cuda")]
type CudaB = CudaBackendImpl<f32, incin_core::prelude::CudaN<incin_core::typenum::U0>>;

fn read_f32<B: Backend>(s: &B::Storage<f32>) -> Vec<f32> {
    let bytes = B::to_bytes::<f32>(s).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

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

#[test]
fn parity_elementwise_add() {
    let shape = vec![2, 3];
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![10.0f32, 20.0, 30.0, 40.0, 5.0, 60.0];

    // CPU
    let cpu_a = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::add::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32,
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

#[test]
fn parity_activations_softmax() {
    let shape = vec![1, 4];
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];

    // CPU
    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::softmax::<f32>(&cpu_in, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::matmul::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::layer_norm::<f32>(&cpu_x, &cpu_w, Some(&cpu_b), 1e-5).unwrap();

    // WGPU
    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::max_pool2d::<f32>(&cpu_in, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_tgt = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::cross_entropy_loss::<f32, f32>(&cpu_pred, &cpu_tgt, Reduction::Mean).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_pred = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_tgt = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_tgt = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt_f32),
        &shape_tgt,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let error = WgpuB::from_bytes::<u32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        DTypeId::U32,
        &DeviceId::wgpu(0),
    )
    .err()
    .expect("WGPU must reject U32 storage");
    assert!(matches!(
        error,
        Error::UnsupportedDType {
            dtype: DTypeId::U32,
            backend: "Wgpu",
            op: "from_bytes",
        }
    ));
}

#[test]
fn parity_activations_relu() {
    let shape = vec![2, 3];
    let data = vec![-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0];

    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::relu::<f32>(&cpu_in).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
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

#[test]
fn parity_activations_gelu() {
    let shape = vec![2, 3];
    let data = vec![-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0];

    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::gelu::<f32>(&cpu_in).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::batch_norm::<f32>(&cpu_x, Some(&cpu_w), Some(&cpu_b), None, None, 1e-5, 0.1).unwrap();

    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32,
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

#[test]
fn parity_reductions_sum_and_mean() {
    let shape = vec![2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Sum dim 1
    let cpu_in = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_sum = CpuB::sum_dim::<f32>(&cpu_in, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_sum).unwrap();

    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::conv2d::<f32>(&cpu_x, &cpu_w, None, 1, 0, 1, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::add::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_a = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::matmul::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_a = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::conv2d::<f32>(&cpu_x, &cpu_w, None, 1, 0, 1, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    let cuda_x = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_w = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        DTypeId::F32,
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
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32,
        &DeviceId::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::batch_norm::<f32>(&cpu_x, Some(&cpu_w), Some(&cpu_b), None, None, 1e-5, 0.1).unwrap();

    let cuda_x = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_w = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_c,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap();
    let cuda_b = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_c,
        DTypeId::F32,
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
