#![cfg(feature = "wgpu")]

use kindle_backends::cpu::CpuBackend;
use kindle_backends::wgpu::WgpuBackend;
use kindle_core::prelude::*;

type CpuB = CpuBackend;
type WgpuB = WgpuBackend<f32, Cpu>;

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
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::add::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
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
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::softmax::<f32>(&cpu_in, 1).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_in = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data),
        &shape,
        KindleDType::F32,
        &KindleDevice::cpu(),
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
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::matmul::<f32>(&cpu_a, &cpu_b).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_a = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_a),
        &shape_a,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_b,
        KindleDType::F32,
        &KindleDevice::cpu(),
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
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_w = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_b = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_out = CpuB::layer_norm::<f32>(&cpu_x, &cpu_w, Some(&cpu_b), 1e-5).unwrap();

    // WGPU
    let wgpu_x = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_x),
        &shape_x,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_w = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_w),
        &shape_w,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_b = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_b),
        &shape_w,
        KindleDType::F32,
        &KindleDevice::cpu(),
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
fn parity_cross_entropy_loss() {
    let shape_pred = vec![2, 3];
    let shape_tgt = vec![2];
    let data_pred = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_tgt = vec![0u32, 2u32];

    // CPU (target is f32 / i64)
    let data_tgt_f32: Vec<f32> = data_tgt.iter().map(|&x| x as f32).collect();
    let cpu_pred = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_tgt = CpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_tgt_f32),
        &shape_tgt,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let cpu_out =
        CpuB::cross_entropy_loss::<f32, f32>(&cpu_pred, &cpu_tgt, Reduction::Mean).unwrap();
    let cpu_grads = CpuB::backward::<f32>(&cpu_out).unwrap();

    // WGPU
    let wgpu_pred = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&data_pred),
        &shape_pred,
        KindleDType::F32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_tgt = WgpuB::from_bytes::<u32>(
        bytemuck::cast_slice(&data_tgt),
        &shape_tgt,
        KindleDType::U32,
        &KindleDevice::cpu(),
    )
    .unwrap();
    let wgpu_out =
        WgpuB::cross_entropy_loss::<f32, u32>(&wgpu_pred, &wgpu_tgt, Reduction::Mean).unwrap();
    let wgpu_grads = WgpuB::backward::<f32>(&wgpu_out).unwrap();

    let cpu_res = read_f32::<CpuB>(&cpu_out);
    let wgpu_res = read_f32::<WgpuB>(&wgpu_out);
    assert!(
        approx_eq_slice(&cpu_res, &wgpu_res, 1e-3),
        "CrossEntropy Loss Forward mismatch: CPU {:?} vs WGPU {:?}",
        cpu_res,
        wgpu_res
    );

    let cpu_gp = read_f32::<CpuB>(
        &CpuB::get_grad::<f32>(&cpu_pred, &cpu_grads)
            .unwrap()
            .unwrap(),
    );
    let wgpu_gp = read_f32::<WgpuB>(
        &WgpuB::get_grad::<f32>(&wgpu_pred, &wgpu_grads)
            .unwrap()
            .unwrap(),
    );
    assert!(
        approx_eq_slice(&cpu_gp, &wgpu_gp, 1e-3),
        "CrossEntropy Loss Grad mismatch: CPU {:?} vs WGPU {:?}",
        cpu_gp,
        wgpu_gp
    );
}
