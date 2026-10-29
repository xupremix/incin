//! WGPU normalization backward values against the CPU twin (#91 Tier-1 remainder).
//!
//! The batch suites prove the norms run forward and leave tape entries
//! behind; a tape entry that replays the wrong math returns right numbers
//! forward and wrong gradients backward, which reads as working until a
//! model silently stops learning. Every case here runs the same
//! forward-then-`backward` on WGPU and on CPU and compares the cotangents,
//! not just their presence: `dx` always, plus `dw`/`db` where the identity
//! takes an affine operand.
//!
//! The backward is seeded symmetrically on both backends (`backward` over
//! the op output itself, the same seeding the CUDA twin tests use), so any
//! seeding convention cancels out of the comparison — what is compared is
//! the gradient recipe, which is what the `training = true` capability rows
//! claim.
//!
//! Requires a WGPU adapter for the WGPU leg:
//! `cargo test -p incin-backends --features wgpu,cpu --test wgpu_norms_backward`.
#![cfg(all(feature = "wgpu", feature = "cpu"))]

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, TapeStorage, op,
};
use incin_core::exec::catalog::{
    AxisAttributes, BatchNormAttributes, EpsilonAttributes, GroupNormAttributes,
    LayerNormAttributes,
};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type Wgpu = WgpuBackendImpl<WgpuN<U0>>;
type Cpu = CpuBackendImpl;
type WStorage = <Wgpu as StorageBackend>::Storage<f32>;
type CStorage = <Cpu as StorageBackend>::Storage<f32>;

/// Aborts unless a WGPU adapter can allocate — same contract as the other
/// WGPU suites: enabling the feature is an explicit request for the backend.
fn require_wgpu() {
    assert!(
        <Wgpu as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled"
    );
}

fn upload_wgpu(values: &[f32], shape: &[usize]) -> WStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <Wgpu as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading f32 values to WGPU must succeed")
}

fn upload_cpu(values: &[f32], shape: &[usize]) -> CStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <Cpu as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading f32 values to CPU must succeed")
}

fn read_wgpu(storage: &WStorage) -> Vec<f64> {
    <Wgpu as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading an f32 buffer back from WGPU must succeed")
}

fn read_cpu(storage: &CStorage) -> Vec<f64> {
    <Cpu as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading an f32 buffer back from CPU must succeed")
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: got {a}, expected {e} (|Δ|={}; tol {tol})",
            (a - e).abs()
        );
    }
}

/// Seed `backward` over the op output itself on both backends and return
/// `(wgpu_grads, cpu_grads)`. Seeding symmetrically keeps the comparison
/// about the recipe rather than about the seed.
fn backward_both(
    w_out: &WStorage,
    c_out: &CStorage,
) -> (
    <Wgpu as AutogradBackend>::Grads,
    <Cpu as AutogradBackend>::Grads,
) {
    let w_grads =
        <Wgpu as AutogradBackend>::backward::<f32>(w_out).expect("WGPU backward must run");
    let c_grads = <Cpu as AutogradBackend>::backward::<f32>(c_out).expect("CPU backward must run");
    (w_grads, c_grads)
}

/// Two rows of four, deliberately non-zero-mean so the centering step is
/// exercised rather than a no-op (same fixture as the batch-B forward test).
const NORM_IN: [f32; 8] = [1.0, 2.0, 3.0, 6.0, -4.0, 0.0, 2.0, 4.0];

#[test]
fn layer_norm_gradients_match_cpu_without_bias() {
    require_wgpu();
    let w_x = upload_wgpu(&NORM_IN, &[2, 4]);
    let w_w = upload_wgpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::LayerNorm, _>(
        &w_ctx,
        LayerNormAttributes {
            normalized_shape: vec![4],
            epsilon: 1e-5,
            has_bias: false,
        },
        &[
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_x),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_w),
        ],
    )
    .expect("layer_norm must execute on WGPU");

    let c_x = upload_cpu(&NORM_IN, &[2, 4]);
    let c_w = upload_cpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::LayerNorm, _>(
        &c_ctx,
        LayerNormAttributes {
            normalized_shape: vec![4],
            epsilon: 1e-5,
            has_bias: false,
        },
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&c_x),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_w),
        ],
    )
    .expect("layer_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-5,
        "layer_norm forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU layer_norm input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU layer_norm input must receive a gradient");
    assert_close(&read_wgpu(w_dx), &read_cpu(c_dx), 1e-4, "layer_norm dx");
    let w_dw = w_grads
        .get(TapeStorage::id(&w_w))
        .expect("WGPU layer_norm weight must receive a gradient");
    let c_dw = c_grads
        .get(TapeStorage::id(&c_w))
        .expect("CPU layer_norm weight must receive a gradient");
    assert_close(&read_wgpu(w_dw), &read_cpu(c_dw), 1e-4, "layer_norm dw");
}

#[test]
fn layer_norm_gradients_match_cpu_with_bias() {
    require_wgpu();
    let w_x = upload_wgpu(&NORM_IN, &[2, 4]);
    let w_w = upload_wgpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let w_b = upload_wgpu(&[0.25, -0.5, 0.0, 1.0], &[4]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::LayerNorm, _>(
        &w_ctx,
        LayerNormAttributes {
            normalized_shape: vec![4],
            epsilon: 1e-5,
            has_bias: true,
        },
        &[
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_x),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_w),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_b),
        ],
    )
    .expect("biased layer_norm must execute on WGPU");

    let c_x = upload_cpu(&NORM_IN, &[2, 4]);
    let c_w = upload_cpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let c_b = upload_cpu(&[0.25, -0.5, 0.0, 1.0], &[4]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::LayerNorm, _>(
        &c_ctx,
        LayerNormAttributes {
            normalized_shape: vec![4],
            epsilon: 1e-5,
            has_bias: true,
        },
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&c_x),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_w),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_b),
        ],
    )
    .expect("biased layer_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-5,
        "biased layer_norm forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    for (what, w_id, c_id) in [
        ("dx", TapeStorage::id(&w_x), TapeStorage::id(&c_x)),
        ("dw", TapeStorage::id(&w_w), TapeStorage::id(&c_w)),
        ("db", TapeStorage::id(&w_b), TapeStorage::id(&c_b)),
    ] {
        let w_g = w_grads
            .get(w_id)
            .unwrap_or_else(|| panic!("WGPU biased layer_norm {what} is missing"));
        let c_g = c_grads
            .get(c_id)
            .unwrap_or_else(|| panic!("CPU biased layer_norm {what} is missing"));
        assert_close(
            &read_wgpu(w_g),
            &read_cpu(c_g),
            1e-4,
            &format!("biased layer_norm {what}"),
        );
    }
}

#[test]
fn rms_norm_gradients_match_cpu() {
    require_wgpu();
    let attributes = EpsilonAttributes { epsilon: 1e-5 };
    let w_x = upload_wgpu(&NORM_IN, &[2, 4]);
    let w_w = upload_wgpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::RmsNorm, _>(
        &w_ctx,
        attributes,
        &[
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_x),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_w),
        ],
    )
    .expect("rms_norm must execute on WGPU");

    let c_x = upload_cpu(&NORM_IN, &[2, 4]);
    let c_w = upload_cpu(&[1.0, 0.5, 2.0, 1.5], &[4]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::RmsNorm, _>(
        &c_ctx,
        EpsilonAttributes { epsilon: 1e-5 },
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&c_x),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_w),
        ],
    )
    .expect("rms_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-5,
        "rms_norm forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU rms_norm input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU rms_norm input must receive a gradient");
    assert_close(&read_wgpu(w_dx), &read_cpu(c_dx), 1e-4, "rms_norm dx");
    let w_dw = w_grads
        .get(TapeStorage::id(&w_w))
        .expect("WGPU rms_norm weight must receive a gradient");
    let c_dw = c_grads
        .get(TapeStorage::id(&c_w))
        .expect("CPU rms_norm weight must receive a gradient");
    assert_close(&read_wgpu(w_dw), &read_cpu(c_dw), 1e-4, "rms_norm dw");
}

#[test]
fn group_norm_gradients_match_cpu() {
    require_wgpu();
    // One sample, C=4 channels, spatial 2x2, two groups of two channels —
    // the batch-B forward fixture, so the forward leg is already pinned
    // there and only the cotangent is new here.
    let values: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let attributes = GroupNormAttributes {
        groups: 2,
        epsilon: 1e-5,
    };
    let w_x = upload_wgpu(&values, &[1, 4, 2, 2]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::GroupNorm, _>(
        &w_ctx,
        attributes,
        &[TensorHandle::from_storage::<Wgpu, f32, _>(&w_x)],
    )
    .expect("group_norm must execute on WGPU");

    let c_x = upload_cpu(&values, &[1, 4, 2, 2]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::GroupNorm, _>(
        &c_ctx,
        GroupNormAttributes {
            groups: 2,
            epsilon: 1e-5,
        },
        &[TensorHandle::from_storage::<Cpu, f32, _>(&c_x)],
    )
    .expect("group_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-4,
        "group_norm forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU group_norm input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU group_norm input must receive a gradient");
    assert_close(&read_wgpu(w_dx), &read_cpu(c_dx), 1e-4, "group_norm dx");
}

#[test]
fn instance_norm_gradients_match_cpu() {
    require_wgpu();
    let values: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let w_x = upload_wgpu(&values, &[1, 4, 2, 2]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::InstanceNorm, _>(
        &w_ctx,
        EpsilonAttributes { epsilon: 1e-5 },
        &[TensorHandle::from_storage::<Wgpu, f32, _>(&w_x)],
    )
    .expect("instance_norm must execute on WGPU");

    let c_x = upload_cpu(&values, &[1, 4, 2, 2]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::InstanceNorm, _>(
        &c_ctx,
        EpsilonAttributes { epsilon: 1e-5 },
        &[TensorHandle::from_storage::<Cpu, f32, _>(&c_x)],
    )
    .expect("instance_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-4,
        "instance_norm forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU instance_norm input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU instance_norm input must receive a gradient");
    assert_close(&read_wgpu(w_dx), &read_cpu(c_dx), 1e-4, "instance_norm dx");
}

#[test]
fn batch_norm_training_gradients_match_cpu() {
    require_wgpu();
    // Same invocation shape as the batch-B training test (running
    // statistics present, training mode, no affine): the forward leg is
    // pinned there and the input cotangent is new here.
    let w_x = upload_wgpu(&NORM_IN, &[2, 4]);
    let w_m = upload_wgpu(&[0.0f32; 4], &[4]);
    let w_v = upload_wgpu(&[1.0f32; 4], &[4]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::BatchNorm, _>(
        &w_ctx,
        BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
            has_weight: false,
            has_bias: false,
            has_running_mean: true,
            has_running_variance: true,
        },
        &[
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_x),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_m),
            TensorHandle::from_storage::<Wgpu, f32, _>(&w_v),
        ],
    )
    .expect("training batch_norm must execute on WGPU");

    let c_x = upload_cpu(&NORM_IN, &[2, 4]);
    let c_m = upload_cpu(&[0.0f32; 4], &[4]);
    let c_v = upload_cpu(&[1.0f32; 4], &[4]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::BatchNorm, _>(
        &c_ctx,
        BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
            has_weight: false,
            has_bias: false,
            has_running_mean: true,
            has_running_variance: true,
        },
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&c_x),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_m),
            TensorHandle::from_storage::<Cpu, f32, _>(&c_v),
        ],
    )
    .expect("training batch_norm must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-4,
        "batch_norm training forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU batch_norm input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU batch_norm input must receive a gradient");
    assert_close(
        &read_wgpu(w_dx),
        &read_cpu(c_dx),
        1e-4,
        "batch_norm training dx",
    );
}

#[test]
fn softmax_gradients_match_cpu() {
    require_wgpu();
    let values: [f32; 8] = [1.0, 2.0, 3.0, 6.0, -4.0, 0.0, 2.0, 4.0];
    let w_x = upload_wgpu(&values, &[2, 4]);
    let w_ctx = ExecutionContext::new(Wgpu::default());
    let w_out = incin_core::exec::dispatch::execute::<op::Softmax, _>(
        &w_ctx,
        AxisAttributes { axis: 1 },
        &[TensorHandle::from_storage::<Wgpu, f32, _>(&w_x)],
    )
    .expect("softmax must execute on WGPU");

    let c_x = upload_cpu(&values, &[2, 4]);
    let c_ctx = ExecutionContext::new(Cpu::new());
    let c_out = incin_core::exec::dispatch::execute::<op::Softmax, _>(
        &c_ctx,
        AxisAttributes { axis: 1 },
        &[TensorHandle::from_storage::<Cpu, f32, _>(&c_x)],
    )
    .expect("softmax must execute on CPU");

    assert_close(
        &read_wgpu(&w_out),
        &read_cpu(&c_out),
        1e-6,
        "softmax forward",
    );
    let (w_grads, c_grads) = backward_both(&w_out, &c_out);
    let w_dx = w_grads
        .get(TapeStorage::id(&w_x))
        .expect("WGPU softmax input must receive a gradient");
    let c_dx = c_grads
        .get(TapeStorage::id(&c_x))
        .expect("CPU softmax input must receive a gradient");
    assert_close(&read_wgpu(w_dx), &read_cpu(c_dx), 1e-4, "softmax dx");
}

/// Acceptance criterion "no row advertising `f64`", pinned rather than
/// assumed: WGSL has no `f64`, so a single `f64` entry in any WGPU rule
/// would be a claim no kernel here honours. (The `f16`/`bf16` half lives
/// in `wgpu_f16_audit.rs`; this is the `f64` half.)
#[test]
fn no_wgpu_rule_advertises_f64() {
    for rule in incin_backends::capability::WGPU_CAPABILITIES {
        assert!(
            !rule.dtypes.contains(&DTypeId::F64.descriptor()),
            "WGPU rule for {:?} advertises f64, but WGSL has no f64 and no WGPU \
             shader reads double precision (#91)",
            rule.operation,
        );
    }
}
