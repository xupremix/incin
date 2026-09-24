//! Issue #84 training behaviours on CUDA: dropout's eval identity and
//! counter-based mask reproducibility, `instance_norm` backward parity with
//! the CPU composition, and one complete Linear -> cross-entropy -> SGD step.
//!
//! Every hardware test is `#[ignore]`d; `require_cuda` fails loudly rather
//! than skipping, because reaching an `#[ignore]`d test is an explicit
//! request for the hardware run. The dropout mask seams exist because
//! `set_dropout_seed` is gated to unit-test builds and the process-wide draw
//! offset cannot be reset from an integration crate: `dropout_mask_seeded`
//! pins the (seed, offset) contract directly, and `dropout_mask` pins the
//! production reservation behaviour on top of it.
#![cfg(feature = "cuda")]

use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{
        download_f32, dropout_mask, dropout_mask_seeded, require_cuda, sgd_step, upload_f32_shaped,
        upload_i64,
    },
};
use incin_core::backend_authoring::{
    AutogradBackend, Execute, HostInterop, HostReadback, StorageBackend,
};
use incin_core::exec::catalog::{
    DropoutAttributes, EpsilonAttributes, LinearAttributes, LossAttributes, LossReduction,
    SgdAttributes,
};
use incin_core::exec::{
    CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle, dispatch, op,
};
use incin_core::prelude::{CudaN, DTypeId, DeviceId};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;
#[cfg(feature = "cpu")]
type Cpu = incin_backends::cpu::CpuBackendImpl<incin_core::tensor::device::Cpu>;

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    download_f32(storage)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn scalar(storage: &TestStorage) -> f64 {
    let values = download_f32(storage);
    assert_eq!(values.len(), 1, "expected a scalar loss");
    f64::from(values[0])
}

/// One f32 input through `dispatch`, returning the output and the tape
/// entries the forward pushed.
fn run1<O, A>(
    context: &ExecutionContext<TestBackend>,
    input: &TestStorage,
    attributes: A,
) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[TensorHandle::from_storage::<TestBackend, f32, _>(input)],
    )
    .expect("an advertised one-input CUDA operation must execute");
    (out, tape_depth() - before)
}

/// Three f32 inputs (`linear`'s input, weight and bias) through `dispatch`.
fn run3<O, A>(
    context: &ExecutionContext<TestBackend>,
    first: &TestStorage,
    second: &TestStorage,
    third: &TestStorage,
    attributes: A,
) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(first),
            TensorHandle::from_storage::<TestBackend, f32, _>(second),
            TensorHandle::from_storage::<TestBackend, f32, _>(third),
        ],
    )
    .expect("an advertised three-input CUDA operation must execute");
    (out, tape_depth() - before)
}

/// `cross_entropy_loss`: f32 logits, i64 class indices.
fn run_ce(
    context: &ExecutionContext<TestBackend>,
    logits: &TestStorage,
    targets: &TestStorage,
) -> (TestStorage, usize) {
    let before = tape_depth();
    let out = dispatch::execute::<op::CrossEntropyLoss, TestBackend>(
        context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(logits),
            TensorHandle::from_storage::<TestBackend, i64, _>(targets),
        ],
    )
    .expect("cross_entropy_loss must execute on CUDA");
    (out, tape_depth() - before)
}

#[test]
#[ignore = "requires CUDA hardware"]
fn dropout_eval_mode_is_the_identity_and_records_nothing() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let input = upload_f32_shaped(&[2, 3], &[1.0, -2.0, 3.5, 0.0, 4.0, -0.5]);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) = run1::<op::Dropout, _>(
        &context,
        &input,
        DropoutAttributes {
            probability: 0.5,
            training: false,
        },
    );
    assert_eq!(
        recorded, 0,
        "eval-mode dropout is the clone-links-identity path: no tape entry"
    );
    assert_eq!(
        TapeStorage::id(&out),
        input_id,
        "eval-mode dropout returns the same allocation"
    );
    assert_eq!(
        read_f32(&out),
        read_f32(&input),
        "eval-mode dropout is the identity"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn dropout_mask_follows_the_explicit_seed_and_offset() {
    require_cuda();
    let first = download_f32(&dropout_mask_seeded(&[16], 7, 0));
    let replay = download_f32(&dropout_mask_seeded(&[16], 7, 0));
    assert_eq!(first, replay, "the same (seed, offset) must replay");
    assert_eq!(first.len(), 16);
    for (i, &v) in first.iter().enumerate() {
        assert!(
            (0.0..1.0).contains(&v),
            "mask[{i}] = {v} must lie in [0, 1)"
        );
    }
    let advanced = download_f32(&dropout_mask_seeded(&[16], 7, 16));
    assert_ne!(
        first, advanced,
        "an advanced offset must draw a disjoint stream"
    );
    let reseeded = download_f32(&dropout_mask_seeded(&[16], 8, 0));
    assert_ne!(
        first, reseeded,
        "a different seed must draw a different stream"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn consecutive_production_dropout_draws_are_disjoint() {
    require_cuda();
    let first = download_f32(&dropout_mask(&[16]));
    let second = download_f32(&dropout_mask(&[16]));
    assert_eq!(first.len(), 16);
    assert_ne!(
        first, second,
        "consecutive draws must reserve disjoint counter ranges even on the \
         fixed default seed"
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn instance_norm_backward_matches_the_cpu_twin_on_cuda() {
    // The descriptor validates [N, C, H, W], so this walks the full
    // statistical path (mean -> sub -> rsqrt-of-variance -> scale) on rank
    // four and pins CUDA's replay against the CPU composition under the
    // same ones seed both backends' `backward` plants.
    require_cuda();
    const VALUES: [f32; 8] = [0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0];
    const SHAPE: [usize; 4] = [1, 2, 2, 2];

    let context = ExecutionContext::new(TestBackend::new());
    let input = upload_f32_shaped(&SHAPE, &VALUES);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) =
        run1::<op::InstanceNorm, _>(&context, &input, EpsilonAttributes { epsilon: 1e-5 });
    assert!(
        recorded >= 1,
        "instance_norm advertises training = true, recorded {recorded}"
    );
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
        .expect("instance_norm backward on CUDA");
    let gx = grads
        .get(input_id)
        .expect("instance_norm input receives a gradient");

    let cpu_input = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(&VALUES),
        &SHAPE,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading the CPU twin must succeed");
    let cpu_context = ExecutionContext::new(Cpu::new());
    let cpu_out = dispatch::execute::<op::InstanceNorm, _>(
        &cpu_context,
        EpsilonAttributes { epsilon: 1e-5 },
        &[TensorHandle::from_storage::<Cpu, f32, _>(&cpu_input)],
    )
    .expect("CPU reference instance_norm executes");
    let cpu_grads =
        <Cpu as AutogradBackend>::backward::<f32>(&cpu_out).expect("instance_norm backward on CPU");
    let cpu_gx = cpu_grads
        .get(TapeStorage::id(&cpu_input))
        .expect("CPU reference is missing a gradient");

    let want_fwd = <Cpu as HostReadback>::float_to_vec1::<f32>(&cpu_out)
        .expect("reading the CPU forward back");
    assert_close(&read_f32(&out), &want_fwd, 1e-5, "instance_norm forward");
    let want_dx =
        <Cpu as HostReadback>::float_to_vec1::<f32>(cpu_gx).expect("reading the CPU gradient back");
    assert_close(&read_f32(gx), &want_dx, 1e-5, "instance_norm dx");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn linear_cross_entropy_sgd_step_lowers_the_loss_on_cuda() {
    // The issue's acceptance criterion: a full training step on CUDA.
    // Linear -> cross-entropy -> backward produces finite, correctly shaped
    // gradients for every operand, the fused SGD kernel moves the weights,
    // and re-forwarding the stepped weights lowers the loss.
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let x = upload_f32_shaped(
        &[4, 3],
        &[
            0.2, -0.4, 0.6, 1.0, 0.5, -0.8, -0.3, 0.9, 0.1, 0.7, -0.2, 0.4,
        ],
    );
    let w = upload_f32_shaped(&[2, 3], &[0.1, 0.2, 0.3, -0.2, 0.4, -0.1]);
    let b = upload_f32_shaped(&[2], &[0.05, -0.05]);
    let targets = upload_i64(&[4], &[0, 1, 1, 0]);
    let (x_id, w_id, b_id) = (
        TapeStorage::id(&x),
        TapeStorage::id(&w),
        TapeStorage::id(&b),
    );

    let (logits, _) =
        run3::<op::Linear, _>(&context, &x, &w, &b, LinearAttributes { has_bias: true });
    let (loss, recorded) = run_ce(&context, &logits, &targets);
    assert!(
        recorded >= 1,
        "the linear + cross-entropy forward must record, recorded {recorded}"
    );
    let before = scalar(&loss);
    assert!(before.is_finite(), "loss must be finite, got {before}");

    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs on CUDA");
    let check = |grad: &TestStorage, want_len: usize, what: &str| {
        let values = download_f32(grad);
        assert_eq!(values.len(), want_len, "{what}: gradient shape");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{what}: gradient must be finite"
        );
    };
    let gx = grads.get(x_id).expect("dL/dx exists");
    let gw = grads.get(w_id).expect("dL/dw exists");
    let gb = grads.get(b_id).expect("dL/db exists");
    check(gx, 12, "dL/dx");
    check(gw, 6, "dL/dw");
    check(gb, 2, "dL/db");

    let w_next = sgd_step(&w, gw, &SgdAttributes { learning_rate: 0.1 })
        .expect("the fused SGD step must run on CUDA");
    assert_ne!(
        download_f32(&w_next),
        download_f32(&w),
        "one SGD step must move the weights"
    );

    let (logits_after, _) = run3::<op::Linear, _>(
        &context,
        &x,
        &w_next,
        &b,
        LinearAttributes { has_bias: true },
    );
    let (loss_after, _) = run_ce(&context, &logits_after, &targets);
    let after = scalar(&loss_after);
    assert!(
        after < before,
        "one SGD step must lower the loss: {before} -> {after}"
    );
}
