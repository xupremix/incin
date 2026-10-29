//! Issue #84 CUDA losses: admission, every reduction mode, tape claims and
//! ones-seeded gradients against host references.
//!
//! The issue body predates the parity commit: `mse_loss`, `l1_loss`,
//! `bce_with_logits_loss` and `cross_entropy_loss` already execute on CUDA
//! and are advertised for training. This file is the runtime half:
//!
//! - an admission test that needs no device - `CudaBackendImpl::new()` is
//!   `const` and `support` is a pure table lookup, so the seven rows issue
//!   #84 tracks are checked on any build with the `cuda` feature;
//! - hardware tests (`#[ignore]`d) that walk each loss under `Mean`, `Sum`
//!   and `None` against an f64 host reference, assert the forward records at
//!   least one tape entry, and check the ones-seeded backward gradient -
//!   including `cross_entropy_loss`'s refusal to differentiate its integer
//!   target and `l1_loss`'s `sign(0) = 0` slope.
#![cfg(feature = "cuda")]

use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{download_f32, download_i64, require_cuda, upload_f32_shaped, upload_i64},
};
use incin_core::backend_authoring::{AutogradBackend, Execute, StorageBackend};
use incin_core::exec::catalog::{LossAttributes, LossReduction};
use incin_core::exec::meta::LayoutClass;
use incin_core::exec::policy::MathMode;
use incin_core::exec::{
    CanonicalOperation, Capabilities, CapabilityQuery, ExecutionContext, OperationIdentity,
    SupportLevel, TapeStorage, TensorHandle, dispatch, op,
};
use incin_core::prelude::{CudaN, DTypeId, OperationKind};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Predictions for the elementwise losses: a mix of signs, fractions and one
/// exact hit so `l1_loss`'s zero slope and `mse_loss`'s zero error surface.
const P: [f32; 6] = [1.0, 2.0, 3.0, -0.5, 0.25, 4.0];
/// Targets paired with [`P`]; `T[5] == P[5]` pins the zero-difference case.
const T: [f32; 6] = [0.5, 2.5, 1.0, 0.0, -1.0, 4.0];
const SHAPE: [usize; 2] = [2, 3];
/// Class indices for the cross-entropy walk over [`P`] as `[2, 3]` logits.
const CE_TARGETS: [i64; 2] = [1, 2];

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

/// Reduce per-element points the way the loss executor does.
fn reduced(points: &[f64], reduction: LossReduction) -> Vec<f64> {
    match reduction {
        LossReduction::None => points.to_vec(),
        LossReduction::Mean => vec![points.iter().sum::<f64>() / points.len() as f64],
        LossReduction::Sum => vec![points.iter().sum::<f64>()],
    }
}

/// `Mean` over the six elementwise entries; every other reduction is 1:1.
fn scale(reduction: LossReduction) -> f64 {
    match reduction {
        LossReduction::Mean => 1.0 / 6.0,
        LossReduction::Sum | LossReduction::None => 1.0,
    }
}

/// `Mean` over cross-entropy divides by the batch, not the logit count.
fn ce_scale(reduction: LossReduction) -> f64 {
    match reduction {
        LossReduction::Mean => 1.0 / 2.0,
        LossReduction::Sum | LossReduction::None => 1.0,
    }
}

/// Run a two-f32-input loss under `reduction`, returning the output and the
/// number of tape entries the forward pushed.
fn run_loss<O>(
    context: &ExecutionContext<TestBackend>,
    reduction: LossReduction,
    pred: &TestStorage,
    target: &TestStorage,
) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = LossAttributes>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        LossAttributes { reduction },
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(pred),
            TensorHandle::from_storage::<TestBackend, f32, _>(target),
        ],
    )
    .expect("an advertised CUDA loss must execute");
    (out, tape_depth() - before)
}

/// The cross-entropy twin: f32 logits against an i64 class-index target.
fn run_ce(
    context: &ExecutionContext<TestBackend>,
    reduction: LossReduction,
    logits: &TestStorage,
    targets: &TestStorage,
) -> (TestStorage, usize) {
    let before = tape_depth();
    let out = dispatch::execute::<op::CrossEntropyLoss, TestBackend>(
        context,
        LossAttributes { reduction },
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(logits),
            TensorHandle::from_storage::<TestBackend, i64, _>(targets),
        ],
    )
    .expect("cross_entropy_loss must execute on CUDA");
    (out, tape_depth() - before)
}

/// The four losses plus dropout and both norms, admitted for training on
/// CUDA without opening a device: `support` reads the capability table and
/// nothing else. The expected level pins the implementation kind the public
/// capability table documents - native where a kernel exists, composed where
/// the operation rewrites into tape-tracked primitives.
#[test]
fn cuda_admits_the_seven_issue_84_rows_for_training() {
    let backend = TestBackend::new();
    let rows: &[(OperationKind, usize, LayoutClass, SupportLevel)] = &[
        (
            OperationKind::MseLoss,
            2,
            LayoutClass::Contiguous,
            SupportLevel::Composed,
        ),
        (
            OperationKind::L1Loss,
            2,
            LayoutClass::Contiguous,
            SupportLevel::Composed,
        ),
        (
            OperationKind::BceWithLogitsLoss,
            2,
            LayoutClass::Contiguous,
            SupportLevel::Composed,
        ),
        (
            OperationKind::CrossEntropyLoss,
            2,
            LayoutClass::Contiguous,
            SupportLevel::Composed,
        ),
        (
            OperationKind::Dropout,
            2,
            LayoutClass::Contiguous,
            SupportLevel::Native,
        ),
        (
            OperationKind::Dropout,
            2,
            LayoutClass::Strided,
            SupportLevel::Native,
        ),
        (
            OperationKind::GroupNorm,
            4,
            LayoutClass::Contiguous,
            SupportLevel::Native,
        ),
        (
            OperationKind::InstanceNorm,
            4,
            LayoutClass::Contiguous,
            SupportLevel::Composed,
        ),
    ];
    for (operation, rank, layout, expected) in rows {
        let level = backend.support(&CapabilityQuery {
            operation: OperationIdentity::Builtin(*operation),
            dtype: DTypeId::F32.descriptor(),
            layout: *layout,
            rank: *rank,
            training: true,
            math_mode: MathMode::default(),
        });
        assert_eq!(
            &level, expected,
            "{operation:?} (rank {rank}, {layout:?}, training) must be admitted on CUDA"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn mse_loss_matches_the_host_reference_under_every_reduction() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let pred = upload_f32_shaped(&SHAPE, &P);
    let targ = upload_f32_shaped(&SHAPE, &T);
    let pred_id = TapeStorage::id(&pred);
    for reduction in [LossReduction::Mean, LossReduction::Sum, LossReduction::None] {
        let (out, recorded) = run_loss::<op::MseLoss>(&context, reduction, &pred, &targ);
        assert!(
            recorded >= 1,
            "mse_loss under {reduction:?} must record a tape entry, recorded {recorded}"
        );
        let points: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| {
                let d = f64::from(p) - f64::from(t);
                d * d
            })
            .collect();
        assert_close(
            &read_f32(&out),
            &reduced(&points, reduction),
            1e-5,
            &format!("mse_loss {reduction:?} forward"),
        );
        let grads =
            <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("mse backward on CUDA");
        let gx = grads.get(pred_id).expect("mse pred receives a gradient");
        let want: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| 2.0 * (f64::from(p) - f64::from(t)) * scale(reduction))
            .collect();
        assert_close(
            &read_f32(gx),
            &want,
            1e-5,
            &format!("mse_loss {reduction:?} dx"),
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn l1_loss_matches_the_host_reference_under_every_reduction() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let pred = upload_f32_shaped(&SHAPE, &P);
    let targ = upload_f32_shaped(&SHAPE, &T);
    let pred_id = TapeStorage::id(&pred);
    // The `sign(0) = 0` convention at index 5 (P[5] == T[5]) is CPU's
    // slope too; a kernel answering `+1` there would diverge.
    let sign = |d: f64| {
        if d > 0.0 {
            1.0
        } else if d < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    for reduction in [LossReduction::Mean, LossReduction::Sum, LossReduction::None] {
        let (out, recorded) = run_loss::<op::L1Loss>(&context, reduction, &pred, &targ);
        assert!(
            recorded >= 1,
            "l1_loss under {reduction:?} must record a tape entry, recorded {recorded}"
        );
        let points: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| (f64::from(p) - f64::from(t)).abs())
            .collect();
        assert_close(
            &read_f32(&out),
            &reduced(&points, reduction),
            1e-5,
            &format!("l1_loss {reduction:?} forward"),
        );
        let grads =
            <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("l1 backward on CUDA");
        let gx = grads.get(pred_id).expect("l1 pred receives a gradient");
        let want: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| sign(f64::from(p) - f64::from(t)) * scale(reduction))
            .collect();
        assert_close(
            &read_f32(gx),
            &want,
            1e-5,
            &format!("l1_loss {reduction:?} dx"),
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn bce_with_logits_loss_matches_the_host_reference_under_every_reduction() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let pred = upload_f32_shaped(&SHAPE, &P);
    let targ = upload_f32_shaped(&SHAPE, &T);
    let pred_id = TapeStorage::id(&pred);
    // Stable form, f64, the same formula CPU's kernel evaluates:
    // max(x, 0) - x*z + ln(1 + exp(-|x|)); gradient sigmoid(x) - z.
    let point = |x: f64, z: f64| x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln();
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    for reduction in [LossReduction::Mean, LossReduction::Sum, LossReduction::None] {
        let (out, recorded) = run_loss::<op::BceWithLogitsLoss>(&context, reduction, &pred, &targ);
        assert!(
            recorded >= 1,
            "bce under {reduction:?} must record a tape entry, recorded {recorded}"
        );
        let points: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| point(f64::from(p), f64::from(t)))
            .collect();
        assert_close(
            &read_f32(&out),
            &reduced(&points, reduction),
            1e-5,
            &format!("bce_with_logits_loss {reduction:?} forward"),
        );
        let grads =
            <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("bce backward on CUDA");
        let gx = grads.get(pred_id).expect("bce pred receives a gradient");
        let want: Vec<f64> = P
            .iter()
            .zip(&T)
            .map(|(&p, &t)| (sigmoid(f64::from(p)) - f64::from(t)) * scale(reduction))
            .collect();
        assert_close(
            &read_f32(gx),
            &want,
            1e-5,
            &format!("bce_with_logits_loss {reduction:?} dx"),
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cross_entropy_loss_matches_the_host_reference_under_every_reduction() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let logits = upload_f32_shaped(&SHAPE, &P);
    let targets = upload_i64(&[2], &CE_TARGETS);
    let logits_id = TapeStorage::id(&logits);
    let targets_id = TapeStorage::id(&targets);

    // Max-shifted log-softmax in f64, then the per-row NLL the CPU kernel
    // gathers; `Mean` averages the two rows, `None` keeps them separate.
    let rows: Vec<Vec<f64>> = P
        .chunks(3)
        .map(|row| row.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let mut softmax = [0.0f64; 6];
    let mut nll = [0.0f64; 2];
    for (r, row) in rows.iter().enumerate() {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        for (c, e) in exp.iter().enumerate() {
            softmax[r * 3 + c] = e / sum;
        }
        let t = usize::try_from(CE_TARGETS[r]).expect("class index fits usize");
        nll[r] = -(row[t] - max - sum.ln());
    }

    for reduction in [LossReduction::Mean, LossReduction::Sum, LossReduction::None] {
        let (out, recorded) = run_ce(&context, reduction, &logits, &targets);
        assert!(
            recorded >= 1,
            "cross_entropy under {reduction:?} must record a tape entry, recorded {recorded}"
        );
        assert_close(
            &read_f32(&out),
            &reduced(&nll, reduction),
            1e-5,
            &format!("cross_entropy_loss {reduction:?} forward"),
        );
        let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
            .expect("cross_entropy backward on CUDA");
        let gx = grads
            .get(logits_id)
            .expect("cross_entropy logits receive a gradient");
        let mut want = vec![0.0f64; 6];
        let s = ce_scale(reduction);
        for (r, &t) in CE_TARGETS.iter().enumerate() {
            let t = usize::try_from(t).expect("class index fits usize");
            for c in 0..3 {
                let one_hot = if c == t { 1.0 } else { 0.0 };
                want[r * 3 + c] = (softmax[r * 3 + c] - one_hot) * s;
            }
        }
        assert_close(
            &read_f32(gx),
            &want,
            1e-5,
            &format!("cross_entropy_loss {reduction:?} dlogits"),
        );
        assert!(
            grads.get(targets_id).is_none(),
            "the i64 class-index target must receive no gradient under {reduction:?}"
        );
        // The integer download path: the target round-trips untouched.
        assert_eq!(download_i64(&targets), CE_TARGETS);
    }
}
