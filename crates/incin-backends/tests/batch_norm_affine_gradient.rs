//! Batch norm's affine parameters receive a gradient on the CPU backend.
//!
//! Found by the `gradcheck_model` example, which checks gradients against
//! central differences and reported that the per-channel gain of a
//! `batch_norm` had no gradient at all while the input did.
//!
//! The cause: both batch-norm kernels expanded `weight` and `bias` from
//! their flat `[C]` to the broadcast shape with the *inherent*
//! `CpuStorage::reshape`, which records no tape node. The parameters were
//! therefore not reachable from the recorded graph, so the reverse walk
//! never produced a gradient for either - and produced no error either,
//! because a missing gradient is only visible to a caller that asks for
//! one. A forward-value test cannot see this at all: the forward was always
//! right, and a batch norm's whole contribution to training *is* the
//! gradient. Every batch norm on this backend kept its scale and shift at
//! their initialization for as long as it trained.
//!
//! The fix routes the affine through a tape-tracked expansion: a reshape to
//! rank-matched `[1, C, 1, ...]` (its own inverse) and then a
//! `broadcast_as` whose inverse is `unbroadcast`, the sum over the axes it
//! expanded. Both steps are needed - `broadcast_as` cannot expand rank, and
//! a tracked reshape alone would hand back a `[1, C, 1, ...]` gradient
//! rather than the per-channel sum a `[C]` parameter needs.
//!
//! The check is a central difference rather than a hand-derived closed form
//! so it does not depend on whether the loss reduces by mean or by sum, and
//! so it keeps working if the reduction changes.
//!
//! This drives the kernel through `dispatch` rather than through the
//! `BatchNorm2d` module because that is where the defect is: the module
//! declares `forward` on a `NoGrad` input and returns `NoGrad`, so a loss
//! built straight from it is not something `backward` can walk, and
//! promoting it afterwards mints a fresh autograd identity that detaches the
//! very graph the walk needs.

use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::{AutogradBackend, HostInterop, HostReadback, StorageBackend};
use incin_core::exec::catalog::{BatchNormAttributes, LossAttributes, LossReduction};
use incin_core::exec::{ExecutionContext, TapeStorage, TensorHandle, dispatch, op};
use incin_core::prelude::DTypeId;
use incin_core::tensor::device::{Cpu, DeviceId};

type B = CpuBackendImpl<Cpu>;

const CHANNELS: usize = 3;
/// `[2, 3, 2, 2]`: two images, three channels, a 2x2 plane each.
const INPUT_SHAPE: [usize; 4] = [2, CHANNELS, 2, 2];
/// Central-difference step and tolerance, matching `gradcheck_model`.
const STEP: f32 = 1e-3;
const RELATIVE: f32 = 2e-2;
const FLOOR: f32 = 1e-4;

type Storage = <B as StorageBackend>::Storage<f32>;

fn upload(values: &[f32], shape: &[usize]) -> Storage {
    <B as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading an f32 operand must succeed")
}

fn download(storage: &Storage) -> Vec<f64> {
    <B as HostReadback>::float_to_vec1::<f32>(storage).expect("readback must succeed")
}

/// One `sum((batch_norm(x; w, b) - t)^2)` forward. Returns the analytic
/// gradient of each operand, or `None` where there is none.
struct Run {
    input: Option<Vec<f64>>,
    weight: Option<Vec<f64>>,
    bias: Option<Vec<f64>>,
    running_var: Option<Vec<f64>>,
}

fn run(x: &[f32], w: &[f32], b: &[f32], t: &[f32]) -> Run {
    let context = ExecutionContext::new(B::new());
    let input = upload(x, &INPUT_SHAPE);
    let weight = upload(w, &[CHANNELS]);
    let bias = upload(b, &[CHANNELS]);
    let zeros = upload(&[0.0; CHANNELS], &[CHANNELS]);
    let ones = upload(&[1.0; CHANNELS], &[CHANNELS]);
    let target = upload(t, &INPUT_SHAPE);

    let normalized = dispatch::execute::<op::BatchNorm, B>(
        &context,
        BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_variance: true,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&input),
            TensorHandle::from_storage::<B, f32, _>(&weight),
            TensorHandle::from_storage::<B, f32, _>(&bias),
            TensorHandle::from_storage::<B, f32, _>(&zeros),
            TensorHandle::from_storage::<B, f32, _>(&ones),
        ],
    )
    .expect("training-mode batch norm must execute");
    let loss = dispatch::execute::<op::MseLoss, B>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&normalized),
            TensorHandle::from_storage::<B, f32, _>(&target),
        ],
    )
    .expect("mse_loss must execute");
    let grads = <B as AutogradBackend>::backward::<f32>(&loss).expect("backward must run");
    // `get` errors on a missing gradient rather than returning `None`,
    // which is what makes the defect this file covers visible at all.
    let take = |storage: &Storage| grads.get(TapeStorage::id(storage)).map(download);
    Run {
        input: take(&input),
        weight: take(&weight),
        bias: take(&bias),
        running_var: take(&ones),
    }
}

/// Values that are neither constant nor symmetric, so a gradient off by a
/// sign, an axis or a factor shows up instead of cancelling.
fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + 3) % 11) as f32 * scale - 5.0 * scale)
        .collect()
}

/// The regression itself: the gain and the shift are reachable, and their
/// gradients agree with a central difference of the same forward.
#[test]
fn the_affine_parameters_receive_their_gradient() {
    let x = ramp(INPUT_SHAPE.iter().product(), 0.4);
    let t = ramp(INPUT_SHAPE.iter().product(), 0.3);
    let w0 = ramp(CHANNELS, 0.6);
    let b0 = ramp(CHANNELS, 0.5);

    let base = run(&x, &w0, &b0, &t);

    // The input was always differentiable; assert it, so a failure below is
    // about the affine and not about the whole op.
    let input_grad = base
        .input
        .clone()
        .expect("the input must still receive a gradient");
    assert_eq!(input_grad.len(), INPUT_SHAPE.iter().product::<usize>());
    assert!(
        input_grad.iter().all(|v| v.is_finite()),
        "the input gradient must be finite, got {input_grad:?}"
    );
    assert!(
        base.running_var.is_none(),
        "a running statistic is a buffer, not a differentiated input, and must \
         not collect a gradient"
    );

    // The two assertions the defect would have failed, as a missing gradient
    // rather than a wrong one: `require` on a missing gradient is an error,
    // so this is where the original bug showed up.
    let gain = base
        .weight
        .clone()
        .expect("the per-channel gain must receive a gradient");
    let shift = base
        .bias
        .clone()
        .expect("the per-channel shift must receive a gradient");
    assert_eq!(gain.len(), CHANNELS);
    assert_eq!(shift.len(), CHANNELS);

    // And the values, against central differences of the same forward. This
    // is the part that stays meaningful if the fix is ever replaced by
    // something that produces a gradient of the right shape and the wrong
    // one.
    for (label, base_values, analytic, is_gain) in
        [("gain", &w0, &gain, true), ("shift", &b0, &shift, false)]
    {
        for index in 0..CHANNELS {
            let mut up = base_values.clone();
            up[index] += STEP;
            let mut down = base_values.clone();
            down[index] -= STEP;
            let (up_loss, down_loss) = if is_gain {
                (loss_of(&x, &up, &b0, &t), loss_of(&x, &down, &b0, &t))
            } else {
                (loss_of(&x, &w0, &up, &t), loss_of(&x, &w0, &down, &t))
            };
            let numeric = (up_loss - down_loss) / (2.0 * f64::from(STEP));
            let got = analytic[index];
            let absolute = (got - numeric).abs();
            assert!(
                absolute < f64::from(FLOOR)
                    || absolute / got.abs().max(numeric.abs()).max(1e-12) <= f64::from(RELATIVE),
                "d(loss)/d({label})[{index}]: analytic {got:.8e}, numeric {numeric:.8e}"
            );
        }
    }
}

/// The scalar loss, so a central difference has something to slope.
fn loss_of(x: &[f32], w: &[f32], b: &[f32], t: &[f32]) -> f64 {
    let context = ExecutionContext::new(B::new());
    let input = upload(x, &INPUT_SHAPE);
    let zeros = upload(&[0.0; CHANNELS], &[CHANNELS]);
    let ones = upload(&[1.0; CHANNELS], &[CHANNELS]);
    let normalized = dispatch::execute::<op::BatchNorm, B>(
        &context,
        BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_variance: true,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&input),
            TensorHandle::from_storage::<B, f32, _>(&upload(w, &[CHANNELS])),
            TensorHandle::from_storage::<B, f32, _>(&upload(b, &[CHANNELS])),
            TensorHandle::from_storage::<B, f32, _>(&zeros),
            TensorHandle::from_storage::<B, f32, _>(&ones),
        ],
    )
    .expect("training-mode batch norm must execute");
    let loss = dispatch::execute::<op::MseLoss, B>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&normalized),
            TensorHandle::from_storage::<B, f32, _>(&upload(t, &INPUT_SHAPE)),
        ],
    )
    .expect("mse_loss must execute");
    let value = download(&loss);
    assert_eq!(value.len(), 1, "mse_loss reduces to a scalar");
    value[0]
}

/// Evaluation mode normalizes by the running statistics, which is a
/// constant with respect to the input - but the affine is still a
/// `Param`, so it must still collect a gradient. The inference kernel had
/// the same untracked reshape as the training kernel and is fixed the same
/// way; a model that trains in one mode and evaluates in the other would
/// otherwise see its batch-norm scale freeze as soon as it switched.
#[test]
fn the_affine_parameters_also_receive_a_gradient_in_inference_mode() {
    let x = ramp(INPUT_SHAPE.iter().product(), 0.4);
    let t = ramp(INPUT_SHAPE.iter().product(), 0.3);
    let w = ramp(CHANNELS, 0.6);
    let b = ramp(CHANNELS, 0.5);
    let mean = vec![0.1f32, -0.2, 0.05];
    let var = vec![0.9f32, 1.1, 0.8];

    let context = ExecutionContext::new(B::new());
    // Every operand is hoisted: a gradient is keyed by the storage's
    // identity, so an operand uploaded inline inside the call below would
    // be a different tensor from the one asked about afterwards.
    let input = upload(&x, &INPUT_SHAPE);
    let weight = upload(&w, &[CHANNELS]);
    let bias = upload(&b, &[CHANNELS]);
    let running_mean = upload(&mean, &[CHANNELS]);
    let running_var = upload(&var, &[CHANNELS]);
    let target = upload(&t, &INPUT_SHAPE);

    let normalized = dispatch::execute::<op::BatchNorm, B>(
        &context,
        BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: false,
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_variance: true,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&input),
            TensorHandle::from_storage::<B, f32, _>(&weight),
            TensorHandle::from_storage::<B, f32, _>(&bias),
            TensorHandle::from_storage::<B, f32, _>(&running_mean),
            TensorHandle::from_storage::<B, f32, _>(&running_var),
        ],
    )
    .expect("inference batch norm must execute");
    let loss = dispatch::execute::<op::MseLoss, B>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&normalized),
            TensorHandle::from_storage::<B, f32, _>(&target),
        ],
    )
    .expect("mse_loss must execute");
    let grads = <B as AutogradBackend>::backward::<f32>(&loss).expect("backward must run");

    let gain = download(
        grads
            .get(TapeStorage::id(&weight))
            .expect("the gain must have a gradient in inference mode"),
    );
    let shift = download(
        grads
            .get(TapeStorage::id(&bias))
            .expect("the shift must have a gradient in inference mode"),
    );
    let input_grad = download(
        grads
            .get(TapeStorage::id(&input))
            .expect("the input must have a gradient in inference mode"),
    );
    assert!(
        gain.iter().all(|v| v.is_finite())
            && shift.iter().all(|v| v.is_finite())
            && input_grad.iter().all(|v| v.is_finite()),
        "inference-mode gradients must be finite"
    );
    // The running statistics are buffers: never differentiated, so they must
    // not collect a gradient even though they are operands.
    assert!(
        grads.get(TapeStorage::id(&running_mean)).is_none()
            && grads.get(TapeStorage::id(&running_var)).is_none(),
        "a running statistic must not collect a gradient"
    );
}
