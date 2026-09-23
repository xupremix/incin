//! `UX-001`: the automatic `Trainer` plans devices honestly and trains on them.
//!
//! The row's deliverable is "an unchanged model runs on CPU and on three GPUs".
//! Both halves are here, and they are different kinds of test on purpose. The
//! CPU half really trains: a model, an optimizer, batches, a loss that goes
//! down. The three-GPU half plans against a machine that does not exist,
//! because the point of the deliverable is that *the model* did not change -
//! and a test that could only describe this runner could never check that.
//!
//! What every test below is really guarding is one sentence of §2: "'Easy' must
//! not mean silent CPU transfer." Almost all of these assert a refusal.

#![cfg(all(feature = "train", feature = "cpu"))]

use incin::backend_authoring::HostReadback;
use incin::experimental::training::{
    GradientSynchronizer, Machine, Plan, SingleRankSynchronizer, SyncError, TrainError, Trainer,
    all_reduce_model_gradients,
};
use incin::nn::{ParameterVisitor, TrainState, VisitParameters};
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Backend = incin::DefaultBackend;

/// The model type both halves of the deliverable name.
///
/// `SeqTy!` expands to the nested container type, which is long enough that
/// spelling it at each use site obscures the signatures it appears in.
type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);

// ============================================================================
// Machines that do not exist
// ============================================================================

/// A machine with three CUDA devices and nothing else.
struct ThreeGpus;

impl Machine for ThreeGpus {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        kind == DeviceKind::Cuda
    }
    fn has_device(&self, device: DeviceId) -> bool {
        device.kind() == DeviceKind::Cuda && device.ordinal() < 3
    }
}

/// A machine with the CUDA feature compiled in and no CUDA hardware - the
/// configuration a silent fallback would hide.
struct CudaCompiledButAbsent;

impl Machine for CudaCompiledButAbsent {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        matches!(kind, DeviceKind::Cuda | DeviceKind::Cpu)
    }
    fn has_device(&self, device: DeviceId) -> bool {
        device.kind() == DeviceKind::Cpu
    }
}

/// An ordinary CPU machine.
struct CpuOnly;

impl Machine for CpuOnly {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        kind == DeviceKind::Cpu
    }
    fn has_device(&self, device: DeviceId) -> bool {
        device == DeviceId::cpu()
    }
}

/// A build with no backend at all.
struct NothingCompiledIn;

impl Machine for NothingCompiledIn {
    fn compiled_in(&self, _kind: DeviceKind) -> bool {
        false
    }
    fn has_device(&self, _device: DeviceId) -> bool {
        false
    }
}

// ============================================================================
// The model, which does not change between the two halves
// ============================================================================

/// The one model both halves of the deliverable use.
///
/// Written once, at the top, and referred to by both the CPU run and the
/// three-GPU plan. If this function ever needed a device argument the row's
/// deliverable would be false, so its signature is itself the assertion.
fn model() -> Result<Model> {
    // The hidden ReLU is deliberately not the last layer. `ReLU(Linear(ones))`
    // can put every unit on the flat side for an unlucky random init, and a
    // model whose gradient is exactly zero would make the "did the parameters
    // move" assertion below fail for a reason that has nothing to do with the
    // trainer.
    Ok(seq![
        Linear::<Dyn, Backend>::build((4, 8))?,
        ReLU,
        Linear::<Dyn, Backend>::build((8, 2))?
    ])
}

/// Four batches of a trivially learnable problem.
fn batches() -> Vec<(Tensor<Dyn, Backend>, Tensor<Dyn, Backend>)> {
    (0..4)
        .map(|i| {
            let input = Tensor::<Dyn, Backend>::ones(vec![2, 4]).expect("a 2x4 input");
            let target = Tensor::<Dyn, Backend>::zeros(vec![2, 2])
                .expect("a 2x2 target")
                .add_scalar(f64::from(i))
                .expect("a shifted target")
                .forget_layout();
            (input, target)
        })
        .collect()
}

// ============================================================================
// Planning
// ============================================================================

/// §2's example, verbatim in its device half.
#[test]
fn the_rfcs_three_gpu_request_plans_three_gpus() {
    let plan = Trainer::plan()
        .devices(DeviceSet::cuda(0..3).expect("three CUDA devices"))
        .epochs(10)
        .build_on(&ThreeGpus)
        .expect("a machine with three CUDA devices can satisfy a three-CUDA request");

    assert_eq!(plan.devices().len(), 3);
    assert_eq!(plan.devices().primary(), DeviceId::cuda(0));
    assert_eq!(plan.epochs(), 10);
    assert!(plan.is_multi_device());
}

/// The row's whole point. Asking for CUDA on a machine without it must be an
/// error, not a CPU run.
#[test]
fn an_absent_device_is_an_error_and_never_a_cpu_fallback() {
    let error = Trainer::plan()
        .devices(DeviceSet::cuda(0..3).expect("three CUDA devices"))
        .build_on(&CudaCompiledButAbsent)
        .expect_err("CUDA is compiled in but absent, so this cannot be satisfied");

    assert_eq!(
        error,
        TrainError::DeviceUnavailable {
            device: DeviceId::cuda(0)
        }
    );
    assert!(error.to_string().contains("not available"));
}

/// A missing *feature* and missing *hardware* are different problems with
/// different fixes, so they are different errors.
#[test]
fn a_missing_feature_is_reported_separately_from_missing_hardware() {
    let error = Trainer::plan()
        .devices(DeviceSet::cuda(0..1).expect("one CUDA device"))
        .build_on(&CpuOnly)
        .expect_err("a CPU-only build has no CUDA backend");

    assert_eq!(
        error,
        TrainError::NotCompiledIn {
            kind: DeviceKind::Cuda,
            feature: "cuda",
        }
    );
    assert!(error.to_string().contains("`cuda`"));
}

/// Asking for four GPUs on a three-GPU machine names the one that is missing,
/// rather than quietly planning three.
#[test]
fn a_partially_available_set_names_the_device_that_is_missing() {
    let error = Trainer::plan()
        .devices(DeviceSet::cuda(0..4).expect("four CUDA devices"))
        .build_on(&ThreeGpus)
        .expect_err("the fourth device does not exist");

    assert_eq!(
        error,
        TrainError::DeviceUnavailable {
            device: DeviceId::cuda(3)
        }
    );
}

/// `Fastest` is the one preference allowed to end up on the CPU, because it is
/// the one where the caller said they did not mind.
#[test]
fn the_fastest_preference_may_fall_back_and_records_every_step() {
    let plan = Trainer::plan()
        .device_preference(DevicePreference::Fastest)
        .build_on(&CudaCompiledButAbsent)
        .expect("the CPU is available");

    assert_eq!(plan.devices().primary(), DeviceId::cpu());

    let codes: Vec<&str> = plan.decisions().iter().map(|d| d.code).collect();
    assert!(
        codes.contains(&"family-unavailable"),
        "the skipped CUDA family has to appear in the report, or the fallback \
         is silent after all: {codes:?}"
    );
    assert!(codes.contains(&"devices-resolved"));

    let cuda = plan
        .decisions()
        .iter()
        .find(|d| d.code == "family-unavailable")
        .expect("CUDA was skipped");
    assert!(cuda.detail.contains("cuda"), "{}", cuda.detail);
}

/// The difference between the two preferences is the whole reason they are two
/// types rather than one.
#[test]
fn an_exact_request_and_a_preference_disagree_on_the_same_machine() {
    let exact = Trainer::plan()
        .devices(DeviceSet::cuda(0..1).expect("one CUDA device"))
        .build_on(&CudaCompiledButAbsent);
    let preferred = Trainer::plan()
        .device_preference(DevicePreference::Fastest)
        .build_on(&CudaCompiledButAbsent);

    assert!(exact.is_err(), "an exact request must not be substituted");
    assert!(preferred.is_ok(), "a preference may resolve elsewhere");
}

#[test]
fn a_build_with_no_backend_has_nowhere_to_run() {
    assert_eq!(
        Trainer::plan()
            .device_preference(DevicePreference::Fastest)
            .build_on(&NothingCompiledIn),
        Err(TrainError::NoDeviceAvailable)
    );
    assert_eq!(
        Trainer::plan().build_on(&NothingCompiledIn),
        Err(TrainError::NotCompiledIn {
            kind: DeviceKind::Cpu,
            feature: "cpu",
        })
    );
}

/// The default has to be the boring one, and it has to be the CPU rather than
/// whatever is fastest - otherwise an unchanged program moves onto a GPU the
/// day one appears.
#[test]
fn the_default_plan_is_one_cpu_and_one_epoch() {
    let plan = Trainer::plan()
        .build_on(&CpuOnly)
        .expect("the CPU is there");
    assert_eq!(plan.devices().devices(), [DeviceId::cpu()]);
    assert_eq!(plan.epochs(), 1);
    assert!(!plan.is_multi_device());
}

/// Every plan explains itself. §2: "Every automatic decision is inspectable."
#[test]
fn every_plan_carries_at_least_a_device_and_an_epoch_decision() {
    for plan in [
        Trainer::plan().build_on(&CpuOnly).expect("cpu"),
        Trainer::plan()
            .device_preference(DevicePreference::Fastest)
            .build_on(&CpuOnly)
            .expect("cpu"),
        Trainer::plan()
            .devices(DeviceSet::cuda(0..3).expect("three"))
            .build_on(&ThreeGpus)
            .expect("three gpus"),
    ] {
        let codes: Vec<&str> = plan.decisions().iter().map(|d| d.code).collect();
        assert!(codes.contains(&"epochs"), "{codes:?}");
        assert!(codes.iter().any(|c| c.starts_with("devices-")), "{codes:?}");
        assert!(
            plan.decisions().iter().all(|d| !d.detail.is_empty()),
            "a decision with no detail explains nothing"
        );
    }
}

/// A plan that cannot execute has to say so at plan time, not only at `fit`.
#[test]
fn a_multi_device_plan_says_it_needs_collectives() {
    let plan = Trainer::plan()
        .devices(DeviceSet::cuda(0..3).expect("three"))
        .build_on(&ThreeGpus)
        .expect("three gpus");

    let collectives = plan
        .decisions()
        .iter()
        .find(|d| d.code == "collectives-required")
        .expect("a three-device plan needs collectives");
    assert!(collectives.detail.contains("DST-005"), "{collectives:?}");
}

// ============================================================================
// Training
// ============================================================================

/// The CPU half of the deliverable: the model from `model()`, really trained.
#[test]
fn the_model_trains_on_the_cpu() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(
        Trainer::plan()
            .epochs(3)
            .build_on(&CpuOnly)
            .expect("the CPU is there"),
    );

    // Taken on this model instance before training, and again after. A fresh
    // model would not do: `Linear::build` initializes randomly, so two
    // instances differ for reasons that have nothing to do with the optimizer.
    let probe = |model: &Model| {
        let (input, target) = &data[0];
        model
            .forward(input.clone())
            .and_then(|out| out.mse_loss(target))
            .and_then(|loss| loss.to_scalar::<f32>())
            .expect("the probe batch evaluates")
    };
    let before = probe(&model);

    let outcome = trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("training on the CPU succeeds");

    assert_eq!(outcome.epochs, 3);
    assert_eq!(outcome.batches, 12, "3 epochs of 4 batches");

    // A finite loss is not evidence of training. A `fit` that ran the forward
    // pass and never called the optimizer would produce one, so what is
    // asserted is that the parameters moved: the same batch, through the same
    // model, gives a different loss after training than before it.
    let after = probe(&model);
    assert_ne!(
        before, after,
        "the optimizer never moved the parameters, so nothing was trained"
    );
    assert!(
        outcome.final_loss.is_some_and(f32::is_finite),
        "got {:?}",
        outcome.final_loss
    );
    Ok(())
}

/// The three-GPU half. Same `model()`, same `batches()`, different plan - and
/// `fit` refuses rather than running a third of the work on one GPU.
#[test]
fn the_same_model_plans_for_three_gpus_and_refuses_to_fake_the_run() -> Result<()> {
    let mut model = model().expect("the same model as the CPU test");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cuda(0..3).expect("three"))
            .build_on(&ThreeGpus)
            .expect("three gpus"),
    );

    assert_eq!(
        trainer
            .fit(
                &mut model,
                &mut optimizer,
                &data,
                |model, (input, target)| { model.forward(input.clone())?.mse_loss(target) }
            )
            .expect_err("collectives do not exist yet"),
        TrainError::CollectivesUnavailable { devices: 3 }
    );
    Ok(())
}

/// An empty dataset is a dataset with no batches, not an error and not a loss
/// of zero.
#[test]
fn an_empty_dataset_reports_no_batches_rather_than_a_zero_loss() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let empty: Vec<(Tensor<Dyn, Backend>, Tensor<Dyn, Backend>)> = Vec::new();

    let trainer = Trainer::new(Trainer::plan().epochs(5).build_on(&CpuOnly).expect("cpu"));
    let outcome = trainer
        .fit(
            &mut model,
            &mut optimizer,
            &empty,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("an empty dataset is not a failure");

    assert_eq!(outcome.batches, 0);
    assert_eq!(outcome.final_loss, None);
    Ok(())
}

/// A failure in the caller's own step has to say where it happened. "Shape
/// mismatch" without a batch number is a bug report nobody can act on.
#[test]
fn a_failing_step_reports_the_epoch_and_batch_it_failed_in() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(Trainer::plan().epochs(2).build_on(&CpuOnly).expect("cpu"));
    let mut seen = 0;
    let error = trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| {
                seen += 1;
                if seen == 3 {
                    // A target whose shape cannot match the output.
                    let wrong = Tensor::<Dyn, Backend>::zeros(vec![7, 7])?;
                    return model.forward(input.clone())?.mse_loss(&wrong);
                }
                model.forward(input.clone())?.mse_loss(target)
            },
        )
        .expect_err("the third batch fails");

    match error {
        TrainError::Step {
            epoch,
            batch,
            ref message,
        } => {
            assert_eq!((epoch, batch), (0, 2), "the third batch of the first epoch");
            assert!(!message.is_empty());
        }
        other => panic!("expected a step failure, got {other:?}"),
    }
    Ok(())
}

/// `epochs(0)` is a plan that trains nothing. It is allowed - planning without
/// training is a use of this API - but it must not silently become one epoch.
#[test]
fn zero_epochs_runs_nothing_rather_than_being_rounded_up() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(Trainer::plan().epochs(0).build_on(&CpuOnly).expect("cpu"));
    let outcome = trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("zero epochs is a valid plan");

    assert_eq!(outcome.batches, 0);
    assert_eq!(outcome.final_loss, None);
    Ok(())
}

/// The plan a trainer reports is the plan it was built with, unmodified.
#[test]
fn a_trainer_reports_the_plan_it_was_given() {
    let plan: Plan = Trainer::plan()
        .devices(DeviceSet::cuda(0..3).expect("three"))
        .epochs(7)
        .build_on(&ThreeGpus)
        .expect("three gpus");

    assert_eq!(Trainer::new(plan.clone()).report(), &plan);
}

// ============================================================================
// Gradient synchronization (#97, DST-005/DST-008 fail-closed surface)
// ============================================================================

/// Collects every parameter's gradient, in `VisitParameters` order, as `f64`.
///
/// Non-f32 parameters and parameters without a gradient are skipped, matching
/// `all_reduce_model_gradients`' collective protocol so peer buffers built
/// with this visitor line up with the synchronizer's call sequence.
struct GradFlatten<'a> {
    grads: &'a Gradients<Backend>,
    chunks: Vec<Vec<f64>>,
}

impl GradFlatten<'_> {
    fn flat(&self) -> Vec<f64> {
        self.chunks.iter().flatten().copied().collect()
    }
}

impl ParameterVisitor<Backend> for GradFlatten<'_> {
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &incin::nn::param::Param<S, Backend, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let tensor = param.as_tensor()?;
        if tensor.dtype() != <f32 as ConstDType>::DESCRIPTOR {
            return Ok(());
        }
        if let Some(grad) = self.grads.get(&tensor)? {
            let values = Backend::float_to_vec1::<f32>(grad.inner())?;
            self.chunks.push(values);
        }
        Ok(())
    }
}

fn flatten_grads(model: &Model, grads: &Gradients<Backend>) -> Result<Vec<f64>> {
    let mut visitor = GradFlatten {
        grads,
        chunks: Vec::new(),
    };
    model.visit_parameters(&StatePath::root(), &mut visitor)?;
    Ok(visitor.flat())
}

/// A synchronizer that claims three ranks and never changes a value.
///
/// Enough to prove a three-device plan executes once counts agree; it is not
/// a claim that three-rank reduction is correct (that needs real transports).
#[derive(Debug)]
struct ThreeRankNoop;

impl GradientSynchronizer for ThreeRankNoop {
    fn world_size(&self) -> usize {
        3
    }
    fn rank(&self) -> usize {
        0
    }
    fn all_reduce_mean(&self, _values: &mut [f64]) -> std::result::Result<(), SyncError> {
        Ok(())
    }
}

/// A synchronizer whose world size disagrees with the plan it is attached to.
#[derive(Debug)]
struct TwoRankNoop;

impl GradientSynchronizer for TwoRankNoop {
    fn world_size(&self) -> usize {
        2
    }
    fn rank(&self) -> usize {
        0
    }
    fn all_reduce_mean(&self, _values: &mut [f64]) -> std::result::Result<(), SyncError> {
        Ok(())
    }
}

/// Rank `1` of a one-rank world: passes the device-count check, fails the
/// rank-inside-world check inside the first collective.
#[derive(Debug)]
struct OutOfRangeRank;

impl GradientSynchronizer for OutOfRangeRank {
    fn world_size(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        1
    }
    fn all_reduce_mean(&self, _values: &mut [f64]) -> std::result::Result<(), SyncError> {
        Ok(())
    }
}

/// The synchronizer's own refusal must surface as a step error naming the
/// epoch and batch, not disappear into a silent skip.
#[derive(Debug)]
struct RefusesEveryReduce;

impl GradientSynchronizer for RefusesEveryReduce {
    fn world_size(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn all_reduce_mean(&self, _values: &mut [f64]) -> std::result::Result<(), SyncError> {
        Err(SyncError::Synchronizer {
            message: "scripted refusal".to_string(),
        })
    }
}

/// The proven single-rank path: a world of one must leave every gradient
/// bit-identical, because that is what makes multi-rank mean reduction
/// trustworthy to build on.
#[test]
fn single_rank_synchronization_leaves_gradients_bit_identical() -> Result<()> {
    let model = model().expect("the model builds");
    let data = batches();
    let (input, target) = &data[0];
    let loss = model.forward(input.clone())?.mse_loss(target)?;
    let mut grads = loss.backward()?;

    let before = flatten_grads(&model, &grads)?;
    assert!(!before.is_empty(), "backward produced no gradients");

    all_reduce_model_gradients(&model, &mut grads, &SingleRankSynchronizer)
        .map_err(|error| incin::Error::Msg(error.to_string()))?;

    let after = flatten_grads(&model, &grads)?;
    assert_eq!(
        before, after,
        "a one-rank mean must be the identity on every gradient element"
    );
    Ok(())
}

/// A single-rank synchronizer runs the full sync path but changes nothing, so
/// the trajectory - parameters, loss, batch count - must match a plain fit.
#[test]
fn a_single_rank_synchronizer_preserves_the_fit_trajectory() -> Result<()> {
    let data = batches();

    let mut model_sync = model().expect("the model builds");
    let initial = collect_state::<Backend, _>(&model_sync)?;
    let mut model_plain = model().expect("the model builds");
    load_state::<Backend, _>(&mut model_plain, &initial)?;

    let mut opt_plain = SGD::<Backend>::from_module(&model_plain, 0.01)?;
    let mut opt_sync = SGD::<Backend>::from_module(&model_sync, 0.01)?;

    let plain = Trainer::new(Trainer::plan().epochs(3).build_on(&CpuOnly).expect("cpu"));
    let synced = Trainer::new(Trainer::plan().epochs(3).build_on(&CpuOnly).expect("cpu"))
        .with_synchronizer(SingleRankSynchronizer);
    assert!(synced.synchronizer().is_some());

    let outcome_plain = plain
        .fit(
            &mut model_plain,
            &mut opt_plain,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("plain fit succeeds");
    let outcome_sync = synced
        .fit(
            &mut model_sync,
            &mut opt_sync,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("synchronized fit succeeds");

    assert_eq!(outcome_plain, outcome_sync);
    assert_eq!(
        collect_state::<Backend, _>(&model_plain)?,
        collect_state::<Backend, _>(&model_sync)?,
        "a one-rank synchronizer must not move the trajectory"
    );
    Ok(())
}

/// `fit_scaled` is the same contract: multi-device without a synchronizer is
/// a refusal, not a silent single-device scaled run.
#[test]
fn fit_scaled_refuses_a_multi_device_plan_without_a_synchronizer() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();
    let mut scaler = LossScaleState::new(LossScaling::None);

    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cuda(0..3).expect("three"))
            .build_on(&ThreeGpus)
            .expect("three gpus"),
    );

    assert_eq!(
        trainer
            .fit_scaled(
                &mut model,
                &mut optimizer,
                &mut scaler,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect_err("no synchronizer is attached"),
        TrainError::CollectivesUnavailable { devices: 3 }
    );
    Ok(())
}

/// When the synchronizer's world size equals the device count, the refusal
/// lifts and the run executes - reducing gradients after every backward pass.
#[test]
fn a_multi_device_plan_with_a_matching_synchronizer_executes() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let probe = |model: &Model| {
        let (input, target) = &data[0];
        model
            .forward(input.clone())
            .and_then(|out| out.mse_loss(target))
            .and_then(|loss| loss.to_scalar::<f32>())
            .expect("the probe batch evaluates")
    };
    let before = probe(&model);

    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cuda(0..3).expect("three"))
            .epochs(2)
            .build_on(&ThreeGpus)
            .expect("three gpus"),
    )
    .with_synchronizer(ThreeRankNoop);

    let outcome = trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("matching world size means the plan can execute");

    assert_eq!(outcome.epochs, 2);
    assert_eq!(outcome.batches, 8, "2 epochs of 4 batches");
    assert_ne!(
        before,
        probe(&model),
        "the synchronized run must still train the parameters"
    );
    Ok(())
}

/// A world size that disagrees with the plan is a hang-or-wrong-answer
/// waiting to happen, so both directions of disagreement are refused before
/// the first batch - three devices against two ranks, and one device
/// against two.
#[test]
fn a_synchronizer_whose_world_size_disagrees_with_the_plan_is_refused() -> Result<()> {
    let data = batches();

    let mut model_three = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model_three, 0.01)?;
    let three = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cuda(0..3).expect("three"))
            .build_on(&ThreeGpus)
            .expect("three gpus"),
    )
    .with_synchronizer(TwoRankNoop);
    assert_eq!(
        three
            .fit(
                &mut model_three,
                &mut optimizer,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect_err("3 devices against a 2-rank world cannot agree"),
        TrainError::SynchronizerMismatch {
            devices: 3,
            world_size: 2
        }
    );

    let mut model_one = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model_one, 0.01)?;
    let one = Trainer::new(Trainer::plan().epochs(1).build_on(&CpuOnly).expect("cpu"))
        .with_synchronizer(TwoRankNoop);
    assert_eq!(
        one.fit(
            &mut model_one,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect_err("1 device against a 2-rank world cannot agree"),
        TrainError::SynchronizerMismatch {
            devices: 1,
            world_size: 2
        }
    );
    Ok(())
}

/// Counts agree but the rank is outside its own world: the device-count
/// check passes, and the first collective refuses with a step error that
/// names where it happened.
#[test]
fn an_out_of_range_rank_surfaces_as_a_step_error_at_the_first_batch() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(Trainer::plan().epochs(1).build_on(&CpuOnly).expect("cpu"))
        .with_synchronizer(OutOfRangeRank);

    match trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect_err("rank 1 does not exist in a world of one")
    {
        TrainError::Step {
            epoch,
            batch,
            ref message,
        } => {
            assert_eq!((epoch, batch), (0, 0), "the very first batch");
            assert!(
                message.contains("rank 1") && message.contains("world size 1"),
                "got: {message}"
            );
        }
        other => panic!("expected a step failure, got {other:?}"),
    }
    Ok(())
}

/// A synchronizer that refuses every collective must fail the step loudly.
#[test]
fn a_refusing_synchronizer_fails_the_step_rather_than_skipping_the_reduce() -> Result<()> {
    let mut model = model().expect("the model builds");
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
    let data = batches();

    let trainer = Trainer::new(Trainer::plan().epochs(1).build_on(&CpuOnly).expect("cpu"))
        .with_synchronizer(RefusesEveryReduce);

    match trainer
        .fit(
            &mut model,
            &mut optimizer,
            &data,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect_err("the scripted refusal must stop the run")
    {
        TrainError::Step {
            epoch,
            batch,
            ref message,
        } => {
            assert_eq!((epoch, batch), (0, 0));
            assert!(
                message.contains("scripted refusal"),
                "the transport's own message must survive: {message}"
            );
        }
        other => panic!("expected a step failure, got {other:?}"),
    }
    Ok(())
}

// ============================================================================
// FSDP sharded execution (#99, distributed-gated)
// ============================================================================

/// Planning, refusal, and wiring coverage for [`ShardingSpec::Fsdp`].
///
/// The CPU-runnable protocol arithmetic (reduce-scatter, byte fanout,
/// trajectory equality) lives in `incin-core`'s `fsdp_exec` tests; this
/// module is the trainer's half - what the builder records, what fit
/// refuses, and that the sharded walk executes in place of the plain
/// seam.
#[cfg(feature = "distributed")]
mod fsdp_sharding {
    use super::*;
    use incin::experimental::distributed::ZeROStage;
    use incin::experimental::training::{FsdpSynchronizer, ShardingSpec};

    /// A machine with two CUDA devices and nothing else.
    struct TwoGpus;

    impl Machine for TwoGpus {
        fn compiled_in(&self, kind: DeviceKind) -> bool {
            kind == DeviceKind::Cuda
        }
        fn has_device(&self, device: DeviceId) -> bool {
            device.kind() == DeviceKind::Cuda && device.ordinal() < 2
        }
    }

    /// A plumbing-only FSDP synchronizer of a given world size.
    ///
    /// Rank 0 of `world`: reduce-scatter hands back the local prefix as
    /// its "owned shard", all-gather concatenates this rank's shard with
    /// itself `world` times, and all-reduce changes nothing. Enough to
    /// prove a plan of that world size executes once counts agree; it is
    /// not a claim that cross-rank reduction is correct (that needs real
    /// transports and lives in the `fsdp_exec` protocol tests).
    #[derive(Debug)]
    struct FsdpNoop {
        world: usize,
    }

    impl GradientSynchronizer for FsdpNoop {
        fn world_size(&self) -> usize {
            self.world
        }
        fn rank(&self) -> usize {
            0
        }
        fn all_reduce_mean(&self, _values: &mut [f64]) -> std::result::Result<(), SyncError> {
            Ok(())
        }
    }

    impl FsdpSynchronizer for FsdpNoop {
        fn reduce_scatter_mean(&self, values: &[f64]) -> std::result::Result<Vec<f64>, SyncError> {
            Ok(values[..values.len() / self.world].to_vec())
        }
        fn all_gather(&self, shard: &[f64]) -> std::result::Result<Vec<f64>, SyncError> {
            let mut full = Vec::with_capacity(shard.len() * self.world);
            for _ in 0..self.world {
                full.extend_from_slice(shard);
            }
            Ok(full)
        }
    }

    /// The builder says what will shard, in the decision list, the
    /// getter, and both explain renderings.
    #[test]
    fn an_fsdp_sharding_decision_is_recorded_and_explained() {
        let plan = Trainer::plan()
            .devices(DeviceSet::cuda(0..2).expect("two CUDA devices"))
            .sharding(ShardingSpec::Fsdp {
                stage: ZeROStage::ZeRO2,
            })
            .build_on(&TwoGpus)
            .expect("two gpus");

        assert_eq!(
            plan.sharding(),
            Some(ShardingSpec::Fsdp {
                stage: ZeROStage::ZeRO2
            })
        );
        let decision = plan
            .decisions()
            .iter()
            .find(|decision| decision.code == "fsdp-sharding")
            .expect("the sharding decision is recorded");
        assert!(
            decision.detail.contains("ZeRO2"),
            "the stage must appear in the detail: {}",
            decision.detail
        );
        let explain = plan.explain();
        assert!(
            explain.contains("Sharding: FSDP / ZeRO2"),
            "the rendered plan must say what will shard: {explain}"
        );
        let json = plan.explain_json();
        assert!(
            json.contains("\"sharding\"") && json.contains("\"fsdp\""),
            "the JSON plan must carry the sharding strategy: {json}"
        );
    }

    /// ZeRO-3 needs parameter sharding this trainer does not have, so the
    /// build refuses instead of describing a run it would approximate as
    /// ZeRO-2.
    #[test]
    fn zero3_sharding_is_refused_at_build() {
        let error = Trainer::plan()
            .sharding(ShardingSpec::Fsdp {
                stage: ZeROStage::ZeRO3,
            })
            .build_on(&CpuOnly)
            .expect_err("ZeRO-3 execution does not exist");
        match error {
            TrainError::UnsupportedShardingStage { stage } => {
                assert_eq!(stage, ZeROStage::ZeRO3);
            }
            other => panic!("expected UnsupportedShardingStage, got {other:?}"),
        }
    }

    /// ZeRO-1 plans build: the all-reduce-then-mask path is executable.
    #[test]
    fn a_zero1_sharding_plan_builds() {
        let plan = Trainer::plan()
            .sharding(ShardingSpec::Fsdp {
                stage: ZeROStage::ZeRO1,
            })
            .build_on(&CpuOnly)
            .expect("ZeRO-1 builds");
        assert_eq!(
            plan.sharding(),
            Some(ShardingSpec::Fsdp {
                stage: ZeROStage::ZeRO1
            })
        );
    }

    /// A sharded plan with no FSDP synchronizer has no reduction to run,
    /// so fit refuses before the first batch rather than stepping on
    /// rank-local gradients.
    #[test]
    fn an_fsdp_plan_without_an_fsdp_synchronizer_refuses_to_fit() -> Result<()> {
        let mut model = model().expect("the model builds");
        let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
        let data = batches();

        let trainer = Trainer::new(
            Trainer::plan()
                .devices(DeviceSet::cuda(0..2).expect("two CUDA devices"))
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .build_on(&TwoGpus)
                .expect("two gpus"),
        );
        match trainer
            .fit(
                &mut model,
                &mut optimizer,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect_err("no FSDP synchronizer is attached")
        {
            TrainError::FsdpUnavailable { devices } => assert_eq!(devices, 2),
            other => panic!("expected FsdpUnavailable, got {other:?}"),
        }
        Ok(())
    }

    /// An FSDP synchronizer whose world size disagrees with the plan is a
    /// hang-or-wrong-answer waiting to happen, exactly like the plain
    /// seam: refused before the first batch.
    #[test]
    fn an_fsdp_synchronizer_with_a_mismatched_world_is_refused() -> Result<()> {
        let mut model = model().expect("the model builds");
        let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
        let data = batches();

        let trainer = Trainer::new(
            Trainer::plan()
                .devices(DeviceSet::cuda(0..2).expect("two CUDA devices"))
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .build_on(&TwoGpus)
                .expect("two gpus"),
        )
        .with_fsdp_synchronizer(FsdpNoop { world: 3 });

        match trainer
            .fit(
                &mut model,
                &mut optimizer,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect_err("world 3 against 2 devices")
        {
            TrainError::SynchronizerMismatch {
                devices,
                world_size,
            } => {
                assert_eq!((devices, world_size), (2, 3));
            }
            other => panic!("expected SynchronizerMismatch, got {other:?}"),
        }
        Ok(())
    }

    /// The sharded path takes its collective through the FSDP seam, so a
    /// multi-device sharded plan executes without the plain seam attached
    /// - and the plumbing really runs: every batch steps.
    #[test]
    fn a_sharded_plan_executes_without_the_plain_seam() -> Result<()> {
        let mut model = model().expect("the model builds");
        let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
        let data = batches();

        let trainer = Trainer::new(
            Trainer::plan()
                .devices(DeviceSet::cuda(0..2).expect("two CUDA devices"))
                .epochs(2)
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .build_on(&TwoGpus)
                .expect("two gpus"),
        )
        .with_fsdp_synchronizer(FsdpNoop { world: 2 });
        assert!(trainer.fsdp_synchronizer().is_some());
        assert!(trainer.synchronizer().is_none(), "the plain seam stays off");

        let outcome = trainer
            .fit(
                &mut model,
                &mut optimizer,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("a matching FSDP world executes");
        assert_eq!(outcome.epochs, 2);
        assert_eq!(outcome.batches, 8, "2 epochs of 4 batches");
        Ok(())
    }

    /// A world-of-one FSDP synchronizer makes every sharded walk the
    /// identity - reduce-scatter owns everything, the mask keeps
    /// everything, the gather reproduces the replica - so the trajectory
    /// must equal a plain fit exactly, bit for bit.
    #[test]
    fn a_one_device_fsdp_plan_preserves_the_fit_trajectory() -> Result<()> {
        let data = batches();

        let mut model_plain = model().expect("the model builds");
        let initial = collect_state::<Backend, _>(&model_plain)?;
        let mut model_fsdp = model().expect("the model builds");
        load_state::<Backend, _>(&mut model_fsdp, &initial)?;

        let mut opt_plain = SGD::<Backend>::from_module(&model_plain, 0.01)?;
        let mut opt_fsdp = SGD::<Backend>::from_module(&model_fsdp, 0.01)?;

        let plain = Trainer::new(Trainer::plan().epochs(3).build_on(&CpuOnly).expect("cpu"));
        let sharded = Trainer::new(
            Trainer::plan()
                .epochs(3)
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .build_on(&CpuOnly)
                .expect("cpu"),
        )
        .with_fsdp_synchronizer(SingleRankSynchronizer);

        let outcome_plain = plain
            .fit(
                &mut model_plain,
                &mut opt_plain,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("plain fit succeeds");
        let outcome_fsdp = sharded
            .fit(
                &mut model_fsdp,
                &mut opt_fsdp,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("sharded fit succeeds");

        assert_eq!(outcome_plain, outcome_fsdp);
        assert_eq!(
            collect_state::<Backend, _>(&model_plain)?,
            collect_state::<Backend, _>(&model_fsdp)?,
            "a one-rank ZeRO-2 walk must not move the trajectory"
        );
        Ok(())
    }

    /// ZeRO-1's all-reduce-then-mask stage takes the same walk and must
    /// preserve the trajectory too.
    #[test]
    fn a_one_device_zero1_plan_preserves_the_fit_trajectory() -> Result<()> {
        let data = batches();

        let mut model_plain = model().expect("the model builds");
        let initial = collect_state::<Backend, _>(&model_plain)?;
        let mut model_fsdp = model().expect("the model builds");
        load_state::<Backend, _>(&mut model_fsdp, &initial)?;

        let mut opt_plain = SGD::<Backend>::from_module(&model_plain, 0.01)?;
        let mut opt_fsdp = SGD::<Backend>::from_module(&model_fsdp, 0.01)?;

        let plain = Trainer::new(Trainer::plan().epochs(2).build_on(&CpuOnly).expect("cpu"));
        let sharded = Trainer::new(
            Trainer::plan()
                .epochs(2)
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO1,
                })
                .build_on(&CpuOnly)
                .expect("cpu"),
        )
        .with_fsdp_synchronizer(SingleRankSynchronizer);

        let outcome_plain = plain
            .fit(
                &mut model_plain,
                &mut opt_plain,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("plain fit succeeds");
        let outcome_fsdp = sharded
            .fit(
                &mut model_fsdp,
                &mut opt_fsdp,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("ZeRO-1 fit succeeds");

        assert_eq!(outcome_plain, outcome_fsdp);
        assert_eq!(
            collect_state::<Backend, _>(&model_plain)?,
            collect_state::<Backend, _>(&model_fsdp)?,
            "a one-rank ZeRO-1 walk must not move the trajectory"
        );
        Ok(())
    }

    /// `fit_scaled` carries the same sharded path: reduce, unscale-and-step,
    /// gather - and a world-of-one run still matches plain scaled training.
    #[test]
    fn fit_scaled_executes_the_fsdp_path() -> Result<()> {
        let data = batches();

        let mut model_plain = model().expect("the model builds");
        let initial = collect_state::<Backend, _>(&model_plain)?;
        let mut model_fsdp = model().expect("the model builds");
        load_state::<Backend, _>(&mut model_fsdp, &initial)?;

        let mut opt_plain = SGD::<Backend>::from_module(&model_plain, 0.01)?;
        let mut opt_fsdp = SGD::<Backend>::from_module(&model_fsdp, 0.01)?;
        let mut scaler_plain = LossScaleState::new(LossScaling::None);
        let mut scaler_fsdp = LossScaleState::new(LossScaling::None);

        let plain = Trainer::new(Trainer::plan().epochs(2).build_on(&CpuOnly).expect("cpu"));
        let sharded = Trainer::new(
            Trainer::plan()
                .epochs(2)
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .build_on(&CpuOnly)
                .expect("cpu"),
        )
        .with_fsdp_synchronizer(SingleRankSynchronizer);

        let outcome_plain = plain
            .fit_scaled(
                &mut model_plain,
                &mut opt_plain,
                &mut scaler_plain,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("plain fit_scaled succeeds");
        let outcome_fsdp = sharded
            .fit_scaled(
                &mut model_fsdp,
                &mut opt_fsdp,
                &mut scaler_fsdp,
                &data,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect("sharded fit_scaled succeeds");

        assert_eq!(outcome_plain, outcome_fsdp);
        assert_eq!(
            collect_state::<Backend, _>(&model_plain)?,
            collect_state::<Backend, _>(&model_fsdp)?,
            "the sharded scaled path must not move the trajectory"
        );
        Ok(())
    }
}
