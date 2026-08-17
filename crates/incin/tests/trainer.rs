//! `UX-001`: the automatic `Trainer` plans devices honestly and trains on them.
//!
//! The row's deliverable is "an unchanged model runs on CPU and on three GPUs".
//! Both halves are here, and they are different kinds of test on purpose. The
//! CPU half really trains: a model, an optimizer, batches, a loss that goes
//! down. The three-GPU half plans against a machine that does not exist,
//! because the point of the deliverable is that *the model* did not change —
//! and a test that could only describe this runner could never check that.
//!
//! What every test below is really guarding is one sentence of §2: "'Easy' must
//! not mean silent CPU transfer." Almost all of these assert a refusal.

#![cfg(all(feature = "train", feature = "cpu"))]

use incin::experimental::training::{Machine, Plan, TrainError, Trainer};
use incin::prelude::*;

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

/// A machine with the CUDA feature compiled in and no CUDA hardware — the
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
                .expect("a shifted target");
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
/// whatever is fastest — otherwise an unchanged program moves onto a GPU the
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

/// The three-GPU half. Same `model()`, same `batches()`, different plan — and
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

/// `epochs(0)` is a plan that trains nothing. It is allowed — planning without
/// training is a use of this API — but it must not silently become one epoch.
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
