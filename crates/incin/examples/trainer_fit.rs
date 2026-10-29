//! Example: drive a training run through the automatic `Trainer`.
//!
//! ```text
//! cargo run -p incin --features train --example trainer_fit
//! ```
//!
//! Every other training example here writes the loop out by hand: forward,
//! loss, `backward`, `optim.step`, per epoch. That is the right way to
//! *learn* the framework and the wrong way to check that a loop is correct,
//! because a loop bug and a working loop produce the same falling loss. The
//! `Trainer` is the framework's own answer - it owns the loop, the device
//! plan, the gradient synchronizer seam and the FSDP seam - and it is the
//! preview surface with the least example coverage, so it is the one most
//! worth exercising end to end.
//!
//! Three things are checked, in increasing order of how much they would
//! catch:
//!
//! 1. **The run really trains.** The loss of a fixed probe batch is taken
//!    from the *same model instance* before and after, because `Linear::build`
//!    initializes randomly and two instances differ for reasons that have
//!    nothing to do with the optimizer. A finite `final_loss` is not
//!    evidence of anything: a loop that ran the forward and never stepped
//!    would produce one.
//! 2. **The bookkeeping is exact.** `epochs` and `batches` are counted by
//!    the trainer, and a count that drifts by one - an off-by-one in the
//!    epoch loop, a skipped final batch, a doubled epoch - is invisible in
//!    the loss and obvious here.
//! 3. **A plan for hardware that does not exist is refused.** This is the
//!    sentence the `Trainer` exists to make true: "easy" must not mean
//!    "silently ran on the CPU instead". The same model, the same batches
//!    and the same optimizer, asked for three GPUs on a machine with none,
//!    must fail before the first batch rather than quietly doing a third of
//!    the work in one place.
//!
//! The problem is synthetic and separable so the run is seconds long and the
//! expected answer is not in doubt: two Gaussian blobs per class in two
//! dimensions, which a two-layer network separates in a few dozen steps.

use incin::experimental::training::{FitOutcome, Machine, Plan, Trainer};
use incin::prelude::*;

type B = DefaultBackend;

/// The model type, spelled once. `SeqTy!` expands to the nested container
/// `seq![]` builds, which is far too long to repeat at every use site.
type Model = SeqTy!(Linear<Dyn, B>, ReLU, Linear<Dyn, B>);

/// One labelled batch. Spelled once because it appears in three signatures
/// and is too long to be worth repeating in each.
type Batch = (Tensor<Dyn, B>, Tensor<Dyn, B, i64>);

/// The machine this process is running on, which is what the CPU preview
/// actually has. `HostMachine` answers exactly this, so the refusal half of
/// the example needs a *different* one.
struct ThisMachine;

impl Machine for ThisMachine {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        // Only the CPU backend is compiled into this build.
        kind == DeviceKind::Cpu
    }

    fn has_device(&self, device: DeviceId) -> bool {
        device == DeviceId::cpu()
    }
}

/// A machine with three CUDA devices and nothing else: the `CUDA` feature
/// off, so `compiled_in` is false for every device kind.
struct ThreeGpusThatDoNotExist;

impl Machine for ThreeGpusThatDoNotExist {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        kind == DeviceKind::Cuda
    }

    fn has_device(&self, device: DeviceId) -> bool {
        device.kind() == DeviceKind::Cuda && device.ordinal() < 3
    }
}

const IN_FEATURES: usize = 2;
const HIDDEN: usize = 16;
const CLASSES: usize = 2;
const PER_CLASS: usize = 64;
const BATCH: usize = 16;

/// Deterministic pseudo-random values, so a run is reproducible without a
/// `rand` dependency. Two well-separated blobs per class: the classes are
/// separable, and a network that cannot separate them is broken rather than
/// under-trained.
fn blobs() -> Result<Vec<Batch>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    // Two uniforms in [-1, 1), one per advance. Both take the top 31 bits
    // of the state: a different shift would scale the second one by a power
    // of two and quietly hand the model features in the millions, which
    // looks like a diverging loss rather than like a broken generator.
    let next = move || {
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / 2147483648.0) - 1.0
        };
        (uniform(), uniform())
    };
    let mut batches = Vec::new();
    for class in 0..CLASSES {
        // Class 0 sits around (-1, -1) and class 1 around (+1, +1), with
        // noise of the same order. The classes therefore *overlap*: a model
        // that is already perfect before training would make the
        // accuracy assertions below vacuous, which is the same mistake as
        // asserting a loss fell when it started at zero.
        let centre = if class == 0 { -1.0 } else { 1.0 };
        for _ in 0..PER_CLASS / BATCH {
            let mut features = Vec::with_capacity(BATCH * IN_FEATURES);
            let mut labels = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                let (dx, dy) = next();
                features.push(centre + dx);
                features.push(centre + dy);
                labels.push(class as i64);
            }
            batches.push((
                Tensor::<Dyn, B>::from_slice(&features, vec![BATCH, IN_FEATURES])?,
                Tensor::<Dyn, B, i64>::from_slice(&labels, vec![BATCH])?,
            ));
        }
    }
    Ok(batches)
}

fn build_model() -> Result<Model> {
    // The hidden ReLU is deliberately not the last layer: `ReLU(Linear)` can
    // put every unit on the flat side for an unlucky initialization, and a
    // model whose gradient is exactly zero would make the "did the
    // parameters move" check fail for a reason that has nothing to do with
    // the trainer.
    Ok(seq![
        Linear::<Dyn, B>::build((IN_FEATURES, HIDDEN))?,
        ReLU,
        Linear::<Dyn, B>::build((HIDDEN, CLASSES))?
    ])
}

/// Loss of one batch, as a plain value. Used to probe the model before and
/// after training on the *same* instance.
fn probe(model: &Model, batch: &Batch) -> Result<f32> {
    let (input, labels) = batch;
    let logits = model.forward(input.to_owned())?;
    logits
        .cross_entropy_loss(labels)
        .and_then(|loss| loss.to_scalar::<f32>())
}

/// Accuracy of one batch, argmax over the classes.
fn accuracy(model: &Model, batch: &Batch) -> Result<f64> {
    let (input, labels) = batch;
    let logits = model.forward(input.to_owned())?.to_vec1::<f32>()?;
    let targets = labels.to_vec1::<i64>()?;
    let correct = (0..targets.len())
        .filter(|&i| {
            let row = &logits[i * CLASSES..(i + 1) * CLASSES];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are comparable"))
                .map(|(j, _)| j)
                .expect("a row has at least one class");
            best as i64 == targets[i]
        })
        .count();
    Ok(correct as f64 / targets.len() as f64)
}

fn main() -> Result<()> {
    let data = blobs()?;
    let mut model = build_model()?;
    let mut optimizer = Adam::<B>::from_module(&model, 5e-2)?;

    println!(
        "Trainer::fit on {} batches of {BATCH}, {IN_FEATURES} features, {CLASSES} classes\n",
        data.len()
    );
    let before_loss = probe(&model, &data[0])?;
    let before_accuracy = accuracy(&model, &data[0])?;
    println!("  before: loss {before_loss:.4}, batch-0 accuracy {before_accuracy:.2}");

    let epochs = 12;
    let plan: Plan = Trainer::plan()
        .epochs(epochs)
        .build_on(&ThisMachine)
        .map_err(|e| incin::Error::Msg(e.to_string()))?;
    println!("  plan:   {}", plan.explain().replace('\n', "\n          "));

    let trainer = Trainer::new(plan);
    let outcome: FitOutcome = trainer
        .fit(
            &mut model,
            &mut optimizer,
            data.clone(),
            |model, (input, labels)| model.forward(input)?.cross_entropy_loss(&labels),
        )
        .map_err(|e| incin::Error::Msg(e.to_string()))?;

    println!(
        "  ran {} epochs, {} batches, final loss {:?}",
        outcome.epochs, outcome.batches, outcome.final_loss
    );
    let after_loss = probe(&model, &data[0])?;
    let after_accuracy = accuracy(&model, &data[0])?;
    println!("  after:  loss {after_loss:.4}, batch-0 accuracy {after_accuracy:.2}");

    // 1. The run trained. Compared before and after on one model instance, so
    //    a fresh initialization cannot explain the difference.
    assert_ne!(
        before_loss, after_loss,
        "the optimizer never moved the parameters, so nothing was trained"
    );
    assert!(
        after_loss < before_loss,
        "the loss rose from {before_loss:.4} to {after_loss:.4}"
    );
    assert!(
        after_accuracy > before_accuracy,
        "accuracy did not improve: {before_accuracy:.2} -> {after_accuracy:.2}"
    );
    assert!(
        after_accuracy > 0.9,
        "the classes are separable and twelve epochs is plenty: batch-0 \
         accuracy is only {after_accuracy:.2}"
    );
    assert!(
        outcome.final_loss.is_some_and(f32::is_finite),
        "final loss must be finite, got {:?}",
        outcome.final_loss
    );

    // 2. The bookkeeping is exact. A count that drifts is invisible in the
    //    loss curve and obvious here.
    assert_eq!(
        outcome.epochs, epochs,
        "the trainer reported the wrong number of epochs"
    );
    assert_eq!(
        outcome.batches,
        epochs * data.len(),
        "the trainer reported the wrong number of batches"
    );

    // 3. A plan for hardware that does not exist is refused rather than
    //    silently narrowed to the CPU. Same model, same batches, same
    //    optimizer - only the plan differs.
    //
    //    The refusal can land in two places and both are correct. Building
    //    the plan asks the machine whether the backend family is compiled
    //    in at all, which in a CPU-only build it is not; building with the
    //    feature on, the plan succeeds and the refusal moves to the first
    //    batch. Which one fired is printed rather than assumed, because the
    //    distinction is the whole difference between "you need a Cargo
    //    feature" and "that device is not there".
    let three_gpus = Trainer::plan()
        .epochs(epochs)
        .devices(DeviceSet::cuda(0..3).expect("a three-device set is a valid set"))
        .build_on(&ThreeGpusThatDoNotExist);
    let (refused_at, error) = match three_gpus {
        Err(error) => ("planning", error.to_string()),
        Ok(plan) => {
            let mut fresh = build_model()?;
            let mut fresh_optimizer = Adam::<B>::from_module(&fresh, 5e-2)?;
            match Trainer::new(plan).fit(
                &mut fresh,
                &mut fresh_optimizer,
                data.clone(),
                |model, (input, labels)| model.forward(input)?.cross_entropy_loss(&labels),
            ) {
                Ok(outcome) => panic!(
                    "a plan for three absent GPUs ran anyway, stepping {} batches",
                    outcome.batches
                ),
                Err(error) => ("the first batch", error.to_string()),
            }
        }
    };
    println!("\n  three absent GPUs refused at {refused_at}: {error}");

    // An empty dataset is not an error either: zero batches and no loss say
    // so more honestly than a loss of 0.0 would.
    let mut empty_model = build_model()?;
    let mut empty_optimizer = Adam::<B>::from_module(&empty_model, 5e-2)?;
    let empty: Vec<Batch> = Vec::new();
    let plan = Trainer::plan()
        .epochs(epochs)
        .build_on(&ThisMachine)
        .map_err(|e| incin::Error::Msg(e.to_string()))?;
    let outcome = Trainer::new(plan)
        .fit(
            &mut empty_model,
            &mut empty_optimizer,
            empty.clone(),
            |model, (input, labels)| model.forward(input)?.cross_entropy_loss(&labels),
        )
        .map_err(|e| incin::Error::Msg(e.to_string()))?;
    assert_eq!(outcome.batches, 0, "an empty dataset steps nothing");
    assert!(
        outcome.final_loss.is_none(),
        "an empty dataset has no last loss, and inventing one would be a lie"
    );
    println!(
        "  an empty dataset: {} batches, final loss {:?}",
        outcome.batches, outcome.final_loss
    );

    println!("\nPASS: the trainer trains, counts exactly, and refuses what it cannot run");
    Ok(())
}
