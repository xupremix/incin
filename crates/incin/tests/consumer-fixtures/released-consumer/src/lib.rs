//! The published crate, used the way the front page tells a reader to use it.
//!
//! Every other consumer fixture resolves `incin` through a path dependency, so
//! all of them together prove that this checkout exports the names the facade
//! promises. None of them touches what a reader actually gets: a version
//! requirement resolved against crates.io, with whatever feature unification,
//! dependency floor, and packaging decisions the published `.crate` files
//! carry. `tools/check-package.sh` compares the packaged file lists, which is a
//! different claim from "the packaged thing computes".
//!
//! So the tests below are deliberately end to end rather than surface checks.
//! A missing re-export is caught fifteen ways already; what is not caught is a
//! release whose exports all resolve and whose training loop does not descend.

use incin::prelude::*;

/// The README's quick-start model, transcribed rather than paraphrased.
///
/// Copying it is the point. The front page block is `rust,ignore`, so nothing
/// compiles it, and it is the single most-read piece of code in the project. If
/// a signature moves under it, this fixture stops compiling and the README is
/// wrong in the same commit rather than a release later.
#[module]
pub struct Mlp {
    fc1: Linear<s![784, 128]>,
    fc2: Linear<s![128, 10]>,
}

impl Mlp {
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::build(())?,
            fc2: Linear::build(())?,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<s![dyn, 784]>,
    ) -> Result<Tensor<s![dyn, 10], DefaultBackend, f32, Grad>> {
        Ok(self.fc2.forward(self.fc1.forward(x)?.relu()?)?)
    }
}

/// A batch whose values depend on the row, so a layer that ignores its input
/// cannot produce the descent the training test asserts.
#[cfg(test)]
fn batch(rows: usize) -> Result<Tensor<s![dyn, 784], DefaultBackend>> {
    let values: Vec<f32> = (0..rows * 784)
        .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
        .collect();
    Tensor::from_slice(&values, (rows, ()))
}

/// Alternating labels, so neither class can be learned by a constant.
#[cfg(test)]
fn labels(rows: usize) -> Result<Tensor<Dyn, DefaultBackend, u32>> {
    let values: Vec<u32> = (0..rows).map(|row| (row % 2) as u32).collect();
    Tensor::from_slice(&values, vec![rows])
}

/// The published crate still resolves, links, and runs a forward pass.
#[test]
fn the_readme_model_runs_a_forward_pass() -> Result<()> {
    let model = Mlp::new()?;
    let logits = model.forward(batch(4)?)?;
    assert_eq!(logits.dims(), vec![4, 10]);
    Ok(())
}

/// Autograd against a closed form, so a tape that runs but computes the wrong
/// thing is a failure rather than a pass.
///
/// `d/dx (3 * sum(x))` is three at every position, whatever `x` holds.
#[test]
fn autograd_matches_a_closed_form_gradient() -> Result<()> {
    let x: Tensor<s![2, 3], DefaultBackend> = Tensor::ones(())?;
    let x = x.require_grad();
    let y = x.mul_scalar(3.0)?.sum_all()?;
    let gradients = y.backward()?;
    let gx: Vec<f32> = gradients.require(&x)?.to_vec1()?;
    assert_eq!(gx, vec![3.0; 6]);
    Ok(())
}

/// A loop of forward, loss, backward, step that has to move the loss down.
///
/// The assertion is descent rather than a threshold. A fixed target would
/// encode this machine's arithmetic and would have to be revisited whenever an
/// initializer or a reduction changed; "thirty steps of SGD on a separable
/// batch reduced the loss" is the property that would actually be broken by a
/// released tape that fails to connect, an optimizer that steps the wrong sign,
/// or a loss that returns a constant.
#[test]
fn thirty_sgd_steps_reduce_the_loss() -> Result<()> {
    let rows = 8;
    let model = Mlp::new()?;
    let mut sgd = SGD::<DefaultBackend>::from_module(&model, 0.05)?;
    let loss_fn = CrossEntropyLoss::new();

    let mut history = Vec::new();
    for _ in 0..30 {
        let logits = model.forward(batch(rows)?)?;
        let loss = loss_fn.forward(&logits, &labels(rows)?)?;
        let value: f32 = loss.to_vec1()?[0];
        assert!(value.is_finite(), "the loss left the reals: {value}");
        history.push(value);
        sgd.step(&loss.backward()?)?;
    }

    let first = history[0];
    let last = history[history.len() - 1];
    assert!(
        last < first,
        "thirty SGD steps did not reduce the loss: {first} -> {last}"
    );
    Ok(())
}

/// A checkpoint written and read back reproduces the logits exactly.
///
/// Exactly, not approximately. Saving and loading moves bytes; it does no
/// arithmetic, so any drift at all is a defect in the format rather than
/// floating-point noise, and an epsilon here would hide it.
#[test]
fn a_safetensors_round_trip_reproduces_the_logits() -> Result<()> {
    let model = Mlp::new()?;
    let before: Vec<f32> = model.forward(batch(4)?)?.to_vec1()?;

    let path = std::env::temp_dir().join("incin-released-consumer.safetensors");
    model.save(Format::Safetensors, &path)?;

    let mut restored = Mlp::new()?;
    let untrained: Vec<f32> = restored.forward(batch(4)?)?.to_vec1()?;
    restored.load(Format::Safetensors, &path)?;
    let after: Vec<f32> = restored.forward(batch(4)?)?.to_vec1()?;

    let _ = std::fs::remove_file(&path);

    // The second model is randomly initialized, so it disagrees before the
    // load. Without this the test would pass against a `load` that did nothing
    // whenever two initializations happened to coincide, and would pass always
    // against a `save` that wrote a file nobody read.
    assert_ne!(
        untrained, before,
        "two fresh initializations agreed, so the round trip proves nothing"
    );
    assert_eq!(after, before, "the checkpoint round trip changed the logits");
    Ok(())
}
