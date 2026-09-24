//! A complete supervised training loop in one file: a fixed synthetic
//! regression dataset, a single linear layer, and twenty gradient steps -
//! forward, loss, backward, optimizer step - with nothing hidden behind a
//! trainer abstraction. Every line below is the actual API you would call
//! to train a real model; MNIST in this directory is the same loop with
//! batches, a larger network, and cross-entropy.
//!
//! The dataset is deterministic on purpose: the coefficients the model
//! recovers are printed next to the ones that generated the data, so a
//! descending loss line is not just a printout but an actual claim about
//! what the optimizer did. One honest asymmetry to note: there is no
//! `zero_grad()` anywhere, because `backward()` constructs a fresh
//! `Gradients` value each call - gradients live per-step, not on the
//! module, so nothing can leak between iterations.
//!
//! Run with: `cargo run -p incin --example training_loop_from_scratch --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]
#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin::prelude::*;
use incin::state::{StatePath, collect_state};

type Backend = DefaultBackend;

// `s![...]` shapes are written as literals below; these consts drive the
// data loop, and `from_slice` validates that the two agree.
const SAMPLES: usize = 64;
const FEATURES: usize = 4;
const EPOCHS: usize = 20;
const LEARNING_RATE: f64 = 0.3;

/// The regression target the data was generated from: one weight per
/// feature plus a bias. The model starts ignorant of all five numbers.
const TRUE_WEIGHTS: [f32; FEATURES] = [0.5, -1.5, 2.0, -0.75];
const TRUE_BIAS: f32 = 0.25;

fn main() -> incin::Result<()> {
    section("1. A fixed dataset: x @ w_true + b_true");
    let (features, targets) = make_dataset();

    section("2. Twenty steps of forward / mse_loss / backward / step");
    let model = Linear::<s![4, 1], Backend>::build(())?;
    let mut optim = SGD::<Backend>::from_module(&model, LEARNING_RATE)?;
    for epoch in 1..=EPOCHS {
        let predicted = model.forward(features.clone())?;
        let loss = predicted.mse_loss(&targets)?;
        // A new `Gradients` value every iteration - there is nothing to
        // zero, and no state on the module to clear between steps.
        let grads = loss.backward()?;
        optim.step(&grads)?;
        println!(
            "  epoch {epoch:>2}: mse = {:.6}{}",
            loss.to_scalar::<f32>()?,
            if epoch == 1 { "   (untrained)" } else { "" }
        );
    }

    section("3. The recovered coefficients, next to the generating ones");
    let snapshot = collect_state::<Backend, _>(&model)?;
    let weight_bytes = snapshot
        .get(&StatePath::new("weight")?)
        .expect("the layer's weight is a state leaf")
        .bytes();
    let learned_weights: Vec<f32> = weight_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let bias_bytes = snapshot
        .get(&StatePath::new("bias")?)
        .expect("the layer's bias is a state leaf")
        .bytes();
    let learned_bias =
        f32::from_le_bytes([bias_bytes[0], bias_bytes[1], bias_bytes[2], bias_bytes[3]]);
    for (feature, (expected, actual)) in TRUE_WEIGHTS.iter().zip(&learned_weights).enumerate() {
        println!("  w[{feature}]: true {expected:>6.3}   learned {actual:>7.4}");
    }
    println!("  bias : true {TRUE_BIAS:>6.3}   learned {learned_bias:>7.4}");
    println!(
        "  max |error| over the five coefficients = {:.4}",
        TRUE_WEIGHTS
            .iter()
            .zip(&learned_weights)
            .map(|(expected, actual)| (expected - actual).abs())
            .chain([(TRUE_BIAS - learned_bias).abs()])
            .fold(0.0f32, f32::max)
    );

    Ok(())
}

/// Builds the training set deterministically: one sinusoid per feature
/// with a distinct frequency, targets from the exact linear rule and no
/// noise. The independent columns matter as much as the noise-free target:
/// collinear features admit infinitely many exact coefficient sets, so a
/// perfect loss would prove nothing about which one was learned.
fn make_dataset() -> (Tensor<s![64, 4], Backend>, Tensor<s![64, 1], Backend>) {
    let mut feature_values = Vec::with_capacity(SAMPLES * FEATURES);
    for sample in 0..SAMPLES {
        for feature in 0..FEATURES {
            let phase = (sample + 1) as f32 * (feature + 1) as f32 * 0.37;
            feature_values.push(phase.sin());
        }
    }
    let target_values: Vec<f32> = feature_values
        .chunks_exact(FEATURES)
        .map(|row| {
            row.iter()
                .zip(TRUE_WEIGHTS)
                .map(|(value, weight)| value * weight)
                .sum::<f32>()
                + TRUE_BIAS
        })
        .collect();

    let features = Tensor::<s![64, 4], Backend>::from_slice(&feature_values, ())
        .expect("the feature block has the exact static shape");
    let targets = Tensor::<s![64, 1], Backend>::from_slice(&target_values, ())
        .expect("the target block has the exact static shape");
    (features, targets)
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
