//! Example: three optimizers race on one quadratic bowl, plus a
//! per-parameter learning-rate pin.
//!
//! A `[4, 1]` linear model fits fixed targets with SGD (momentum), Adam
//! and RMSprop side by side; a fourth run pins its weight at lr 0 and
//! must leave it frozen while the bias trains. Loss traces print per
//! step; every run must descend.
//!
//! Run it with `cargo run -p incin --example optimizer_showdown`.

use incin::prelude::*;

fn main() -> Result<()> {
    let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.7).collect();
    let y_data: Vec<f32> = (0..4).map(|i| i as f32 * 0.05 - 0.2).collect();
    let x = Tensor::<Dyn, DefaultBackend>::from_slice(&x_data, vec![4, 4])?;
    let y = Tensor::<Dyn, DefaultBackend>::from_slice(&y_data, vec![4, 1])?;

    let sgd_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let mut sgd = SGD::<DefaultBackend>::from_module(&sgd_model, 0.01)?;
    sgd.momentum = 0.9;
    let mut sgd_losses = Vec::new();
    for _ in 0..10 {
        let pred = sgd_model.forward(x.clone().require_grad())?;
        let loss = MSELoss::new().forward(&pred, &y)?;
        sgd_losses.push(loss.to_scalar::<f32>()? as f64);
        let grads = loss.backward()?;
        sgd.step(&grads)?;
    }

    let adam_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let mut adam = Adam::<DefaultBackend>::from_module(&adam_model, 0.05)?;
    let mut adam_losses = Vec::new();
    for _ in 0..10 {
        let pred = adam_model.forward(x.clone().require_grad())?;
        let loss = MSELoss::new().forward(&pred, &y)?;
        adam_losses.push(loss.to_scalar::<f32>()? as f64);
        let grads = loss.backward()?;
        adam.step(&grads)?;
    }

    let rms_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let mut rms = RMSprop::<DefaultBackend>::from_module(&rms_model, 0.02)?;
    let mut rms_losses = Vec::new();
    for _ in 0..10 {
        let pred = rms_model.forward(x.clone().require_grad())?;
        let loss = MSELoss::new().forward(&pred, &y)?;
        rms_losses.push(loss.to_scalar::<f32>()? as f64);
        let grads = loss.backward()?;
        rms.step(&grads)?;
    }

    println!("step  sgd+momentum   adam          rmsprop");
    for i in 0..10 {
        println!(
            "{i:<4}  {:<13.6} {:<13.6} {:.6}",
            sgd_losses[i], adam_losses[i], rms_losses[i]
        );
    }
    for (name, trace) in [
        ("sgd", &sgd_losses),
        ("adam", &adam_losses),
        ("rmsprop", &rms_losses),
    ] {
        assert!(
            trace[9] < trace[0],
            "{name} must descend: {} -> {}",
            trace[0],
            trace[9]
        );
    }

    // Per-parameter pin: freeze the weight, train the bias.
    let pinned = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let w_before = pinned.weight.as_tensor()?.to_vec1::<f32>()?;
    let mut frozen = SGD::<DefaultBackend>::from_module(&pinned, 0.05)?;
    frozen.set_param_lr("weight", 0.0);
    assert_eq!(frozen.lr_for("weight"), 0.0);
    assert_eq!(frozen.lr_for("bias"), 0.05);
    let pred = pinned.forward(x.clone().require_grad())?;
    let loss = MSELoss::new().forward(&pred, &y)?;
    frozen.step(&loss.backward()?)?;
    let w_after = pinned.weight.as_tensor()?.to_vec1::<f32>()?;
    assert_eq!(w_before, w_after, "pinned weight must not move");

    println!("PASS: all three optimizers descend; the pinned weight froze");
    Ok(())
}
