//! Example: train, checkpoint, resume, and prove the resumed run is
//! bit-identical to an uninterrupted one.
//!
//! A `[4, 1]` linear model trains 3 steps, writes one envelope file
//! (model + optimizer + scheduler + epoch), then a fresh model and
//! optimizer load it back and train 2 more steps. A reference run trains
//! 5 steps straight from identical initial weights; both parameter
//! snapshots must match exactly.
//!
//! Run it with `cargo run -p incin --example checkpoint_resume`.
#![allow(clippy::type_complexity)]

use incin::nn::VisitParameters;
use incin::prelude::*;
use incin::state::{collect_state, load_state};
use incin_core::serialization::{
    SchedulerState, TrainingCheckpoint, load_training_checkpoint, optimizer_tensors_to_snapshot,
    save_training_checkpoint, snapshot_to_optimizer_tensors,
};
use std::collections::BTreeMap;

fn train_steps(
    model: &Linear<s![4, 1], DefaultBackend>,
    optim: &mut SGD<DefaultBackend>,
    x: &Tensor<Dyn, DefaultBackend>,
    y: &Tensor<Dyn, DefaultBackend>,
    steps: usize,
) -> Vec<f64> {
    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let pred = model.forward(x.clone().require_grad()).unwrap();
        let loss = MSELoss::new().forward(&pred, y).unwrap();
        losses.push(loss.to_scalar::<f32>().unwrap() as f64);
        let grads = loss.backward().unwrap();
        optim.step(&grads).unwrap();
    }
    losses
}

fn fresh_optimizer<M>(model: &M) -> SGD<DefaultBackend>
where
    M: VisitParameters<DefaultBackend>,
{
    let mut optim = SGD::<DefaultBackend>::from_module(model, 0.05).unwrap();
    optim.momentum = 0.9;
    optim
}

fn main() -> Result<()> {
    let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.7).collect();
    let y_data: Vec<f32> = (0..4).map(|i| i as f32 * 0.05 - 0.2).collect();
    let x = Tensor::<Dyn, DefaultBackend>::from_slice(&x_data, vec![4, 4])?;
    let y = Tensor::<Dyn, DefaultBackend>::from_slice(&y_data, vec![4, 1])?;

    // Identical initial weights for both runs.
    let proto = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let init = collect_state::<DefaultBackend, _>(&proto)?;

    // Reference: 5 uninterrupted steps.
    let mut ref_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    load_state::<DefaultBackend, _>(&mut ref_model, &init)?;
    let mut ref_optim = fresh_optimizer(&ref_model);
    let ref_losses = train_steps(&ref_model, &mut ref_optim, &x, &y, 5);
    let reference = collect_state::<DefaultBackend, _>(&ref_model)?;

    // Interrupted run: 3 steps, checkpoint, fresh objects, 2 more steps.
    let mut int_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    load_state::<DefaultBackend, _>(&mut int_model, &init)?;
    let mut int_optim = fresh_optimizer(&int_model);
    let mut int_losses = train_steps(&int_model, &mut int_optim, &x, &y, 3);

    let path = std::env::temp_dir().join("incin-checkpoint-resume.bin");
    let mut dict = BTreeMap::new();
    int_optim.state_dict("sgd", &mut dict)?;
    save_training_checkpoint(
        &path,
        &TrainingCheckpoint::new(
            3,
            SchedulerState {
                kind: "manual".to_string(),
                steps: 3,
                last_lr: Some(0.05),
            },
            collect_state::<DefaultBackend, _>(&int_model)?,
            optimizer_tensors_to_snapshot(&dict)?,
        ),
    )?;

    let mut res_model = Linear::<s![4, 1], DefaultBackend>::build(())?;
    let mut res_optim = fresh_optimizer(&res_model);
    let loaded = load_training_checkpoint(&path)?;
    assert_eq!(loaded.epoch, 3, "epoch travels with the envelope");
    load_state::<DefaultBackend, _>(&mut res_model, &loaded.model)?;
    let res_dict = snapshot_to_optimizer_tensors(&loaded.optimizer, &DeviceId::cpu())?;
    res_optim.load_state_dict("sgd", &res_dict)?;
    int_losses.extend(train_steps(&res_model, &mut res_optim, &x, &y, 2));
    let resumed = collect_state::<DefaultBackend, _>(&res_model)?;
    let _ = std::fs::remove_file(&path);

    println!("reference losses: {ref_losses:.6?}");
    println!("resumed   losses: {int_losses:.6?}");
    assert_eq!(resumed, reference, "resumed run must match bit-for-bit");
    println!("PASS: checkpoint at epoch 3 resumes to identical parameters");
    Ok(())
}
