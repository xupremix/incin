use incin::experimental::training::Trainer;
use incin::prelude::*;
use incin_core::backend_authoring::{AutogradBackend, HostReadback};

type Backend = DefaultBackend;
type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
fn test_static_loss_scaling_numeric_parity() -> TestResult {
    let model = Linear::<s![4, 2], Backend>::build(())?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;

    let input =
        Tensor::<s![2, 4], Backend>::from_slice(&[1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], ())?;

    // 1. Measure unscaled analytical gradient first
    let weight_initial = model.weight.as_tensor()?.to_vec1::<f32>()?;
    let out_u = model.forward(input.clone())?;
    let loss_u = out_u.sum_all()?;
    let grads_u = loss_u.backward()?;

    let grad_w_storage =
        Backend::get_grad::<f32>(model.weight.as_tensor()?.inner(), grads_u.as_backend())?.unwrap();
    let grad_w_vec = Backend::float_to_vec1::<f32>(&grad_w_storage)?;

    // Expected weight after SGD update with lr = 0.05: W_expected = W_initial - 0.05 * Grad
    let expected_weights: Vec<f32> = weight_initial
        .iter()
        .zip(&grad_w_vec)
        .map(|(&w, &g)| w - 0.05 * (g as f32))
        .collect();

    // 2. Perform scaled backward pass (scale = 1024.0) and step_scaled
    let mut scaler = LossScaleState::new(LossScaling::Static(1024.0));
    assert_eq!(scaler.scale(), 1024.0);

    let out_s = model.forward(input)?;
    let loss_s_unscaled = out_s.sum_all()?;
    let loss_s_scaled = loss_s_unscaled.mul_scalar(1024.0)?;
    let mut grads_s = loss_s_scaled.backward()?;

    // Verify raw gradient before step_scaled is scaled by 1024x
    let raw_scaled_grad_storage =
        Backend::get_grad::<f32>(model.weight.as_tensor()?.inner(), grads_s.as_backend())?.unwrap();
    let raw_scaled_grad_vec = Backend::float_to_vec1::<f32>(&raw_scaled_grad_storage)?;
    for (g_raw, g_orig) in raw_scaled_grad_vec.iter().zip(&grad_w_vec) {
        assert!(
            (g_raw - g_orig * 1024.0).abs() < 1e-3,
            "Raw gradient must be scaled by 1024.0"
        );
    }

    // Step with scaler (unscales gradients and steps)
    let stepped = optimizer.step_scaled(&mut grads_s, &mut scaler)?;
    assert!(stepped, "step_scaled should succeed for finite gradients");

    // 3. Verify actual weight matches expected unscaled update
    let weight_after = model.weight.as_tensor()?.to_vec1::<f32>()?;
    for (actual, expected) in weight_after.iter().zip(&expected_weights) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Weight mismatch: actual={actual}, expected={expected}"
        );
    }

    Ok(())
}

#[test]
fn test_dynamic_loss_scaling_overflow_skip_and_backoff() -> TestResult {
    let model = Linear::<s![2, 2], Backend>::build(())?;
    let mut optimizer = AdamW::<Backend>::from_module(&model, 1e-3)?;
    let mut scaler = LossScaleState::new(LossScaling::dynamic(65536.0, 2.0, 0.5, 10));

    assert_eq!(scaler.scale(), 65536.0);
    assert_eq!(scaler.steps_since_last_overflow(), 0);

    let input = Tensor::<s![1, 2], Backend>::ones(())?;
    let loss = model.forward(input)?.sum_all()?;
    let mut grads = loss.backward()?;

    // Manually inject Infinity into a gradient handle to simulate overflow
    let weight_tensor = model.weight.as_tensor()?;
    let nan_storage = Tensor::<s![2, 2], Backend>::from_slice(&[f32::INFINITY, 0.0, 0.0, 0.0], ())?;
    <Backend as AutogradBackend>::set_grad::<f32>(
        weight_tensor.inner(),
        grads.as_backend_mut(),
        nan_storage.into_inner(),
    )?;

    // Snapshot parameters before step
    let weight_before = model.weight.as_tensor()?.to_vec1::<f32>()?;

    // Execute step_scaled - should detect overflow, back off scale, and skip update
    let stepped = optimizer.step_scaled(&mut grads, &mut scaler)?;

    assert!(
        !stepped,
        "step_scaled must return false on gradient overflow"
    );
    assert_eq!(scaler.scale(), 32768.0, "scale must back off by factor 0.5");
    assert_eq!(scaler.steps_since_last_overflow(), 0);

    // Parameters must be unchanged
    let weight_after = model.weight.as_tensor()?.to_vec1::<f32>()?;
    assert_eq!(
        weight_before, weight_after,
        "Parameters must not change when step is skipped"
    );

    Ok(())
}

#[test]
fn test_dynamic_loss_scaling_growth() -> TestResult {
    let model = Linear::<s![2, 2], Backend>::build(())?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 1e-2)?;
    // growth_interval = 3
    let mut scaler = LossScaleState::new(LossScaling::dynamic(1024.0, 2.0, 0.5, 3));

    assert_eq!(scaler.scale(), 1024.0);

    // Step 1
    let input = Tensor::<s![1, 2], Backend>::ones(())?;
    let loss = model.forward(input.clone())?.sum_all()?;
    let mut grads = loss.backward()?;
    assert!(optimizer.step_scaled(&mut grads, &mut scaler)?);
    assert_eq!(scaler.steps_since_last_overflow(), 1);
    assert_eq!(scaler.scale(), 1024.0);

    // Step 2
    let loss = model.forward(input.clone())?.sum_all()?;
    let mut grads = loss.backward()?;
    assert!(optimizer.step_scaled(&mut grads, &mut scaler)?);
    assert_eq!(scaler.steps_since_last_overflow(), 2);
    assert_eq!(scaler.scale(), 1024.0);

    // Step 3 -> threshold 3 reached, scale should double to 2048.0
    let loss = model.forward(input)?.sum_all()?;
    let mut grads = loss.backward()?;
    assert!(optimizer.step_scaled(&mut grads, &mut scaler)?);
    assert_eq!(scaler.steps_since_last_overflow(), 0);
    assert_eq!(
        scaler.scale(),
        2048.0,
        "scale must double after growth_interval steps"
    );

    Ok(())
}

#[test]
fn test_trainer_fit_scaled_integration() -> TestResult {
    let plan = Trainer::plan()
        .epochs(2)
        .loss_scaling(LossScaling::Static(512.0))
        .build()?;

    assert_eq!(plan.loss_scaling(), LossScaling::Static(512.0));
    assert!(plan.explain().contains("Static(512.0) policy configured"));

    let trainer = Trainer::new(plan);
    let mut model = Linear::<s![2, 2], Backend>::build(())?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 1e-2)?;
    let mut scaler = LossScaleState::new(trainer.report().loss_scaling());

    let dataset = vec![
        Tensor::<s![1, 2], Backend>::ones(())?,
        Tensor::<s![1, 2], Backend>::zeros(())?,
    ];

    let outcome = trainer.fit_scaled(
        &mut model,
        &mut optimizer,
        &mut scaler,
        dataset,
        |m, batch| m.forward(batch)?.sum_all(),
    )?;

    assert_eq!(outcome.epochs, 2);
    assert_eq!(outcome.batches, 4);
    assert!(outcome.final_loss.is_some());

    Ok(())
}

#[test]
fn verify_growth_boundary_unscales_by_the_pre_growth_scale() -> TestResult {
    // Regression check: `LossScaleState::unscale_and_update_vars` must unscale
    // gradients by the scale that was actually used to produce them (the
    // value *before* `update()` runs), not the post-growth value `update()`
    // may have just written. This isolates the exact step where dynamic
    // growth triggers and checks the applied SGD update against the analytic
    // expectation using the *pre-growth* scale.
    let model = Linear::<s![2, 2], Backend>::build(())?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 1e-2)?;
    let mut scaler = LossScaleState::new(LossScaling::dynamic(1024.0, 2.0, 0.5, 3));

    let input = Tensor::<s![1, 2], Backend>::ones(())?;

    // Steps 1-2: warm up steps_since_last_overflow to 2 (no growth yet).
    for _ in 0..2 {
        let loss = model.forward(input.clone())?.sum_all()?;
        let mut grads = loss.backward()?;
        assert!(optimizer.step_scaled(&mut grads, &mut scaler)?);
    }
    assert_eq!(scaler.scale(), 1024.0);
    assert_eq!(scaler.steps_since_last_overflow(), 2);

    // Step 3: growth triggers (scale becomes 2048.0), but the gradients
    // being unscaled *this step* were produced under scale = 1024.0. Two
    // separate forward/backward passes, mirroring
    // `test_static_loss_scaling_numeric_parity`'s pattern: a single
    // `backward()` call drains the (thread-local, single-use) tape, so the
    // analytic-baseline pass and the scaled pass cannot share one forward.
    let weight_before = model.weight.as_tensor()?.to_vec1::<f32>()?;
    let scale_used_for_this_step = scaler.scale(); // 1024.0, captured pre-step

    let loss_for_baseline = model.forward(input.clone())?.sum_all()?;
    let grads_for_analytic = loss_for_baseline.backward()?; // unscaled analytic grad
    let grad_w = Backend::get_grad::<f32>(
        model.weight.as_tensor()?.inner(),
        grads_for_analytic.as_backend(),
    )?
    .unwrap();
    let grad_w_vec = Backend::float_to_vec1::<f32>(&grad_w)?;
    let expected_weights: Vec<f32> = weight_before
        .iter()
        .zip(&grad_w_vec)
        .map(|(&w, &g)| w - 1e-2_f32 * (g as f32))
        .collect();

    let loss_scaled = model
        .forward(input)?
        .sum_all()?
        .mul_scalar(scale_used_for_this_step as f64)?;
    let mut grads_scaled = loss_scaled.backward()?;
    assert!(optimizer.step_scaled(&mut grads_scaled, &mut scaler)?);
    assert_eq!(
        scaler.scale(),
        2048.0,
        "growth should have triggered this step"
    );

    let weight_after = model.weight.as_tensor()?.to_vec1::<f32>()?;
    for (actual, expected) in weight_after.iter().zip(&expected_weights) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "growth-boundary step used the wrong unscale factor: actual={actual}, expected={expected}"
        );
    }

    Ok(())
}
