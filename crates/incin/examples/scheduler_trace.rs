//! Example: learning-rate schedules traced step by step.
//!
//! A linear warmup ramp, a cosine-with-warmup curve and a step decay print
//! their first values so you can see the schedule shape; key positions are
//! asserted against the closed form.
//!
//! Run it with `cargo run -p incin --example scheduler_trace`.

use incin::prelude::*;

fn main() -> Result<()> {
    // Linear warmup over 4 steps to 0.2: 0.0, 0.05, 0.10, 0.15, then flat.
    let mut warmup = LinearWarmup::new(0.2, 4);
    let mut ramp = Vec::new();
    for _ in 0..6 {
        ramp.push(warmup.get_lr());
        warmup.step();
    }
    println!("linear warmup: {ramp:.4?}");
    assert_eq!(ramp[0], 0.0, "warmup starts at zero");
    assert!((ramp[1] - 0.05).abs() < 1e-12);
    assert!((ramp[4] - 0.2).abs() < 1e-12, "ramp reaches base at step 4");
    assert_eq!(ramp[5], 0.2, "warmup holds the base rate after the ramp");

    // Cosine with warmup: ramps like above, then decays toward min_lr.
    let mut cosine = CosineWithWarmup::new(0.2, 0.0, 4, 14);
    let mut curve = Vec::new();
    for _ in 0..15 {
        curve.push(cosine.get_lr());
        cosine.step();
    }
    println!(
        "cosine+warmup first/last: {:.4?} ... {:.4?}",
        &curve[..5],
        &curve[12..]
    );
    assert_eq!(curve[0], 0.0);
    assert!(curve[14] < curve[5], "cosine must decay after warmup");
    assert!(curve[14] >= 0.0, "cosine never goes negative");

    // Step decay: 0.1, gamma 0.5 every 3 steps.
    let mut step = StepLR::new(0.1, 3, 0.5);
    let mut decay = Vec::new();
    for _ in 0..7 {
        decay.push(step.get_lr());
        step.step();
    }
    println!("step decay: {decay:.4?}");
    assert_eq!(decay[0], 0.1);
    assert!((decay[3] - 0.05).abs() < 1e-12, "first decay at step 3");
    assert!((decay[6] - 0.025).abs() < 1e-12, "second decay at step 6");

    println!("PASS: warmup ramps, cosine decays, step decays on schedule");
    Ok(())
}
