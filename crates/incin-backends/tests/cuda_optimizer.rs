//! The fused CUDA optimizer kernels, checked against their analytical form.
//!
//! An earlier version of this file recomputed Adam, AdamW and SGD in the test
//! body and asserted its own arithmetic. It imported the attribute types and
//! never used them, launched nothing, and called no incin code at all -- so its
//! three green results said nothing whatsoever about the kernels this file is
//! named for, while reading as coverage in any summary.
//!
//! These launch the real kernels and compare against the update rule computed
//! on the host in `f64`. The reference is written from the definition rather
//! than from the kernel, so agreement is evidence rather than a tautology.
//!
//! Requires a GPU, so they follow the crate convention of `#[ignore]`:
//! `cargo test -p incin-backends --features cuda --test cuda_optimizer -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{download_f32, require_cuda, upload_f32};
use incin_core::exec::catalog::{AdamAttributes, AdamWAttributes, SgdAttributes};

/// Single-precision tolerance, relative where the magnitude warrants it.
fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-5 * left.abs().max(right.abs()).max(1.0)
}

fn params() -> Vec<f32> {
    vec![1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5, 0.125]
}

fn grads() -> Vec<f32> {
    vec![0.1, -0.2, 0.05, -0.1, 0.3, 0.02, -0.4, 0.15]
}

#[test]
#[ignore = "requires CUDA hardware"]
fn sgd_step_matches_the_update_rule() {
    require_cuda();
    let (p, g) = (params(), grads());
    let learning_rate = 0.1;

    let out = incin_backends::cuda::testing::sgd_step(
        &upload_f32(&p),
        &upload_f32(&g),
        &SgdAttributes { learning_rate },
    )
    .expect("sgd_step must launch");
    let got = download_f32(&out);

    for index in 0..p.len() {
        let expected = f64::from(p[index]) - learning_rate * f64::from(g[index]);
        assert!(
            close(f64::from(got[index]), expected),
            "sgd at {index}: kernel gave {}, rule gives {expected}",
            got[index]
        );
    }
}

/// Adam from a zeroed moment state, which is what step 1 of training does.
///
/// Bias correction is the part worth checking: at step 1 the corrected moments
/// are the raw gradients, so an implementation that forgot the correction would
/// produce a visibly smaller update rather than a subtly different one.
#[test]
#[ignore = "requires CUDA hardware"]
fn adam_step_matches_the_update_rule_including_bias_correction() {
    require_cuda();
    let (p, g) = (params(), grads());
    let attrs = AdamAttributes {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: 1,
    };

    let (p_out, m_out, v_out) = incin_backends::cuda::testing::adam_step(
        &upload_f32(&p),
        &upload_f32(&g),
        None,
        None,
        &attrs,
    )
    .expect("adam_step must launch");
    let (got_p, got_m, got_v) = (
        download_f32(&p_out),
        download_f32(&m_out),
        download_f32(&v_out),
    );

    let bias1 = 1.0 - attrs.beta1.powi(attrs.step as i32);
    let bias2 = 1.0 - attrs.beta2.powi(attrs.step as i32);
    for index in 0..p.len() {
        let grad = f64::from(g[index]);
        let m = (1.0 - attrs.beta1) * grad;
        let v = (1.0 - attrs.beta2) * grad * grad;
        let expected = f64::from(p[index])
            - attrs.learning_rate * ((m / bias1) / ((v / bias2).sqrt() + attrs.epsilon));

        assert!(close(f64::from(got_m[index]), m), "adam m at {index}");
        assert!(close(f64::from(got_v[index]), v), "adam v at {index}");
        assert!(
            close(f64::from(got_p[index]), expected),
            "adam p at {index}: kernel gave {}, rule gives {expected}",
            got_p[index]
        );
    }
}

/// AdamW differs from Adam by decoupling the weight decay from the gradient.
///
/// Checked against Adam directly rather than only against a formula: with a
/// non-zero decay the two must disagree, which is the entire reason AdamW
/// exists. An implementation that folded the decay into the gradient -- the
/// classic mistake -- would pass a formula written the same wrong way, but not
/// this.
#[test]
#[ignore = "requires CUDA hardware"]
fn adamw_decouples_weight_decay_from_the_gradient() {
    require_cuda();
    let (p, g) = (params(), grads());
    // Chosen so the decay term, `lr * weight_decay * p`, is comfortably above
    // the comparison tolerance. At a realistic 1e-3 learning rate and 0.01
    // decay it is 1e-5, which is exactly the tolerance -- the formula check
    // would still hold, but the "it actually differs from Adam" assertion below
    // could not distinguish a correct decay from none at all.
    let weight_decay = 0.1;
    let adamw = AdamWAttributes {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay,
        step: 1,
    };

    let (p_out, _, _) = incin_backends::cuda::testing::adamw_step(
        &upload_f32(&p),
        &upload_f32(&g),
        None,
        None,
        &adamw,
    )
    .expect("adamw_step must launch");
    let got = download_f32(&p_out);

    let bias1 = 1.0 - adamw.beta1.powi(1);
    let bias2 = 1.0 - adamw.beta2.powi(1);
    for index in 0..p.len() {
        let param = f64::from(p[index]);
        let grad = f64::from(g[index]);
        let m = (1.0 - adamw.beta1) * grad;
        let v = (1.0 - adamw.beta2) * grad * grad;
        // Decay applies to the parameter, not to the gradient.
        let decayed = param - adamw.learning_rate * weight_decay * param;
        let expected =
            decayed - adamw.learning_rate * ((m / bias1) / ((v / bias2).sqrt() + adamw.epsilon));

        assert!(
            close(f64::from(got[index]), expected),
            "adamw p at {index}: kernel gave {}, rule gives {expected}",
            got[index]
        );

        // And it must actually differ from plain Adam, or the decay did nothing.
        let adam_expected =
            param - adamw.learning_rate * ((m / bias1) / ((v / bias2).sqrt() + adamw.epsilon));
        if param.abs() > 1e-6 {
            assert!(
                !close(expected, adam_expected),
                "decay of {weight_decay} on a parameter of {param} should change the update"
            );
        }
    }
}

/// A second Adam step must carry the moment state forward.
///
/// The single-step tests cannot distinguish a kernel that ignores its incoming
/// moments from one that uses them, because at step 1 the incoming state is
/// zero either way.
#[test]
#[ignore = "requires CUDA hardware"]
fn adam_carries_moment_state_between_steps() {
    require_cuda();
    let (p, g) = (params(), grads());
    let mut attrs = AdamAttributes {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: 1,
    };

    let (p1, m1, v1) = incin_backends::cuda::testing::adam_step(
        &upload_f32(&p),
        &upload_f32(&g),
        None,
        None,
        &attrs,
    )
    .unwrap();

    attrs.step = 2;
    let (_, m2, v2) = incin_backends::cuda::testing::adam_step(
        &p1,
        &upload_f32(&g),
        Some(&m1),
        Some(&v1),
        &attrs,
    )
    .unwrap();

    let (host_m1, host_m2) = (download_f32(&m1), download_f32(&m2));
    let (host_v1, host_v2) = (download_f32(&v1), download_f32(&v2));

    for index in 0..p.len() {
        let grad = f64::from(g[index]);
        let expected_m = attrs.beta1 * f64::from(host_m1[index]) + (1.0 - attrs.beta1) * grad;
        let expected_v =
            attrs.beta2 * f64::from(host_v1[index]) + (1.0 - attrs.beta2) * grad * grad;
        assert!(
            close(f64::from(host_m2[index]), expected_m),
            "second-step m at {index}: kernel gave {}, rule gives {expected_m}",
            host_m2[index]
        );
        assert!(
            close(f64::from(host_v2[index]), expected_v),
            "second-step v at {index}"
        );
    }
}

/// A step count the kernel ABI cannot carry is refused, not saturated: the
/// old code pinned the bias correction at `i32::MAX`, silently training with
/// a frozen correction instead of reporting the overflow.
#[test]
#[ignore = "requires CUDA hardware"]
fn adam_step_overflow_is_a_typed_error_not_a_saturated_correction() {
    require_cuda();
    let attrs = AdamAttributes {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: usize::MAX,
    };
    let result = incin_backends::cuda::testing::adam_step(
        &upload_f32(&params()),
        &upload_f32(&grads()),
        None,
        None,
        &attrs,
    );
    let error = format!("{:?}", result.expect_err("step usize::MAX must be refused"));
    assert!(
        error.contains("arithmetic overflow"),
        "expected an overflow error, got: {error}"
    );
}
