//! Example: what the framework does at the edges of the number line.
//!
//! `cargo run -p incin --example numerical_edge_cases`
//!
//! Every other example here stays comfortably inside the reals: finite
//! inputs, well-scaled targets, a loss that falls. That is the right way
//! to demonstrate a framework and the wrong way to find out what it does
//! at zero, at infinity, and at the boundary between them - which is where
//! a kernel's special-case handling lives, and where an optimization that
//! handles the common case faster can quietly change the uncommon one.
//!
//! The framework is explicit about which of those edges it guards and which
//! it leaves to IEEE 754, and this example pins both so that a change to
//! either is a visible diff rather than a surprise in someone else's
//! training run.
//!
//! - **Guarded: non-finite gradients.** `NanPolicy::Reject` (and the
//!   `check_gradients` shorthand) turns a non-finite gradient into a typed
//!   error naming the tensor and the recipe, instead of letting it into
//!   the optimizer. The default is the opposite - a plain `backward`
//!   returns the infinities - and that asymmetry is deliberate and worth
//!   stating, because a run that silently poisoned its weights looks
//!   exactly like a run with a bad learning rate.
//! - **Not guarded: the forward.** Division by zero, `log(0)`, `sqrt` of
//!   a negative and `inf - inf` all produce the IEEE answer and no error.
//!   A framework whose house rule is fail-closed refusals still lets value
//!   errors through as values, because a model that legitimately computes
//!   `log(0)` during a warmup should not be stopped by the type system.
//!   The cost is that a NaN has to be *looked for*; the guarded gradient
//!   pass is the cheap place to look.
//! - **Worth knowing: `x^0 == 1` even at `x == 0`,** and `(-1)^-1 == -1`.
//!   A power with a non-integer exponent follows the C library and yields
//!   `NaN` on a negative base, which is a different answer from the one
//!   `powf` on a strictly positive tensor gives.
//! - **Worth knowing: `softmax` and `log_softmax` are stable** in the
//!   presence of a very negative logit, because they subtract the maximum.
//!   An implementation that did not would return `NaN` for perfectly
//!   ordinary logits, and the check below is here so that it cannot start.

use incin::prelude::*;
// The scoped policy has no facade alias by design, so the example names the
// `incin_core` path - the same choice the book documents.
use incin_core::exec::policy::{NanPolicy, check_gradients};

type B = DefaultBackend;

/// The probe vector: one positive, one zero, one negative, one larger
/// positive. It contains a zero and a sign change, which is what most of
/// the edges below need.
fn probe() -> Result<Tensor<Dyn, B>> {
    Tensor::<Dyn, B>::from_slice(&[1.0, 0.0, -1.0, 2.0], vec![4])
}

/// Prints one row of the table. Written as a macro rather than a function
/// because the tensor's layout parameter differs between a fresh
/// `from_slice` and the result of a reduction, and a generic function would
/// have to name the placement type the facade prelude does not export.
macro_rules! values {
    ($name:literal, $expr:expr) => {
        match $expr {
            Ok(tensor) => match tensor.forget_layout().to_vec1::<f32>() {
                Ok(v) => println!("  {:<26} {v:?}", $name),
                Err(e) => println!("  {:<26} read failed: {}", $name, e),
            },
            Err(error) => println!(
                "  {:<26} refused: {}",
                $name,
                error.to_string().lines().next().unwrap_or("")
            ),
        }
    };
}

/// True where `got` is a NaN, so a case can assert *which* elements are
/// NaN rather than merely that the answer has one.
fn nan_mask(values: &[f32]) -> Vec<bool> {
    values.iter().map(|v| v.is_nan()).collect()
}

fn main() -> Result<()> {
    let a = probe()?;
    let zero = Tensor::<Dyn, B>::zeros(vec![4])?;

    println!("IEEE 754 at the edges - no error, the standard's answer:\n");
    // `1/0 = inf`, `-1/0 = -inf`, `0/0 = NaN`: the sign of the zero decides
    // the sign of the infinity, and only the indeterminate case is NaN.
    values!("a / 0", a.try_div(&zero));
    values!("0 / 0", zero.try_div(&zero));
    // `x/x` is 1 everywhere except at the zero, which is the whole of 0/0.
    values!("a / a", a.try_div(&a));
    values!("a.sqrt()", a.sqrt());
    values!("a.log()", a.log());
    values!("0.log()", zero.log());
    let a_inf = a.add_scalar(f64::INFINITY)?;
    values!("(a + inf) - inf", a_inf.sub_scalar(f64::INFINITY));

    println!("\npowers follow the C library, not the algebra textbook:\n");
    // `0^0 = 1` and `(-1)^0 = 1`: exponent zero short-circuits before the
    // base is examined, so a negative base is not an error here even
    // though `(-1)^0.5` is.
    values!("a.powf(0.0)", a.powf(0.0));
    // A negative base with a non-integer exponent is NaN, and so is `0^-1`.
    values!("a.powf(0.5)", a.powf(0.5));
    values!("a.powf(-1.0)", a.powf(-1.0));

    println!("\nreductions stay finite where the naive form would not:\n");
    // A zero row is not a division-by-zero hazard: the mean of zeros is
    // zero, and dividing *by* that mean is what produces the infinity, so
    // the reduction and the reciprocal are checked separately.
    values!("a.mean()", a.mean(0isize));
    values!("zero.mean()", zero.mean(0isize));
    values!("zero.var_all()", zero.var_all(false));
    values!("zero.std_all()", zero.std_all(false));
    let one = Tensor::<Dyn, B>::ones(vec![1])?;
    let mean = a.mean(0isize)?;
    values!("1 / a.mean()", one.try_div(&mean));
    // A mean over a row containing a NaN is a NaN: reductions propagate,
    // they do not skip. Worth stating because "skip NaN" is a reasonable
    // thing to want and is not what happens.
    let poisoned = Tensor::<Dyn, B>::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0], vec![4])?;
    let poisoned_inf = poisoned.add_scalar(f64::INFINITY)?;
    values!("mean with a NaN", poisoned.mean(0isize));
    values!("sum with an inf", poisoned_inf.sum(0isize));

    println!("\nsoftmax is stable, which is a property worth pinning:\n");
    // Without a max subtraction this is all NaN, and with one it is a
    // distribution: the exponentials of a very negative logit underflow to
    // zero, which is correct, not a failure.
    let extreme = Tensor::<Dyn, B>::from_slice(&[1000.0f32, -1000.0], vec![2])?;
    let soft = extreme.softmax(0isize)?.forget_layout().to_vec1::<f32>()?;
    let log_soft = extreme
        .log_softmax(0isize)?
        .forget_layout()
        .to_vec1::<f32>()?;
    println!("  softmax([1000, -1000])      {soft:?}");
    println!("  log_softmax([1000, -1000])  {log_soft:?}");
    assert!(
        soft.iter().all(|v| v.is_finite()) && soft.iter().all(|v| *v >= 0.0),
        "softmax of extreme logits must stay a finite distribution, got {soft:?}"
    );
    let total: f32 = soft.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax must sum to one, got {total}"
    );
    assert!(
        log_soft.iter().all(|v| v.is_finite()),
        "log_softmax of extreme logits must stay finite, got {log_soft:?}"
    );

    println!("\nthe guarded edge: non-finite gradients\n");
    // The default is fail-open on purpose. A plain backward through a
    // division by zero hands the optimizer infinities, and nothing complains:
    // the policy that rejects them is opt-in.
    let x = a.clone().require_grad();
    let poisoned_loss = x.try_div(&zero)?.sum_all()?;
    let gradients = poisoned_loss.backward()?;
    let unguarded = gradients.require(&x)?.forget_layout().to_vec1::<f32>()?;
    println!("  plain backward through x/0    {unguarded:?}");
    assert!(
        unguarded.iter().all(|v| v.is_infinite()),
        "d/dx of x/0 is 1/0, so every gradient is an infinity"
    );
    assert!(
        !nan_mask(&unguarded).iter().any(|v| *v),
        "and none of them is a NaN, because the derivative of x/0 is well defined"
    );

    // Under the policy the same run is a typed error naming the tensor and
    // the recipe, which is the whole point of the feature: the failure
    // arrives at the operation that produced it rather than at the
    // optimizer step twenty iterations later.
    // `check_gradients` installs the policy and returns the closure's own
    // result: the rejection arrives as the `backward` error itself, which is
    // the point - it names the tensor and the recipe rather than surfacing
    // later at the optimizer.
    let rejected = check_gradients(|| {
        x.try_div(&zero)
            .and_then(|quotient| quotient.sum_all())
            .and_then(|loss| loss.backward())
            .and_then(|gradients| {
                gradients
                    .require(&x)
                    .map(|t| t.forget_layout().to_vec1::<f32>())
            })
    });
    match &rejected {
        Ok(_) => panic!("a non-finite gradient passed NanPolicy::Reject"),
        Err(error) => {
            let text = error.to_string();
            println!("  NanPolicy::Reject            {text}");
            assert!(
                text.contains("not finite"),
                "the error should say what was wrong, got {text}"
            );
        }
    }

    // The same policy must not disturb a finite run, or it would be
    // unusable.
    let finite = check_gradients(|| {
        x.mul_scalar(3.0)
            .and_then(|scaled| scaled.sum_all())
            .and_then(|loss| loss.backward())
            .and_then(|gradients| {
                gradients
                    .require(&x)
                    .map(|t| t.forget_layout().to_vec1::<f32>())
            })
    })??;
    println!("  the same policy, finite loss  {finite:?}");
    assert_eq!(finite, vec![3.0, 3.0, 3.0, 3.0]);

    // The two axes are independent: the policy is about the backward pass,
    // so it rejects a poisoned gradient even when the forward was the one
    // that produced the non-finite value.
    let explicit = incin_core::exec::policy::ExecutionPolicy::current()
        .with_nan_policy(NanPolicy::Reject)
        .scope(|| {
            x.try_div(&zero)
                .and_then(|quotient| quotient.sum_all())
                .and_then(|loss| loss.backward())
                .and_then(|gradients| {
                    gradients
                        .require(&x)
                        .map(|t| t.forget_layout().to_vec1::<f32>())
                })
        });
    assert!(
        explicit.is_err(),
        "naming the policy directly must reject the same gradient"
    );
    assert!(
        incin_core::exec::policy::ExecutionPolicy::current().nan_policy != NanPolicy::Reject,
        "the scope must restore the policy it replaced"
    );

    println!("\nPASS: the edges behave as documented, and the guarded one is guarded");
    Ok(())
}
