//! Example: check gradients against central differences.
//!
//! `cargo run -p incin --example gradcheck_model`
//!
//! Every per-operation gradient test in this repository checks one
//! backward recipe against a hand-computed answer. That leaves the thing
//! that actually breaks in practice untested: the *composition*. Six
//! individually-correct recipes wired together in the wrong order, or with
//! one axis transposed, train to a loss that decreases and means nothing -
//! it just decreases a little more slowly than it should, which no "the
//! loss went down" threshold can catch.
//!
//! This example closes that gap with an oracle independent of the tape. A
//! central difference never asks the framework what the derivative is:
//!
//! ```text
//!     dL/dw_i  ~=  ( L(w + h*e_i) - L(w - h*e_i) ) / (2h)
//! ```
//!
//! It perturbs one weight element, re-runs the forward, and measures the
//! slope. If the analytic gradient is right the two agree to within the
//! step's own truncation and rounding error. If it is wrong - a transposed
//! operand, a dropped term, a sign, a constant factor - the central
//! difference is a stranger to that mistake, and the report names the
//! element that disagreed.
//!
//! The cost is two forward passes per weight element, so every case here
//! is deliberately tiny. What it buys is a check that no amount of
//! loss-descent evidence can stand in for.
//!
//! Each case below is a *composition* rather than a single operation,
//! because that is what composition bugs look like:
//!
//! - `matmul` -> `relu` -> `sum_all`, the shape of every MLP;
//! - `batch_norm` -> `relu` -> `sum_all`, where the batch norm reduces
//!   across the batch and the spatial axes at once and so is the one step
//!   in a convolutional network whose gradient is a reduction over its own
//!   input; the tracked quantity is its per-channel gain;
//! - a broadcast `mul` then `sum_all`, which is a reduction over an
//!   expanded axis - the step that gets the right gradient shape and the
//!   wrong values when it sums the wrong axis;
//! - `index_select` with a thrice-repeated index, whose gradient is a
//!   scatter: a last-write-wins scatter is invisible on a batch with no
//!   repeats and wrong on every real one;
//! - `matmul` -> `sub` -> `mul` -> `sum_all` with a broadcast operand on
//!   both sides, so a gradient that ignores which operand was shared
//!   cannot pass.

use incin::prelude::*;
// Two names the facade prelude does not carry, both of which a helper that
// takes or returns a scalar loss has to name. `Nil` is the rank-0 shape a
// loss is, and `backward()` is defined only on it - every example that
// calls `.backward()` does so inline and never has to spell the type, which
// is why the gap has stayed invisible. `GradMode::Disabled` is the
// documented way to keep a forward off the tape.
use incin_core::dist::Local;
use incin_core::exec::GradMode;
use incin_core::shapes::Nil;

/// A gradient-tracked, layout-erased `f32` tensor: the quantity a check
/// perturbs and asks the backward pass about.
///
/// The layout parameter is `Dyn` rather than the `RowMajor` the `Dense`
/// alias gives, because that is the type `Gradients::require` is declared
/// against; a tensor from `from_slice` arrives `RowMajor` and needs
/// `forget_layout` before the gradient API will take it.
type T = Tensor<Dyn, DefaultBackend, f32, Grad, Local, Dyn>;

/// A scalar loss. `backward()` is defined on exactly this shape, so a loss
/// is a rank-0 tensor and nothing else will do.
type Loss = Tensor<Nil, DefaultBackend, f32, Grad, Local, Dyn>;

/// Central-difference step. A central difference trades truncation error
/// (`h^2`) against rounding error (`1/h`); they balance near
/// `(6 * f32::EPSILON).cbrt()`, about `8.9e-3` at `f32`. This is a little
/// larger still, trading a little truncation for a lot less cancellation.
const STEP: f32 = 1e-3;

/// Relative error a comparison may show and still pass. Finite differences
/// over an `f32` forward are good to a few parts in `1e5`; `2e-2` leaves
/// room for a differently-ordered summation and is still nowhere near a
/// transposed operand, a dropped term or a factor of two.
const RELATIVE: f32 = 2e-2;

/// Below this absolute difference a comparison passes regardless. Where the
/// true gradient is zero the numeric estimate is pure rounding noise, and
/// dividing it by itself would report an error near one while both sides
/// correctly agree the derivative is nothing.
const FLOOR: f32 = 1e-4;

/// One element that disagreed.
struct Disagreement {
    index: usize,
    analytic: f64,
    numeric: f64,
    relative: f64,
}

/// What one case measured.
struct Report {
    name: &'static str,
    compared: usize,
    worst: f64,
    disagreements: Vec<Disagreement>,
}

impl Report {
    fn passed(&self) -> bool {
        self.disagreements.is_empty()
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.<50} {:>3} weights, worst relative error {:.2e}",
            self.name, self.compared, self.worst
        )?;
        for d in &self.disagreements {
            write!(
                f,
                "\n      element {}: analytic {:.8e}, numeric {:.8e}, relative {:.2e}",
                d.index, d.analytic, d.numeric, d.relative
            )?;
        }
        Ok(())
    }
}

/// Compares the analytic gradient of a scalar loss with respect to the
/// weight `w0` against a central difference of the same loss.
///
/// `loss` takes the weight as a tensor - tracked on the analytic pass, a
/// plain tensor on each numeric one - and returns a scalar. That is the
/// shape that makes the check independent: the numeric passes re-run the
/// framework's own forward and can only disagree with its backward if the
/// backward is wrong.
fn check<F>(name: &'static str, w0: &[f32], shape: Vec<usize>, loss: F) -> Result<Report>
where
    F: Fn(T) -> Result<Loss>,
{
    // `forget_layout` is what a created tensor needs before the gradient
    // API will take it: `Gradients::require` is declared against the
    // layout-erased `Tensor`, and a tensor from `from_slice` arrives
    // `RowMajor`.
    let weight = Dense::<Dyn, DefaultBackend>::from_slice(w0, shape.clone())?
        .forget_layout()
        .require_grad();
    // The handle clone shares storage, so the gradient the backward
    // produces is keyed to the tensor `require` is asked about below.
    let value = loss(weight.clone())?;
    let gradients = value.backward()?;
    let analytic = gradients.require(&weight)?.to_vec1::<f32>()?;
    assert_eq!(
        analytic.len(),
        w0.len(),
        "the backward produced one gradient element per weight element"
    );

    let mut worst = 0.0f64;
    let mut disagreements = Vec::new();
    for (index, &got) in analytic.iter().enumerate() {
        let mut up = w0.to_vec();
        up[index] += STEP;
        let mut down = w0.to_vec();
        down[index] -= STEP;
        let numeric = (f64::from(value_of(&loss, &up, shape.clone())?)
            - f64::from(value_of(&loss, &down, shape.clone())?))
            / (2.0 * f64::from(STEP));

        let absolute = (f64::from(got) - numeric).abs();
        if absolute < f64::from(FLOOR) {
            continue;
        }
        let denominator = f64::from(got).abs().max(numeric.abs()).max(1e-12);
        let relative = absolute / denominator;
        worst = worst.max(relative);
        if relative > f64::from(RELATIVE) {
            disagreements.push(Disagreement {
                index,
                analytic: f64::from(got),
                numeric,
                relative,
            });
        }
    }
    Ok(Report {
        name,
        compared: analytic.len(),
        worst,
        disagreements,
    })
}

/// Evaluates `loss` on an untracked weight and reads the scalar. This is
/// the one thing a numeric pass does differently: nothing is recorded, so
/// the two passes per element leave nothing behind on the tape. Without
/// the disabled scope a numeric pass would leave a graph on every call -
/// two per weight element, none of them ever walked.
fn value_of<F>(loss: &F, values: &[f32], shape: Vec<usize>) -> Result<f32>
where
    F: Fn(T) -> Result<Loss>,
{
    let tracked = Dense::<Dyn, DefaultBackend>::from_slice(values, shape)?
        .forget_layout()
        .require_grad();
    let scalar = GradMode::Disabled.scope(|| loss(tracked))?;
    scalar.to_scalar::<f32>()
}

/// Values that are neither constant nor symmetric, so a gradient that is
/// off by a sign, an axis or a factor shows up instead of cancelling.
fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + 3) % 11) as f32 * scale - 5.0 * scale)
        .collect()
}

fn main() -> Result<()> {
    type B = DefaultBackend;
    println!("central-difference gradient check: step {STEP}, relative {RELATIVE}\n");

    // Each case is reported as soon as it finishes, so a case that cannot
    // even produce a gradient names itself before the run stops.
    let mut failed = 0;
    macro_rules! checked {
        ($name:literal, $values:expr, $shape:expr, $loss:expr) => {{
            let report = check($name, $values, $shape, $loss)?;
            println!("  {report}");
            if !report.passed() {
                failed += 1;
            }
        }};
    }

    // ------------------------------------------------------------ matmul
    // The shape of every MLP, and the baseline: if this case fails, the
    // failure is in the most ordinary thing a user can write. The gradient
    // asked for is the one w.r.t. the *left* operand, which is the tracked
    // tensor, so the right one stays an untracked constant.
    {
        let w = Dense::<Dyn, B>::from_slice(&ramp(3 * 4, 0.3), vec![3, 4])?;
        let loss = |x: T| -> Result<Loss> { Ok(x.matmul(&w)?.relu()?.sum_all()?.forget_layout()) };
        checked!(
            "matmul -> relu -> sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            loss
        );
    }

    // ---------------------------------------------------------- batch norm
    // The most interesting backward in this repository, because it is the
    // only step whose gradient is a *reduction* over its own input: batch
    // norm normalizes across the batch and the spatial axes at once, so
    // dL/dgain sums over every element the layer ever saw, and a recipe
    // that drops the batch term produces a gradient of exactly the right
    // shape and a plausible-looking wrong value.
    //
    // The tracked tensor is the per-channel gain. The receiver and the
    // other four operands are tracked too because `batch_norm` declares
    // every operand at the receiver's gradient mark; their gradients are
    // not asked for.
    {
        let input = Dense::<Dyn, B>::from_slice(&ramp(2 * 3 * 2 * 2, 0.4), vec![2, 3, 2, 2])?
            .forget_layout()
            .require_grad();
        let loss = |gain: T| -> Result<Loss> {
            let zeros = Dense::<Dyn, B>::zeros(vec![3])?
                .forget_layout()
                .require_grad();
            let ones = Dense::<Dyn, B>::ones(vec![3])?
                .forget_layout()
                .require_grad();
            let bias = zeros.clone();
            Ok(input
                .batch_norm(&gain, &bias, &zeros, &ones, 1e-5)?
                .relu()?
                .sum_all()?
                .forget_layout())
        };
        checked!(
            "batch_norm (per-channel gain) -> relu -> sum_all",
            &ramp(3, 0.6),
            vec![3],
            loss
        );
    }

    // ------------------------------------------------ broadcast and reduce
    // A reduction over an expanded axis: the column is broadcast over eight
    // positions, so its gradient is the same three numbers summed eight
    // ways. Summing the wrong axis gives a gradient of the right shape and
    // the wrong values, which is exactly what this catches.
    {
        let rows = Dense::<Dyn, B>::from_slice(&ramp(2 * 3 * 4, 0.5), vec![2, 3, 4])?;
        let loss = |column: T| -> Result<Loss> {
            Ok(rows.broadcast_mul(&column)?.sum_all()?.forget_layout())
        };
        checked!("broadcast mul -> sum_all", &ramp(3, 0.5), vec![3, 1], loss);
    }

    // -------------------------------------------------------- index gather
    // The gradient of a gather is a scatter. This batch repeats index 1
    // three times, so a scatter that overwrites instead of accumulating
    // fails on the second and third occurrence rather than silently
    // agreeing on a batch of distinct indices.
    {
        let indices = Dense::<Dyn, B, i64>::from_slice(&[1i64, 1, 1, 3, 0], vec![5])?;
        let target = Dense::<Dyn, B>::from_slice(&ramp(5 * 3, 0.2), vec![5, 3])?;
        let loss = |table: T| -> Result<Loss> {
            Ok(table
                .index_select(0isize, &indices)?
                .mse_loss(&target)?
                .forget_layout())
        };
        checked!(
            "index_select (thrice-repeated index) -> mse_loss",
            &ramp(5 * 3, 0.4),
            vec![5, 3],
            loss
        );
    }

    // --------------------------------------------- broadcast after a matmul
    // A weight multiplied into a matmul's result, then reduced. The
    // gradient has to route back through a broadcast expansion and a matmul
    // at once, and a recipe that mishandles the expansion produces the
    // right shape and the wrong values.
    {
        let a = Dense::<Dyn, B>::from_slice(&ramp(2 * 3, 0.4), vec![2, 3])?;
        let b = Dense::<Dyn, B>::from_slice(&ramp(3 * 2, 0.3), vec![3, 2])?;
        // [2,3] @ [3,2] -> [2,2], stretched against a [2,1] weight.
        let loss = |w: T| -> Result<Loss> {
            Ok(a.matmul(&b)?.broadcast_mul(&w)?.sum_all()?.forget_layout())
        };
        checked!(
            "matmul -> broadcast mul -> sum_all",
            &ramp(2, 0.4),
            vec![2, 1],
            loss
        );
    }

    // ------------------------------------------------------------ summary
    if failed == 0 {
        println!("\nPASS: every composition agrees with the numeric derivative");
        return Ok(());
    }
    Err(incin::Error::Msg(format!(
        "{failed} composition(s) disagreed with the central difference"
    )))
}
