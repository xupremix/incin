//! Example: check many gradients against central differences at once.
//!
//! `cargo run -p incin --example gradcheck_ops`
//!
//! `gradcheck_model` checks a few *compositions* - the shapes of model a
//! user writes. This one is the breadth-first companion: a sweep over the
//! operation surface, one case per row, each checking the gradient of a
//! single tracked tensor against a central difference of the same
//! forward. The oracle is identical; only the shape of the question
//! differs.
//!
//! Why a sweep rather than more per-operation tests: a per-operation test
//! pins the operation someone thought to test, against an answer someone
//! wrote down. A sweep asks a *mechanical* question of every row - does
//! this gradient match the derivative of the function it claims to be the
//! derivative of - and a row that disagrees is a bug nobody had to think
//! to look for. The batch-norm defect fixed in `51123813` was found this
//! way, and it is the kind that no value test can see: the forward was
//! right, and the whole contribution of the layer was the gradient.
//!
//! Two facts about the framework shape every row here. A loss has to be
//! rank 0, because `backward()` is defined on exactly that shape - so a
//! row that reduces over an axis ends in `sum_all()`, and weights the
//! reduced result so the axis matters (a plain `sum(axis)` followed by
//! `sum_all` would be a total sum, whose gradient is 1 everywhere and
//! could not tell axis 0 from axis 1). And the tracked tensor is
//! layout-erased, because `Gradients` is keyed by storage identity, so the
//! value being perturbed and the value being differentiated have to be
//! constructed the same way.
//!
//! The rows are grouped by what they are most likely to get wrong:
//!
//! - **axis reductions**, where transposing the reduction axis produces a
//!   gradient of exactly the right shape and entirely the wrong values;
//! - **structural moves** (narrow, pad, expand, concat, stack, split,
//!   repeat, unfold, gather), where the backward has to undo the forward
//!   and the undo is a scatter - and a scatter that forgets an element
//!   loses a gradient silently;
//! - **elementwise derivatives**, where the chain rule is easy to get
//!   right for a function and easy to get wrong for its derivative;
//! - **whole-tensor reductions** (norm, variance, logsumexp), where the
//!   gradient flows back through the reduction itself;
//! - **norms and losses**, the two things every model ends in.
//!
//! Every row is cheap: two forward passes per element of a handful of
//! elements. If a row ever disagrees, the report names the element, and
//! that is the whole debugging loop.
//!
//! Two things about an oracle like this are worth stating, because a check
//! that cannot fail is worse than no check.
//!
//! **The tolerance is derived, not guessed.** The absolute floor comes from
//! the loss's own f32 roundoff amplified by the step, so it is a property of
//! the row rather than a constant chosen to make the output look tidy - see
//! `STEP` and `NOUN_ROUNDOFF`.
//!
//! **The row set was validated by breaking things on purpose.** A sweep that
//! has only ever agreed with the framework has not been shown to disagree
//! with anything. Two faults were injected into the CPU kernels and the
//! sweep was re-run:
//!
//! - `reverse_cumsum` in `cpu/ops/reduce/dim.rs` scanning forward instead of
//!   backward, which is the exact confusion between a prefix and a suffix
//!   scan that `cumsum`'s Jacobian turns on. One row of 47 failed, the
//!   others 46 stayed green, and the failing row named all four offending
//!   elements with a 175% relative error.
//! - `tape::unbroadcast` returning the cotangent unreduced - "forgot to sum
//!   the broadcast axes", which is the shape of the batch-norm affine defect
//!   fixed in `51123813`. That one failed loudly rather than numerically,
//!   because a gradient of the wrong shape cannot be compared elementwise.
//!
//! Worth noting what the second injection could *not* do: scaling or negating
//! a whole loss is not a fault this oracle can see, because both the analytic
//! and the numeric gradient scale with it. A uniform factor of two in a
//! backward recipe is invisible to any central difference - it is caught by
//! comparing against a loss value instead, or by a training run that behaves
//! like a learning rate twice as large as the one that was asked for.

use incin::prelude::*;
// The facade prelude does not export the rank-0 shape a loss has to be, the
// placement parameter every `Tensor` type spells, or the scoped gradient
// policy - so a helper that takes a loss has to name all three itself.
use incin_core::dist::Local;
use incin_core::exec::GradMode;
use incin_core::shapes::Nil;

type B = DefaultBackend;
/// A gradient-tracked, layout-erased tensor: the quantity a row perturbs.
type T = Tensor<Dyn, B, f32, Grad, Local, Dyn>;
/// A rank-0 loss. `backward()` is defined on exactly this shape.
type Loss = Tensor<Nil, B, f32, Grad, Local, Dyn>;

/// Central-difference step.
///
/// This is the number most worth getting right, and the first value used
/// here was wrong. A central difference carries two errors that pull in
/// opposite directions: truncation, which grows as `h^2`, and the
/// roundoff of the loss evaluation, which is amplified by `1/(2h)`. The
/// balance point is near `(3 * eps * |loss|)^(1/3)`, which for an f32 loss
/// of order 10 is about `1.5e-2`. The first version of this example used
/// `h = 1e-3`, ten times too small, and duly reported a `tanh` element as
/// failing by 5% - when both sides were `2e-3` and differed by `1e-4`,
/// which is the noise floor and not a defect. `1e-2` sits near the
/// balance point: the roundoff term falls to about `4e-5` while truncation
/// is still negligible even for `exp`, whose third derivative is the
/// largest in the sweep.
const STEP: f32 = 1e-2;

/// Relative tolerance. Loose enough to survive a differently-ordered
/// summation, tight enough that a transposed axis, a dropped term or a
/// factor of two is nowhere near it.
const RELATIVE: f32 = 3e-2;

/// How many times the loss's own roundoff a row is allowed to disagree by.
///
/// The absolute floor is derived per row rather than fixed, because the
/// smallest defensible absolute tolerance depends on the loss's magnitude
/// and on the step: `|f32::EPSILON * |loss|| / (2 * STEP)`. A fixed floor
/// would have to be either too tight for a large loss or too loose for a
/// small one, and either way the tolerance would be a guess dressed up as a
/// constant. This factor of 8 leaves room for a sum reordered between the
/// analytic and numeric paths - which is a difference in the last bits, not
/// in the answer.
const NOUN_ROUNDOFF: f32 = 8.0;

/// One element that disagreed.
struct Disagreement {
    index: usize,
    analytic: f64,
    numeric: f64,
    relative: f64,
}

struct Row {
    name: &'static str,
    worst: f64,
    worst_absolute: f64,
    floor: f64,
    compared: usize,
    disagreements: Vec<Disagreement>,
}

impl Row {
    fn passed(&self) -> bool {
        self.disagreements.is_empty()
    }
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "  {:.<46} {:>4} elements, worst {:.2e} rel / {:.2e} abs (floor {:.1e})",
            self.name, self.compared, self.worst, self.worst_absolute, self.floor
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

/// Builds the tracked tensor, runs the loss, and compares the analytic
/// gradient against a central difference of the same closure.
fn check<F>(name: &'static str, values: &[f32], shape: Vec<usize>, loss: F) -> Result<Row>
where
    F: Fn(T) -> Result<Loss>,
{
    let input = Tensor::<Dyn, B>::from_slice(values, shape.clone())?
        .forget_layout()
        .require_grad();
    // A handle clone shares the storage, so the gradient the backward
    // produces is keyed to the tensor asked about below.
    let value = loss(input.clone())?;
    let gradients = value.backward()?;
    let analytic = gradients.require(&input)?.to_vec1::<f32>()?;
    assert_eq!(
        analytic.len(),
        values.len(),
        "{name}: the backward produced one gradient element per input element"
    );

    // Two forward passes per element, untracked and with gradients
    // disabled, so neither leaves anything on the tape.
    let evaluate = |v: &[f32]| -> Result<f64> {
        let plain = Tensor::<Dyn, B>::from_slice(v, shape.clone())?
            .forget_layout()
            .require_grad();
        Ok(f64::from(
            GradMode::Disabled
                .scope(|| loss(plain))?
                .to_scalar::<f32>()?,
        ))
    };

    // The tolerance floor is this loss's own roundoff, amplified by the step
    // - see `NOUN_ROUNDOFF`. Everything is measured against the numeric
    // value, because that is the quantity being treated as the truth.
    let base = f64::from(loss(input.clone())?.to_scalar::<f32>()?);
    let floor = f64::from(NOUN_ROUNDOFF) * f64::from(f32::EPSILON) * base.abs().max(1.0)
        / (2.0 * f64::from(STEP));

    let mut worst = 0.0f64;
    let mut worst_absolute = 0.0f64;
    let mut disagreements = Vec::new();
    for (index, &got) in analytic.iter().enumerate() {
        let mut up = values.to_vec();
        up[index] += STEP;
        let mut down = values.to_vec();
        down[index] -= STEP;
        let numeric = (evaluate(&up)? - evaluate(&down)?) / (2.0 * f64::from(STEP));
        let absolute = (f64::from(got) - numeric).abs();
        let relative = absolute / numeric.abs().max(1e-12);
        worst_absolute = worst_absolute.max(absolute);
        if absolute > floor {
            worst = worst.max(relative);
        }
        // The standard allclose criterion: a discrepancy is a real one only
        // if it exceeds both the loss's own noise and a share of the answer.
        // Where the true gradient is zero - relu on its negative half, clamp
        // outside its range - the absolute term is what admits it, so those
        // elements are checked rather than skipped.
        if absolute > floor + f64::from(RELATIVE) * numeric.abs() {
            disagreements.push(Disagreement {
                index,
                analytic: f64::from(got),
                numeric,
                relative,
            });
        }
    }
    Ok(Row {
        name,
        worst,
        worst_absolute,
        floor,
        compared: analytic.len(),
        disagreements,
    })
}

/// Values that are neither constant nor symmetric, and - importantly -
/// all distinct, so a max or min that has to break a tie does not, and so
/// a gradient that is silently permuted does not land on equal values.
fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + 3) % 13) as f32 * scale - 6.0 * scale)
        .collect()
}

fn tensor(values: &[f32], shape: Vec<usize>) -> Result<Tensor<Dyn, B>> {
    Ok(Tensor::<Dyn, B>::from_slice(values, shape)?.forget_layout())
}

/// A second *tracked* operand, for the operations that require both sides of
/// a binary op to carry the same grad mode as the differentiated one. It
/// accumulates a gradient of its own, which nothing reads.
fn tracked(values: &[f32], shape: Vec<usize>) -> Result<T> {
    Ok(tensor(values, shape)?.require_grad())
}

fn labels(values: &[i64], shape: Vec<usize>) -> Result<Tensor<Dyn, B, i64>> {
    Tensor::<Dyn, B, i64>::from_slice(values, shape)
}

fn main() -> Result<()> {
    println!("central-difference gradient sweep: step {STEP}, relative {RELATIVE}\n");
    let mut rows: Vec<Row> = Vec::new();
    // Each row below sits in its own block for a mundane reason: a closure
    // moves the tensor it captured, so two closures in one scope cannot
    // share one. The blocks are for the borrow checker.
    macro_rules! row {
        ($name:literal, $values:expr, $shape:expr, $loss:expr) => {{
            let row = check($name, $values, $shape, $loss)?;
            println!("{row}");
            rows.push(row);
        }};
    }

    // ================================================== axis reductions
    // A reduction over the wrong axis gives a gradient of the right shape
    // and entirely the wrong values. This is the most common silent
    // gradient bug, which is why these rows come first - and why each
    // weights the reduced result, since an unweighted one cannot tell the
    // axes apart.
    {
        let weights = tensor(&ramp(3, 0.25), vec![3])?;
        row!(
            "sum over axis 0, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .sum(0isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(2, 0.25), vec![2])?;
        row!(
            "sum over axis 1, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .sum(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // mean divides by the reduced extent, so a wrong extent shows up
        // as a uniform scale factor on the gradient.
        let weights = tensor(&ramp(3, 0.25), vec![3])?;
        row!(
            "mean over axis 0, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .mean(0isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(2, 0.25), vec![2])?;
        row!(
            "mean over axis 1, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .mean(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // A max sends the whole cotangent to the argmax element, so this
        // row also pins that exactly one element per reduced row is
        // non-zero. The absolute term of the tolerance is what admits the
        // zeros, rather than dividing them by themselves.
        let weights = tensor(&ramp(2, 0.25), vec![2])?;
        row!(
            "max over axis 1, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .max(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(2, 0.25), vec![2])?;
        row!(
            "min over axis 1, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .min(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // cumsum's backward is a *truncated* reverse cumsum: a plain
        // reverse over-counts, and an axis mix-up shows it here too.
        let weights = tensor(&ramp(2 * 4, 0.25), vec![2, 4])?;
        row!(
            "weighted cumsum over axis 1",
            &ramp(2 * 4, 0.4),
            vec![2, 4],
            |x: T| Ok(x
                .cumsum(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // softmax's backward subtracts the cotangent-weighted output, which
        // is where a missing term shows up as a small constant offset on
        // every element rather than as anything local.
        let weights = tensor(&ramp(2 * 4, 0.25), vec![2, 4])?;
        row!(
            "softmax over axis 1, then weighted sum_all",
            &ramp(2 * 4, 0.4),
            vec![2, 4],
            |x: T| Ok(x
                .softmax(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(2 * 4, 0.25), vec![2, 4])?;
        row!(
            "log_softmax over axis 1, then weighted sum_all",
            &ramp(2 * 4, 0.4),
            vec![2, 4],
            |x: T| Ok(x
                .log_softmax(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(2, 0.25), vec![2])?;
        row!(
            "logsumexp over axis 1, then weighted sum_all",
            &ramp(2 * 4, 0.4),
            vec![2, 4],
            |x: T| Ok(x
                .logsumexp(1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }

    // ============================================== structural moves
    // Each of these has to undo itself in the backward, and the undo is a
    // scatter. A scatter that drops an element loses a gradient with no
    // error, so these rows are worth more than they look.
    {
        let weights = tensor(&ramp(2 * 4, 0.25), vec![2, 4])?;
        row!(
            "narrow (a view) then weighted sum_all",
            &ramp(3 * 4, 0.4),
            vec![3, 4],
            |x: T| Ok(x
                .try_narrow(0isize, 1, 2)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // The backward of a constant pad is a slice, and the padded
        // region's gradient is dropped rather than scattered.
        let weights = tensor(&ramp(4 * 7, 0.25), vec![4, 7])?;
        row!(
            "pad then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .pad(&[(1usize, 1), (2usize, 2)], 0.0f32)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // expand's backward is the unbroadcast sum over the expanded axes,
        // so the gradient w.r.t. a [1, 3] row is the column-wise total. This
        // is the shape a broadcast bias actually takes.
        let block = tensor(&ramp(2 * 3, 0.3), vec![2, 3])?;
        row!(
            "expand a row, then broadcast multiply",
            &ramp(3, 0.4),
            vec![1, 3],
            |x: T| Ok(x
                .expand(vec![2usize, 3usize])?
                .broadcast_mul(&block)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let other = tracked(&ramp(2 * 3, 0.3), vec![2, 3])?;
        let weights = tensor(&ramp(4 * 3, 0.25), vec![4, 3])?;
        row!(
            "concat along axis 0, then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .concat_axis(&other, 0isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let other = tracked(&ramp(3, 0.3), vec![3])?;
        let weights = tensor(&ramp(2 * 3, 0.5), vec![2, 3])?;
        row!(
            "stack on a new axis 0, then weighted sum_all",
            &ramp(3, 0.4),
            vec![3],
            |x: T| Ok(x
                .stack(&other, 0isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // split's backward is a concat of the cotangents, so a half that
        // is not gathered back loses its gradient.
        let weights = tensor(&ramp(3, 0.5), vec![3])?;
        row!(
            "split in two, weighted sum_all of both halves",
            &ramp(6, 0.4),
            vec![6],
            |x: T| {
                let parts = x.split(3usize, 0isize)?;
                let mut total = parts[0].broadcast_mul(&weights)?.sum_all()?;
                for part in &parts[1..] {
                    total = total
                        .broadcast_add(&part.broadcast_mul(&weights)?.sum_all()?.forget_layout())?;
                }
                Ok(total.forget_layout())
            }
        );
    }
    {
        let weights = tensor(&ramp(2 * 4, 0.25), vec![2, 4])?;
        row!(
            "repeat_interleave 2 on axis 1",
            &ramp(2 * 2, 0.4),
            vec![2, 2],
            |x: T| Ok(x
                .repeat_interleave(2, 1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // unfold's backward is a scatter-add back into the overlapping
        // windows, so a window that is skipped is a lost gradient.
        let weights = tensor(&ramp(4 * 2, 0.25), vec![4, 2])?;
        row!(
            "unfold windows of 2, then weighted sum_all",
            &ramp(5, 0.4),
            vec![5],
            |x: T| Ok(x
                .unfold(0usize, 2usize, 1usize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // gather's backward is a scatter-add, so a repeated index must
        // accumulate rather than overwrite. The index repeats element 0
        // twice and element 2 twice, so a non-accumulating backward would
        // report half the gradient for those two.
        let index = labels(&[0i64, 0, 2, 2, 1], vec![5])?;
        let weights = tensor(&ramp(5, 0.4), vec![5])?;
        row!(
            "gather with a repeated index, then weighted sum_all",
            &ramp(3, 0.4),
            vec![3],
            |x: T| Ok(x
                .gather(0isize, &index)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // transpose, then reduce: the two mistakes cancel in the value and
        // not in the gradient, which is why this is worth a row.
        let weights = tensor(&ramp(3 * 2, 0.25), vec![3, 2])?;
        row!(
            "transpose(0,1) then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .transpose(0isize, 1isize)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // A reshape is a view, so its backward is the inverse reshape. If
        // the element *order* were transposed, a symmetric loss would still
        // give the right value and the gradient would be permuted.
        let weights = tensor(&ramp(6, 0.4), vec![6])?;
        row!(
            "reshape [2,3] -> [6], then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .reshape([6usize])?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }

    // ============================================ elementwise derivatives
    // The chain rule is easy to get right for a function and easy to get
    // wrong for its derivative, so each row is a function followed by a
    // weighted sum that makes every element's gradient distinct.
    {
        let weights = tensor(&ramp(6, 0.3), vec![6])?;
        // abs sends the gradient through the sign, so this catches a
        // missing `sign` or a missing zero at the origin.
        let shifted = tensor(&[1.5f32, -2.0, 0.5, -0.25, 3.0, -1.0], vec![6])?;
        row!(
            "abs (backward through the sign)",
            &ramp(6, 0.4),
            vec![6],
            |x: T| Ok(x
                .broadcast_add(&shifted)?
                .abs()?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // clamp's backward is zero outside the range and one inside, so
        // this row also pins that the boundary is handled.
        let weights = tensor(&ramp(6, 0.3), vec![6])?;
        row!(
            "clamp(-1, 1) then weighted sum_all",
            &ramp(6, 0.4),
            vec![6],
            |x: T| Ok(x
                .clamp(-1.0, 1.0)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        let weights = tensor(&ramp(5, 0.3), vec![5])?;
        // Shifted into the positive reals, so a row can only fail on the
        // derivative under test and not on a domain error.
        let positive = tensor(&[0.4f32, 0.9, 1.3, 2.0, 0.2], vec![5])?;
        macro_rules! unary {
            ($name:literal, $method:ident) => {
                row!($name, &ramp(5, 0.4), vec![5], |x: T| {
                    Ok(x.broadcast_add(&positive)?
                        .$method()?
                        .broadcast_mul(&weights)?
                        .sum_all()?
                        .forget_layout())
                });
            };
        }
        unary!("sqrt then weighted sum_all", sqrt);
        unary!("exp then weighted sum_all", exp);
        unary!("tanh then weighted sum_all", tanh);
        unary!("sigmoid then weighted sum_all", sigmoid);
        unary!("gelu then weighted sum_all", gelu);
        unary!("swish then weighted sum_all", swish);
        unary!("elu then weighted sum_all", elu);
        unary!("log then weighted sum_all", log);
        unary!("neg then weighted sum_all", neg);
        // relu is the easy case and the one most often assumed: its
        // gradient is zero on the negative half, which is what the
        // absolute term of the tolerance has to accommodate.
        unary!("relu then weighted sum_all", relu);
    }
    {
        // mul_scalar's backward is a broadcast of the cotangent times the
        // scalar, and div_scalar's is a division - the two scalar
        // backward rules, which are the ones most often missing entirely.
        let weights = tensor(&ramp(6, 0.3), vec![6])?;
        row!(
            "mul_scalar then broadcast_divide",
            &ramp(6, 0.4),
            vec![6],
            |x: T| Ok(x
                .mul_scalar(2.5)?
                .broadcast_div(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        row!(
            "add_scalar then mul_scalar",
            &ramp(6, 0.4),
            vec![6],
            |x: T| Ok(x
                .add_scalar(1.5)?
                .mul_scalar(-0.75)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        // lerp interpolates, so its backward is a weighted mix of both
        // cotangents; dropping one gives half the gradient.
        let other = tracked(&ramp(6, 0.2), vec![6])?;
        row!("lerp then sum_all", &ramp(6, 0.4), vec![6], |x: T| Ok(x
            .lerp(&other, 0.35)?
            .sum_all()?
            .forget_layout()));
    }

    // =========================================== whole-tensor reductions
    // The gradient flows back through the reduction itself here, which is
    // a different path from a reduction over an axis.
    row!("L2 norm", &ramp(6, 0.4), vec![6], |x: T| Ok(x
        .norm(2.0)?
        .forget_layout()));
    row!("L1 norm", &ramp(6, 0.4), vec![6], |x: T| Ok(x
        .norm(1.0)?
        .forget_layout()));
    row!("variance (biased)", &ramp(6, 0.4), vec![6], |x: T| Ok(x
        .var_all(false)?
        .forget_layout()));
    row!(
        "standard deviation (biased)",
        &ramp(6, 0.4),
        vec![6],
        |x: T| Ok(x.std_all(false)?.forget_layout())
    );
    row!(
        "logsumexp over the whole tensor",
        &ramp(6, 0.4),
        vec![6],
        |x: T| Ok(x.logsumexp(0isize)?.sum_all()?.forget_layout())
    );

    // =================================================== norms, losses
    {
        let weights = tensor(&ramp(2 * 4 * 2 * 2, 0.3), vec![2, 4, 2, 2])?;
        // group_norm normalizes over groups of channels, so its backward
        // mixes a per-group mean and variance reduction with the input -
        // the same shape of computation as batch norm, whose affine bug is
        // the reason this row exists.
        row!(
            "group_norm(2) then weighted sum_all",
            &ramp(2 * 4 * 2 * 2, 0.4),
            vec![2, 4, 2, 2],
            |x: T| Ok(x
                .group_norm(2usize, 1e-5)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }
    {
        row!(
            "instance_norm then weighted sum_all",
            &ramp(2 * 4 * 2 * 2, 0.4),
            vec![2, 4, 2, 2],
            |x: T| {
                let weights = tensor(&ramp(2 * 4 * 2 * 2, 0.3), vec![2, 4, 2, 2])?;
                Ok(x.instance_norm(1e-5)?
                    .broadcast_mul(&weights)?
                    .sum_all()?
                    .forget_layout())
            }
        );
        // mse_loss reduces by the mean, so its gradient is scaled by 1/n; a
        // missing reduction factor is invisible in the value and obvious
        // here. l1_loss's derivative is the sign, which is the whole test.
        let target = tensor(&ramp(2 * 3, 0.2), vec![2, 3])?;
        row!(
            "mse_loss (mean reduction)",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x.mse_loss(&target)?.forget_layout())
        );
        row!(
            "l1_loss (mean reduction)",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x.l1_loss(&target)?.forget_layout())
        );
    }
    {
        // Cross entropy is the one loss where a subtly wrong gradient still
        // trains: softmax's Jacobian is the interesting part, and a missing
        // off-diagonal term makes it worse rather than broken.
        let labels = labels(&[0i64, 1], vec![2])?;
        row!(
            "cross_entropy_loss from logits",
            &ramp(4, 0.6),
            vec![2, 2],
            |x: T| Ok(x.cross_entropy_loss(&labels)?.forget_layout())
        );
    }
    {
        // A two-layer matmul chain: the composition of two transposes,
        // where a single swapped factor gives a plausible number.
        let middle = tensor(&ramp(3 * 4, 0.3), vec![3, 4])?;
        let out = tensor(&ramp(4 * 2, 0.3), vec![4, 2])?;
        let weights = tensor(&ramp(2 * 2, 0.5), vec![2, 2])?;
        row!(
            "matmul then matmul then weighted sum_all",
            &ramp(2 * 3, 0.4),
            vec![2, 3],
            |x: T| Ok(x
                .matmul(&middle)?
                .matmul(&out)?
                .broadcast_mul(&weights)?
                .sum_all()?
                .forget_layout())
        );
    }

    // ================================================================ end
    let failed = rows.iter().filter(|r| !r.passed()).count();
    println!();
    if failed == 0 {
        println!(
            "PASS: all {} rows agree with the numeric derivative",
            rows.len()
        );
        return Ok(());
    }
    Err(incin::Error::Msg(format!(
        "{failed} of {} rows disagreed with the central difference",
        rows.len()
    )))
}
