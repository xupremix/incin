//! Numerical gradient checking for hand-written backward rules.
//!
//! A backward recipe is the part of a custom operation that fails quietly. A
//! wrong forward kernel produces visibly wrong numbers; a wrong recipe
//! produces a model that trains slightly worse, which is not a signal anyone
//! can act on. The check is a central difference: perturb one input element,
//! re-run the forward, and compare the slope against the analytic gradient at
//! that element.
//!
//! This lives in the core, next to [`DifferentiableOp`], rather than in a
//! backend, because it is part of the custom-operation contract rather than a
//! CPU convenience. Everything it needs of a storage type is
//! [`GradCheckStorage`]: read one element, write one element, and run the
//! backward pass. A backend that implements those gets the sweep, the step
//! size, and the report for free.
//!
//! [`DifferentiableOp`]: crate::tensor::backend::DifferentiableOp
//!
//! The shape of a check, in a backend-agnostic sketch. A runnable one against
//! real storage is `the_public_gradcheck_agrees_with_the_internal_one` in
//! `incin-backends`, and the custom-operations chapter walks through a whole
//! recipe checked this way.
//!
//! ```text
//! let report = gradcheck(
//!     |inputs| my_scalar_loss(&inputs[0]),
//!     &[input.clone()],
//!     GradCheckOptions::for_f32(),
//! )?;
//! assert!(report.passed(), "{report}");
//! ```
//!
//! `report` is worth printing rather than only asserting on: it names the
//! input, the element, both values and their relative error, and a failure
//! that is a constant factor across every element reads differently from one
//! that is a single boundary.

use alloc::vec::Vec;
use core::fmt;

use crate::err::Result;
use crate::exec::tape::{GradientMap, TapeStorage};

/// What gradient checking needs of a backend's storage, and nothing else.
pub trait GradCheckStorage: TapeStorage + Sized {
    /// Run the backward pass from `loss` on whatever graph this backend
    /// recorded, and return the accumulated gradients.
    ///
    /// This is the backend's own entry point, the one `Tensor::backward`
    /// reaches. Naming it here rather than taking a closure keeps the check's
    /// signature free of a parameter every caller would spell the same way,
    /// and means the sweep exercises the real path rather than a
    /// reconstruction of it.
    fn backward_from(loss: &Self) -> Result<GradientMap<Self>>;

    /// How many logical elements this value holds.
    fn element_count(&self) -> usize;

    /// Read one element in logical row-major order, as `f64`.
    fn element(&self, index: usize) -> Result<f64>;

    /// A copy of `self` with one element moved by `delta`.
    ///
    /// A copy rather than a mutation: the original is still on the graph, and
    /// perturbing it in place would move the value the recipe saved.
    fn with_element_perturbed(&self, index: usize, delta: f64) -> Result<Self>;
}

/// The step, the ceiling, and the floor a sweep runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradCheckOptions {
    /// The central-difference step.
    pub step: f64,
    /// The relative-error ceiling a comparison must stay under.
    pub tolerance: f64,
    /// Below this absolute difference a comparison passes regardless.
    ///
    /// Where the true gradient is zero the numeric estimate is pure rounding
    /// noise, and a relative comparison against it would report an error near
    /// one while both values correctly agree the gradient is nothing. The
    /// floor is what sets the check's absolute sensitivity, which is why it
    /// is a knob rather than a constant.
    pub absolute_floor: f64,
}

impl GradCheckOptions {
    /// Defaults for `f32` storage.
    ///
    /// The step is the one that minimizes total error at `f32` precision:
    /// central-difference rounding grows as `1 / step`, truncation as
    /// `step^2`, and the two meet near `(6 * f32::EPSILON).cbrt()`, about
    /// `9e-3`. The `1e-4` that looks conservative is roughly a hundredth of
    /// that and sits at its own noise floor, where a real defect and a
    /// rounding artifact are indistinguishable.
    #[must_use]
    pub const fn for_f32() -> Self {
        Self {
            step: 1e-2,
            tolerance: 1e-3,
            absolute_floor: 5e-5,
        }
    }

    /// Defaults for `f64` storage, where a much smaller step is affordable.
    #[must_use]
    pub const fn for_f64() -> Self {
        Self {
            step: 1e-5,
            tolerance: 1e-6,
            absolute_floor: 1e-9,
        }
    }

    /// The same options with a different step.
    #[must_use]
    pub const fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// The same options with a different relative-error ceiling.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Default for GradCheckOptions {
    fn default() -> Self {
        Self::for_f32()
    }
}

/// One element where the analytic and numeric gradients disagreed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disagreement {
    /// Position of the input in the slice handed to the check.
    pub input: usize,
    /// Element index within that input, in logical row-major order.
    pub element: usize,
    /// What the recipe produced.
    pub analytic: f64,
    /// What the central difference measured.
    pub numeric: f64,
    /// Their difference relative to the larger magnitude.
    pub relative_error: f64,
}

/// What one sweep concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCheckReport {
    /// Every element whose relative error exceeded the ceiling, worst first.
    pub disagreements: Vec<Disagreement>,
    /// How many element comparisons were made.
    pub compared: usize,
    /// The largest relative error seen, including passing ones.
    pub worst_relative_error: f64,
    /// The options the sweep ran with.
    pub options: GradCheckOptions,
}

impl GradCheckReport {
    /// Whether every element agreed within the ceiling.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.disagreements.is_empty()
    }
}

impl fmt::Display for GradCheckReport {
    /// Reads as a verdict, and names where to look when it is a bad one.
    ///
    /// The count matters as much as the worst case. One disagreeing element
    /// out of hundreds is usually a boundary the perturbation stepped across;
    /// every element disagreeing by a constant factor is a recipe that
    /// forgot a term.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed() {
            return write!(
                f,
                "gradcheck passed: {} elements compared, worst relative error {:.3e} \
                 (ceiling {:.3e}, step {:.3e})",
                self.compared, self.worst_relative_error, self.options.tolerance, self.options.step
            );
        }
        writeln!(
            f,
            "gradcheck failed: {} of {} elements disagree (ceiling {:.3e}, step {:.3e})",
            self.disagreements.len(),
            self.compared,
            self.options.tolerance,
            self.options.step
        )?;
        for bad in self.disagreements.iter().take(8) {
            writeln!(
                f,
                "  input {} element {}: recipe {:+.6e}, finite difference {:+.6e}, \
                 relative error {:.3e}",
                bad.input, bad.element, bad.analytic, bad.numeric, bad.relative_error
            )?;
        }
        if self.disagreements.len() > 8 {
            writeln!(f, "  ... and {} more", self.disagreements.len() - 8)?;
        }
        write!(
            f,
            "A constant factor across every element is usually a missing term in the \
             recipe. A single element is usually the perturbation stepping across a \
             boundary the derivative does not exist at."
        )
    }
}

/// Compare a recorded backward pass against central differences.
///
/// `op` must return a scalar, so that one central difference approximates the
/// whole gradient contribution of the element it perturbed. Build the scalar
/// the way the model does, by reducing whatever the operation returns.
///
/// Every element of every input is swept. A spot check is not enough: an
/// accumulation that overwrites instead of summing agrees everywhere except
/// at the one input two operations share, which is exactly the element a spot
/// check skips.
///
/// # Errors
///
/// Returns whatever `op` or the backward pass returns, and
/// [`BackwardError::Recipe`](crate::err::BackwardError::Recipe) if `op`
/// produced no gradient for one of the inputs. A missing gradient is a
/// finding rather than a zero: it is what a recipe that returned too few
/// contributions, or an operation that recorded no node at all, looks like
/// from here.
pub fn gradcheck<S: GradCheckStorage>(
    op: impl Fn(&[S]) -> Result<S>,
    inputs: &[S],
    options: GradCheckOptions,
) -> Result<GradCheckReport> {
    let output = op(inputs)?;
    let grads = S::backward_from(&output)?;

    let mut disagreements = Vec::new();
    let mut compared = 0usize;
    let mut worst_relative_error = 0.0f64;

    for (position, input) in inputs.iter().enumerate() {
        let Some(analytic_storage) = grads.get(input.id()) else {
            return Err(crate::err::BackwardError::Recipe {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "gradcheck: the backward pass produced no gradient for an input",
            }
            .into());
        };

        for element in 0..input.element_count() {
            let analytic = analytic_storage.element(element)?;
            let numeric = central_difference(&op, inputs, position, element, options.step)?;
            compared += 1;

            let absolute = (analytic - numeric).abs();
            if absolute < options.absolute_floor {
                continue;
            }
            let denominator = analytic.abs().max(numeric.abs()).max(1e-12);
            let relative_error = absolute / denominator;
            worst_relative_error = worst_relative_error.max(relative_error);
            if relative_error > options.tolerance {
                disagreements.push(Disagreement {
                    input: position,
                    element,
                    analytic,
                    numeric,
                    relative_error,
                });
            }
        }
    }

    disagreements.sort_by(|a, b| {
        b.relative_error
            .partial_cmp(&a.relative_error)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    Ok(GradCheckReport {
        disagreements,
        compared,
        worst_relative_error,
        options,
    })
}

/// `(f(x + step) - f(x - step)) / (2 * step)` for one element.
///
/// Central rather than forward: its truncation error is `O(step^2)` where a
/// forward difference is `O(step)`, which is the difference between a check
/// that can see a small defect and one that cannot.
fn central_difference<S: GradCheckStorage>(
    op: &impl Fn(&[S]) -> Result<S>,
    inputs: &[S],
    position: usize,
    element: usize,
    step: f64,
) -> Result<f64> {
    let mut plus: Vec<S> = inputs.to_vec();
    let mut minus: Vec<S> = inputs.to_vec();
    plus[position] = inputs[position].with_element_perturbed(element, step)?;
    minus[position] = inputs[position].with_element_perturbed(element, -step)?;

    let high = op(&plus)?.element(0)?;
    let low = op(&minus)?.element(0)?;
    Ok((high - low) / (2.0 * step))
}
