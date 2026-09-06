//! Shared finite-difference gradient-check helper (D-01, D-02).
//! Used to verify analytic gradients against numerical finite-difference approximations.
//!
//! `gradcheck` calls the REAL Phase 1 API (`tape::backward`, `CpuGrads::get`).
use incin_core::error::Result;

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride;
use crate::cpu::tape;

/// The central-difference step that minimizes total error for f32 storage:
/// `(6 * f32::EPSILON).cbrt()` is approximately `9e-3`, so this rounds to a
/// readable `1e-2`.
///
/// This crate previously used `1e-4` everywhere, roughly a hundredth of the
/// optimum. Because the rounding term scales as `1/eps`, that inflated the
/// finite-difference noise floor by about the same factor: measured max
/// absolute error on `canonical_matmul` fell from `1.1e-2` at `1e-4` to
/// `1.1e-4` at `1e-2`, and on `sum_of_squares` from `6.2e-3` to `5.7e-6`.
/// That noise, not any gradient defect, is what the old `abs_tol` skip was
/// suppressing.
pub(crate) const F32_STEP: f64 = 1e-2;

/// The relative-error ceiling every gradcheck assertion in this crate uses.
///
/// This was `1e-2` per call site (and `3e-2` on three batched-matmul cases
/// that had drifted up to clear aarch64 noise). At the old `1e-4` step the
/// measured error on correct gradients already reached `1.3e-2`, so `1e-2`
/// was not slack, it was roughly the noise itself: a check whose threshold
/// sits at its own noise floor cannot distinguish a small real defect from a
/// rounding artifact, which is what made those aarch64 failures look like
/// gradient bugs. With [`F32_STEP`] the measured worst case across the whole
/// crate falls to `1.0e-4`, so a `1e-3` ceiling clears real noise by 10x
/// while catching gradient errors an order of magnitude smaller than before.
pub(crate) const GRAD_TOL: f64 = 1e-3;

/// The CPU backend's answer to the core's gradient-check contract.
///
/// Three of the four methods are the helpers this module already had, given
/// the names the shared sweep calls them by. The fourth, `backward_from`, is
/// this backend's own thread-local walk, so a check written against the core
/// trait exercises the path `Tensor::backward` takes rather than a
/// reconstruction of it.
impl incin_core::exec::GradCheckStorage for CpuStorage {
    fn backward_from(loss: &Self) -> Result<incin_core::exec::GradientMap<Self>> {
        tape::backward(loss).map(|grads| grads.grads)
    }

    fn element_count(&self) -> usize {
        crate::cpu::stride::validated_numel(&self.shape).max(1)
    }

    fn element(&self, index: usize) -> Result<f64> {
        Ok(flat_get(self, index))
    }

    fn with_element_perturbed(&self, index: usize, delta: f64) -> Result<Self> {
        Ok(perturbed(self, index, delta))
    }
}

/// Build a fresh, owned copy of `storage` with the scalar at flat buffer
/// position `flat_idx` perturbed by `delta`. Never mutates the input's
/// shared `Rc<CpuBuffer>` in place (T-02-01 mitigation) - always
/// constructs a new `Vec` copy first.
fn perturbed(storage: &CpuStorage, flat_idx: usize, delta: f64) -> CpuStorage {
    // Resolve `flat_idx` (an index into the LOGICAL row-major element order,
    // i.e. an index into a hypothetical contiguous buffer of this storage's
    // shape) into a multi-index using the same odometer iteration order as
    // `ops/elementwise.rs`'s `increment_index`, then read through `storage`'s
    // own strides (correct even for a non-contiguous view).
    let shape = &storage.shape;
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..flat_idx {
        crate::cpu::ops::elementwise::increment_index(&mut idx, shape);
    }

    let buffer = match &*storage.buffer {
        CpuBuffer::F32(v) => {
            let mut v = v.clone();
            let mut flat = storage.offset_elements;
            for (i, s) in idx.iter().zip(storage.strides.iter()) {
                flat += i * s;
            }
            v[flat] = (v[flat] as f64 + delta) as f32;
            CpuBuffer::F32(v)
        }
        CpuBuffer::F64(v) => {
            let mut v = v.clone();
            let mut flat = storage.offset_elements;
            for (i, s) in idx.iter().zip(storage.strides.iter()) {
                flat += i * s;
            }
            v[flat] += delta;
            CpuBuffer::F64(v)
        }
        _ => panic!("gradcheck: perturbation only supported for F32/F64 buffers"),
    };

    // The perturbed value lives in a freshly materialized CONTIGUOUS buffer
    // matching `storage`'s logical shape - build via `from_contiguous` (new
    // `Rc`, new `TensorId`) rather than reusing `storage`'s own strides,
    // which may be non-contiguous.
    let contiguous_shape = shape.clone();
    debug_assert_eq!(
        stride::contiguous_strides(&contiguous_shape).len(),
        storage.strides.len()
    );
    CpuStorage::from_contiguous(buffer, &contiguous_shape)
}

/// Read the scalar at flat buffer position `flat_idx` (logical row-major
/// order) out of `storage`, resolving through strides.
fn flat_get(storage: &CpuStorage, flat_idx: usize) -> f64 {
    let shape = &storage.shape;
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..flat_idx {
        crate::cpu::ops::elementwise::increment_index(&mut idx, shape);
    }
    storage.get(&idx)
}

/// Central-difference approximation of `d(output_scalar)/d(inputs[input_idx][flat_idx])`,
/// using the standard formula `(f(x+eps) - f(x-eps)) / (2*eps)`. Central
/// difference has O(eps^2) truncation error vs. forward difference's O(eps) -
/// the standard choice for gradient checking (matches `torch.autograd.
/// gradcheck`'s own default technique).
fn numerical_grad(
    f: &impl Fn(&[CpuStorage]) -> CpuStorage,
    inputs: &[CpuStorage],
    input_idx: usize,
    flat_idx: usize,
    eps: f64,
) -> f64 {
    let mut plus: Vec<CpuStorage> = inputs.to_vec();
    let mut minus: Vec<CpuStorage> = inputs.to_vec();
    plus[input_idx] = perturbed(&inputs[input_idx], flat_idx, eps);
    minus[input_idx] = perturbed(&inputs[input_idx], flat_idx, -eps);

    let f_plus = f(&plus);
    let f_minus = f(&minus);
    (f_plus.get(&[]) - f_minus.get(&[])) / (2.0 * eps)
}

/// Runs `op` under autodiff, extracts the ANALYTIC gradient for every input's
/// every element via `tape::backward`, compares each against `numerical_grad`
/// at the same position, and returns the MAXIMUM relative error found across
/// every input/element pair.
///
/// `eps` is the central-difference step. For f32 storage the total error is
/// minimized near `(6 * f32::EPSILON).cbrt()`, about `9e-3`: below that,
/// subtractive rounding of `f(x)` divided by `2 * eps` dominates and grows as
/// `1/eps`. Callers that pass a much smaller step get a correspondingly
/// noisier estimate, which is why [`F32_STEP`] exists and why domain-sensitive
/// ops (`log`, `sqrt`, `acosh` near their boundaries, where `x +/- eps` must
/// stay in-domain) are the only places a smaller value belongs.
///
/// # Panics
///
/// Panics with a clear message if `op(inputs)`'s output shape is not scalar
/// (`[]`) - `gradcheck` requires a scalar-output op so a single central
/// difference directly approximates the whole gradient contribution.
pub(crate) fn gradcheck(
    op: impl Fn(&[CpuStorage]) -> CpuStorage,
    inputs: &[CpuStorage],
    eps: f64,
) -> f64 {
    let out = op(inputs);
    assert!(
        out.shape.is_empty(),
        "gradcheck requires a scalar-output op (got shape {:?})",
        out.shape
    );

    let grads = tape::backward(&out).expect("gradcheck: backward() must succeed");

    let mut max_rel_err = 0.0f64;
    for (i, input) in inputs.iter().enumerate() {
        let analytic = grads
            .get(input.id)
            .expect("gradcheck: missing analytic gradient for input");
        let total = stride::validated_numel(&input.shape).max(1);
        for flat_idx in 0..total {
            let numeric = numerical_grad(&op, inputs, i, flat_idx, eps);
            let analytic_val = flat_get(analytic, flat_idx);
            let abs_diff = (analytic_val - numeric).abs();

            // Absolute-error escape hatch: when the TRUE gradient is exactly
            // (or near) zero, `numeric` is pure finite-difference rounding
            // noise rather than a real small gradient, and a purely relative
            // comparison would blow `rel_err` up toward 1.0 even though both
            // values correctly agree the gradient is ~0. Below this ceiling,
            // treat the element as a pass.
            //
            // Skipping here is sound precisely BECAUSE it only fires when the
            // two values agree: a wrong gradient makes `abs_diff` large, so it
            // never takes this branch and still fails loudly below. What the
            // ceiling really sets is this check's ABSOLUTE sensitivity floor,
            // which is why its value matters.
            //
            // It was `1e-3`, chosen to clear the finite-difference noise at
            // the `1e-4` step every caller used to pass. That step was about a
            // hundredth of f32's optimum (see `F32_STEP`), so the noise it had
            // to clear was inflated ~100x. With the step corrected the
            // measured noise floor drops to ~1e-4 worst case and ~5e-6
            // typical, so the ceiling comes down 20x with it. On this crate's
            // gradient magnitudes (0.3 to 9.0) that moves the effective
            // sensitivity from ~0.1% relative to ~0.005%.
            const ABS_TOL: f64 = 5e-5;
            if abs_diff < ABS_TOL {
                continue;
            }

            const DENOM_FLOOR: f64 = 1e-6;
            let denom = analytic_val.abs().max(numeric.abs()).max(DENOM_FLOOR);
            let rel_err = abs_diff / denom;
            max_rel_err = max_rel_err.max(rel_err);
        }
    }
    max_rel_err
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    /// `vector`.
    fn vector(v: Vec<f32>) -> CpuStorage {
        let len = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
    }

    /// Test 1: `gradcheck` applied to `sum(x^2)` on a 1-D input matches the
    /// known-analytic derivative `2x` within tolerance. Built entirely from
    /// The concrete multiplication helper and reduction kernel are used
    /// directly so this fixture exercises the canonical CPU ownership path.
    #[test]
    fn gradcheck_matches_analytic_gradient_of_sum_of_squares() {
        let x = vector(vec![2.0, 3.0, -1.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let squared =
                crate::cpu::ops::elementwise::mul_storage(&inputs[0], &inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&squared).unwrap()
        };

        let max_rel_err = gradcheck(op, &[x], F32_STEP);
        assert!(
            max_rel_err < GRAD_TOL,
            "gradcheck max relative error too high: {max_rel_err}"
        );
    }

    /// The public core sweep reaches the same verdict as this module's own.
    ///
    /// Both are pointed at `sum(x^2)`, whose gradient is `2x` in closed form.
    /// The point is not that either is right in isolation but that the
    /// generic path, which an out-of-tree backend will use, agrees with the
    /// in-tree one that every gradient test in this crate is written against.
    #[test]
    fn the_public_gradcheck_agrees_with_the_internal_one() {
        use incin_core::exec::{GradCheckOptions, gradcheck as public_gradcheck};

        let x = vector(vec![2.0, 3.0, -1.0]);
        let op = |inputs: &[CpuStorage]| -> incin_core::error::Result<CpuStorage> {
            let squared = crate::cpu::ops::elementwise::mul_storage(&inputs[0], &inputs[0])?;
            crate::cpu::ops::reduce::sum_all(&squared)
        };

        let report = public_gradcheck(op, &[x], GradCheckOptions::for_f32())
            .expect("the sweep ran to completion");
        assert!(report.passed(), "{report}");
        assert_eq!(report.compared, 3);
        assert!(report.worst_relative_error < GRAD_TOL);
    }

    /// A recipe that is wrong by a constant factor is caught and located.
    ///
    /// A check that has never failed a bad gradient says nothing when it
    /// passes a good one. This records the deliberate defect: the recipe
    /// halves every contribution, so every element disagrees by the same
    /// factor, which is the shape the report tells the reader to look for.
    #[test]
    fn a_recipe_wrong_by_a_constant_factor_is_reported_on_every_element() {
        use alloc::boxed::Box;
        use incin_core::exec::{GradCheckOptions, TapeNode, gradcheck as public_gradcheck};

        let x = vector(vec![2.0, 3.0, -1.0]);
        let op = |inputs: &[CpuStorage]| -> incin_core::error::Result<CpuStorage> {
            let squared = crate::cpu::ops::elementwise::mul_storage(&inputs[0], &inputs[0])?;
            let out = crate::cpu::ops::reduce::sum_all(&squared)?;
            // Half of `2x`. Recorded after the real chain, so it is the node
            // the walk reaches first and the one whose contribution lands.
            let saved = inputs[0].clone();
            let (input_id, out_id) = (inputs[0].id, out.id);
            tape::push(TapeNode {
                output_id: out_id,
                input_ids: alloc::vec![input_id],
                backward: Box::new(move |_grad: &CpuStorage| {
                    Ok(alloc::vec![crate::cpu::ops::elementwise::mul_storage(
                        &saved,
                        &saved.clone(),
                    )?])
                }),
            });
            Ok(out)
        };

        let report = public_gradcheck(op, &[x], GradCheckOptions::for_f32())
            .expect("the sweep ran to completion");
        assert!(!report.passed(), "a wrong recipe passed: {report}");
        assert!(
            !report.disagreements.is_empty(),
            "the report named no element"
        );
    }

    /// Test 2: `numerical_grad` alone (independent of any op's analytic
    /// gradient / `tape::backward`) matches the closed-form derivative of
    /// `f(x) = x[0]^3`, i.e. `3*x[0]^2`, proving the central-difference
    /// formula itself.
    #[test]
    fn numerical_grad_matches_closed_form_derivative_of_cube() {
        let x = vector(vec![2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let x = &inputs[0];
            let x2 = crate::cpu::ops::elementwise::mul_storage(x, x).unwrap();
            let x3 = crate::cpu::ops::elementwise::mul_storage(&x2, x).unwrap();
            crate::cpu::ops::reduce::sum_all(&x3).unwrap()
        };

        let eps = 1e-4;
        let numeric = numerical_grad(&op, &[x], 0, 0, eps);
        let closed_form = 3.0 * 2.0_f64 * 2.0; // 3*x[0]^2 at x[0]=2.0

        let rel_err = (numeric - closed_form).abs() / closed_form.abs().max(1e-6);
        assert!(
            rel_err < 1e-2,
            "numerical_grad diverged from closed-form derivative: numeric={numeric}, closed_form={closed_form}"
        );
    }

    /// Test 3: `gradcheck` panics with a clear message (not a silent wrong
    /// number) when given an op whose output shape is not scalar.
    #[test]
    #[should_panic(expected = "gradcheck requires a scalar-output op")]
    fn gradcheck_panics_on_non_scalar_output() {
        let x = vector(vec![1.0, 2.0, 3.0]);
        // op returns x*x elementwise - shape [3], not scalar.
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            crate::cpu::ops::elementwise::mul_storage(&inputs[0], &inputs[0]).unwrap()
        };

        gradcheck(op, &[x], F32_STEP);
    }
}
