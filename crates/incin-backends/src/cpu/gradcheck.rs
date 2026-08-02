//! Shared finite-difference gradient-check helper (D-01, D-02).
//! Used to verify analytic gradients against numerical finite-difference approximations.
//!
//! Exposed as a public API in `incin-cpu` for users who implement custom operations
//! and want to verify their backward rules using standard central-difference checks.
//!
//! `gradcheck` calls the REAL Phase 1 API (`tape::backward`, `CpuGrads::get`).

#![allow(dead_code)]
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride;
use crate::cpu::tape;

/// Build a fresh, owned copy of `storage` with the scalar at flat buffer
/// position `flat_idx` perturbed by `delta`. Never mutates the input's
/// shared `Rc<CpuBuffer>` in place (T-02-01 mitigation) — always
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
    // matching `storage`'s logical shape — build via `from_contiguous` (new
    // `Rc`, new `TensorId`) rather than reusing `storage`'s own strides,
    // which may be non-contiguous.
    let contiguous_shape = shape.clone();
    debug_assert_eq!(
        stride::contiguous_strides(&contiguous_shape).len(),
        storage.strides.len()
    );
    CpuStorage::from_contiguous(buffer, contiguous_shape.to_vec())
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
/// difference has O(eps^2) truncation error vs. forward difference's O(eps) —
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
/// # Panics
///
/// Panics with a clear message if `op(inputs)`'s output shape is not scalar
/// (`[]`) — `gradcheck` requires a scalar-output op so a single central
/// difference directly approximates the whole gradient contribution.
pub fn gradcheck(op: impl Fn(&[CpuStorage]) -> CpuStorage, inputs: &[CpuStorage], eps: f64) -> f64 {
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
        let total: usize = input.shape.iter().product::<usize>().max(1);
        for flat_idx in 0..total {
            let numeric = numerical_grad(&op, inputs, i, flat_idx, eps);
            let analytic_val = flat_get(analytic, flat_idx);
            let abs_diff = (analytic_val - numeric).abs();

            // Absolute-error escape hatch: when the TRUE gradient is exactly
            // (or near) zero, `numeric` is pure f32 finite-difference
            // rounding noise (observed up to ~3e-4 magnitude at eps=1e-4 on
            // scalar outputs near 1.0) rather than a real small gradient. A
            // purely relative comparison makes `rel_err` blow up toward 1.0
            // in that regime even though both values correctly agree the
            // gradient is ~0 (mirrors PyTorch's `gradcheck` atol+rtol
            // combination, not a bare relative-error ratio). If the absolute
            // difference itself is below this noise ceiling, treat it as a
            // pass (contributes 0 to `max_rel_err`) regardless of the ratio
            // — a genuinely wrong non-zero gradient still produces an
            // `abs_diff` far above this ceiling and fails loudly via the
            // relative check below.
            let abs_tol = 1e-3;
            if abs_diff < abs_tol {
                continue;
            }

            let floor = 1e-6;
            let denom = analytic_val.abs().max(numeric.abs()).max(floor);
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
    use crate::cpu::CpuBackendImpl;
    use incin_core::backend_authoring::{NumericOps, ReductionOps};
    use incin_core::prelude::Cpu;

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<f32, Cpu>;

    /// `vector`.
    fn vector(v: Vec<f32>) -> CpuStorage {
        let len = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
    }

    /// Test 1: `gradcheck` applied to `sum(x^2)` on a 1-D input matches the
    /// known-analytic derivative `2x` within tolerance. Built entirely from
    /// Phase 1's already-implemented `NumericOps::mul`/`ReductionOps::sum_all`
    /// — no new op is implemented in this file.
    #[test]
    fn gradcheck_matches_analytic_gradient_of_sum_of_squares() {
        let x = vector(vec![2.0, 3.0, -1.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let squared = TestBackend::mul::<f32>(&inputs[0], &inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&squared).unwrap()
        };

        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gradcheck max relative error too high: {max_rel_err}"
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
            let x2 = TestBackend::mul::<f32>(x, x).unwrap();
            let x3 = TestBackend::mul::<f32>(&x2, x).unwrap();
            TestBackend::sum_all::<f32>(&x3).unwrap()
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
        // op returns x*x elementwise — shape [3], not scalar.
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            TestBackend::mul::<f32>(&inputs[0], &inputs[0]).unwrap()
        };

        gradcheck(op, &[x], 1e-4);
    }
}
