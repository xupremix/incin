//! Does the IR agree with the hand-written CUDA C, on the GPU?
//!
//! `cuda::backend::elementwise` declares each pointwise operation as CUDA C
//! literals, and `codegen::catalog` declares the same operations as `IrExpr`.
//! Two independent definitions of the same arithmetic are only useful if they
//! agree, and until this file existed nothing checked that they did: the IR had
//! no executor at all, so its expressions had never produced a number.
//!
//! Each test here runs both definitions through the *same* launcher on the
//! *same* input and compares the results elementwise, then compares both against
//! a host reference evaluated in `f64`. The host reference is what makes a
//! disagreement actionable: when the two GPU paths differ, it says which one is
//! wrong rather than only that they differ.
//!
//! These require a GPU and follow the crate convention of `#[ignore]`, so
//! `cargo test --features cuda -- --ignored` is what runs them.

use super::elementwise::{
    launch_binary_body, launch_binary_op, launch_unary_body, launch_unary_op,
};
use crate::codegen::catalog;
use crate::cuda::backend::{cuda_from_f32, download_f32_host};
use crate::cuda::storage::CudaStorage;
use crate::kernel::{KernelSpecialization, lower_binary_body, lower_unary_body};

/// These probes construct storage directly rather than going through a typed
/// frontend, so there is no shape type behind them and nothing to specialize on.
const NO_PROOF: KernelSpecialization = KernelSpecialization::NONE;
use alloc::vec::Vec;
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeId;

fn storage(shape: &[usize], values: Vec<f32>) -> CudaStorage {
    cuda_from_f32(
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values,
        "test",
    )
    .unwrap()
}

fn has_cuda() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

/// A distinct `&'static str` kernel name per (path, operation) pair.
///
/// The CUDA module cache is keyed on `KernelKey::cache_id`, which includes the
/// operation *name* but not the kernel source, so two kernels launched under one
/// name on the same dtype and layout resolve to whichever compiled first. Every
/// probe therefore needs its own name, and the launchers want `&'static str`.
/// Leaking is confined to the fixed, small set of names this file uses.
fn probe_name(path: &str, op_name: &str) -> &'static str {
    alloc::boxed::Box::leak(alloc::format!("{path}_{op_name}").into_boxed_str())
}

/// Inputs chosen to stay inside every operation's domain at once.
///
/// `log`, `sqrt` and `rsqrt` need a strictly positive argument, so the sample is
/// positive throughout; the sign-dependent operations are covered separately by
/// [`sign_dependent_ops_agree_across_the_zero_crossing`], which uses a signed
/// sample against the operations whose domain allows it.
fn positive_sample() -> Vec<f32> {
    vec![
        0.125, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.25, 8.0, 10.0, 0.0625, 1.25, 3.5, 5.0,
    ]
}

/// A sample that straddles zero, for operations defined on the whole line.
fn signed_sample() -> Vec<f32> {
    vec![
        -4.0, -3.5, -2.0, -1.25, -0.5, -0.125, 0.125, 0.5, 1.25, 2.0, 3.5, 4.0, -6.0, 6.0, -0.75,
        0.75,
    ]
}

/// The hand-written CUDA C forward literal for `op_name`.
///
/// These are copied verbatim from the `cuda_pointwise!` invocation in
/// `cuda::backend::elementwise`. Duplicating them is deliberate: the point of
/// the test is to compare against exactly the text the backend ships, and
/// reaching into the macro would couple the test to the macro's shape rather
/// than to its output.
fn unary_literal(op_name: &str) -> Option<&'static str> {
    Some(match op_name {
        "relu" => "x > 0.0f ? x : 0.0f",
        "neg" => "-x",
        "abs" => "fabsf(x)",
        "log" => "logf(x)",
        "exp" => "expf(x)",
        "sqrt" => "sqrtf(x)",
        "rsqrt" => "rsqrtf(x)",
        "sin" => "sinf(x)",
        "cos" => "cosf(x)",
        "tanh" => "tanhf(x)",
        "sigmoid" => "1.0f / (1.0f + expf(-x))",
        "swish" => "x / (1.0f + expf(-x))",
        "gelu" => "0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))",
        "step" => "x > 0.0f ? 1.0f : 0.0f",
        "sign" => "x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)",
        "elu" => "x >= 0.0f ? x : (expf(x) - 1.0f)",
        "mish" => "x * tanhf(log1pf(expf(x)))",
        "log2" => "log2f(x)",
        "log10" => "log10f(x)",
        _ => return None,
    })
}

fn binary_literal(op_name: &str) -> Option<&'static str> {
    Some(match op_name {
        "add" => "a + b",
        "sub" => "a - b",
        "mul" => "a * b",
        "div" => "a / b",
        "maximum" => "a > b ? a : b",
        "minimum" => "a < b ? a : b",
        "abs_diff" => "fabsf(a - b)",
        _ => return None,
    })
}

/// The host reference for `op_name`, evaluated in `f64`.
fn unary_reference(op_name: &str, x: f64) -> f64 {
    match op_name {
        "relu" => x.max(0.0),
        "neg" => -x,
        "abs" => x.abs(),
        "log" => x.ln(),
        "exp" => x.exp(),
        "sqrt" => x.sqrt(),
        "rsqrt" => 1.0 / x.sqrt(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tanh" => x.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "swish" => x / (1.0 + (-x).exp()),
        "gelu" => {
            let inner = 0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        "square" => x * x,
        "reciprocal" => 1.0 / x,
        "elu" => {
            if x >= 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        "mish" => x * x.exp().ln_1p().tanh(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "step" => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "sign" => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        other => panic!("no host reference for {other}"),
    }
}

fn binary_reference(op_name: &str, a: f64, b: f64) -> f64 {
    match op_name {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        "maximum" => a.max(b),
        "minimum" => a.min(b),
        "abs_diff" => (a - b).abs(),
        other => panic!("no host reference for {other}"),
    }
}

/// Single-precision tolerance, relative where the magnitude warrants it.
///
/// The two paths emit different but mathematically equal expressions -- the IR
/// spells `relu` as `fmaxf(0.0f, x)` where the literal uses a conditional, and
/// binds intermediates to named temporaries the literal recomputes inline -- so
/// bitwise equality is not the contract. Agreement to single-precision rounding
/// is.
fn close(left: f64, right: f64) -> bool {
    let tolerance = 1e-5 * left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= tolerance
}

/// The IR-lowered forward must match the shipped literal, and both must match a
/// host reference, for every unary operation the catalog claims.
#[test]
#[ignore = "requires CUDA hardware"]
fn ir_unary_forward_matches_the_hand_written_literal() {
    if !has_cuda() {
        return;
    }
    for &op_name in catalog::UNARY_OPS {
        // `square` and `reciprocal` have IR definitions but no literal in the
        // backend's macro table, so there is nothing to compare them against
        // here; the host reference still covers them below.
        let Some(literal) = unary_literal(op_name) else {
            continue;
        };
        let sample = match op_name {
            "log" | "log2" | "log10" | "sqrt" | "rsqrt" => positive_sample(),
            _ => signed_sample(),
        };
        let input = storage(&[sample.len()], sample.clone());

        let ir = catalog::unary_forward(op_name)
            .unwrap_or_else(|| panic!("{op_name} is listed in UNARY_OPS but has no IR"));
        let body = lower_unary_body(&ir, DTypeId::F32).unwrap();

        let from_ir =
            launch_unary_body(probe_name("ir", op_name), &body, &input, NO_PROOF).unwrap();
        let from_literal = launch_unary_op(probe_name("lit", op_name), literal, &input).unwrap();

        let ir_values = download_f32_host(&from_ir).unwrap();
        let literal_values = download_f32_host(&from_literal).unwrap();

        for (index, &x) in sample.iter().enumerate() {
            let expected = unary_reference(op_name, f64::from(x));
            let got_ir = f64::from(ir_values[index]);
            let got_literal = f64::from(literal_values[index]);
            assert!(
                close(got_ir, got_literal),
                "{op_name}({x}): IR gave {got_ir}, shipped literal gave {got_literal}"
            );
            assert!(
                close(got_ir, expected),
                "{op_name}({x}): IR gave {got_ir}, host reference is {expected}"
            );
        }
    }
}

/// The same comparison for binary operations.
#[test]
#[ignore = "requires CUDA hardware"]
fn ir_binary_forward_matches_the_hand_written_literal() {
    if !has_cuda() {
        return;
    }
    let lhs_sample = signed_sample();
    // Rotated so no lane divides by zero and the max/min comparisons see both
    // orderings.
    let rhs_sample: Vec<f32> = lhs_sample.iter().rev().copied().collect();

    for &op_name in catalog::BINARY_OPS {
        let Some(literal) = binary_literal(op_name) else {
            continue;
        };
        let lhs = storage(&[lhs_sample.len()], lhs_sample.clone());
        let rhs = storage(&[rhs_sample.len()], rhs_sample.clone());
        let out_shape = [lhs_sample.len()];

        let ir = catalog::binary_forward(op_name)
            .unwrap_or_else(|| panic!("{op_name} is listed in BINARY_OPS but has no IR"));
        let body = lower_binary_body(&ir, DTypeId::F32).unwrap();

        let from_ir = launch_binary_body(
            probe_name("ir", op_name),
            &body,
            &lhs,
            &rhs,
            &out_shape,
            NO_PROOF,
        )
        .unwrap();
        let from_literal =
            launch_binary_op(probe_name("lit", op_name), literal, &lhs, &rhs, &out_shape).unwrap();

        let ir_values = download_f32_host(&from_ir).unwrap();
        let literal_values = download_f32_host(&from_literal).unwrap();

        for index in 0..lhs_sample.len() {
            let (a, b) = (f64::from(lhs_sample[index]), f64::from(rhs_sample[index]));
            let expected = binary_reference(op_name, a, b);
            let got_ir = f64::from(ir_values[index]);
            let got_literal = f64::from(literal_values[index]);
            assert!(
                close(got_ir, got_literal),
                "{op_name}({a}, {b}): IR gave {got_ir}, shipped literal gave {got_literal}"
            );
            assert!(
                close(got_ir, expected),
                "{op_name}({a}, {b}): IR gave {got_ir}, host reference is {expected}"
            );
        }
    }
}

/// The symbolically differentiated backward must match a host reference.
///
/// This is the property the IR exists for. The backend currently ships a
/// separate hand-written derivative literal per operation, so an agreement here
/// is what would let those literals be deleted rather than maintained.
#[test]
#[ignore = "requires CUDA hardware"]
fn ir_symbolic_derivatives_match_a_numerical_reference() {
    if !has_cuda() {
        return;
    }
    // Operations whose derivative is well-defined and continuous on the signed
    // sample. `relu`, `abs`, `step` and `sign` are excluded: their derivatives
    // are discontinuous at zero and a central difference across the kink does
    // not converge to the symbolic answer.
    let differentiable = [
        "neg",
        "log",
        "exp",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "swish",
        "gelu",
        "square",
        "reciprocal",
        "elu",
        "mish",
        "log2",
        "log10",
    ];

    for op_name in differentiable {
        let sample = match op_name {
            "log" | "log2" | "log10" | "sqrt" | "rsqrt" => positive_sample(),
            _ => signed_sample(),
        };
        let input = storage(&[sample.len()], sample.clone());

        let forward = catalog::unary_forward(op_name).unwrap();
        let derivative = forward.diff(0);
        let body = lower_unary_body(&derivative, DTypeId::F32).unwrap();
        let from_ir =
            launch_unary_body(probe_name("irgrad", op_name), &body, &input, NO_PROOF).unwrap();
        let ir_values = download_f32_host(&from_ir).unwrap();

        for (index, &x) in sample.iter().enumerate() {
            let x = f64::from(x);
            // Central difference, with a step scaled to the magnitude of the
            // point so it stays well above f64 rounding without leaving the
            // region where the function is locally linear.
            let h = 1e-4 * x.abs().max(1.0);
            let numerical =
                (unary_reference(op_name, x + h) - unary_reference(op_name, x - h)) / (2.0 * h);
            let symbolic = f64::from(ir_values[index]);
            let tolerance = 2e-3 * symbolic.abs().max(numerical.abs()).max(1.0);
            assert!(
                (symbolic - numerical).abs() <= tolerance,
                "d/dx {op_name}({x}): IR gave {symbolic}, central difference gives {numerical}"
            );
        }
    }
}

/// Two kernels that differ only in a baked-in constant must not share a cache
/// entry.
///
/// `KernelKey::cache_id` is built from the operation name, dtype, layout and
/// access pattern. It does not include the kernel source. Every caller that
/// formats a runtime value into its expression while passing a fixed name
/// therefore collides with itself: `cuda_powf_storage` renders
/// `powf(x, <exp>)` under the constant name `"powf"`, `cuda_clamp_storage`
/// renders its bounds under `"clamp"`, and `mean`'s backward renders
/// `x * <1/axis_len>` under `"mul_scalar"`. The first exponent, bound or axis
/// length to be compiled in a process wins, and every later call silently reuses
/// it.
///
/// This test pins the behaviour so a fix is observable. It is written as the
/// correctness assertion, so it fails while the defect is present.
#[test]
#[ignore = "requires CUDA hardware"]
fn a_baked_in_constant_participates_in_the_kernel_cache_key() {
    use crate::cuda::backend::elementwise::{cuda_clamp_storage, cuda_powf_storage};

    if !has_cuda() {
        return;
    }
    let values = vec![2.0_f32, 3.0, 4.0];
    let input = storage(&[values.len()], values.clone());

    let squared = download_f32_host(&cuda_powf_storage(&input, 2.0).unwrap()).unwrap();
    let cubed = download_f32_host(&cuda_powf_storage(&input, 3.0).unwrap()).unwrap();

    for (index, &x) in values.iter().enumerate() {
        assert!(
            close(f64::from(squared[index]), f64::from(x).powf(2.0)),
            "powf({x}, 2) gave {}",
            squared[index]
        );
        assert!(
            close(f64::from(cubed[index]), f64::from(x).powf(3.0)),
            "powf({x}, 3) gave {} after powf(.., 2) was compiled under the same \
             cache key",
            cubed[index]
        );
    }

    let narrow = download_f32_host(&cuda_clamp_storage(&input, 0.0, 2.5).unwrap()).unwrap();
    let wide = download_f32_host(&cuda_clamp_storage(&input, 0.0, 10.0).unwrap()).unwrap();
    assert!(
        close(f64::from(narrow[2]), 2.5),
        "clamp(4, 0, 2.5) gave {}",
        narrow[2]
    );
    assert!(
        close(f64::from(wide[2]), 4.0),
        "clamp(4, 0, 10) gave {} after clamp(.., 0, 2.5) was compiled under the \
         same cache key",
        wide[2]
    );
}

/// One fused kernel must equal the two-launch backward it replaces.
///
/// The shipped backward for a unary operation launches `f'(x)` into a temporary
/// and then multiplies that temporary by the incoming gradient.
/// `catalog::unary_fused_backward` emits `grad_out * f'(x)` as a single binary
/// kernel. This checks the fused result against the two-step one computed
/// through the same launchers, so any difference is the fusion's fault and not a
/// difference in how the operands were read.
#[test]
#[ignore = "requires CUDA hardware"]
fn the_fused_backward_matches_the_two_launch_backward() {
    if !has_cuda() {
        return;
    }
    let fusible = [
        "neg",
        "log",
        "exp",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "swish",
        "gelu",
        "square",
        "reciprocal",
        "relu",
        "abs",
    ];

    for op_name in fusible {
        let sample = match op_name {
            "log" | "log2" | "log10" | "sqrt" | "rsqrt" => positive_sample(),
            _ => signed_sample(),
        };
        // A non-constant incoming gradient, so a fusion that dropped the
        // multiply entirely would be caught.
        let gradient: Vec<f32> = (0..sample.len())
            .map(|index| 0.5 + index as f32 * 0.25)
            .collect();

        let x = storage(&[sample.len()], sample.clone());
        let grad_out = storage(&[gradient.len()], gradient.clone());
        let out_shape = [sample.len()];

        // Two launches: derivative into a temporary, then multiply.
        let derivative = catalog::unary_forward(op_name).unwrap().diff(0);
        let deriv_body = lower_unary_body(&derivative, DTypeId::F32).unwrap();
        let deriv =
            launch_unary_body(probe_name("twostep_d", op_name), &deriv_body, &x, NO_PROOF).unwrap();
        let two_step = launch_binary_body(
            probe_name("twostep_m", op_name),
            &lower_binary_body(&catalog::binary_forward("mul").unwrap(), DTypeId::F32).unwrap(),
            &grad_out,
            &deriv,
            &out_shape,
            NO_PROOF,
        )
        .unwrap();

        // One launch.
        let fused_ir = catalog::unary_fused_backward(op_name).unwrap();
        let fused_body = lower_binary_body(&fused_ir, DTypeId::F32).unwrap();
        let fused = launch_binary_body(
            probe_name("fused", op_name),
            &fused_body,
            &grad_out,
            &x,
            &out_shape,
            NO_PROOF,
        )
        .unwrap();

        let two_step_values = download_f32_host(&two_step).unwrap();
        let fused_values = download_f32_host(&fused).unwrap();

        for index in 0..sample.len() {
            let expected = f64::from(two_step_values[index]);
            let got = f64::from(fused_values[index]);
            assert!(
                close(got, expected),
                "fused d/dx {op_name}({}) * {} gave {got}, two-launch gave {expected}",
                sample[index],
                gradient[index]
            );
        }
    }
}

/// The shipped autograd path must still produce correct gradients now that the
/// fused kernel is wired into `cuda_pointwise!`.
///
/// [`the_fused_backward_matches_the_two_launch_backward`] proves the fused
/// expression is right. This proves the *wiring* is right: that the real tape
/// entry passes the incoming gradient and the forward input in the order the
/// fused kernel expects, at the shape it expects. A transposed operand pair
/// would still be a valid kernel and would still return numbers, so only running
/// the actual backward catches it.
#[test]
#[ignore = "requires CUDA hardware"]
fn the_shipped_backward_is_correct_for_every_fused_operation() {
    use crate::cuda::backend::elementwise as ops;

    if !has_cuda() {
        return;
    }

    // The `unary_wrt_input` operations the catalog covers, so exactly the set
    // that now takes the fused path. Anything else in the macro still runs its
    // hand-written two-launch backward.
    #[allow(clippy::type_complexity)]
    let fused: [(
        &str,
        fn(&CudaStorage, KernelSpecialization) -> incin_core::error::Result<CudaStorage>,
    ); 12] = [
        ("relu", ops::cuda_relu_storage),
        ("gelu", ops::cuda_gelu_storage),
        ("abs", ops::cuda_abs_storage),
        ("neg", ops::cuda_neg_storage),
        ("log", ops::cuda_log_storage),
        ("swish", ops::cuda_swish_storage),
        ("sin", ops::cuda_sin_storage),
        ("cos", ops::cuda_cos_storage),
        ("elu", ops::cuda_elu_storage),
        ("mish", ops::cuda_mish_storage),
        ("log2", ops::cuda_log2_storage),
        ("log10", ops::cuda_log10_storage),
    ];

    for (op_name, forward) in fused {
        assert!(
            catalog::unary_fused_backward(op_name).is_some(),
            "{op_name} is expected to take the fused path but the catalog does not cover it"
        );
        let sample = match op_name {
            "log" | "log2" | "log10" => positive_sample(),
            _ => signed_sample(),
        };
        let input = storage(&[sample.len()], sample.clone());
        let input_id = input.id;

        let out = forward(&input, NO_PROOF).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let grad = grads
            .get(input_id)
            .unwrap_or_else(|| panic!("{op_name} produced no gradient for its input"));
        let grad_values = download_f32_host(grad).unwrap();

        // `backward` seeds the output gradient with ones, so the result is
        // exactly f'(x).
        for (index, &x) in sample.iter().enumerate() {
            let x = f64::from(x);
            let h = 1e-4 * x.abs().max(1.0);
            let numerical =
                (unary_reference(op_name, x + h) - unary_reference(op_name, x - h)) / (2.0 * h);
            let got = f64::from(grad_values[index]);
            let tolerance = 2e-3 * got.abs().max(numerical.abs()).max(1.0);
            assert!(
                (got - numerical).abs() <= tolerance,
                "shipped d/dx {op_name}({x}) gave {got}, central difference gives {numerical}"
            );
        }
    }
}

/// A proven element count must remove the packed kernel's ragged-tail branch,
/// and must not change a single result.
///
/// This is the first thing in any backend that reads the frontend's shape proof.
/// `ShapeEvidence` has ridden along on every `Validated` descriptor since it was
/// introduced, carrying `proof`, `static_rank` and `static_numel`, and until now
/// nothing opened it.
///
/// The distinction being tested is not "does the kernel know the numel" -- the
/// launcher always knows that at runtime. It is that a *statically* known count
/// is a constant of the program, so specialising on it produces a bounded number
/// of kernels rather than one per observed shape.
#[test]
#[ignore = "requires CUDA hardware"]
fn a_proven_element_count_elides_the_packed_tail() {
    if !has_cuda() {
        return;
    }
    // f32 packs four lanes wide, so 16 divides it and 14 does not.
    let divisible = KernelSpecialization {
        static_numel: Some(16),
        static_extents: &[Some(16)],
    };
    let indivisible = KernelSpecialization {
        static_numel: Some(14),
        static_extents: &[Some(14)],
    };
    assert!(divisible.packed_tail_is_dead(4));
    assert!(!indivisible.packed_tail_is_dead(4));
    assert!(
        !NO_PROOF.packed_tail_is_dead(4),
        "an unproven count must never license eliding the tail"
    );

    let values: Vec<f32> = (0..16).map(|index| index as f32 - 8.0).collect();
    let input = storage(&[values.len()], values.clone());
    let body = lower_unary_body(&catalog::unary_forward("relu").unwrap(), DTypeId::F32).unwrap();

    let specialized = launch_unary_body(probe_name("spec", "relu"), &body, &input, divisible)
        .expect("a specialized packed kernel must compile and launch");
    let general = launch_unary_body(probe_name("gen", "relu"), &body, &input, NO_PROOF)
        .expect("the unspecialized kernel must still work");

    let specialized_values = download_f32_host(&specialized).unwrap();
    let general_values = download_f32_host(&general).unwrap();

    for (index, &x) in values.iter().enumerate() {
        let expected = f64::from(x).max(0.0);
        assert!(
            close(f64::from(specialized_values[index]), expected),
            "specialized relu({x}) gave {}",
            specialized_values[index]
        );
        assert!(
            close(
                f64::from(specialized_values[index]),
                f64::from(general_values[index])
            ),
            "specialization changed relu({x})"
        );
    }
}

/// Absent evidence must specialize nothing.
///
/// A backend cannot fabricate a `ShapeEvidence`: `ShapeEvidence::of` is
/// `pub(crate)` to `incin-core`, so the only way to obtain one is from a
/// `Validated` descriptor that a lowering rule produced. That is the provenance
/// guarantee `exec::proof` exists to provide, and it is why this test can check
/// the `None` case but not construct the `Static` one -- the corresponding
/// assertion for a real shape type lives in `incin-core`'s `proof_provenance`
/// suite, on the side of the boundary that is allowed to build the value.
#[test]
fn absent_evidence_specializes_nothing() {
    assert_eq!(
        KernelSpecialization::from_evidence(None),
        KernelSpecialization::NONE
    );
    assert!(!KernelSpecialization::NONE.packed_tail_is_dead(4));
}

/// The specialization must be visible in the emitted source, not merely
/// harmless.
///
/// A passing numerical test would also pass if the specialization silently did
/// nothing, so this asserts on the text: the scalar tail disappears when the
/// count is proven divisible, and survives when it is not.
#[test]
fn the_elided_tail_is_absent_from_the_emitted_source() {
    use crate::codegen::ScalarFragment;
    use crate::kernel::render_cuda_unary_packed_body;
    use incin_core::exec::LayoutClass;

    let body = ScalarFragment::literal("x");
    let render = |spec| {
        render_cuda_unary_packed_body("probe", &body, DTypeId::F32, LayoutClass::Contiguous, spec)
            .unwrap()
            .source
    };

    let general = render(NO_PROOF);
    let specialized = render(KernelSpecialization {
        static_numel: Some(16),
        static_extents: &[Some(16)],
    });

    assert!(
        general.contains("for (int lane"),
        "the general kernel should carry a scalar tail loop"
    );
    assert!(
        !specialized.contains("for (int lane"),
        "a proven count should remove the scalar tail loop:\n{specialized}"
    );
    assert!(
        specialized.contains("packed_output"),
        "the packed body must survive specialization:\n{specialized}"
    );
    // Both still guard the grid overhang, which a proven numel does not license
    // removing: the launch rounds up to whole blocks regardless.
    assert!(general.contains("if (base >= numel)"));
    assert!(specialized.contains("if (base >= numel)"));
}

/// Proven extents must replace the strided kernel's loaded divisors with
/// literals, and must not change an address.
///
/// The strided walk costs one modulo and one division per axis per element, and
/// a divisor living in device memory cannot be strength-reduced -- integer
/// division stays integer division. Substituting the extents as literals lets
/// nvcc lower each one to a multiply-and-shift.
///
/// Strides are deliberately *not* substituted: they describe the view, so a
/// transposed or narrowed tensor has strides no shape type can settle.
#[test]
fn proven_extents_replace_the_strided_divisors_with_literals() {
    use crate::codegen::ScalarFragment;
    use crate::kernel::render_cuda_unary_for_layout_body;
    use incin_core::exec::LayoutClass;

    let body = ScalarFragment::literal("x");
    let render = |extents| {
        render_cuda_unary_for_layout_body(
            "probe",
            &body,
            DTypeId::F32,
            LayoutClass::Strided,
            1,
            extents,
        )
        .unwrap()
        .source
    };

    let general = render(None);
    let unrolled = render(Some(alloc::vec![3, 4]));

    assert!(
        general.contains("temp % shape[i]"),
        "the general kernel divides by a loaded extent"
    );
    assert!(
        !unrolled.contains("shape[i]"),
        "a proven shape must not read extents from memory:\n{unrolled}"
    );
    assert!(
        unrolled.contains("temp % 4") && unrolled.contains("temp % 3"),
        "each extent should appear as a literal divisor:\n{unrolled}"
    );
    // Strides stay dynamic in both: they are a property of the view.
    assert!(general.contains("strides[i]"));
    assert!(unrolled.contains("strides[1]") && unrolled.contains("strides[0]"));

    // The signature must shed the parameters the unrolled walk no longer reads.
    // This is the half the launcher has to agree with: it supplies a shorter
    // argument list for exactly this kernel, and a mismatch would be read as
    // corrupt output rather than reported as an error.
    assert!(
        general.contains("const int* shape,") && general.contains("int ndim"),
        "the general kernel takes the extents as parameters"
    );
    assert!(
        !unrolled.contains("const int* shape,"),
        "a baked-extent kernel must not declare the shape parameter:\n{unrolled}"
    );
    assert!(
        !unrolled.contains("int ndim"),
        "a baked-extent kernel must not declare ndim:\n{unrolled}"
    );
    assert!(
        unrolled.contains("const int* strides,"),
        "strides are still a parameter, since the view supplies them:\n{unrolled}"
    );
    // The outermost axis consumes what is left, so its division is dead.
    assert!(
        !unrolled.contains("temp /= 3"),
        "the outermost division is dead and should not be emitted:\n{unrolled}"
    );
}

/// The unrolled index walk must address a genuinely non-contiguous view
/// identically to the loaded-divisor one.
///
/// The view is built directly from parts rather than by calling `transpose`,
/// and that is the point. Every CUDA operation that could produce a
/// non-contiguous result -- `transpose`, `narrow`, `broadcast` -- materialises
/// into a fresh contiguous buffer instead, so no public operation reaches the
/// strided kernel at all. An earlier version of this test called `transpose`
/// and believed it was exercising that path; it was not, and it passed for the
/// wrong reason. See `docs/plan/research/0.2.0/proof-directed-codegen.md`.
///
/// Constructing the view by hand keeps the kernel honest while the path is
/// otherwise unreachable, and it is exactly the case that becomes reachable if
/// views ever stop materialising.
#[test]
#[ignore = "requires CUDA hardware"]
fn the_unrolled_index_walk_addresses_a_strided_view_correctly() {
    if !has_cuda() {
        return;
    }
    // A 3x4 row-major buffer, viewed as its 4x3 transpose without copying:
    // shape [4, 3] with strides [1, 4] reads element (r, c) from base (c, r).
    let values: Vec<f32> = (0..12).map(|index| index as f32).collect();
    let base = storage(&[3, 4], values.clone());
    let view =
        CudaStorage::try_from_parts(base.buffer.clone(), alloc::vec![4, 3], alloc::vec![1, 4], 0)
            .expect("a transposed view of a 3x4 buffer is valid metadata");

    // Confirm this really is the strided path before relying on the test.
    let plan = crate::iteration::UnaryIterationPlan::new(crate::iteration::OperandLayout {
        shape: &view.shape,
        strides: &view.strides,
        offset: view.offset_elements,
    })
    .unwrap();
    assert_eq!(
        plan.layout_class(),
        incin_core::exec::LayoutClass::Strided,
        "the view must be non-contiguous or this test proves nothing"
    );

    let body = lower_unary_body(&catalog::unary_forward("neg").unwrap(), DTypeId::F32).unwrap();
    let proven = KernelSpecialization {
        static_numel: Some(12),
        static_extents: &[Some(4), Some(3)],
    };

    let unrolled = launch_unary_body(probe_name("unroll", "neg"), &body, &view, proven)
        .expect("the unrolled strided kernel must compile and launch");
    let general = launch_unary_body(probe_name("dynidx", "neg"), &body, &view, NO_PROOF)
        .expect("the general strided kernel must still work");

    let unrolled_values = download_f32_host(&unrolled).unwrap();
    let general_values = download_f32_host(&general).unwrap();

    for row in 0..4usize {
        for col in 0..3usize {
            let index = row * 3 + col;
            let expected = -(values[col * 4 + row]);
            assert!(
                close(f64::from(unrolled_values[index]), f64::from(expected)),
                "unrolled neg at [{row},{col}] gave {}, expected {expected}",
                unrolled_values[index]
            );
            assert!(
                close(
                    f64::from(unrolled_values[index]),
                    f64::from(general_values[index])
                ),
                "the unrolled walk disagreed with the loaded-divisor walk at [{row},{col}]"
            );
        }
    }
}

/// The unroll must decline when the proof disagrees with the iterated shape.
///
/// A mismatch would emit literal divisors for a geometry the kernel is not
/// walking, which silently computes wrong addresses rather than failing. The
/// guard is what makes substituting literals safe.
#[test]
fn the_unroll_declines_when_the_proof_does_not_match_the_iterated_shape() {
    let proven = KernelSpecialization {
        static_numel: Some(12),
        static_extents: &[Some(3), Some(4)],
    };
    assert_eq!(proven.unrollable_extents(&[3, 4]), Some(alloc::vec![3, 4]));
    assert_eq!(
        proven.unrollable_extents(&[4, 3]),
        None,
        "a permuted shape is a different geometry and must not be unrolled"
    );
    assert_eq!(
        proven.unrollable_extents(&[3, 4, 1]),
        None,
        "rank must agree"
    );
    assert_eq!(proven.unrollable_extents(&[]), None);

    // One runtime axis disqualifies the whole unroll: the loop needs every
    // divisor to be a literal.
    let partly = KernelSpecialization {
        static_numel: None,
        static_extents: &[None, Some(4)],
    };
    assert_eq!(partly.unrollable_extents(&[3, 4]), None);
    assert_eq!(NO_PROOF.unrollable_extents(&[3, 4]), None);
}

/// SSA lowering must not duplicate a repeated subexpression's text.
///
/// This is the property that makes the fragment renderer usable for fusion: a
/// chain of operations that reuse a value has to stay linear in the node count.
/// The direct renderer in `ir.rs` inlines each use, so the same expression grows
/// exponentially there.
#[test]
fn repeated_subexpressions_are_evaluated_once() {
    use crate::codegen::ir::IrExpr;

    // gelu(gelu(gelu(x))): each level uses its operand four times.
    let deep = IrExpr::arg(0).gelu().gelu().gelu();
    let fragment = lower_unary_body(&deep, DTypeId::F32).unwrap();

    // Three nested gelus with four textual uses each would be 4^3 = 64
    // occurrences of `x` under the inlining renderer. SSA gives one binding per
    // distinct node, so the count is bounded by the node count.
    let occurrences = fragment
        .prologue
        .iter()
        .map(|statement| statement.matches('x').count())
        .sum::<usize>();
    assert!(
        occurrences <= 8,
        "SSA lowering duplicated the operand {occurrences} times: {:#?}",
        fragment.prologue
    );

    // And the shared subterm inside one gelu -- `k * (v + c * v * v * v)` --
    // must be bound once rather than recomputed for the tanh and for the
    // multiply.
    let single = lower_unary_body(&IrExpr::arg(0).gelu(), DTypeId::F32).unwrap();
    assert_eq!(
        single.prologue.len(),
        1,
        "a single gelu should need exactly one binding: {:#?}",
        single.prologue
    );
}
