//! `GRD-002`: `G` to `GradMode` propagation, and the guarantee that follows.
//!
//! PROPOSALS.md sec. 1.2.5 states it as an obligation rather than a
//! preference: "`NoGrad` must produce no autograd node and save no
//! backward-only tensor." Before this row nothing carried the marker down to
//! the layer that records — `cpu/tape.rs` said so at the declaration of
//! `push`, per D-05 — so the obligation was unmet and untestable in the same
//! breath.
//!
//! The tests below are in three groups, and the order matters. The first two
//! cover the derivation and the scope in isolation, because a failure there
//! should not present as a mysterious tape count. The third runs real kernels
//! through the CPU backend and counts what landed on its tape, which is the
//! only thing that actually discharges the obligation: everything above it is
//! a claim about a mechanism, and this is a claim about a result.

extern crate incin_core as incin;

use std::panic;
use std::sync::mpsc;
use std::thread;

use incin_backends::cpu::{CpuBackendImpl, tape_depth};
use incin_core::exec::{
    AllocatorPolicy, Determinism, ExecutionContext, ExecutionPolicy, FallbackPolicy, GradMode,
    MathMode,
};
use incin_core::prelude::*;
use incin_macros::s;

type B = CpuBackendImpl;

// ── The derivation ───────────────────────────────────────────────────────────

#[test]
fn each_marker_derives_the_mode_its_name_promises() {
    assert_eq!(Grad::grad_mode(&Default::default()), GradMode::Enabled);
    assert_eq!(NoGrad::grad_mode(&Default::default()), GradMode::Disabled);
    assert_eq!(<Dyn as RequiresGrad>::grad_mode(&true), GradMode::Enabled);
    assert_eq!(<Dyn as RequiresGrad>::grad_mode(&false), GradMode::Disabled);
}

#[test]
fn the_mode_cannot_disagree_with_requires_grad() {
    // The derivation is a default body over `requires_grad`, so this holds by
    // construction — which is the point. Asserting it here is what makes a
    // future impl that overrides `grad_mode` into a test failure rather than a
    // tensor that claims gradients and silently records none.
    for (field, marker) in [(true, "Dyn(true)"), (false, "Dyn(false)")] {
        assert_eq!(
            <Dyn as RequiresGrad>::grad_mode(&field).records(),
            <Dyn as RequiresGrad>::requires_grad(&field),
            "{marker}"
        );
    }
    assert!(Grad::grad_mode(&Default::default()).records());
    assert!(!NoGrad::grad_mode(&Default::default()).records());
}

#[test]
fn combining_two_modes_takes_the_stricter_one() {
    use GradMode::{Disabled, Enabled};
    assert_eq!(Enabled.and(Enabled), Enabled);
    assert_eq!(Enabled.and(Disabled), Disabled);
    assert_eq!(Disabled.and(Enabled), Disabled);
    assert_eq!(Disabled.and(Disabled), Disabled);
}

// ── The scope ────────────────────────────────────────────────────────────────

#[test]
fn a_fresh_thread_permits_recording() {
    assert_eq!(GradMode::current(), GradMode::Enabled);
    assert_eq!(ExecutionPolicy::new().grad_mode, GradMode::Enabled);
    assert_eq!(
        ExecutionContext::new(B::default()).grad_mode(),
        GradMode::Enabled
    );
}

#[test]
fn a_context_carries_the_mode_it_was_built_with() {
    let context = ExecutionContext::new(B::default()).with_grad_mode(GradMode::Disabled);
    assert_eq!(context.grad_mode(), GradMode::Disabled);
    // The other four axes are untouched. Grouping them into one policy value
    // makes it possible to set one and clobber the rest, so this is the
    // assertion that catches a builder written as a whole-policy assignment.
    assert_eq!(context.math_mode(), MathMode::Precise);
    assert_eq!(context.determinism(), Determinism::Permitted);
    assert_eq!(context.fallback(), FallbackPolicy::Deny);
    assert_eq!(context.allocator(), AllocatorPolicy::Direct);
}

#[test]
fn a_no_grad_scope_leaves_the_other_policy_axes_alone() {
    let moved = ExecutionPolicy::new()
        .with_math_mode(MathMode::Fast)
        .with_determinism(Determinism::Required);

    moved.scope(|| {
        GradMode::Disabled.scope(|| {
            assert_eq!(GradMode::current(), GradMode::Disabled);
            assert_eq!(ExecutionPolicy::current().math_mode, MathMode::Fast);
            assert_eq!(
                ExecutionPolicy::current().determinism,
                Determinism::Required
            );
        });
        assert_eq!(GradMode::current(), GradMode::Enabled);
    });
}

#[test]
fn restrict_tightens_and_never_loosens() {
    // This is the asymmetry the whole design rests on. An operand's mode may
    // switch recording off; it may not switch it back on, or a `no_grad` block
    // would be undone by the first `Grad` tensor inside it.
    GradMode::Disabled.scope(|| {
        assert_eq!(GradMode::current(), GradMode::Disabled);
        GradMode::Enabled.restrict(|| {
            assert_eq!(GradMode::current(), GradMode::Disabled);
        });
    });

    GradMode::Disabled.restrict(|| assert_eq!(GradMode::current(), GradMode::Disabled));
    assert_eq!(GradMode::current(), GradMode::Enabled);
}

#[test]
fn an_explicit_scope_does_re_enable_recording() {
    // The other half of the same rule, and the reason `restrict` and `scope`
    // are separate: a caller who names `GradMode::Enabled` is asking for it,
    // whereas an operand that merely permits recording is not asking for
    // anything.
    GradMode::Disabled.scope(|| {
        GradMode::Enabled.scope(|| assert_eq!(GradMode::current(), GradMode::Enabled));
        assert_eq!(GradMode::current(), GradMode::Disabled);
    });
}

#[test]
fn nested_scopes_restore_their_enclosing_mode_not_the_default() {
    GradMode::Disabled.scope(|| {
        GradMode::Enabled.scope(|| {
            GradMode::Disabled.scope(|| assert_eq!(GradMode::current(), GradMode::Disabled));
            assert_eq!(GradMode::current(), GradMode::Enabled);
        });
        assert_eq!(GradMode::current(), GradMode::Disabled);
    });
    assert_eq!(GradMode::current(), GradMode::Enabled);
}

#[test]
fn a_panic_out_of_a_no_grad_scope_does_not_poison_the_thread() {
    let escaped = panic::catch_unwind(|| {
        GradMode::Disabled.scope(|| panic!("unwinding out of a disabled gradient scope"));
    });
    assert!(escaped.is_err());
    assert_eq!(GradMode::current(), GradMode::Enabled);
}

#[test]
fn one_threads_no_grad_scope_is_invisible_to_another() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();

    let worker = thread::spawn(move || {
        GradMode::Disabled.scope(|| {
            entered_tx.send(GradMode::current()).unwrap();
            release_rx.recv().unwrap();
            GradMode::current()
        })
    });

    // Read the main thread's mode while the worker is provably still inside
    // its scope, not after it has left.
    assert_eq!(entered_rx.recv().unwrap(), GradMode::Disabled);
    assert_eq!(GradMode::current(), GradMode::Enabled);
    release_tx.send(()).unwrap();
    assert_eq!(worker.join().unwrap(), GradMode::Disabled);
}

// ── The guarantee ────────────────────────────────────────────────────────────

/// The chain every recording test below runs, over whichever `G` its operands
/// carry.
///
/// Deliberately more than one operation and more than one family: an
/// elementwise binary op, a scalar op, a shape op, and a reduction. A gate
/// installed on one family and forgotten on another passes a one-op test.
fn chain<G: RequiresGrad>(
    a: &Tensor<s![2, 3], B, f32, G>,
    b: &Tensor<s![2, 3], B, f32, G>,
) -> Result<f32> {
    let sum = a.add(b)?;
    let scaled = sum.mul_scalar(2.0)?;
    let flat = scaled.reshape::<s![6]>(((), ()))?;
    flat.sum_all()?.to_scalar::<f32>()
}

const OPERAND: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

type Operands<G> = (Tensor<s![2, 3], B, f32, G>, Tensor<s![2, 3], B, f32, G>);

fn grad_operands() -> Operands<Grad> {
    (
        Tensor::from_slice(&OPERAND, ()).unwrap(),
        Tensor::ones(()).unwrap(),
    )
}

fn nograd_operands() -> Operands<NoGrad> {
    (
        Tensor::from_slice(&OPERAND, ()).unwrap(),
        Tensor::ones(()).unwrap(),
    )
}

/// A `Dyn`-grad pair, retagged from a `NoGrad` one.
///
/// The argument system lifts a bare `bool` into the grad slot only for a shape
/// whose own argument is `()`, and `s![2, 3]`'s is not. Retagging through the
/// witnessed constructor is the supported way to say "same storage, this grad
/// field" and keeps the chain's shape the same as every other case here.
fn dyn_operands(requires_grad: bool) -> Operands<Dyn> {
    let retag = |t: Tensor<s![2, 3], B, f32, NoGrad>| {
        Tensor::<s![2, 3], B, f32, Dyn>::try_from_storage(
            t.into_inner(),
            <s![2, 3] as Shape>::try_from_dims(&[2, 3]).unwrap(),
            Default::default(),
            Default::default(),
            requires_grad,
        )
        .unwrap()
    };
    let (a, b) = nograd_operands();
    (retag(a), retag(b))
}

/// Entries this thread's tape gained while `body` ran.
///
/// A delta rather than an absolute count: the tape is thread-local and
/// `backward()` drains it, but nothing in the contract says a test starts on a
/// pristine one, and a test that silently depends on that would be a flake
/// waiting for a harness change.
fn recorded<R>(body: impl FnOnce() -> R) -> usize {
    let before = tape_depth();
    let _ = body();
    tape_depth() - before
}

#[test]
fn a_grad_chain_records_and_a_nograd_chain_records_nothing() {
    let (ga, gb) = grad_operands();
    let (na, nb) = nograd_operands();

    let with_grad = recorded(|| chain(&ga, &gb).unwrap());
    let without = recorded(|| chain(&na, &nb).unwrap());

    // The `Grad` count is asserted as "more than none" rather than as an exact
    // number, because how many entries four operations produce is the
    // backend's business and it changes when a kernel is fused. That it is
    // nonzero is this test's business: a `NoGrad` count of zero proves nothing
    // if the `Grad` one is also zero.
    assert!(
        with_grad > 0,
        "the Grad chain recorded nothing, so the NoGrad assertion below is vacuous"
    );
    assert_eq!(without, 0, "a NoGrad chain recorded {without} tape entries");
}

#[test]
fn a_dyn_tensor_records_according_to_its_runtime_flag() {
    // `Dyn` is the case a compile-time gate cannot cover, and the one where a
    // derivation from `requires_grad` earns its keep.
    let (ta, tb) = dyn_operands(true);
    let (fa, fb) = dyn_operands(false);

    assert!(recorded(|| chain(&ta, &tb).unwrap()) > 0);
    assert_eq!(recorded(|| chain(&fa, &fb).unwrap()), 0);
}

#[test]
fn a_mixed_binary_operation_records_when_any_operand_requires_grad() {
    let no_grad = Tensor::<s![2, 3], B, f32, NoGrad>::ones(()).unwrap();
    let grad = Tensor::<s![2, 3], B, f32, Grad>::ones(()).unwrap();

    assert!(recorded(|| no_grad.add(&grad).unwrap()) > 0);
}

#[test]
fn a_no_grad_scope_silences_a_grad_chain() {
    let (a, b) = grad_operands();

    assert!(recorded(|| chain(&a, &b).unwrap()) > 0);
    assert_eq!(
        recorded(|| GradMode::Disabled.scope(|| chain(&a, &b).unwrap())),
        0
    );
    // And the scope ends where it says it does.
    assert!(recorded(|| chain(&a, &b).unwrap()) > 0);
}

#[test]
fn the_index_returning_reductions_record_nothing_even_on_grad_tensors() {
    // `argmax`, `argmin`, `topk`, and `argsort` return `NoGrad` whatever they
    // were called on, so sec. 1.2.5 applies to them regardless of the
    // receiver. They are the reason propagation reads the *result's* marker
    // rather than the receiver's.
    let t = Tensor::<s![2, 3], B>::from_slice(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], ()).unwrap();

    assert_eq!(recorded(|| t.topk(2, 1, true).unwrap()), 0);
    assert_eq!(recorded(|| t.argsort(1, false).unwrap()), 0);

    // These used to be asserted without unwrapping, because the CPU kernel
    // filled an I64 buffer while `Tensor::argmax` types its result `u32`: the
    // frontend rejected the storage its own backend had just produced, and the
    // public method could not succeed at all. The kernel builds the index
    // dtype it is asked for now, so the two agree and the results are
    // unwrapped and checked.
    assert_eq!(recorded(|| t.argmax(Some(1)).unwrap()), 0);
    assert_eq!(recorded(|| t.argmax(None).unwrap()), 0);
    assert_eq!(recorded(|| t.argmin(Some(1)).unwrap()), 0);
    assert_eq!(recorded(|| t.argmin(None).unwrap()), 0);

    // Row 0 is [1, 5, 3] and row 1 is [4, 2, 6], so the maxima sit at 1 and 2
    // and the minima at 0 and 1. Asserting the values rather than only the
    // tape count is what makes the unwrap above worth having.
    assert_eq!(
        t.argmax(Some(1)).unwrap().to_vec1::<u32>().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        t.argmin(Some(1)).unwrap().to_vec1::<u32>().unwrap(),
        vec![0, 1]
    );
}

#[test]
fn silencing_the_tape_does_not_change_what_the_forward_pass_computes() {
    // The failure this rules out is a gate that skips the kernel rather than
    // the recording. All three spellings must return the identical number.
    let (ga, gb) = grad_operands();
    let (na, nb) = nograd_operands();

    let eager = chain(&ga, &gb).unwrap();
    let detached = chain(&na, &nb).unwrap();
    let scoped = GradMode::Disabled.scope(|| chain(&ga, &gb).unwrap());

    assert_eq!(eager, 54.0);
    assert_eq!(detached, eager);
    assert_eq!(scoped, eager);
}

#[test]
fn backward_still_reaches_the_parameters_it_did_before() {
    // The regression that matters most: a gate that is too eager silences the
    // ordinary training path, and every autograd test in the workspace that
    // asserts a shape rather than a movement would still pass. This one
    // asserts the movement.
    let model = incin_core::nn::Linear::<s![2, 2], B>::build(()).unwrap();
    let x = Tensor::<s![1, 2], B>::from_slice(&[1.0, 2.0], ()).unwrap();
    let y = Tensor::<s![1, 2], B>::from_slice(&[3.0, 4.0], ()).unwrap();

    let before = model
        .forward(x.clone())
        .unwrap()
        .mse_loss(&y)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let mut optim = incin_core::optim::SGD::<B>::new(model.parameters(), 0.1);
    let loss = model.forward(x.clone()).unwrap().mse_loss(&y).unwrap();
    optim.step(&loss.backward().unwrap()).unwrap();

    let after = model
        .forward(x)
        .unwrap()
        .mse_loss(&y)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        after < before,
        "one SGD step did not reduce the loss ({before} -> {after}); the tape gate is silencing a Grad path"
    );
}
