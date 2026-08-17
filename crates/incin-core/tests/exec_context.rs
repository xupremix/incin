//! GRD-001: the explicit execution context and the scoped policy default.
//!
//! Two things are under test and they are not the same thing. An
//! `ExecutionContext` is an owned value a caller passes explicitly, and its
//! policy is whatever that caller put there. An `ExecutionPolicy` scope is a
//! per-thread ambient default for the eager convenience form. PROPOSALS.md
//! sec. 1.2.5 calls the explicit context canonical and thread-safe, and the
//! scope a convenience, so the interesting cases are the ones where a scope
//! could leak: out of a nested scope, across a thread, or out of a panic.

use std::sync::mpsc;
use std::thread;

use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::{ExecutionContext, ExecutionPolicy, FallbackPolicy, MathMode, TensorMeta};
use incin_core::prelude::{Cpu, DType};

/// A backend that owns no device and executes nothing. The context under test
/// is generic over `StorageBackend`, so what this one does is irrelevant; what
/// matters is that it is a distinct value the context can be observed to own.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Probe(u32);

#[derive(Clone)]
struct Storage {
    metadata: TensorMeta,
}

impl StorageBackend for Probe {
    const BACKEND_NAME: &'static str = "Probe";
    type Storage<K: DType> = Storage;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.metadata
    }
}

/// A policy with every axis set away from its default, so a test asserting one
/// was carried cannot pass by accident on a value that happens to be the
/// default.
fn every_axis_moved() -> ExecutionPolicy {
    ExecutionPolicy::new()
        .with_math_mode(MathMode::Fast)
        .with_fallback(FallbackPolicy::AllowTransfer)
        .with_training(true)
}

#[test]
fn a_fresh_context_allows_composition_but_denies_transfer() {
    let context = ExecutionContext::new(Probe(1));

    // Spelled out rather than compared against `ExecutionPolicy::default()` so
    // the selected composition/transfer boundary remains visible.
    assert_eq!(context.math_mode(), MathMode::Precise);
    assert_eq!(context.fallback(), FallbackPolicy::AllowComposition);
    assert!(context.fallback().allows_composition());
    assert!(!context.fallback().allows_transfer());
    assert!(!context.training());
}

#[test]
fn each_builder_moves_one_axis_and_leaves_the_others_alone() {
    let base = ExecutionContext::new(Probe(2));

    let fast = ExecutionContext::new(Probe(2)).with_math_mode(MathMode::Fast);
    assert_eq!(fast.math_mode(), MathMode::Fast);
    assert_eq!(fast.fallback(), base.fallback());

    let composing = ExecutionContext::new(Probe(2)).with_fallback(FallbackPolicy::AllowComposition);
    assert_eq!(composing.fallback(), FallbackPolicy::AllowComposition);

    let training = ExecutionContext::new(Probe(2)).with_training(true);
    assert!(training.training());
    assert_eq!(training.grad_mode(), base.grad_mode());
}

#[test]
fn transfer_implies_composition_but_composition_does_not_imply_transfer() {
    assert!(FallbackPolicy::AllowTransfer.allows_composition());
    assert!(FallbackPolicy::AllowTransfer.allows_transfer());
    assert!(FallbackPolicy::AllowComposition.allows_composition());
    assert!(!FallbackPolicy::AllowComposition.allows_transfer());
    assert!(!FallbackPolicy::Deny.allows_composition());
    assert!(!FallbackPolicy::Deny.allows_transfer());
}

#[test]
fn the_backend_survives_every_builder_and_comes_back_out_unchanged() {
    let context = ExecutionContext::new(Probe(7))
        .with_math_mode(MathMode::Fast)
        .with_fallback(FallbackPolicy::AllowTransfer)
        .with_training(true);

    assert_eq!(context.backend(), &Probe(7));
    assert_eq!(context.policy(), every_axis_moved());
    assert_eq!(context.into_backend(), Probe(7));
}

#[test]
fn a_context_built_from_a_scope_keeps_its_policy_after_the_scope_ends() {
    let context = every_axis_moved().scope(|| ExecutionContext::from_scope(Probe(3)));

    // The scope is over. An explicit context is an owned value, so it must
    // still hold what it read, not fall back to the ambient default.
    assert_eq!(ExecutionPolicy::current(), ExecutionPolicy::new());
    assert_eq!(context.policy(), every_axis_moved());
}

#[test]
fn scopes_nest_and_the_inner_one_restores_the_outer_rather_than_the_default() {
    let outer = ExecutionPolicy::new().with_math_mode(MathMode::Fast);
    let inner = ExecutionPolicy::new().with_math_mode(MathMode::Precise);

    assert_eq!(ExecutionPolicy::current(), ExecutionPolicy::new());

    outer.scope(|| {
        assert_eq!(ExecutionPolicy::current(), outer);

        inner.scope(|| {
            assert_eq!(ExecutionPolicy::current(), inner);

            // A third level, to prove the restore is a stack and not a single
            // saved slot that the second entry would have overwritten.
            let innermost = inner.with_training(true);
            innermost.scope(|| assert_eq!(ExecutionPolicy::current(), innermost));

            assert_eq!(ExecutionPolicy::current(), inner);
        });

        assert_eq!(
            ExecutionPolicy::current(),
            outer,
            "leaving the inner scope restored the default instead of the enclosing scope"
        );
    });

    assert_eq!(ExecutionPolicy::current(), ExecutionPolicy::new());
}

#[test]
fn a_panic_unwinding_out_of_a_scope_still_restores_the_enclosing_policy() {
    let outer = ExecutionPolicy::new().with_fallback(FallbackPolicy::AllowComposition);

    outer.scope(|| {
        let panicked = std::panic::catch_unwind(|| {
            every_axis_moved().scope(|| panic!("the body of a scope failed"));
        });
        assert!(panicked.is_err(), "the test's own panic did not happen");

        assert_eq!(
            ExecutionPolicy::current(),
            outer,
            "an unwinding panic left a policy no enclosing scope asked for"
        );
    });

    assert_eq!(ExecutionPolicy::current(), ExecutionPolicy::new());
}

#[test]
fn one_thread_s_scope_is_invisible_to_every_other_thread() {
    // Each worker installs a policy no other worker uses, holds it across a
    // rendezvous so all of them are inside their scope at the same moment,
    // then reports what it sees. Overlapping the scopes is the point: a
    // process-global default would pass a test where the scopes ran in
    // sequence.
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = std::sync::Mutex::new(release_rx);

    let workers = [
        ExecutionPolicy::new().with_math_mode(MathMode::Fast),
        ExecutionPolicy::new().with_fallback(FallbackPolicy::AllowTransfer),
        ExecutionPolicy::new().with_training(true),
    ];

    let observed = thread::scope(|scope| {
        let handles: Vec<_> = workers
            .iter()
            .map(|policy| {
                let ready_tx = ready_tx.clone();
                let release_rx = &release_rx;
                scope.spawn(move || {
                    policy.scope(|| {
                        ready_tx.send(()).expect("the test thread is still alive");
                        release_rx
                            .lock()
                            .expect("no worker panics while holding this")
                            .recv()
                            .expect("the test thread releases every worker");
                        ExecutionPolicy::current()
                    })
                })
            })
            .collect();

        for _ in &workers {
            ready_rx.recv().expect("every worker reaches its scope");
        }
        // Every worker is now inside its own scope simultaneously.
        assert_eq!(
            ExecutionPolicy::current(),
            ExecutionPolicy::new(),
            "a worker's scope was visible from the thread that spawned it"
        );
        for _ in &workers {
            release_tx.send(()).expect("every worker is still waiting");
        }

        handles
            .into_iter()
            .map(|handle| handle.join().expect("no worker panics"))
            .collect::<Vec<_>>()
    });

    assert_eq!(observed, workers.to_vec());
    assert_eq!(ExecutionPolicy::current(), ExecutionPolicy::new());
}

#[test]
fn an_explicit_context_ignores_whatever_scope_it_is_used_inside() {
    // The canonical interface is the explicit one. A context constructed with
    // `new` carries the default policy, and entering a scope must not reach
    // into a value someone already holds.
    let context = ExecutionContext::new(Probe(9));

    every_axis_moved().scope(|| {
        assert_eq!(ExecutionPolicy::current(), every_axis_moved());
        assert_eq!(context.policy(), ExecutionPolicy::new());
        assert_eq!(context.fallback(), FallbackPolicy::AllowComposition);
    });
}
