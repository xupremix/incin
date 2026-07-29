//! The backend-neutral autograd tape (`GRD-003`).
//!
//! PROPOSALS.md sec. 1.2.5 asks the core to own "a backend-neutral graph
//! containing operation kind, dependencies, saved-value handles, and backward
//! recipe", with each backend supplying only the kernels the recipe calls.
//! Before this module there were three graphs and the core owned none of them:
//! `cpu/tape.rs`, `wgpu/tape.rs`, and `cuda/tape.rs` each declared their own
//! `TensorId`, their own entry type, and their own copy of the same reverse
//! walk.
//!
//! They were not similar by accident. The walk is one algorithm — seed, drain,
//! reverse, accumulate — and writing it three times is how the CPU one earned
//! the comment marking the exact line where a bare `insert` silently dropped
//! one of two gradient contributions (`CPUBACK-05`). It is written once here,
//! and the accumulation goes through an API that has no overwrite spelling.
//!
//! What stays with the backends is what genuinely differs: how storage is
//! summed, seeded, and inspected for a non-finite value. [`TapeStorage`] is
//! exactly that list and nothing more.
//!
//! # Scope
//!
//! The thread-locals that *own* a tape are still per-backend; `GRD-006` moves
//! saved-tensor lifetime to the graph and deletes them. `GRD-004` migrates the
//! WGPU and CUDA walks onto this type, on hardware. This module is the node
//! and the walk.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::err::Result;
use crate::exec::policy::GradMode;

/// A monotonic identity tag for one backend allocation.
///
/// Backed by a global counter rather than a pointer address: pointer identity
/// is reused after a drop, which produces a tape that credits one tensor's
/// gradient to an unrelated later one, and the resulting bug is
/// hard to reproduce by construction.
///
/// One counter for the whole workspace, where each backend previously had its
/// own. Three independent counters handing out the same integers is harmless
/// only for as long as no two backends share a tape, which is precisely what
/// `GRD-006` ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    /// Allocate a fresh, never-before-seen id.
    #[must_use]
    pub fn next() -> Self {
        // Relaxed is sufficient: this counter is an identity source, not a
        // synchronization primitive guarding shared data.
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// The underlying integer, for diagnostics and stable ordering.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// What a reverse walk needs of a backend's storage, and nothing else.
///
/// Every method here is a place where the three backends genuinely differ.
/// Seeding is a device allocation, summing is a kernel, and reading a value
/// back to test it for `NaN` costs a WGPU readback and a CPU slice walk. The
/// walk itself does not differ, which is why it is not in this trait.
pub trait TapeStorage: Clone + 'static {
    /// This allocation's identity.
    fn id(&self) -> TensorId;

    /// A value shaped like `self` and filled with ones — the seed a backward
    /// pass starts from.
    fn ones_like(&self) -> Result<Self>;

    /// Sum two gradient contributions for the same tensor.
    ///
    /// Fallible because one of the three backends already is: WGPU allocates
    /// to add and CPU does not. A shared walk has to carry the weaker
    /// guarantee, and an accumulation that cannot report a failure is how a
    /// dropped contribution becomes a wrong gradient rather than an error.
    ///
    /// Both operands have already been shape-matched to their target by the
    /// recipe that produced them, so this never broadcasts.
    fn accumulate(&self, contribution: &Self) -> Result<Self>;

    /// Whether this gradient holds a `NaN` or an infinity.
    ///
    /// Only [`backward`] under [`NanCheck::Enforce`] calls this, and only because
    /// finding the operation that first produced a `NaN` is otherwise a
    /// bisection over the whole graph.
    fn has_non_finite(&self) -> bool;
}

/// A backward recipe: given the accumulated gradient of a node's output,
/// produce one gradient per input, in the same order as [`TapeNode::input_ids`].
///
/// Still infallible. `EXE-008` recorded that making it fallible touches
/// nineteen WGPU closures and belongs with the explicit gradient context;
/// `GRD-005` owns that, along with the `panic!` the NaN check still uses.
pub type BackwardFn<S> = Box<dyn Fn(&S) -> Vec<S> + Send + Sync>;

/// One recorded operation: what it produced, what it consumed, and how to run
/// it backwards.
///
/// Saved values are captured by the recipe closure rather than listed in a
/// field. That is what makes the node backend-neutral — the core never names a
/// storage type it would have to hold — and it is why refusing a push releases
/// the saved tensors: they live in the `Box` that is dropped.
pub struct TapeNode<S> {
    /// The id of the value this operation produced.
    pub output_id: TensorId,
    /// The ids of the values it consumed, in the recipe's output order.
    pub input_ids: Vec<TensorId>,
    /// How to turn an output gradient into one gradient per input.
    pub backward: BackwardFn<S>,
}

impl<S> core::fmt::Debug for TapeNode<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TapeNode")
            .field("output_id", &self.output_id)
            .field("input_ids", &self.input_ids)
            .finish_non_exhaustive()
    }
}

/// Accumulated gradients, keyed by the id of the tensor each belongs to.
///
/// The map is private. A caller asks for one tensor's gradient; nothing
/// outside needs to rewrite the result of a backward pass in place.
pub struct GradientMap<S> {
    grads: BTreeMap<TensorId, S>,
}

impl<S> GradientMap<S> {
    /// The accumulated gradient for `id`, if the backward pass reached it.
    #[must_use]
    pub fn get(&self, id: TensorId) -> Option<&S> {
        self.grads.get(&id)
    }

    /// How many tensors received a gradient.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether the backward pass reached nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Every `(id, gradient)` pair, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &S)> {
        self.grads.iter().map(|(id, g)| (*id, g))
    }
}

impl<S> Default for GradientMap<S> {
    fn default() -> Self {
        Self {
            grads: BTreeMap::new(),
        }
    }
}

/// Whether a backward walk validates each gradient as it is produced.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum NanCheck {
    /// Walk without inspecting values. The normal path.
    #[default]
    Skip,
    /// Test every contribution and every accumulation for a non-finite value.
    ///
    /// A debugging aid whose whole purpose is to fail at the operation that
    /// first produced the `NaN` rather than at the end of the pass.
    Enforce,
}

/// The recorded operations of one backward-reachable graph.
///
/// A tape is drained into the walk that consumes it, so a second [`backward`]
/// from the same loss returns only the seed. That is `D-06`, and it is
/// deliberate: invoking a recipe twice would double every gradient it feeds.
pub struct Tape<S> {
    nodes: Vec<TapeNode<S>>,
}

impl<S> Default for Tape<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Tape<S> {
    /// An empty tape.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Record `node`, unless the ambient [`GradMode`] forbids it (`GRD-002`).
    ///
    /// The gate is here, on the one function every kernel in every backend
    /// funnels through, rather than at the call sites. There are 116 of those,
    /// and a guarantee that depends on 116 correct edits — and on the next
    /// kernel author knowing the convention — is not a guarantee. A refused
    /// node is dropped on the spot, which releases the saved values its recipe
    /// captured.
    pub fn push(&mut self, node: TapeNode<S>) {
        if !GradMode::current().records() {
            return;
        }
        self.nodes.push(node);
    }

    /// How many nodes are currently recorded.
    ///
    /// Public because `GRD-002`'s guarantee is that a `NoGrad` chain records
    /// nothing, and a guarantee nothing can count is not a guarantee.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.nodes.len()
    }

    /// Whether anything is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Take every node, leaving the tape empty.
    ///
    /// The only way to obtain the input [`backward`] needs, and deliberately
    /// so: a caller holding this tape through a walk would be holding it while
    /// a recipe records onto it.
    #[must_use]
    pub fn drain(&mut self) -> Vec<TapeNode<S>> {
        core::mem::take(&mut self.nodes)
    }
}

/// Walk `nodes` backward from `loss`, returning the accumulated gradients.
///
/// Taking the nodes by value rather than borrowing a [`Tape`] is the whole
/// point of the signature. A recipe may itself record — every convolution
/// backward on the CPU backend does — so a walk that still held the tape would
/// either re-enter it or, with the tape behind a `RefCell`, panic on the
/// second borrow. `D-06` says drain before invoking anything; this makes that
/// structural, because there is no way to call the walk without having already
/// taken the nodes out. Nodes recorded *during* the walk land on the fresh
/// tape and belong to the next pass.
///
/// The order of the remaining steps is the rest of the contract:
///
/// 1. Seed `grads[loss]` with ones.
/// 2. Walk in reverse insertion order. An output nothing reached is skipped
///    rather than failed: an unreached branch is ordinary, not an error.
/// 3. A tensor consumed by two later operations receives the **sum** of both
///    contributions. Writing that as an insert instead of an accumulate is the
///    `CPUBACK-05` defect, which is why [`Entry`] is matched here rather than a
///    `contains_key` guarding an assignment.
pub fn backward<S: TapeStorage>(
    nodes: Vec<TapeNode<S>>,
    loss: &S,
    check: NanCheck,
) -> Result<GradientMap<S>> {
    let mut grads: BTreeMap<TensorId, S> = BTreeMap::new();
    grads.insert(loss.id(), loss.ones_like()?);

    for node in nodes.into_iter().rev() {
        let Some(grad_out) = grads.get(&node.output_id).cloned() else {
            continue;
        };
        for (input, contribution) in node.input_ids.into_iter().zip((node.backward)(&grad_out)) {
            if check == NanCheck::Enforce {
                assert_finite(&contribution, input);
            }
            match grads.entry(input) {
                Entry::Occupied(mut slot) => {
                    let summed = slot.get().accumulate(&contribution)?;
                    if check == NanCheck::Enforce {
                        assert_finite(&summed, input);
                    }
                    slot.insert(summed);
                }
                Entry::Vacant(slot) => {
                    slot.insert(contribution);
                }
            }
        }
    }

    Ok(GradientMap { grads })
}

/// Panic naming the tensor whose gradient went non-finite.
///
/// A panic rather than an error because that is what all three backends did
/// before this module and moving code is not the place to change behaviour.
/// `GRD-005` owns replacing it: "structured backward and NaN failures; no
/// expected-failure panic paths".
fn assert_finite<S: TapeStorage>(grad: &S, id: TensorId) {
    assert!(
        !grad.has_non_finite(),
        "NaN or Infinity detected in gradient for TensorId {id:?}"
    );
}
