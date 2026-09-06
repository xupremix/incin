# Deep autograd: tapes, recipes, and custom training ops

[Autograd](./autograd.md) shows the user surface and [Custom and fused
operations](./custom_operations.md) shows the authoring contract. This
chapter is the idea underneath both: what a tape is, what a backward recipe
owes the walk, how a custom operation earns a place on the tape, and where
the current design deliberately stops. Concept-first by design; the runnable
proofs live in `crates/incin-core/tests/custom_training.rs` (a custom
operation training end to end from a user crate) and
`crates/incin-backends/examples/polar_cartesian.rs` (a two-input,
two-output operation with hand-derived gradients, run by CI).

## Three switches, not one

Gradient behavior is decided by three orthogonal mechanisms, and most
confusion comes from blaming the wrong one:

- The type parameter `G` (`Grad`, `NoGrad`, `Dyn`) decides whether a tensor
  *can* record. A `NoGrad` tensor never consults anything at runtime.
- The tape decides *how* a recorded operation runs backward. It is a list of
  nodes plus one reverse walk, shared by every backend.
- `GradMode` decides *right now*. It is a scoped, thread-local override that
  can only tighten recording, never loosen it. Evaluation loops, target
  tensors, and the backward walk itself run under `Disabled`.

A missing gradient is therefore always one of three distinct events: the
tensor could not record (`G`), nothing was recorded (mode), or the walk
never reached it (graph). The optimizer's refusal of a step that reaches no
parameter exists because the third case used to look exactly like success.

## Anatomy of a node

One recorded operation is a `TapeNode<S>` with three fields:

- `output_id`: the identity of the value produced.
- `input_ids`: the identities consumed, in recipe output order.
- `backward`: the recipe mapping one output gradient to one gradient per
  input, in that same order.

Identities are `TensorId`, a workspace-global monotonic counter rather than
a pointer address. Pointer identity is reused after a drop, which credits
one tensor's gradient to an unrelated later allocation in a way that is
unreproducible by construction; the counter cannot do that. It is global
across backends today in anticipation of backends one day sharing a tape,
which is precisely what they do not do yet (see below).

The recipe type is `BackwardFn<S> = Box<dyn Fn(&S) -> Result<Vec<S>>>`.
Three properties of that signature are load-bearing:

- **Order, not names.** The `Vec` lines up with `input_ids` positionally.
  A recipe returning fewer gradients than inputs, or more, fails the whole
  pass with an invariant violation rather than starving an input or dropping
  a surplus silently — the walk checks the counts before it zips. Getting
  the order itself right is still on the author, which is why every custom
  recipe is cross-checked against central finite differences, per element.
- **Fallible.** A recipe that cannot produce a gradient returns a structured
  error rather than panicking. The infallible alternative was tried: it gave
  every backend exactly one way to report failure, and over a hundred sites
  took it as `.expect()` on kernels that genuinely can fail.
- **Opaque.** The core never inspects the closure. Saved values are captured
  by move rather than listed in a field, which is what makes the node
  backend-neutral: the core never names a storage type it would have to
  hold. Refusing a push drops the closure on the spot, releasing the saved
  values with it.

## The walk

`backward(nodes, loss)` takes the nodes **by value**. That signature is the
whole of the `D-06` decision: a recipe may itself record (several CPU
backward kernels do), so a walk that still held the tape would either
re-enter it or panic on the second borrow. Draining first makes that
structural — there is no way to call the walk without having taken the nodes
out — and nodes recorded *during* the walk land on the fresh tape, belonging
to the next pass. A second `backward()` from the same loss therefore finds
an empty graph and fails with `BackwardError::GraphConsumed` instead of
returning a seed-only gradient: a seed-only `Ok` reads like a successful
gradient while training on nothing. There is no `retain_graph`, no
`create_graph`, and no second-order gradient; see [Autograd](./autograd.md).

The remaining steps are the rest of the contract, in order:

1. Seed `grads[loss]` with ones (or the explicit cotangent for
   `backward_with_seed`).
2. Walk in reverse insertion order. An output nothing reached is skipped
   rather than failed: an unreached branch is ordinary, not an error.
3. A tensor consumed by two later operations receives the **sum** of both
   contributions. Writing that as an insert instead of an accumulate dropped
   one of two gradients silently, which is why the walk matches on the map
   entry rather than guarding an assignment.
4. Under `NanPolicy::Reject`, every contribution and every accumulation is
   checked for non-finite values and reported with the tensor id and the
   site, rather than aborting the process.

What a backend supplies to all of this is exactly the `TapeStorage` trait
and nothing else: an identity, a ones-like seed, a fallible sum, and a
non-finiteness test. Seeding is a device allocation, summing is a kernel,
and reading a value back costs a WGPU readback and a CPU slice walk. The
walk itself does not differ between backends, which is why it is written
once in the core instead of three times.

## Who owns the tape

The nodes and the walk live in the core. The thread-local that *owns* a
tape still lives in each backend: one for CPU storage, one for CUDA, one
for WGPU, one for Metal. A custom operation on the CPU backend records onto
the CPU tape; a fully foreign backend runs its own thread-local over the
public core `Tape` type and walks it with the same core walk. Cross-backend
differentiable graphs — one `backward()` spanning two backends' tapes — do
not exist yet; neither do custom forward-mode, batching, or higher-order
rules. Those are tracked as future architecture, not as gaps in the path
below.

## Recording: the custom-operation contract

The ergonomic form is the `DifferentiableOp` trait: a forward kernel plus a
backward rule over one backend's storage, with a blanket `Execute`
implementation building the node, deriving its identities from the
storages, and recording through the backend's `RecordingBackend`
implementation. The author writes no `execute` body, names no ids, and
cannot forget to record. The raw entry points underneath —
`incin_backends::cpu::tape_record` (plus lazy `tape_record_with`) and the
WGPU, CUDA, and Metal twins — remain for multi-output operations, where one
node per output cannot be derived from a single return type.
The shape of the trait, abbreviated from
`crates/incin-core/tests/custom_training.rs`:

```rust,ignore
use incin_core::backend_authoring::DifferentiableOp;

impl DifferentiableOp<CpuBackendImpl<Cpu>> for Square {
    type Dtype = f32;          // one implementation trains one dtype
    type Saved = CpuStorage;   // what forward saves: the input itself

    fn forward(inputs: &[CpuStorage], _: &NoAttributes)
        -> Result<(CpuStorage, Self::Saved), BackendError>;
    fn backward(saved: &CpuStorage, grad_out: &CpuStorage)
        -> Result<Vec<CpuStorage>, incin_core::error::Error>;
}
```

Four rules keep recorded graphs sound, and each has a failure mode worth
naming:

1. **Save by move.** Clone what the recipe needs out of the forward inputs
   before returning. Borrowing them ties the recipe's lifetime to values
   the caller is free to drop.
2. **Match the input order.** The walk pairs recipe outputs with inputs
   positionally and refuses a count mismatch instead of zipping silently.
   A swapped pair still trains the wrong tensor with the right numbers —
   that half is what finite differences catch.
3. **Shape-match before returning.** Accumulation sums without broadcasting,
   so a recipe that returns a broadcast-shaped gradient where a smaller one
   belongs breaks the sum. Reduce in the recipe, and re-broadcast only where
   the target genuinely needs the width. Each backend keeps an `unbroadcast`
   for this and all four are `pub(crate)` on purpose: they are four
   implementations, not one contract, differing in how a reduced-all-the-way
   scalar seed is expanded back and in which reduce kernel they reach for.
   Exporting them as one API is how a recipe comes to depend on the CPU
   backend's edge cases and meets CUDA's. Write the reduction your kernel
   needs; it is a sum over the axes the forward broadcast.
4. **One node per output.** A two-output operation records two nodes sharing
   the inputs — that is how the polar example's `x` and `y` nodes work, and
   the walk's summation is what combines their contributions. A single node
   cannot name two outputs.

## Checking that the gradient is actually right

A backward recipe is the part of a custom operation that fails quietly. A
wrong forward kernel produces visibly wrong numbers; a wrong recipe produces
a model that trains slightly worse, which is not a signal anyone can act on.
Check it numerically before trusting it, with
`incin_core::exec::gradcheck`:

```rust,ignore
use incin_core::exec::{GradCheckOptions, gradcheck};

let report = gradcheck(
    |inputs| my_scalar_loss(&inputs[0]),
    &[input.clone()],
    GradCheckOptions::for_f32(),
)?;
assert!(report.passed(), "{report}");
```

It perturbs one element, re-runs the forward, and compares the slope against
the recipe's own gradient at that element, for every element of every input.
Sweeping all of them rather than spot-checking is the point: an accumulation
that overwrites instead of summing agrees everywhere except at the one input
two operations share.

Two things it does for you that are easy to get wrong alone. The step size in
`for_f32()` is the one that minimises total error at `f32` precision, near
`1e-2`; the `1e-4` that looks conservative is roughly a hundredth of that and
sits at its own noise floor, where a real defect and a rounding artifact are
indistinguishable. And the report names the input, the element, both values
and the relative error rather than returning one number, because a constant
factor across every element is a missing term in the recipe while a single
element is usually the perturbation stepping across a boundary the derivative
does not exist at. Print it; it says which.

`GradCheckStorage` is the whole of what the sweep asks of a backend: read one
element, perturb one element, run the backward pass. A foreign backend that
implements those three gets the sweep for free.

`incin_core::exec::check_gradients` is *not* this check, despite the name. It
installs `NanPolicy::Reject` for the enclosed code, so the walk stops at the
first non-finite contribution and names the tensor it appeared on. That finds
where a gradient exploded; it says nothing about whether a finite gradient is
the right number.

Then validate the rest like any other kernel: a `NoGrad` run asserting nothing
is recorded, and capability refusals for every dtype the kernel does not hold.
An operation that neither records nor composes from existing differentiable
tensor operations is forward-only and should say so.

## The framework checks that a training row records

A capability row claiming `training` is enforced in both directions now. The
dispatcher refuses a training query against a row that does not claim it, and
the conformance oracle runs every training tuple with recording enabled and
fails the row if the tape did not grow.

That second direction is the one that used to be missing, and it matters more
than it sounds. A kernel that runs correctly and records nothing does not
produce a wrong number, it produces a graph with a hole in it: `backward`
reaches everything below the hole and nothing above, the optimizer skips the
parameters it found no gradient for, and the run finishes having trained part
of the model. `sin`, `cos`, `log2`, `log10`, `to_dtype`, `frac`, `fmod` and
`remainder` were all in that state until the check went in.

Operations whose derivative is genuinely zero or undefined are declared by
hand, with a reason each, in the oracle's `carries_no_gradient`: `sign`,
`floor`, `ceil`, `round` and `trunc` are piecewise constant, and `one_hot`
reads indices. A new operation is a finding until somebody writes down which
group it belongs to, because there is no way to tell "has no derivative" from
"forgot to write one" by reading a kernel.

## From storage back to `Tensor`

A recorded output is backend storage, but models are written in typed
tensors. The bridge is `Tensor::try_from_storage`, which is public and
moves the storage rather than rebuilding it: the tape identity survives the
lift, so downstream typed operations record against the custom node's
output id and one `backward()` walks the mixed graph. It re-checks shape,
dtype, and device rather than trusting the caller, which is what makes the
bridge safe to use inside a `#[module]::forward` between ordinary typed
calls:

```rust,ignore
// Inside Module::forward: handles in, storage out, tensor back.
let handle = TensorHandle::from_storage::<CpuBackend, f32, Local>(&x.inner());
let out: CpuStorage = execute(&context, FusedAttrs { .. }, &[handle])?;
let y = Tensor::<Dyn, Cpu, f32, Grad>::try_from_storage(
    out, ShapeBuf::from_slice(&[4, 8]), (), Cpu::init(()), Grad::init(()),
)?;
y.relu() // typed ops continue; backward flows through the custom node
```

## Shared weights fall out of the model

Calling the same parameters K times — a looped block, a recurrent step, an
iterated refinement — records K nodes against the same storages, and the
walk sums K contributions into one gradient. No custom machinery is needed
for weight sharing; it is the accumulation rule applied K times. The cost
is K sets of live activations with no recompute valve yet (see below), so
loop depth is a memory decision, not a ceremony decision.

## What stays future

- **Second-order and retained graphs.** Drained tapes, no `retain_graph`,
  no `create_graph`. Gradient penalties and meta-learning wait on this.
- **Foreign-backend and cross-backend graphs.** The pieces are public (core
  `Tape`, the walk, `AutogradBackend`), but each backend still owns its
  thread-local, and no walk spans two.
- **Activation checkpointing.** No recompute primitive; a model whose
  forward activations do not fit has no compute-for-memory trade to reach
  for.
- **In-place mutation and aliasing.** Ownership, views, allocation
  identity, and autograd versioning need their own design first.
- **Custom forward-mode, batching, and higher-order rules.** Recipes are
  reverse-mode, first-order, one-shot.

Each of these is a design project with its own invariants, not a missing
method on the tape. What the tape does today — one reverse walk, summed
contributions, typed errors, no silent drops — is the foundation they will
all build on.
