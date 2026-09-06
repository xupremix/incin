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
  Returning gradients in the wrong order credits the wrong tensor, and
  nothing downstream can detect it — which is why every custom recipe is
  cross-checked against central finite differences, per element.
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
an empty graph and returns only the seed. There is no `retain_graph`, no
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

A custom operation trains when its `Execute` implementation records a node
for every output it produces. Each backend with a training tape exposes the
same entry points — `incin_backends::cpu::tape_record`,
`incin_backends::wgpu::tape_record`, `incin_backends::cuda::tape_record`,
and `incin_backends::metal::tape_record` — plus a lazy `tape_record_with`
variant that builds the entry only when the ambient mode records.
The shape of the call, abbreviated from
`crates/incin-core/tests/custom_training.rs`:

```rust,ignore
use incin_backends::cpu::tape_record;
use incin_core::backend_authoring::{TapeNode, TapeStorage};

// Inside Execute<YourOperation>::execute, after the forward kernel:
let node = TapeNode {
    output_id: out.id(),       // the fresh output storage's identity
    input_ids: vec![x.id()],   // every consumed input, in recipe order
    backward: Box::new(move |grad_out: &CpuStorage| {
        // One gradient per input, same order. Saved values (x_saved) are
        // owned by this closure, cloned out of the forward inputs above.
        Ok(vec![/* dL/dx for this grad_out */])
    }),
};
tape_record(node);
```

Four rules keep recorded graphs sound, and each has a failure mode worth
naming:

1. **Save by move.** Clone what the recipe needs out of the forward inputs
   before returning. Borrowing them ties the recipe's lifetime to values
   the caller is free to drop.
2. **Match the input order.** The walk zips recipe outputs with `input_ids`
   positionally. A swapped pair trains the wrong tensor with the right
   numbers.
3. **Shape-match before returning.** Accumulation sums without broadcasting,
   so a recipe that returns a broadcast-shaped gradient where a smaller one
   belongs breaks the sum. Reduce in the recipe (the CPU backend's
   `unbroadcast` is the in-tree pattern), and re-broadcast only where the
   target genuinely needs the width.
4. **One node per output.** A two-output operation records two nodes sharing
   the inputs — that is how the polar example's `x` and `y` nodes work, and
   the walk's summation is what combines their contributions. A single node
   cannot name two outputs.

Then validate like any other kernel: hand-derived gradients swept against
central finite differences per element, a `NoGrad` run asserting nothing is
recorded, and capability refusals for every dtype the kernel does not hold.
An operation that neither records nor composes from existing differentiable
tensor operations is forward-only and should say so.

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
