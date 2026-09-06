# Custom and fused operations

There are three different forms of fusion:

1. A new semantic operation, such as a real `BiasGelu` implementation, is an `Operation` with
   attributes and output inference, plus a backend `Execute<YourOperation>`.
2. A faster implementation of an existing operation stays behind that
   operation's existing `Execute` implementation. It does not add a second
   public operation hierarchy.
3. Combining several graph nodes belongs to compiler lowering and is outside
   this API task.

The executable authoring contract is exercised in
`crates/incin-core/tests/custom_operation.rs` and the downstream backend
fixture under `crates/incin/tests/consumer-fixtures`. Those fixtures cover
operation identity, serializable attributes, output inference, validation,
capability admission, and execution payloads.

The smallest useful implementation has four pieces:

1. Define an `Operation` type and its serializable attributes.
2. Add output inference and validation to the canonical operation catalog.
3. Implement `Execute<YourOperation>` for each backend that supports it.
4. Register capability admission and test the operation through the public
   consumer fixture.

The compact `CompanyIdentity` operation in
`crates/incin-core/tests/custom_operation.rs` is the reference implementation:
it defines serializable attributes, implements the canonical operation
contract, validates the input metadata, and executes through the backend
dispatch path. The downstream fixture invokes that operation through the
public authoring API, which is the important compatibility check for external
backend crates.

Here is the compact shape of the author-facing operation declaration. The
backend implementation uses the same `Execute<Identity>` request path as the
built-in catalog and returns its backend storage type.

```rust,ignore
use incin_core::backend_authoring::{
    Descriptor, DescriptorError, Execute, ExecutionRequest, LogicalTensorMeta, Operation,
    OperationKey,
};
use incin_core::err::BackendError;
use incin_core::exec::catalog::NoAttributes;
use std::borrow::Cow;

#[derive(Clone, Debug)]
struct Identity;

impl Operation for Identity {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("identity"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

// The backend supplies the storage type and executes the validated request.
impl Execute<Identity> for MyBackend {
    type Output = MyStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Identity, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _input = request.inputs.first().ok_or(BackendError::InvalidInput {
            operation: incin_core::shapes::error::OperationKind::Storage,
            reason: "identity needs one input",
        })?;
        // Decode the checked handle and launch the backend copy kernel here.
        todo!("backend-specific storage copy")
    }
}

// `infer_invocation` creates a validated `Descriptor<Identity>` before the
// caller invokes the same canonical dispatch path as built-in operations.
let invocation = Identity::infer_invocation(NoAttributes, logical_inputs)?;
let descriptor: &Descriptor<Identity> = invocation.descriptor();
let output = incin_core::exec::dispatch::execute_shaped::<
    Identity,
    MyBackend,
    incin_core::shapes::Dyn,
>(
    &context,
    NoAttributes,
    &[input_handle],
    &output_shape,
)?;
```

The real fixture fills in metadata validation, capability admission, and
backend execution. Keep the operation key stable once published, serialize
all attributes, and route execution through the validated descriptor request.

The runnable compact example is
`crates/incin-core/examples/custom_operation.rs`. It uses a metadata-only
backend so the operation contract can be exercised without pretending that a
backend kernel exists. A real backend replaces its proof-only executor with a
kernel implementation while keeping the same operation, descriptor, and
dispatch boundaries.

For a real fused operation such as `BiasGelu`, the same pattern applies. The
operation accepts activation and bias handles, validates their broadcast
relationship, infers the output metadata, and dispatches one backend kernel.
It should not introduce a parallel executor or construct a `TensorMeta` from
unchecked fields. If the operation is built from existing tensor methods
instead, document it as a composition rather than as a new fused catalog
entry.

## Gradients: the other half of a custom operation

Everything above stops at the forward pass. An operation that only ever runs
under inference can stop there too, and the seven-operand
`crates/incin-backends/examples/calibration_update.rs` does. An operation that
has to sit inside a training graph needs the other half: a rule that turns the
gradient of its output back into one gradient per input.

There is no registration hook for that. Nothing discovers an
`impl Backward for MyOperation`, and the catalog has no differentiability
column a custom row could set. What there is instead is the tape the built-in
operations already use, and every piece of it except one push site is public:
you write the backward recipe, assemble it into core `TapeNode`s, and hand
those to the same reverse walk the CPU backend calls.

### First, check whether you need to write one at all

If the operation is expressible as a composition of existing differentiable
tensor methods, write it that way and stop there. Each method in the
composition records its own tape entry as it runs, the `loss.backward()` of
[Autograd](./autograd.md) already walks through the whole chain, and there is
no recipe of yours to get wrong. Document the result as a composition rather
than as a new fused catalog entry, as above.

The rest of this section is for the case where a single fused kernel *is* the
point. That kernel runs as one opaque step, so nothing gets recorded unless
you record it.

### The recipe is a `TapeNode`

```rust,ignore
use incin_core::exec::TapeStorage as _; // brings `.id()` into scope
use incin_core::exec::tape::TapeNode;

// The kernel has already produced `out` from `input`. Record how to run it
// backwards. `saved` is moved into the closure: the recipe needs the forward
// input, and nothing else is going to keep it alive.
let saved = input.clone();
let node = TapeNode {
    output_id: out.id(),
    input_ids: vec![input.id()],
    backward: Box::new(move |grad_out: &MyStorage| {
        // One contribution per id in `input_ids`, in that order.
        Ok(vec![my_elementwise_mul(grad_out, &saved)?])
    }),
};
```

The type is three fields and no more: the id of the value the operation
produced, the ids of the values it consumed, and a
`Fn(&S) -> Result<Vec<S>>` that maps the output's gradient to one contribution
per input. The returned vector is positional -- `contributions[i]` belongs to
`input_ids[i]` -- and returning a shorter vector silently drops the gradients
of the inputs it does not cover, because the walk zips the two together.

Two properties of that shape are worth stating outright, because in both
cases the natural first guess is the other one.

**Saved values are captured, not declared.** There is no `saved_tensors`
field. Whatever the derivative needs -- the forward inputs, an argmax index
map, a normalization constant -- is moved into the closure, which is what
keeps `TapeNode` backend-neutral: the core never names a storage type it would
otherwise have to hold. It is also why a node that gets refused releases its
saved values immediately, since they live in the `Box` that is dropped.

**One node names one output.** `output_id` is a single id, so an operation
that returns two values records *two* nodes that share the same `input_ids`.
Each recipe receives only the gradient of its own output:

```rust,ignore
// x = r cos(theta) and y = r sin(theta): two outputs, two nodes, both naming
// the same two inputs.
let node_x = TapeNode {
    output_id: x.id(),
    input_ids: vec![radius.id(), theta.id()],
    // dx/dr = cos(theta), dx/dtheta = -r sin(theta)
    backward: Box::new(move |grad_x: &MyStorage| { /* ... */ }),
};
let node_y = TapeNode {
    output_id: y.id(),
    input_ids: vec![radius.id(), theta.id()],
    // dy/dr = sin(theta), dy/dtheta =  r cos(theta)
    backward: Box::new(move |grad_y: &MyStorage| { /* ... */ }),
};
```

Neither recipe adds the two contributions to `radius` together. That is the
walk's job.

### The walk accumulates; the recipe shapes

```rust,ignore
use incin_core::exec::tape;

let grads = tape::backward(vec![node_x, node_y, node_loss], &loss)?;
let d_radius = grads
    .get(radius.id())
    .expect("the backward pass reached the radius");
```

`tape::backward` seeds the loss with ones, walks the nodes in reverse
insertion order, and for an input that already holds a gradient **sums** the
new contribution into it rather than overwriting. That accumulation is the
reason both halves of the two-output operation above can name the same inputs
and still come out right. It is also the step that was silently wrong back
when each backend carried its own copy of the walk: one contribution was
inserted over the other rather than added to it. The walk is written once in
the core now, and its accumulation has no overwrite spelling.

The division of labour between the two is precise, and getting it backwards
produces gradients that are wrong only for some shapes:

- **The walk never broadcasts.** `TapeStorage::accumulate` assumes both
  operands already have the target's shape.
- **The recipe must therefore un-broadcast.** If the forward pass broadcast an
  operand -- a bias vector added across a batch, a scalar multiplied into a
  matrix -- the recipe has to sum the incoming gradient back down to that
  operand's shape before returning it. The CPU backend keeps an `unbroadcast`
  helper for this, but it is `pub(crate)`; an out-of-tree backend writes its
  own reduction.

Three more contract details the signature encodes:

- `backward` takes the nodes **by value**. A recipe may itself record -- every
  convolution backward on the CPU backend does -- so a walk still holding the
  tape would re-enter it. Drain first; there is no spelling that lets you
  avoid it.
- A second `backward` from the same loss returns only the seed. The nodes were
  consumed by the first one, and running a recipe twice would double every
  gradient it feeds.
- `tape::backward_with_seed(nodes, &loss, &seed)` takes an explicit output
  cotangent instead of ones, which is what you want for a
  vector-Jacobian product or a non-scalar output.

An operation whose recipes call tape-recording operations should run the walk
inside `GradMode::Disabled.scope(..)`, as the CPU backend's `backward` does,
so the backward pass does not record a graph of its own.

### The one seam: where the nodes get pushed

An in-tree backend owns a thread-local tape and pushes onto it from inside
`Execute`, so its custom nodes and its built-in ones land on one graph and a
single `loss.backward()` walks all of them:

```rust,ignore
// Inside `impl Execute<PolarToCartesian> for CpuBackendImpl`. `tape` here is
// the backend's own `crate::cpu::tape`, not `incin_core::exec::tape`; both
// `push_with` and `TapeEntry` are `pub(crate)` to that backend.
let (x, y) = /* the fused kernel */;
tape::push_with(|| TapeEntry { output_id: x.id(), /* ... */ });
tape::push_with(|| TapeEntry { output_id: y.id(), /* ... */ });
Ok((x, y))
```

`push_with` takes a closure rather than a node because it consults the ambient
`GradMode` first: under `NoGrad` the entry is never built, so a `NoGrad`
forward pass does not allocate a boxed closure and an id vector per operation
for a value nothing can read. On the way out, the backend drains only the
graph reachable from the loss, leaving any unrelated live graph on the same
thread for its own `backward`.

That push is the seam an external crate cannot reach: the thread-local is
`pub(crate)` by design. An out-of-tree backend therefore owns its node list
itself and calls `tape::backward` on it directly. Prefer a core
`incin_core::exec::Tape<S>` for that, rather than a bare `Vec<TapeNode<S>>`:
`Tape::push` is where the `GradMode` gate lives, so a plain vector records
nodes under `GradMode::Disabled` that should never have been recorded, and
you would have to check `GradMode::current().records()` at every call site
yourself. `Tape::drain_reachable(loss.id())` then hands `backward` exactly
the nodes it needs.

The recipes and the walk are identical either way -- only the ownership of the
node list differs -- so a recipe developed out-of-tree moves into an `Execute`
impl unchanged if the operation is later upstreamed.

### A new backend implements `TapeStorage`

The walk is generic over storage, and the trait is the complete list of what
it needs -- four methods, each one a place where backends genuinely differ:

```rust,ignore
use incin_core::exec::{TapeStorage, TensorId};

impl TapeStorage for MyStorage {
    /// This allocation's identity, from a monotonic counter, never a pointer
    /// address: reused addresses credit one tensor's gradient to another.
    fn id(&self) -> TensorId { self.id }

    /// The seed a backward pass starts from.
    fn ones_like(&self) -> Result<Self> { /* ... */ }

    /// Sum two contributions for the same tensor. Fallible because on some
    /// backends adding allocates.
    fn accumulate(&self, contribution: &Self) -> Result<Self> { /* ... */ }

    /// Only consulted under `NanPolicy::Reject`.
    fn has_non_finite(&self) -> Result<bool> { /* ... */ }
}
```

Reusing the built-in `CpuStorage` -- as a custom operation on the CPU backend
does -- means this is already implemented and there is nothing to write.

### Checking that the gradient is actually right

A backward recipe is the part of a custom operation that fails quietly. A
wrong forward kernel produces visibly wrong numbers; a wrong recipe produces a
model that trains slightly worse, which is not a signal anyone can act on.
Check it numerically before trusting it.

There is no public gradcheck helper -- the crate's own lives in
`crates/incin-backends/src/cpu/gradcheck.rs` and is `pub(crate)` -- so the
check is a short central-difference sweep you write yourself:

```rust,ignore
// Perturb one input element, re-run the forward, and compare the slope
// against the analytic gradient for that element. Sweep every element:
// a single spot check will not catch an accumulation that overwrites
// instead of summing.
let eps = 1e-5;
let numeric = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
let denominator = analytic.abs().max(numeric.abs()).max(1e-6);
assert!((analytic - numeric).abs() / denominator < 1e-4);
```

Two practical notes on the step size. In `f64`, `1e-5` is comfortable. In
`f32` it is not: the rounding term of a central difference scales as `1/eps`,
and the step that minimizes total error is around `(6 * f32::EPSILON).cbrt()`,
roughly `1e-2` -- two orders of magnitude larger than the `1e-4` that looks
conservative and in fact sits at the noise floor, where a real defect and a
rounding artifact are indistinguishable. Prefer `f64` for the check where the
operation admits it.

`incin_core::exec::check_gradients` is *not* this check, despite the name. It
installs `NanPolicy::Reject` for the enclosed code, which makes the walk stop
at the first non-finite contribution and name the tensor it appeared on
instead of letting a `NaN` propagate to the optimizer. That finds where a
gradient exploded; it says nothing about whether a finite gradient is the
right number.

### The worked example

`crates/incin-backends/examples/polar_cartesian.rs` is all of the above end to
end, and CI runs it rather than only building it. Polar-to-Cartesian takes two
inputs and returns two outputs -- the multi-output inference a single-output
catalog row cannot express, dispatched through `execute_shaped_n` with one
shape proof per output -- with a backward recipe per output assembled into
core `TapeNode`s and walked by `incin_core::exec::tape::backward`. It checks
the forward values and the hand-derived gradients against textbook answers,
sweeps every input element against central finite differences, asserts the
per-operand contract refusals, and then fits `(r, theta)` to a target point by
gradient descent through nothing but its own backward:

```text
cargo run -p incin-backends --features cpu --example polar_cartesian
```
