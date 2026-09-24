# Custom and fused operations

There are three different forms of fusion:

1. A new semantic operation, such as a real `BiasGelu` implementation, is an `Operation` with
   attributes and output inference, plus a backend `Execute<YourOperation>`.
2. A faster implementation of an existing operation stays behind that
   operation's existing `Execute` implementation. It does not add a second
   public operation hierarchy.
3. Combining several graph nodes belongs to compiler lowering and is outside
   this API task. The preview `compiled` pipeline does exactly this via
   `FusionPass` and the pointwise fuser — see
   [Lowering: from descriptor to kernel](./deep_lowering.md#fusion-legality-checked-pointwise-groups-cmp-005).

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

A custom operation that only supports certain dtypes enforces that at two
layers, and both must agree. `supports_custom` answers the capability query
carrying the invocation's real dtype descriptor: return `Native` for the
dtypes the kernel holds and `Unsupported` with a `CustomOperation` reason for
everything else, so planners and dispatch refuse before launch. The kernel
then re-checks the storage descriptor it actually received instead of
trusting the advertisement. Compile-time bounds on the typed frontend are the
third layer where the operation is reached through a generic tensor method.
There is no silent narrowing anywhere in that chain: an unsupported dtype is
a typed refusal naming the operation, never a quiet cast or a fallback
backend.

Custom autodiff works through public tape-record entry points, one per
backend with a training tape: `incin_backends::cpu::tape_record` (and the
lazy `tape_record_with`), `incin_backends::wgpu::tape_record`,
`incin_backends::cuda::tape_record`, and `incin_backends::metal::tape_record`.
Inside `Execute<YourOperation>`, run the forward kernel, then record a core
`TapeNode`: the output id, the input ids in recipe order, and a `backward`
closure mapping one output gradient to one gradient per input, capturing its
saved values by move. The node joins the same thread-local tape the
built-in kernels record on, under the same `GradMode` gate, so mixed graphs
walk as one graph and `AutogradBackend::backward` returns gradients for
custom inputs alongside built-in ones. The recipe must be validated like any
other: hand-derived gradients cross-checked against central finite
differences, `NoGrad` asserting nothing is recorded, and capability refusals
for dtypes the kernel does not hold. Unless a custom operation records this
way or is composed from existing differentiable tensor operations, document
it as forward-only.

## Experimental: expression-DSL pointwise ops

Separate from the three forms above — and **experimental**, not part of the
stable authoring contract — `incin_backends::codegen::dsl` exposes an
expression DSL for single-kernel pointwise custom ops. `define_unary_custom_op`,
`define_unary_custom_op` and `define_binary_custom_op` take a closure over
the codegen IR (`IrExpr`) and build a `KernelDefinition`: the forward
expression plus one symbolically derived backward derivative per input,
computed by the same `IrExpr::diff` that the shipped fused-unary-backward
path runs.

```rust,ignore
use incin_backends::codegen::{define_unary_custom_op, sigmoid};
use incin_core::tensor::dtype::DTypeId;

let swish = define_unary_custom_op("swish", DTypeId::F32, |x| {
    let s = sigmoid(x.clone());
    x * s
});
```

Two executors take that definition:

- `codegen::CpuJitKernel` evaluates forward and backward on the host as a
  per-element tree-walking interpreter over `f64`. It is a reference — the CPU
  twin a hardware result is checked against — not a zero-overhead runtime and
  not the path a production CPU operation takes.
- `codegen::CudaJitKernel` renders the definition to CUDA C and compiles it
  through the production NVRTC dispatcher, caching the module per device.

What it deliberately does not do: it defines no `Operation`, adds no catalog
row, and participates in no capability admission — a DSL op is not reachable
through tensor methods or `dispatch::execute`. Wiring one into the library is
form 1 above; the DSL is for kernels launched directly while the surface
moves, and it may change without a major bump. The whole path — symbolic
derivatives, CPU reference numbers, NVRTC compilation — is exercised by
`crates/incin-backends/tests/codegen_ir_pipeline.rs` and the ignored
hardware suite in `crates/incin-backends/tests/codegen_nvrtc_smoke.rs`.

The same IR ships inside ordinary kernels: `codegen::fragment::lower_scalar`
renders an `IrExpr` into the body slot the scalar kernel templates fill — see
[Lowering: from descriptor to kernel](./deep_lowering.md#fusion-legality-checked-pointwise-groups-cmp-005).

## Training a custom operation, end to end

What follows is the whole pattern, condensed from the executed fixture in
`crates/incin-core/tests/custom_training.rs` (CPU; WGPU and CUDA twins live
beside it in `crates/incin-backends/tests/`). The `DifferentiableOp` trait
carries the pattern: implement a forward kernel and a backward rule as pure
functions over one backend's storage, and a blanket `Execute` implementation
builds the node, derives its identities, checks admission, and records. A
custom `square` operation, `y = x^2`, training through the standard backward
pass:

```rust,ignore
use incin_core::backend_authoring::DifferentiableOp;

impl DifferentiableOp<CpuBackendImpl<Cpu>> for Square {
    type Dtype = f32;
    type Saved = CpuStorage; // what forward saves: the input itself

    fn supports(query: &CapabilityQuery) -> SupportLevel {
        // f32 only: anything else is refused before launch, never executed
        // against a dtype the kernel was not written for.
        ...
    }

    fn forward(inputs: &[CpuStorage], _attributes: &NoAttributes)
        -> Result<(CpuStorage, Self::Saved), BackendError>
    {
        let x = /* the single input */;
        let out = /* elementwise x^2 into fresh storage (fresh id) */;
        Ok((out, x))
    }

    // Both halves receive the invocation's attributes. `Square` has none, but
    // an operation whose derivative depends on its configuration (a leaky
    // ReLU's slope) reads it here rather than copying it into `Saved`.
    fn backward(saved: &CpuStorage, _attributes: &NoAttributes, grad_out: &CpuStorage)
        -> Result<Vec<CpuStorage>, incin_core::error::Error>
    {
        Ok(vec![/* 2 * saved * grad_out, same shape */])
    }
}

// Call it, and train through the standard pass. No handle and no execution
// context, and the shape, dtype, device and gradient marker are not restated,
// because `x` already carries all four:
let y = x.apply_op::<Square>(NoAttributes)?;
let grads = y.sum_all()?.backward()?; // walks built-in and custom nodes as one

// Several inputs, or an output shape that differs from the input's:
// `apply_op_n` adds the one thing that cannot be inherited from `x`. The extra
// operands are borrowed storage rather than borrowed tensors, so each is free
// to have a shape of its own.
let joined = x.apply_op_n::<Concat2<f32>, s![6]>(&[w.inner()], NoAttributes, expected)?;
```

How many implementations an operation needs is a question about how its
kernel is written, not a property of the trait. A recipe written as a hand
loop over one buffer variant covers that dtype on that backend, so covering a
second means a second implementation, conventionally via a generic wrapper. A
recipe written as dispatched built-in operations covers every backend that
implements them and every dtype they accept, from one implementation; the
dtype has to appear in the self type rather than only in the associated type,
because an impl type parameter that appears nowhere in the self type is
rejected (E0207). `crates/incin-core/tests/custom_op_composition.rs` is the
worked example. Which to write is a performance question: a fused kernel is
still one implementation per backend, and a composed one is not. The same file
holds the two-input, shape-changing case, which is `apply_op_n`'s reason to
exist: a concatenation whose operands are `s![4]` and `s![2]` and whose result
is `s![6]`, three shapes that are three distinct Rust types, none of them
derivable from the receiver. Multi-output
operations keep the explicit `tape_record` path — one node per output
cannot be derived from a single return type, and that shape is rare enough
to deserve spelling out (the polar example is the reference).

Three details carry the soundness, and all three are pinned by the fixture
rather than left as advice. First, the recipe returns one gradient per
input in input order, and the walk refuses a count mismatch instead of
zipping silently — while a swapped pair still trains the wrong tensor with
the right numbers, which is what the finite-difference sweep is for.
Second, `try_from_storage` moves the storage rather than rebuilding it, so
the recorded output id still matches and the custom node stays reachable.
Third, the fixture sweeps the hand-derived gradient against central finite
differences, asserts a `NoGrad` forward records nothing, and drives an
`f16` input at the `f32`-only kernel to prove the refusal happens before
any kernel runs. That sweep is `incin_core::exec::gradcheck`, which is public
and backend generic: hand it a scalar-output closure and its inputs and it
walks every element, with a step size already chosen for the dtype and a
report that names which element disagreed and by how much. The deep autograd
chapter covers what its output means. Run it before trusting a recipe: a
wrong forward kernel produces visibly wrong numbers, and a wrong recipe
produces a model that trains slightly worse.

A recipe that records nothing at all is caught earlier, and not by you. The
conformance oracle runs every tuple whose capability row claims `training`
with recording enabled and fails the row if no node reached the tape, so an
operation that advertises training and forgets its backward is a test
failure rather than a hole in somebody's graph.

The remaining seam for a multi-output operation is the explicit per-backend
`tape_record` path, which is public on all four training backends: one node
per output cannot be derived from a single return type, so there is no
`DifferentiableOp` blanket impl for this shape. An in-tree backend moves its
backward recipes into its `Execute` impl and records them there via
`tape_record`, so custom and built-in nodes share one graph. A foreign backend
does the same with its own thread-local over the public core `Tape` type,
walked by the same `incin_core::exec::tape::backward` the CPU backend calls.
A single-output in-tree operation skips the hand-built nodes entirely and
implements `DifferentiableOp` instead. Everything else a differentiable custom
operation needs is public, and `crates/incin-backends/examples/polar_cartesian.rs`
shows it end to end, run by CI rather than only built. Polar-to-Cartesian takes
two inputs and returns two outputs -- the multi-output inference a
single-output catalog row cannot express, run through the typed dispatch path
via `execute_shaped_n` with one `ShapeValue` per output -- with a backward
recipe per output assembled into core `TapeNode`s and walked by the same
`incin_core::exec::tape::backward` the CPU backend calls. The example checks
the forward values and the hand-derived gradients against textbook answers,
sweeps every input element against central finite differences, asserts the
contract refusals, and then fits `(r, theta)` to a target point by gradient
descent through nothing but its own backward.

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
new contribution into it rather than overwriting. That accumulation is why
both halves of a two-output operation can name the same inputs and still come
out right. It is also the step that was silently wrong when each backend
carried its own copy of the walk: one contribution was inserted over the
other rather than added to it. The walk is written once in the core now, and
its accumulation has no overwrite spelling.

The division of labour between the walk and the recipe is precise, and
getting it backwards produces gradients that are wrong only for some shapes:

- **The walk never broadcasts.** `TapeStorage::accumulate` assumes both
  operands already have the target's shape.
- **The recipe must therefore un-broadcast.** If the forward pass broadcast an
  operand -- a bias vector added across a batch, a scalar multiplied into a
  matrix -- the recipe has to sum the incoming gradient back down to that
  operand's shape before returning it. The CPU backend keeps an `unbroadcast`
  helper for exactly this, but it is `pub(crate)`; an out-of-tree backend
  writes its own reduction.

Three more contract details the signature encodes:

- `backward` takes the nodes **by value**. A recipe may itself record -- every
  convolution backward on the CPU backend does -- so a walk still holding the
  tape would re-enter it. Drain first; there is no spelling that avoids it.
- A second `backward` from the same loss **fails** with
  `BackwardError::GraphConsumed`. The nodes were consumed by the first walk,
  and running a recipe twice would double every gradient it feeds; a
  seed-only `Ok` would read like a successful gradient while training on
  nothing.
- `tape::backward_with_seed(nodes, &loss, &seed)` takes an explicit output
  cotangent instead of ones, which is what a vector-Jacobian product or a
  non-scalar output needs.

An operation whose recipes call tape-recording operations should run the walk
inside `GradMode::Disabled.scope(..)`, as the CPU backend's `backward` does,
so the backward pass does not record a graph of its own.

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

A custom operation on an in-tree backend reuses that backend's storage, so
this is already implemented and there is nothing to write.

### One note on the step size

`gradcheck` picks its own step from the dtype, and the number is worth knowing
before you override it or write a sweep by hand. In `f64`, `1e-5` is
comfortable. In `f32` it is not: the rounding term of a central difference
grows as `1 / step` while truncation grows as `step^2`, and the two meet near
`(6 * f32::EPSILON).cbrt()`, about `9e-3`. The `1e-4` that looks conservative
is roughly a hundredth of that and sits at its own noise floor, where a real
defect and a rounding artifact are indistinguishable. `GradCheckOptions::for_f32`
therefore steps by `1e-2`, and `for_f64` by `1e-5`.

`incin_core::exec::check_gradients` is *not* this check, despite the name. It
installs `NanPolicy::Reject` for the enclosed code, so the walk stops at the
first non-finite contribution and names the tensor it appeared on instead of
letting a `NaN` reach the optimizer. That finds where a gradient exploded; it
says nothing about whether a finite gradient is the right number.
