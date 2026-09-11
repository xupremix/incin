# Architecture review: custom operations, autodiff, and the construction surfaces

The review found one silently wrong number, and fixing it dissolves the design
weakness the record itself calls "the one axis where incin is the outlier".
That is the finding. Everything else below is sequencing and surface work.

A review of the custom-operation contract and how it reaches autodiff, plus
the tensor construction, target, dtype and layout surfaces around it.

Everything in section 1 was measured on this checkout by writing a test,
running it, and reading the number. Everything in section 3 is a count or a
source quotation. Where a finding is already in
`custom-op-autograd-decisions.md`, `typed-layout-decisions.md`,
`what-to-take-from-sota.md`, `layout-at-construction.md`,
`grd-006-graph-owned-tapes.md` or `96-custom-dtypes-devices.md`, it is filed
under "corrections to the record" and says which decision it revisits and what
new evidence moves it.

**What is applied.** 1.1, 1.2, 1.4 and the documentation corrections they
imply are in the tree, each with a test that fails without them. Everything
else is a sketch: a block marked "proposed" has been compiled where it says so
and is not a diff against something that landed. Section 4 says which is
which.

## 1. Not in the record

### 1.1 A composed forward kernel gets its gradient counted twice

`DifferentiableOp::forward` runs under whatever `GradMode` is ambient. The
blanket `Execute` implementation in
`crates/incin-core/src/tensor/backend/differentiable.rs` calls it directly:

```rust
let attributes = request.operation.descriptor().attributes();
let (out, saved) = O::forward(&owned, attributes)?;
...
B::record_custom(node);
```

Nothing in the trait says a forward kernel may not be built out of built-in
operations, and building one that way is the obvious thing to reach for: it is
how an author avoids writing the same loop once per backend. But a built-in
operation records a tape node of its own, against the very storage this
implementation is about to record a custom node for. Both nodes then carry the
same `output_id`. The reverse walk finds both, hands each the same output
gradient, and sums two contributions into every input.

The symptom is not a crash, a missing gradient, or an arity error. It is a
plausible number that is exactly twice the right one.

The operation that exposes it is unremarkable. This is the whole kernel:

```rust
fn forward(
    inputs: &[CpuStorage],
    _attributes: &NoAttributes,
) -> Result<(CpuStorage, Self::Saved), BackendError> {
    let x = inputs.first().ok_or(/* ... */)?;
    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(x);
    let out = incin_core::backend_authoring::execute::<op::Neg, _>(
        &ctx, NoAttributes, &[handle],
    )?;
    Ok((out, ()))
}

fn backward(_saved: &(), grad_out: &CpuStorage) -> Result<Vec<CpuStorage>> {
    // The correct rule for negation.
    Ok(vec![negate(grad_out)?])
}
```

Measured:

```
d(-x)/dx observed = [-2.0, -2.0, -2.0, -2.0]   (correct is [-1, -1, -1, -1])
```

And with `y = 3x^2` composed from `op::Mul` and `op::MulScalar`:

```
d(3x^2)/dx observed = [12.0, 24.0, 36.0, 48.0]   (correct is [6, 12, 18, 24])
```

PyTorch runs `Function.forward` under `no_grad` for exactly this reason.

**Proposed.** One wrapper, and `restrict` rather than `scope` for the reason
recorded in D12, since this module is not gated on `std`:

```rust
// Before
let (out, saved) = O::forward(&owned, attributes)?;

// After
let (out, saved) = GradMode::Disabled.restrict(|| O::forward(&owned, attributes))?;
```

`B::record_custom` stays outside the scope, so the custom node itself still
records under the ambient mode and a `NoGrad` chain still records nothing.

**Landed.** Both probes return the correct gradients, the existing custom
operation suites (`custom_training`, `custom_operation`,
`nograd_records_nothing`) stay green, and `cargo test -p incin-core -p
incin-backends --lib --tests` reports no failures.
`crates/incin-core/tests/custom_op_composition.rs` is the regression test: all
three of its cases fail without the wrapper and pass with it.

**Why nobody looked.** The module documentation waves the case off in one
sentence: "Composing existing differentiable tensor operations needs no trait
at all: the graph is inherited." That is true of composing at the *tensor*
level, where there is no custom node and the built-ins' own nodes are the
graph. It is not true of composing *inside* a `DifferentiableOp`, where there
is a custom node as well. The sentence is correct about the case it names and
is the reason the adjacent case was never tested.

### 1.2 The conformance oracle is a one-sided test

F6 was the review's most valuable finding: a capability row could advertise
training while its kernel recorded nothing, which leaves a hole in the graph
that no forward number reveals. The oracle now poses 2286 training tuples and
fails a row whose kernel records nothing.

It measures recording like this, in `conformance/mod.rs:311-317`:

```rust
let before = tape::depth();
let outcome = incin_core::exec::GradMode::Enabled.scope(|| { /* run the tuple */ });
let recorded = tape::depth() > before;
```

That detects a missing node. It cannot detect a surplus one. Finding 1.1 grows
the tape, by two nodes instead of one, so it passes the oracle cleanly. The
class of defect the oracle was built to catch has a mirror image with the same
symptom, the same invisibility, and no coverage at all.

**What does not work, measured rather than assumed.** The obvious tightening
is a node count, and it is wrong. Instrumenting the oracle over its whole
training set gives this distribution:

```
 361 tuples recorded 0 nodes
1178 tuples recorded 1 node
 203 tuples recorded 2 to 13 nodes
```

A composed implementation records one node per intermediate, so several nodes
is ordinary. The natural next move is to separate the two with the
`ImplementationKind` that `CapabilityRule` carries and `AdvertisedTuple`
discards. That does not work either: of the tuples recording more than one
node, **90 are declared `Native`**, up to thirteen nodes each. `Native` answers
"does the backend have a dedicated kernel", which is a different question from
"how many tape nodes does it push". A count-based gate keyed on it would have
produced 90 false findings.

**What does work.** The defect in 1.1 is not a surplus of nodes. It is two
nodes claiming the *same output identity*. Those are distinguishable, and the
distinction is exact rather than heuristic: every intermediate allocation mints
a fresh `TensorId`, so a repeated output identity is never a composition and
always two recipes claiming one value.

Measured across all 1381 recording tuples on CPU: zero duplicates. So the check
has no false positives to suppress, needs no new field on `AdvertisedTuple`,
and costs one sort of the identities the tape already holds.

```rust
// Drained rather than discarded, so the identities can be inspected.
let mut output_ids = tape::drain_output_ids();
let node_count = output_ids.len();
output_ids.sort_unstable();
output_ids.dedup();
let duplicate_output = output_ids.len() != node_count;
```

**Landed**, as `Verdict::RecordedOneOutputTwice`, alongside a
`cpu::tape::drain_output_ids` that gives the harness the identities rather than
only the count. Verified in both directions: silent on the tree as it stands,
and fired on every affected tuple when a CPU kernel was temporarily made to
record its output twice.

### 1.3 A backward recipe never sees its operation's attributes

`forward` receives `&Self::Attributes`. `backward` receives only
`&Self::Saved` and the output gradient. Any operation whose derivative depends
on its own configuration has nowhere to read it from.

A leaky ReLU is the smallest case that shows it. Today the slope has to be
copied into `Saved` and cloned into every recorded node:

```rust
// Today. `Saved` carries a copy of the configuration because `backward`
// cannot see the attributes, so every node on the tape holds its own f64.
type Saved = (B::Storage<f32>, f64);

fn forward(
    inputs: &[B::Storage<f32>],
    attributes: &LeakySlope,
) -> Result<(B::Storage<f32>, Self::Saved), BackendError> {
    let y = leaky(inputs[0], attributes.slope)?;
    Ok((y, (inputs[0].clone(), attributes.slope)))
}

fn backward(saved: &Self::Saved, g: &B::Storage<f32>) -> Result<Vec<B::Storage<f32>>> {
    let (x, slope) = saved;
    Ok(vec![leaky_grad(x, g, *slope)?])
}
```

```rust
// Proposed. `Saved` carries what forward computed; the descriptor carries
// what the caller configured, and each is read from where it lives.
type Saved = B::Storage<f32>;

fn forward(
    inputs: &[B::Storage<f32>],
    attributes: &LeakySlope,
) -> Result<(B::Storage<f32>, Self::Saved), BackendError> {
    let y = leaky(inputs[0], attributes.slope)?;
    Ok((y, inputs[0].clone()))
}

fn backward(
    saved: &Self::Saved,
    attributes: &Self::Attributes,   // the added parameter
    g: &B::Storage<f32>,
) -> Result<Vec<B::Storage<f32>>> {
    Ok(vec![leaky_grad(saved, g, attributes.slope)?])
}
```

PyTorch's `ctx` carries both. JAX's residuals are returned from the forward
rule that already had the parameters in scope. The window is open now: the
tree holds three `DifferentiableOp` implementations, one per backend fixture,
and no downstream crate has written one. Once one has, the same change is a
break for everybody who did.

Mechanically the attributes are already in hand at the record site, so the
blanket `Execute` only has to clone them into the closure it is already
building:

```rust
let attributes = request.operation.descriptor().attributes().clone();
let node = TapeNode {
    output_id: out.id(),
    input_ids,
    backward: Box::new(move |g| O::backward(&saved, &attributes, g)),
};
```

### 1.4 A gate rewrote the tree and reported success

Found by running the gates rather than by reading them.
`tools/build-api-examples.py --check` exits zero while modifying
`crates/incin/tests/api_examples.rs`, a tracked file. Two separate faults sat
on top of each other.

The generator emitted one more blank line than the committed harness carried,
because `PRELUDE` ended with a blank line and every test part also opens with
one. So regenerating always produced a one-line diff that no gate explained,
which is the drift D14 recorded and left alone.

Underneath it, `--check` never compared anything. It called the same `emit`
the write path calls, so it overwrote the harness with the generator's version
and then checked *that* for compile errors. A committed harness that had
drifted arbitrarily far from the generator would pass, because the thing being
checked was silently replaced first.

**Landed.** `render` is split from `emit`; check mode compares the committed
file against `render` and fails with the regeneration command; the writes of
both the harness and `docs/api-examples.json` are guarded on `not check`; and
the prelude's trailing blank line is gone so the two agree. Verified in both
directions: passing and tree-clean on the committed harness, exit 1 on a
harness given one extra line.

The general point is worth more than the fix. **A check that regenerates
before comparing is not a check**, and this one had been reporting success for
as long as it had existed.

## 2. Corrections to the record

### 2.1 The per-backend and per-dtype multiplication is not a property of the design

`custom-op-autograd.md` names this "the one axis where incin is the outlier",
and it is the most self-critical paragraph in the record:

> `DifferentiableOp` has an associated `Dtype`, so a recipe is written for one
> element type: an author who wants `Square` in `f32` and `f64` writes
> `Square<f32>` and `Square<f64>` and two impls. None of Candle, PyTorch or
> Burn asks that.

The module documentation states the same conclusion, that an operation
training in two dtypes is two implementations "each with its own recipe". The
tree bears it out: `y = x^2` appears three times, in `custom_training.rs`,
`cuda_custom_training.rs` and `wgpu_custom_training.rs`, with identical
`Operation` implementations, identical dtype guards and identical arithmetic,
differing only in how bytes are read out of one storage type.

That is true of a recipe written as a hand loop over one buffer variant. It is
not true of a recipe written as dispatched built-in operations. Measured: one
implementation covers every backend that can multiply and every float dtype it
supports. This is the complete operation, and it replaces all three fixtures:

```rust
/// `y = 3 * x^2`. The dtype rides in a `PhantomData` parameter rather than
/// only in the associated type: an impl type parameter that appears nowhere
/// in the self type is rejected (E0207), and `type Dtype = K` does not
/// constrain `K`.
#[derive(Debug, Clone)]
struct ScaledSquare<K>(PhantomData<K>);

impl<K: DType> Operation for ScaledSquare<K> {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("company.example"),
        name: Cow::Borrowed("scaled_square"),
        version: 1,
    };
    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl<B, K> DifferentiableOp<B> for ScaledSquare<K>
where
    B: Backend + Capabilities + SupportsDType<K>
        + Execute<op::Mul> + Execute<op::MulScalar> + RecordingBackend<K>,
    K: FloatDType,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    B::Storage<K>: Any + StorageOutput + TapeStorage + Send + Sync,
{
    type Dtype = K;
    type Saved = B::Storage<K>;

    fn forward(
        inputs: &[B::Storage<K>],
        _attributes: &NoAttributes,
    ) -> Result<(B::Storage<K>, Self::Saved), BackendError> {
        let x = inputs.first().ok_or(/* ... */)?;
        let squared = run::<op::Mul, B, K>(NoAttributes, &[x, x])?;
        let scaled = run::<op::MulScalar, B, K>(ScalarAttributes { value: 3.0 }, &[&squared])?;
        Ok((scaled, x.clone()))
    }

    fn backward(saved: &Self::Saved, g: &B::Storage<K>) -> Result<Vec<B::Storage<K>>> {
        let slope = run::<op::MulScalar, B, K>(ScalarAttributes { value: 6.0 }, &[saved])?;
        Ok(vec![run::<op::Mul, B, K>(NoAttributes, &[&slope, g])?])
    }
}

/// The whole of the plumbing, written once.
fn run<O, B, K>(
    attributes: O::Attributes,
    inputs: &[&B::Storage<K>],
) -> Result<B::Storage<K>, BackendError>
where
    O: Operation,
    B: Backend + Execute<O> + Capabilities + SupportsDType<K>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let ctx = ExecutionContext::from_scope(B::default());
    let handles: Vec<_> = inputs
        .iter()
        .map(|s| TensorHandle::from_storage::<B, K, Local>(s))
        .collect();
    incin_core::backend_authoring::execute::<O, B>(&ctx, attributes, &handles)
        .map(Into::into)
        .map_err(/* ... */)
}
```

The call site is one line, and the same line for every dtype:

```rust
let y = x.apply_op::<ScaledSquare<f32>>(NoAttributes)?;
let loss = y.sum_all()?;
let grads = loss.backward()?;
assert_eq!(grads.require(&x)?.to_vec1::<f32>()?, vec![6.0, 12.0, 18.0, 24.0]);
```

`f32` runs end to end through `Tensor::apply_op`, `sum_all` and `backward`.
`f64` runs through the same implementation at the storage level, returning the
same numbers, and only stops short of `sum_all` because the CPU `reduction`
capability group is declared f32-only, which is a built-in coverage boundary
rather than anything about the custom operation.

The two findings are the same finding from opposite directions, and that is
why neither was noticed. The composition that collapses the multiplication
silently returned twice the correct gradient, so an author who tried it and
gradchecked would have concluded that composing built-ins does not work, and
an author who did not gradcheck would have shipped a model that trains on
doubled gradients. One `GradMode::Disabled.restrict` turns a recorded design
weakness into a non-issue.

What this does not change: a recipe that genuinely needs a fused kernel still
writes one per backend, and should. The correction is that composition is a
real option rather than a trap, so the multiplication becomes a choice about
performance rather than a tax the design imposes.

The record should also drop the claim that dtype generality is a design
question for GRD-006. It is answerable today, and the answer is that the
associated type is fine as long as the struct carries the parameter. What is
missing is not machinery but a documented example: the module doc currently
teaches the two-impl pattern, and should teach this one beside it, with the
one-line note about why `PhantomData<K>` is there.

### 2.2 D7, D9 and finding 1.3 want one edit, and should be bought together

D7 rejects a per-input `Option` in backward returns on edit count: `Vec<S>` is
returned from 127 sites across four backends, and changing only
`DifferentiableOp::backward` does not work because `input_ids` is fixed at
record time. D9 rejects higher-order gradients because `BackwardFn` is over
storage rather than tensors and lifting it means changing that type and every
push site. Finding 1.3 wants a parameter on `backward`. All three are edits to
`BackwardFn` and `TapeNode`.

`grd-006-graph-owned-tapes.md` already commits to reshaping exactly those:
stage 2 is "move saved-storage ownership into the graph ... and reshape
`TapeNode` to a checked gradient structure, consuming the parked arity item."
D7 itself says "Revisit: with GRD-006, when `BackwardFn` is being changed
anyway."

The argument the record does not make is the sequencing one. Each was rejected
separately against the cost of touching every push site, and each rejection is
correct in isolation. Taken together they are not three costs but one, and
paying it three times is what happens if they land separately.

The node that absorbs all three at once, sketched so the shape is arguable
rather than abstract:

```rust
// Today.
pub type BackwardFn<S> = Box<dyn Fn(&S) -> Result<Vec<S>> + Send + Sync>;

pub struct TapeNode<S> {
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub backward: BackwardFn<S>,
}
```

```rust
// Proposed, at the one point TapeNode is already being reshaped.
//
// `Vec<Option<S>>` is D7: a rule says "this input takes no gradient" rather
// than materializing a zero tensor for a frozen embedding table and throwing
// it away. The walk skips a `None` instead of accumulating it, which is why
// this cannot be done on `DifferentiableOp::backward` alone.
pub type BackwardFn<S> = Box<dyn Fn(&S) -> Result<Vec<Option<S>>> + Send + Sync>;

pub struct TapeNode<S> {
    pub output_id: TensorId,
    // Paired rather than parallel: the arity check the walk performs today
    // becomes unrepresentable-if-wrong instead of checked-at-run-time. This
    // is the "checked gradient structure" GRD-006 already names.
    pub inputs: Vec<TensorId>,
    pub backward: BackwardFn<S>,
    // Co-owned by the graph rather than captured in the closure, which is
    // what gives early drop, offload and recompute somewhere to attach.
    pub saved: SavedValues<S>,
}
```

Deciding not to do higher-order gradients is fine. Deciding it *after* the
node has been reshaped once for other reasons is paying twice.

### 2.3 Layout: the sequencing is what is missing, not the design

`typed-layout-decisions.md` and `layout-at-construction.md` between them
already contain the analysis, including the CuTe layout algebra and
hierarchical shapes as a 0.3.0-scale question gated on nesting, and including
the measurement that justifies the parameter: at a million elements a
transposed view beats materializing by roughly 45% for a single consumer and
loses by roughly 23% by eight, so the two differ in opposite directions
depending on something the producing operation cannot know. That is the
strongest argument in the record for anything, and it holds.

The observation to add is only about what the parameter is currently earning.
`Contiguous` has one consumer, `reshape_view`. `ChannelsLast` is nameable but
not constructible. `AlignedTo<N>` was dropped for being surface without a
consumer. The specced work that converts the parameter from a down-payment
into a return is a trio, all three specced in the record and none built:

```rust
// 1. transpose_view as a catalog operation on every backend, so the caller
//    picks and the type records which they got.
let materialized: Dense<s![4, 3], B> = x.transpose::<U0, U1>()?;  // copies, RowMajor
let view = x.transpose_view::<U0, U1>()?;                         // no copy, Dyn
let flat = view.reshape_view::<s![12]>()?;                        // correctly refused

// 2. to_layout as the copying counterpart to into_row_major, which checks.
let dense: Dense<s![1, 3, 4, 4], B> = nhwc.to_layout()?;   // copies, succeeds
let same: Dense<s![1, 3, 4, 4], B> = nhwc.into_row_major()?;  // checks, Err here

// 3. An allocatable ChannelsLast, with a conformance test asserting values.
let x: Nhwc<s![16, 3, 224, 224], B> = Cpu.zeros(shape![16, 3, 224, 224])?;
```

Until those exist the parameter is carrying one bound and a measurement.

### 2.4 Custom dtypes are open at dispatch and closed at persistence

`96-custom-dtypes-devices.md` names five subsystems demanding `BuiltinDType`
and recommends widening those bounds to `ConstDType` and dispatching on
`DTypeKey`. Worth confirming one thing that recommendation leaves implicit:
the capability layer is already descriptor-keyed. `CapabilityRule.dtypes` is
`&'static [DTypeDescriptor]` and `CapabilityQuery.dtype` is a
`DTypeDescriptor`, so a custom dtype can already reach a capability answer.

The piece that is not a bound widening is persistence. `DTypeKey`'s
`Deserialize` refuses any namespace other than `incin` outright:

```rust
Err(serde::de::Error::custom(alloc::format!(
    "Deserializing custom DTypeKey ({}, {}, {}) is not supported without a \
     persistent DType registry", ns, name, version
)))
```

So a custom dtype can be defined, dispatched and executed, and cannot survive
a checkpoint round trip. That is the difference between an extension point and
a demo.

**Proposed.** The error message already names the missing piece, and the
registry it asks for is small, because `DTypeDescriptor` is already the whole
of what a dtype is:

```rust
/// Process-wide map from a wire key back to the descriptor that defines it.
/// Built-ins are pre-registered; a downstream crate registers its own once,
/// at startup, before loading anything that mentions it.
pub struct DTypeRegistry { /* spin::Mutex<BTreeMap<DTypeKey, DTypeDescriptor>> */ }

impl DTypeRegistry {
    /// Registering the same key twice with the same descriptor is idempotent.
    /// Registering it with a different one is an error rather than a silent
    /// replacement, because the second caller's tensors would then be read
    /// with the first caller's encoding.
    pub fn register(descriptor: DTypeDescriptor) -> Result<()>;
    pub fn lookup(key: DTypeKey) -> Option<DTypeDescriptor>;
}

// Deserialize then becomes a lookup with the same refusal as its fallback,
// so an unregistered key still fails loudly rather than guessing.
impl<'de> Deserialize<'de> for DTypeKey {
    fn deserialize<D>(d: D) -> Result<Self, D::Error> {
        let key = DTypeKey::new(/* from the wire triple */);
        DTypeRegistry::lookup(key)
            .map(|descriptor| descriptor.key())
            .ok_or_else(|| /* the existing message, now actionable */)
    }
}
```

This should be sequenced ahead of the rest of #96, because everything else
there is only useful once a custom dtype can outlive the process that made it.

## 3. The construction surfaces

### 3.1 There are two construction paths and neither is retired

`Tensor::<s![2, 3], B>::zeros(())` and `Cpu.zeros(shape![2, 3])` both build a
tensor. Counting current uses in examples, `incin-core/src/nn` and the book:
90 for the tuple form, 121 for the target form. The `Tensor` type's own
rustdoc teaches the tuple form; `docs/book/src/quickstart.md` teaches the
target form and never mentions the other.

`crates/incin-backends/src/target/mod.rs` is unusually direct about why the
target form exists, and the sentence is worth quoting because it is the
project's own assessment of the surface it kept:

> Get it wrong and the diagnostic is an unsatisfied
> `ArgInto<TensorArgsData<..>>` bound that names none of the four things it is
> actually talking about.

Side by side, on the case the device module has to spend a paragraph
explaining:

```rust
// Tuple form. The leading unit is not decoration: a fully static shape's
// Arg is a tuple of units, NotUnit counts that as a supplied argument, so
// the device selector has to shift into second position.
let t = Tensor::<s![2, 3], IncinBackend<Cuda>, f32, Grad>::zeros(((), Cuda::new(2)))?;

// Passing the selector alone is read as the *shape*, and the diagnostic
// names ArgInto and TensorArgsData rather than the device.
let t = Tensor::<s![2, 3], IncinBackend<Cuda>>::zeros(Cuda::new(2))?;  // unreadable error

// Target form. Backend and device come from the value, geometry from the
// shape argument, dtype from the target or the data. None can be written in
// another's position, so there is no order to remember.
let gpu = Target::<Native, Cuda>::new(Cuda::new(2));
let t = gpu.zeros(shape![2, 3])?.require_grad();
```

`docs/plan/UX-ARCHITECTURE-HANDOFF.md` records the review that produced the
target API and is explicit that the answer to "which object should be the
user-facing allocation target" is a device value. That question was answered.
What did not happen is retiring the other answer.

One clarification on tiering, since it is easy to overstate. `API_TIERS.md`
gives `incin_backends::target` tier X and `incin::prelude` tier S, and
`TargetExt` is re-exported into the prelude. That is not an inversion; it is
what a facade is for. What is true is narrower: the tier table assigns tiers
to modules and says nothing about re-exported items, so the compatibility
status of `Cpu.zeros` is unstated rather than wrong. Worth stating, since it is
the call most users will write first.

### 3.2 `ArgInto`'s positional lifting is the largest remaining UX liability

The tuple form's cost is not only the diagnostic. `tensor/arg_into.rs` is
roughly 700 hand-written lines: identity conversions per primitive, a
`NotUnit` marker to distinguish a supplied argument from an omitted one, and
one lifting impl per subset of the four construction parameters, which is
`C(4,1) + C(4,2) + C(4,3) + C(4,4)` = 15 before the arity-specific shape tuple
conversions.

The generated half is larger. `incin_macros::impl_layer_args!(9)` passes a
`MaxRank` of 9, and the generator loops every subset mask of every rank up to
it:

```rust
for rank in 1..=max_rank {
    for mask in 0..(1usize << rank) { /* one impl per mask */ }
}
```

That is `2^1 + ... + 2^9` = 1022 impls, plus one extra scalar projection per
single-dynamic-position mask. Roughly a thousand trait implementations exist
so that a layer constructor can take its arguments in a compressed tuple.

The mechanism is ingenious and the failure mode is the worst kind: a bound
mentioning a type the user never wrote, about a parameter it does not name.

**Proposed.** Not a replacement design. One already exists, is used more than
the thing it replaced, and is what the book teaches. Finish the migration:

```rust
// 1. Move the `Tensor` rustdoc examples onto the target form, so the type's
//    own documentation stops teaching the path with the bad diagnostics.
//    Today, in tensor/base/types.rs:
//        let t = Tensor::<s![2, 5, 10], DefaultBackend>::zeros(()).unwrap();
//    Proposed:
//        let t: Dense<s![2, 5, 10], _> = Cpu.zeros(shape![2, 5, 10])?;

// 2. Mark the tuple constructors as the compatibility path.
#[deprecated(since = "0.2.0", note = "use a target value: `Cpu.zeros(shape![..])`")]
pub fn zeros<A>(args: A) -> Result<Self> { ... }

// 3. Leave `ArgInto` in place and stop growing it. The 1022 layer impls stay
//    until the layer constructors move too, which is a separate step and does
//    not block this one.
```

Documentation, examples, and a deprecation. No architecture moves, and it
removes the largest single source of unreadable diagnostics in the crate.

### 3.3 `apply_op` covers one shape of custom operation

D8 scopes `apply_op` to a single input, shape preserving, deliberately, on the
grounds that a shape-changing operation's output geometry is something only
the caller knows. That is right. The consequence worth naming is the size of
the cliff on the other side:

```rust
// One input, shape preserving. The whole call.
let y = x.apply_op::<Square>(NoAttributes)?;

// Two inputs, or a shape change. Everything apply_op exists to avoid.
let lhs = TensorHandle::from_storage::<B, f32, Local>(x.inner());
let rhs = TensorHandle::from_storage::<B, f32, Local>(w.inner());
let expected = ShapeValue::<S2>::try_new(dims)?;
let context = ExecutionContext::from_scope(B::default());
let storage = dispatch::execute_shaped_n::<MyOp, B, S2>(
    &context, attributes, &[lhs, rhs], &expected,
)?;
let y = Tensor::<S2, B, f32, Grad>::try_from_storage(
    storage.into(), shape_buf, dtype_field, device_field, grad_field,
)?;
```

The last four arguments are the four things `apply_op` exists to stop the
caller restating, and they are restated here because the shape changed.

**Proposed.** A middle rung that does not reopen D8's objection. The caller
still states the geometry, because only they know it; they do not also restate
dtype, device or gradient marker, and they never touch a handle:

```rust
impl<S, B, K, G, L> Tensor<S, B, K, G, Local, L> {
    /// `apply_op` for several inputs and a caller-stated output shape.
    ///
    /// The shape is a parameter because D8 is right that nothing else knows
    /// it. Everything else is inherited from `self` exactly as in `apply_op`.
    pub fn apply_op_n<O, S2>(
        &self,
        others: &[&Self],
        attributes: O::Attributes,
        shape: ShapeValue<S2>,
    ) -> Result<Tensor<S2, B, K, G, Local>>
    where
        S2: Shape,
        O: Operation,
        B: Execute<O> + Capabilities,
        <B as Execute<O>>::Output: Into<B::Storage<K>>;
}

// The same operation as above.
let y = x.apply_op_n::<MyOp, S2>(&[&w], attributes, expected)?;
```

That is strictly smaller than "a fully general `apply_op_n` returning
arbitrary shapes", which is what D8 declined, because the geometry is still
the caller's to supply.

### 3.4 Repeating `Output: Into<Storage>` once per operation

A recipe written as dispatched built-ins pays two bounds per operation it
uses, one to say the backend executes it and one to say the result converts to
storage:

```rust
B: Execute<op::Mul> + Execute<op::MulScalar>,
<B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
<B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
```

A ten-operation kernel carries twenty. The constraint on any answer is that
`Execute<O>::Output` must stay an associated type: it is what lets an
operation return a pair, a `ShapeBuf` or a scalar, which
`FROZEN_FOUNDATIONS.md` names as the reason the contract has the shape it has.
So collapsing `Output` to storage is not available.

Three candidates were compiled and run against the CPU backend.

**An empty trait alias does not work,** and this is the part worth recording
because it is the obvious first attempt:

```rust
pub trait ExecuteInto<O: Operation, K: DType>: Execute<O> {}
impl<B, O, K> ExecuteInto<O, K> for B
where B: Execute<O>, <B as Execute<O>>::Output: Into<B::Storage<K>> {}
```

The bound `B: ExecuteInto<op::Mul, K>` is accepted, but Rust does not
propagate a blanket impl's where clause back to a generic caller, so any
helper that actually writes `.map(Into::into)` still demands the `Into` bound
and the error reappears one level down:

```
error[E0277]: the trait bound `<Bk as StorageBackend>::Storage<K>:
              From<<Bk as Execute<O>>::Output>` is not satisfied
```

**Putting the conversion on the trait does work.** The obligation is then
discharged once, inside the blanket impl where it is in scope, and generic
code never names it again:

```rust
pub trait ExecuteInto<O: Operation, K: DType>: Execute<O> + Capabilities + Sized {
    fn dispatch_into(
        context: &ExecutionContext<Self>,
        attributes: O::Attributes,
        inputs: &[TensorHandle<'_>],
    ) -> Result<Self::Storage<K>, BackendError>;
}

impl<B, O, K> ExecuteInto<O, K> for B
where
    B: Execute<O> + Capabilities + StorageBackend,
    O: Operation,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    fn dispatch_into(/* ... */) -> Result<Self::Storage<K>, BackendError> {
        execute::<O, B>(context, attributes, inputs).map(Into::into).map_err(/* ... */)
    }
}
```

Two bounds per operation become one, and nothing is given up: `Execute` is
untouched, `Output` keeps its associated type, and multi-output operations are
unaffected because they simply do not implement the alias.

**One bound for a whole capability group** is the version that stops scaling
with the recipe. The groups already exist, declared once in
`capability/declarations.rs` as rule shapes, and
`assert_every_advertised_row_executes!` already proves a backend advertising a
group has an `Execute` behind every member. So a group trait asks for exactly
what a backend already implements as a unit:

```rust
// Generated from the same declaration the capability rows come from.
pub trait ElementwiseOps<K: DType>:
    ExecuteInto<op::Add, K> + ExecuteInto<op::Mul, K> + ExecuteInto<op::MulScalar, K> /* ... */
{}
impl<B, K> ElementwiseOps<K> for B where B: /* the same list */ {}
```

```rust
// Before: two bounds per operation, growing with the kernel.
impl<B, K> DifferentiableOp<B> for ScaledSquare<K>
where
    B: Backend + Capabilities + SupportsDType<K>
        + Execute<op::Mul> + Execute<op::MulScalar> + RecordingBackend<K>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,

// After: one, and a fourth operation costs nothing.
impl<B, K> DifferentiableOp<B> for ScaledSquare<K>
where
    B: Backend + SupportsDType<K> + RecordingBackend<K> + ElementwiseOps<K>,
```

The trade is over-constraint: asking for the group when the kernel uses three
of its members refuses a backend that has those three and not the rest. That
is not a real loss here, because the group is the unit a backend implements
and the completeness proof enforces it, but it is the reason the alias should
exist alongside the group rather than be replaced by it. A kernel that wants
exactly two operations names them; a kernel that wants pointwise arithmetic
names the group.

**Not landed.** All three compile and run, but the group traits should be
generated from `capability/declarations.rs` rather than hand-written, which is
a macro change in `incin-backends` and a public API addition in both crates.
It wants its own change with its own baseline regeneration.

> **Correction, 2026-09-11.** `ExecuteInto` landed. The group traits did not,
> and should not: the claim above that "the group is the unit a backend
> implements and the completeness proof enforces it" is false. The proof
> enforces that a backend implements what *it* advertises, not that the four
> advertise the same members, and they differ sharply -- `elementwise` is 48
> members on CPU and CUDA, 17 on WGPU and 4 on Metal. A group trait generated
> from the CPU declaration is unsatisfiable for two of four backends. See D16
> in `custom-op-autograd-decisions.md` for the full count and for the five
> groups that could never have one regardless.

## 4. What landed, and what is next

### Landed, each with a test that fails without it

1. **The `GradMode` wrapper on `forward` (1.1).** One line. It was a silently
   doubled gradient, and it is what makes composing built-ins inside a custom
   kernel a technique rather than a trap, which is the whole of 2.1.
2. **The duplicate-output verdict in the oracle (1.2).** Sound as measured,
   with no false positives and no new field on `AdvertisedTuple`. A count-based
   gate is not an option and the reasoning is in 1.2.
3. **The api-examples gate (1.4).** It now compares instead of regenerating,
   and no longer writes to the tree.
4. **The `DifferentiableOp` module documentation.** It claimed composition
   needed no trait and that two dtypes meant two recipes. Both are corrected
   against what 1.1 and 2.1 measured.

### Next, in the order I would take them

1. **Attributes on `backward` (1.3).** Three implementations in tree and none
   downstream. The cost only goes up.
2. **`ExecuteInto` and generated group traits (3.4).** Compiled and run;
   what remains is generating the groups from the declaration rather than by
   hand, plus the public API baselines.
3. **Retire the tuple construction path (3.1, 3.2).** Documentation, examples
   and a deprecation. No architecture moves.
4. **`apply_op_n` (3.3).** Small, and it removes the only remaining reason a
   user touches `TensorHandle`.
5. **The custom-dtype registry (2.4).** Before the rest of #96, because a dtype
   that cannot be checkpointed is not an extension point.
6. **The layout trio (2.3),** whenever the transpose measurement is worth
   collecting on.
7. **GRD-006 with the full node reshape (2.2).** The largest, and the one that
   should absorb D7, D9 and 1.3 if 1.3 has not landed by then.

## Reproducing the measurements

The numbers in 1.1 and 2.1 came from two test files written for this review
and not kept. Their essential shape: a type implementing `Operation` with a
custom `OperationKey`, plus `DifferentiableOp<B>` whose `forward` dispatches a
built-in through `incin_core::backend_authoring::execute` against an
`ExecutionContext::from_scope(B::default())`, and whose `backward` returns the
analytically correct rule. Run the forward, call `AutogradBackend::backward`,
and read the input's gradient. Without the `GradMode` wrapper it is exactly
twice the analytic value; with it, correct. The listings in 1.1 and 2.1 are
those files with the error plumbing elided.
