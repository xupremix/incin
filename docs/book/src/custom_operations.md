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

## Training a custom operation, end to end

What follows is the whole pattern, condensed from the executed fixture in
`crates/incin-core/tests/custom_training.rs` (CPU; WGPU and CUDA twins live
beside it in `crates/incin-backends/tests/`). A custom `square` operation,
`y = x^2`, training through the standard backward pass:

```rust,ignore
use incin_backends::cpu::{tape_record, CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    AutogradBackend, Execute, ExecutionRequest, Operation, OperationKey,
    TapeNode, TapeStorage,
};
use incin_core::exec::catalog::NoAttributes;

#[derive(Clone, Debug)]
struct Square;

impl Operation for Square {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("square"),
        version: 1,
    };
    fn infer_outputs(_, inputs: &[LogicalTensorMeta]) -> Result<..> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl Execute<Square> for CpuBackendImpl<Cpu> {
    type Output = CpuStorage;

    fn supports_custom(&self, query: &CapabilityQuery) -> SupportLevel {
        // f32 only: anything else is refused before launch, never executed
        // against a dtype the kernel was not written for.
        if query.dtype != DTypeId::F32.descriptor() {
            SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                operation: Square::KEY,
            })
        } else {
            SupportLevel::Native
        }
    }

    fn execute(&self, request: ExecutionRequest<'_, Square, Self>)
        -> Result<CpuStorage, BackendError>
    {
        let x = /* downcast the single input handle to &CpuStorage */;
        let out = /* elementwise x^2 into fresh storage (fresh id) */;
        // The training half: dy/dx = 2x, with x saved by move.
        let x_saved = x.clone();
        tape_record(TapeNode {
            output_id: out.id(),
            input_ids: vec![x.id()],
            backward: Box::new(move |grad_out: &CpuStorage| {
                Ok(vec![/* 2 * x_saved * grad_out, same shape */])
            }),
        });
        Ok(out)
    }
}

// Dispatch, lift back into a typed tensor, train through the standard pass:
let out: CpuStorage = execute(&context, NoAttributes, &[handle])?;
let y = Tensor::<Dyn, Cpu, f32, Grad>::try_from_storage(out, shape, ..)?;
let grads = y.sum_all()?.backward()?; // walks built-in and custom nodes as one
```

Three details carry the soundness, and all three are pinned by the fixture
rather than left as advice. First, the recipe returns one gradient per
input in input order — the walk zips positionally, so a swapped pair trains
the wrong tensor with the right numbers. Second, `try_from_storage` moves
the storage rather than rebuilding it, so the recorded output id still
matches and the custom node stays reachable. Third, the fixture sweeps the
hand-derived gradient against central finite differences, asserts a
`NoGrad` forward records nothing, and drives an `f16` input at the
`f32`-only kernel to prove the refusal happens before any kernel runs.

The remaining seam for a fully foreign backend is its own thread-local tape
push, which stays `pub(crate)` by design: an in-tree backend moves its
backward recipes into its `Execute` impl and records them there, so custom
and built-in nodes share one graph. A foreign backend does the same with its
own thread-local over the public core `Tape` type, walked by the same
`incin_core::exec::tape::backward` the CPU backend calls. Everything else a
differentiable custom operation needs is public, and
`crates/incin-backends/examples/polar_cartesian.rs` shows it
end to end, run by CI rather than only built. Polar-to-Cartesian takes two
inputs and returns two outputs -- the multi-output inference a single-output
catalog row cannot express, run through the runtime dispatch path because the
typed one requires exactly one output -- with a backward recipe per output
assembled into core `TapeNode`s and walked by the same
`incin_core::exec::tape::backward` the CPU backend calls. The example checks
the forward values and the hand-derived gradients against textbook answers,
sweeps every input element against central finite differences, asserts the
contract refusals, and then fits `(r, theta)` to a target point by gradient
descent through nothing but its own backward.
