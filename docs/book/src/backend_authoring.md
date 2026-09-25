# Backend authoring

Feature `backend-authoring`. This chapter is for someone adding a new device
to Incin, not for someone using it.

The backend authoring contract is the descriptor executor. Implement one
`Execute<op::X>` implementation for each built-in operation the backend
advertises. `op::X` is the operation type; `Descriptor<op::X>` carries its
attributes and inferred output metadata.
Backend authors do not implement historical operation-family traits: reusable
backend helpers are ordinary functions behind each descriptor executor.

## `StorageBackend`: the minimum

```rust,ignore
use incin::backend_authoring::*;

impl StorageBackend for MyBackend {
    // Required, deliberately undefaulted: a refusal that cannot name who
    // refused is not actionable. "dtype F64 is unsupported for zeros" leaves
    // the reader guessing whether their device, build features, or dtype is
    // the thing to change.
    const BACKEND_NAME: &'static str = "MyBackend";

    type Storage<K: DType> = MyStorage;
    type Device = MyDevice;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}
```

`TensorMeta` is a proof token (see [Invariants](./invariants.md)): shape,
strides, offset, dtype, device, layout, alignment and capacity, all checked to
agree. Your storage type must be able to produce one. If your device's native
tensor type is foreign and carries no such metadata, pair it with one in a
wrapper: that is exactly what the Candle adapter does, validating the foreign
tensor's geometry once at the boundary.

`Backend` itself only combines storage, capability admission, and execution.
`HostInterop`, `VariableBackend`, and `AutogradBackend` are optional capability
owners: add them only when the backend supports readback, mutable parameters,
or training. An inference-only backend can stop after `StorageBackend`,
`Capabilities`, `Backend`, and the `Execute` implementations it advertises.

If you do implement `AutogradBackend`, note that it requires `set_grad`
alongside `backward` and `get_grad`. It is required rather than defaulted on
purpose: post-backward transforms such as `clip_grad_norm` are written once
against the trait, and a default returning `Ok(())` would turn clipping into a
silent no-op the caller could not detect. A backend that records no gradients
should return `Error::UnsupportedBackendOperation` rather than accept the
write.

## Capabilities: claim only what you run

This handwritten example admits only F32 matrix multiplication. A real backend
must also check any layout, rank, training, or math-mode restrictions it has.

```rust,ignore
use incin::backend_authoring::{
    Capabilities, CapabilityQuery, OperationIdentity, SupportLevel, UnsupportedReason,
};
use incin::prelude::{DTypeId, OperationKind};

impl Capabilities for MyBackend {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        match &query.operation {
            OperationIdentity::Builtin(OperationKind::MatMul) => {
                if query.dtype == DTypeId::F32.descriptor() {
                    SupportLevel::Native
                } else {
                    SupportLevel::Unsupported(UnsupportedReason::DType {
                        operation: OperationKind::MatMul,
                        dtype: query.dtype,
                    })
                }
            }
            OperationIdentity::Builtin(operation) => {
                SupportLevel::Unsupported(UnsupportedReason::Operation {
                    operation: *operation,
                })
            }
            OperationIdentity::Custom(operation) => {
                SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                    operation: operation.clone(),
                })
            }
        }
    }
}
```

A `CapabilityQuery` carries operation, dtype, layout, rank, training flag and
math mode. `SupportLevel` is `Native`, `Composed` (you rewrite it into other
operations), `Fallback`, or `Unsupported(reason)`, and the reason is typed,
so a refusal names the specific constraint that failed.

Canonical dispatch admits `Native` unconditionally. `Composed` requires an
execution context with `FallbackPolicy::AllowComposition` (the default) or
`AllowTransfer`; `Fallback` requires `AllowTransfer`. A policy refusal is
`CanonicalError::Policy(PolicyViolation)`, distinct from unsupported capability
and execution failures, and occurs before `Execute<O>::execute` is called.

The rule the whole design rests on: **an advertised operation must execute.**
For built-ins, `declare_capabilities!` generates both `Capabilities::support`
routing and a compile-time `Execute<op::X>` obligation for every listed entry,
even without a dispatch call. This checks that an executor exists, not that its
kernel honors every advertised constraint; capability-matrix tests still matter.

## Declaring executors and built-in capabilities

Import `declare_executors!` and `declare_capabilities!` from
`incin::backend_authoring` with the `backend-authoring` feature enabled.
`declare_executors!` accepts built-in or custom operation types. Each entry
names an exact `Output: ExecuteOutput` and a handler path. The generated
implementation passes `&self` and the original validated
`ExecutionRequest<'_, O, Self>` by value to the handler and returns its
`Result<Output, BackendError>` unchanged. It does not convert scalar, vector,
or tuple outputs into storage, or wrap errors.

The companion `declare_capabilities!` accepts only built-in
`CanonicalOperation` types. It routes queries by `CanonicalOperation::ID` to
handlers taking `(&Backend, &CapabilityQuery)` and returning `SupportLevel`
unchanged. Unlisted built-ins return `UnsupportedReason::Operation`; custom
queries return `UnsupportedReason::CustomOperation`. This does not reject
custom execution: canonical dispatch uses the `Execute` hooks for that, as
explained below. Execution policy still applies after admission.

Both macros require a concrete backend type. A specialization such as
`MyBackend<Cpu>` is accepted; generic impl parameters and `where` clauses are
not. Use handwritten implementations when those are needed.

This runnable example demonstrates routing and an exact non-storage output.
It returns the requested shape, not an allocated zero tensor; it is an
executor-contract example rather than a tensor backend.

```rust
use incin::backend_authoring::{
    declare_capabilities, declare_executors, execute, CapabilityQuery,
    ExecutionContext, ExecutionRequest, ShapeBuf, StorageBackend, SupportLevel,
    TensorMeta,
    operations::{op, CreationAttributes},
};
use incin::prelude::{BackendError, Cpu, DType, DTypeId, DeviceId};

struct ShapeBackend;

impl StorageBackend for ShapeBackend {
    const BACKEND_NAME: &'static str = "shape-example";
    type Storage<K: DType> = TensorMeta;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta {
        storage
    }
}

fn zeros_shape(
    _: &ShapeBackend,
    request: ExecutionRequest<'_, op::Zeros, ShapeBackend>,
) -> Result<ShapeBuf, BackendError> {
    Ok(ShapeBuf::from_slice(
        &request.operation.descriptor().attributes().shape,
    ))
}

fn support(_: &ShapeBackend, _: &CapabilityQuery) -> SupportLevel {
    SupportLevel::Native
}

declare_executors! {
    for ShapeBackend {
        op::Zeros => ShapeBuf = zeros_shape;
    }
}

declare_capabilities! {
    for ShapeBackend {
        op::Zeros => support;
    }
}

let context = ExecutionContext::new(ShapeBackend);
let shape = execute::<op::Zeros, _>(
    &context,
    CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    },
    &[],
)?;
assert_eq!(shape, ShapeBuf::from_slice(&[2, 3]));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The macro rustdoc examples in
`crates/incin/src/backend_authoring_macros.rs` and this chapter are included
in `cargo test -p incin --features backend-authoring --doc`.

## Writing an executor

For kernels that need handwritten implementations, the request and output
contract is the same. This skeleton leaves the storage type and kernel to the
backend author.

```rust,ignore
use incin::backend_authoring::{Execute, ExecutionRequest, operations::op};
use incin::prelude::{BackendError, OperationKind};

impl Execute<op::Add> for MyBackend {
    type Output = MyStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Add, Self>,
    ) -> Result<MyStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(BackendError::InvalidInput {
                operation: OperationKind::Add,
                reason: "add expects two operands",
            });
        };
        let outputs = request.operation.descriptor().outputs();
        todo!("run the kernel")
    }
}
```

The backend executor receives a validated descriptor and checked tensor
handles. Shape-typed callers use `incin::backend_authoring::execute_shaped`
before this boundary; the executor itself reads the validated output metadata
rather than re-deriving a shape from the Rust type. The output associated type is not fixed
to storage: readback can return an `f64` or a vector, and multi-output
operations can return a tuple.

## The checklist

1. `StorageBackend`: name, storage type, device type, metadata accessor.
2. Your storage type produces a valid `TensorMeta`.
3. `Capabilities`: claim exactly what you execute, refuse with typed reasons.
4. `Execute<op::X>` for each advertised operation.
5. `AutogradBackend` including `set_grad`, if the backend trains at all.
6. A capability-matrix test that *runs* each advertised row rather than
   asserting the table against itself.

Step 6 is the one that catches real mistakes. The repository's own capability
registration tests execute the boundary cases of every registered rule, which
is how it found rows advertising ranks their kernels refused and dtypes their
kernels silently narrowed.

Executable fixtures show the contract in context:
`crates/incin-core/tests/custom_operation.rs` implements a custom operation,
`crates/incin-core/tests/custom_training.rs` trains one end to end -- forward
kernel, recorded backward recipe, standard backward pass, finite-difference
cross-check, `NoGrad` silence, and an `f16` refusal -- and
`crates/incin/tests/consumer-fixtures/backend-authoring-pass/` implements
both a small custom backend and an inference-only backend. They are compiled
as part of focused integration suites rather than presented as unchecked
pseudocode.

## Custom operations

A custom operation supplies an `Operation` identity, serializable attributes,
and output inference. A backend opts into that identity by implementing
`Execute<YourOperation>`, either by hand or with `declare_executors!`.
Custom admission uses `Execute::supports_custom` for metadata-based queries
and `Execute::supports_custom_operation` when no input or usable output
metadata is available. Both hooks default to `SupportLevel::Native`; custom
admission does not use `Capabilities::support`.

The executor macro retains those defaults. For restricted custom admission,
write the `Execute` implementation by hand and override the relevant hooks;
do not put a custom operation in `declare_capabilities!`. The downstream
fixture demonstrates descriptor creation, attribute validation, admission,
and execution against the public authoring traits.

A custom operation trains by implementing
`DifferentiableOp`: a forward kernel plus a backward rule over one
backend's storage, with the blanket `Execute` building the node, deriving
its identities, and recording (`crates/incin-core/tests/custom_training.rs`
is the worked fixture, with WGPU and CUDA twins beside it in
`crates/incin-backends/tests/`). Multi-output operations keep the explicit
`tape_record` path instead. A custom operation that neither implements the
trait nor composes from existing differentiable tensor operations should be
documented as forward-only.

## Custom dtypes and devices: current boundaries (issue #96)

A custom logical dtype is definable outside the workspace today: implement
`DType` and `ConstDType` with a unique `DTypeKey` (it has no built-in
`DTypeId`), register it, and use it anywhere the stack is descriptor-driven.
`TensorMeta`, capability queries (`CapabilityQuery.dtype`), the operation
catalog, and the postcard state envelope all carry `DTypeDescriptor` and
accept it unchanged. Devices likewise have an open identity:
`DeviceKind::External(DeviceKey)` and `DeviceId::external(key, ordinal)`
let a downstream backend name hardware Incin has never heard of with a
versioned namespace/name key (D-110, issue #96); Incin interprets the key
only for equality, digest, and refuse-on-mismatch. Maintained backends
keep first-class `DeviceKind` variants.

The surfaces that still carry the closed built-in `DTypeId` vocabulary as a
`const` require the `BuiltinDType` bound explicitly, so a custom dtype is
refused at compile time rather than coerced onto a built-in's key (issue
#96):

- the distributed static planners — `HybridPlanner::plan_data_static` and the
  collective, pipeline, tensor-parallel, data-parallel, and FSDP static
  pushes — fingerprint plans by `DTypeId`;
- `CollectiveTuningProblem::new_static` keys its measurement cache by
  `DTypeId`.

Two more sites match the closed builtin-ID set at runtime and return a typed
error for `builtin_id() == None` instead: the backend kernel-table lookup and
the `safetensors` snapshot export. `TensorElement` is open under the POD
bound (D-110, issue #96): a downstream POD newtype gets element access
(slicing, typed construction, host interop) while the four closed
subsystems above keep their `BuiltinDType` gates, so execution on
unadvertised dtypes still refuses loudly. Each of these is a documented
compile-time or typed-runtime refusal, never a silent fallback;
descriptor-keying the remaining sites is tracked as a future phase under
issue #96.
