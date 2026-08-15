# Backend authoring

Feature `backend-authoring`. This chapter is for someone adding a new device
to Incin, not for someone using it.

The backend authoring contract is the descriptor executor. Implement one
`Execute<Descriptor<op::X>>` instance for each operation the backend advertises.
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

`TensorMeta` is a proof token (see [Invariants](./invariants.md)) — shape,
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

## Capabilities: claim only what you run

```rust,ignore
impl Capabilities for MyBackend {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        match query.operation {
            OperationKind::MatMul if query.dtype == DTypeId::F32.descriptor() => SupportLevel::Native,
            operation => SupportLevel::Unsupported(
                UnsupportedReason::Operation { operation },
            ),
        }
    }
}
```

A `CapabilityQuery` carries operation, dtype, layout, rank, training flag and
math mode. `SupportLevel` is `Native`, `Composed` (you rewrite it into other
operations), `Fallback`, or `Unsupported(reason)` — and the reason is typed,
so a refusal names the specific constraint that failed.

The rule the whole design rests on: **an advertised operation must execute.**
The CPU backend makes this mechanical — the same declaration that generates
its capability rows generates a compile-time obligation that each has an
`Execute` impl, so advertising something unimplemented does not build. Copy
that pattern if you can; the alternative is a capability table that is
documentation rather than a contract.

## Writing an executor

```rust,ignore
impl Execute<op::Add> for MyBackend {
    type Output = MyStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Add, Self>,
    ) -> Result<MyStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(OperationKind::Add, "add expects two operands"));
        };
        // `request.operation` is `Validated` — its output shape was derived
        // and checked before you were reached. Read it rather than
        // re-deriving it.
        let out_shape = request.operation.descriptor().output_shape();
        todo!("run the kernel")
    }
}
```

The backend executor receives a validated descriptor and checked tensor
handles. Shape-typed callers use `exec::dispatch::execute_shaped` before this
boundary; the executor itself reads the validated output metadata rather than
re-deriving a shape from the Rust type. The output associated type is not fixed
to storage: readback can return an `f64` or a vector, and multi-output
operations can return a tuple.

## The checklist

1. `StorageBackend` — name, storage type, device type, metadata accessor.
2. Your storage type produces a valid `TensorMeta`.
3. `Capabilities` — claim exactly what you execute, refuse with typed reasons.
4. `Execute<op::X>` for each advertised operation.
5. A capability-matrix test that *runs* each advertised row rather than
   asserting the table against itself.

Step 6 is the one that catches real mistakes. The repository's own
`capability_matrix` suite executes the boundary cases of every registered
rule, which is how it found rows advertising ranks their kernels refused and
dtypes their kernels silently narrowed.

Two executable downstream fixtures show the contract in context:
`crates/incin-core/tests/custom_operation.rs` implements a custom operation,
and `crates/incin/tests/consumer-fixtures/backend-authoring-pass/` implements
both a small custom backend and an inference-only backend. They are compiled
as part of focused integration suites rather than presented as unchecked
pseudocode.

## Custom operations

A custom operation supplies an `Operation` identity, serializable attributes,
and output inference. A backend opts into that identity through
`Capabilities` and implements `Execute<YourOperation>` with an
`ExecutionRequest`. The downstream fixture demonstrates descriptor creation,
attribute validation, capability admission, and execution against the public
authoring traits. Custom autodiff extension points are not part of this
contract yet, so a custom operation should be documented as forward-only
unless it is composed from existing differentiable tensor operations.
