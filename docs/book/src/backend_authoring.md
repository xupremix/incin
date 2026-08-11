# Backend authoring

Feature `backend-authoring`. This chapter is for someone adding a new device
to Incin, not for someone using it.

Be aware going in that the library currently has **two** execution
architectures, and a new backend has to reckon with both. That is a known,
in-progress state, not a design you should imitate for its own sake —
[What's not finished](./whats_not_finished.md) tracks the convergence.

## The two paths

**The legacy path**: `Backend` requires nine operation-family supertraits —
`CreationOps`, `NumericOps`, `FloatOps`, `TensorOps`, `ReductionOps`,
`ModuleOps`, `LossOps`, `QuantizedOps`, `OptimizerOps`. Implementing `Backend`
means implementing all nine. This is what the stable `Tensor` surface calls
today, so a backend that skips it cannot run most user code.

**The canonical path**: one `Execute<Descriptor<op::X>>` implementation per
exact operation identity, dispatched through
`incin_core::exec::dispatch::execute_shaped`. Narrower, checked, and where the
library is heading — but adoption is still partial.

A caution the trait signatures will not tell you: **some default methods on
the op-family traits are real implementations and some are unsupported
stubs.** An empty `impl NumericOps for MyBackend {}` compiles and tells you
nothing about what actually works. Read the default bodies before relying on
them.

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
impl<T: DType, D: Device> Execute<Descriptor<op::Add>> for MyBackend<T, D> {
    type Output = MyStorage;

    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Add>, Self>,
    ) -> Result<MyStorage, BackendError> {
        let [lhs, rhs] = request.inputs else {
            return Err(invalid(OperationKind::Add, "add expects two operands"));
        };
        // `request.operation` is `Validated` — its output shape was derived
        // and checked before you were reached. Read it rather than
        // re-deriving it.
        let out_shape = request.operation.descriptor().outputs();
        todo!("run the kernel")
    }
}
```

Three things worth knowing about that signature:

**`execute_shaped<S>` is the required method; `execute` is provided.** The
delegation runs `execute_shaped::<Dyn>`. It is deliberately this way round: a
required `execute` with a defaulted `execute_shaped` would let a backend
implement only the erased form and silently never specialize.

**`S` carries compile-time facts.** `S::PROOF` says how much of the shape the
compiler settled; `S::STATIC_NUMEL` is `Some(n)` when the element count is a
constant. Because `S` is a type parameter, `if let Some(n) = S::STATIC_NUMEL`
collapses to one arm at monomorphization rather than branching at run time.
The CPU creation executors use exactly this to skip a runtime element-count
loop for statically-shaped allocations.

**The output associated type is not fixed to storage.** A readback returns an
`f64` or a `Vec<f64>`; `chunk`/`split` return several storages; `topk` returns
a pair. Naming the output as an associated type is what lets those exist
without a wrapper that gets immediately unwrapped.

## The checklist

1. `StorageBackend` — name, storage type, device type, metadata accessor.
2. Your storage type produces a valid `TensorMeta`.
3. `Capabilities` — claim exactly what you execute, refuse with typed reasons.
4. The nine op-family traits, for the stable surface to work at all.
5. `Execute<Descriptor<op::X>>` for whatever you want on the canonical path.
6. A capability-matrix test that *runs* each advertised row rather than
   asserting the table against itself.

Step 6 is the one that catches real mistakes. The repository's own
`capability_matrix` suite executes the boundary cases of every registered
rule, which is how it found rows advertising ranks their kernels refused and
dtypes their kernels silently narrowed.
