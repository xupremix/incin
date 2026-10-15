# Custom autograd: writing an operation that trains

The [stable custom-operations chapter](../#/custom_operations) shows how
to add an operation and execute it. This chapter is about the other half:
teaching it to train. The mechanism is `DifferentiableOp`, the design
tension is per-dtype recipes, and the reference implementation is a
sixteen-line `square` that trains on CPU.

## The trait: forward plus what backward needs

`DifferentiableOp<B>` (`crates/incin-core/src/tensor/backend/differentiable.rs`)
extends `Operation` with three items:

- `type Dtype: DType` — the dtype whose storage this recipe is written
  against. One recipe per dtype, not one recipe polymorphic over dtypes.
- `type Saved` — what forward saves for backward. Owned values, never
  handles: the recipe closure moves them and must be self-contained. What
  the kernel computed, not what the caller configured — attributes reach
  `backward` on their own, so configuration stored here would live on the
  node twice.
- `supports` — the capability answer planners see, forwarded by the
  blanket `Execute` through `supports_custom`. Override it to refuse
  dtypes (or ranks, layouts, training modes) the kernel does not hold.

The `Attributes: Send + Sync` bound sits on this trait deliberately
rather than on the blanket `Execute` or the open `Operation` contract, so
a missing bound is reported against the `impl DifferentiableOp` that
introduced it instead of surfacing as a mysterious "no `Execute` for this
operation" — and so the other 170-odd catalog entries do not pay for the
few operations that record custom recipes.

## The tension: one recipe per dtype

This is the design decision the research record
(`docs/plan/research/0.2.0/custom-op-autograd.md` and
`custom-op-autograd-decisions.md`) spends the most ink on, because it cuts
against the grain of Candle and PyTorch: a custom operation's backward is
monomorphized per dtype through the associated type. Per-input gradient
`Option`s (the `needs_input_grad` question) are settled the same way —
explicitly, per operation, rather than by runtime introspection. Higher-order
gradients stay out of scope until GRD-006. None of this is accidental
complexity; it is what "fail closed" costs when the recipe, the storage,
and the capability answer must all agree before launch.

`Tensor::apply_op` (unary) and `apply_op_n` (N-ary) are the call sites
(D8/D15): descriptor-checked dispatch through `execute_shaped`, with the
gradient mode scoped around the call. Broad operation-family traits are
out (D16 rejected them): per-operation descriptors all the way down.

## The reference: `square` that trains

`crates/incin-core/tests/custom_training.rs` is written entirely against
the public extension surface, from a user crate's perspective, and this
chapter follows its structure. The operation declares its contract —
`y = x^2`, elementwise, same shape out — then implements the trait with
the dtype story pointing both directions:

```rust,ignore
impl DifferentiableOp<CpuBackendImpl<Cpu>> for Square {
    type Dtype = f32;
    /// The saved input itself: the recipe needs every `x` next to its gradient.
    type Saved = CpuStorage;

    fn supports(query: &CapabilityQuery) -> SupportLevel {
        // Layer 3 of the dtype contract: this kernel is `f32` only, so say
        // so per query. Anything else is refused before launch, never
        // executed against a dtype it was not written for.
        if query.dtype != DTypeId::F32.descriptor() {
            SupportLevel::Unsupported(UnsupportedReason::CustomOperation { ... })
        } else {
            SupportLevel::Native
        }
    }
    // forward runs the kernel and saves the input; backward multiplies the
    // cotangent by 2x through tape_record; the f16 twin of the test proves
    // the refusal fires before any kernel runs.
}
```

Run both directions yourself — the training proof and the refusal proof:

```sh
cargo test -p incin-core --test custom_training
cargo test -p incin-core --test custom_op_composition
```

The boundary that remains is the same one the stable chapter ends on:
custom operations execute and differentiate on backends whose capability
rows admit them, and the NVRTC device-dispatch tail for *compiled* custom
graphs is still ahead (see [Compiled graphs](./next_compiled_fusion.md)).
