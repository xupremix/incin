what to take from CuTe, ATen and the rest

Grounded against gaps found by working in this codebase rather than by reading
the competition's feature list. Ordered by what it would buy, not by how
impressive it sounds.

Two things worth saying before the list. Incin already has several mechanisms
these frameworks are admired for, sometimes in stronger form -- the comparison
below says so where it is true, because a list of everything the competition has
is not useful. And the single largest lesson is not a feature at all: every
framework here separates *what an operation means* from *how it is iterated*,
and incin does this for meaning (descriptors, `Validated`) but not for
iteration.

## 1. ATen's `TensorIterator`: reorder, do not just coalesce

**The gap, proven rather than asserted.** `crate::iteration::coalesce_dimensions`
merges adjacent axes when their strides allow it. It never reorders them -- there
is no `sort`, `reorder` or `permute` in the planner. `TensorIterator` does both,
and the reordering is what *creates* the merging opportunities.

For the transposed view that `transpose_view` now produces:

```
shape [4, 3], strides [1, 4]
  as given        -> nothing merges, rank stays 2, strided kernel
  axes reordered  -> merges to rank 1, stride 1: fully contiguous
```

A transposed view is a *contiguous* iteration if you are willing to walk its
axes in a different order. Incin currently walks it strided and pays for the
uncoalesced access, which is most of what the view/copy measurement in
`view_cost_bench` is measuring.

**The catch, which is the real lesson.** The output for shape `[4, 3]` is
contiguous with strides `[3, 1]`. Under the same permutation it becomes `[1, 3]`,
which does *not* merge -- so reordering makes the input contiguous by making the
output strided. Incin's strided kernels write `output[idx]` linearly, assuming
the output is enumerated in logical order, so the permutation would silently
write to the wrong places.

That is exactly why `TensorIterator` gives **every operand its own strides,
including the output**, and iterates a permuted space rather than the logical
one. Adopting the reordering means adopting that: the strided templates need an
output stride array and a computed store index, not a linear one.

**Worth doing.** It converts the common transposed-pointwise case from strided
to contiguous on one side, and the machinery is the same shape as the extent
folding already landed. Measure with `view_cost_bench`, which now exists and has
a harness that does not lie.

## 2. CuTe's layout algebra: four operations instead of seventy-four traits

Incin has 74 type-level shape traits -- `BroadcastShape`, `SpatialConv2d`,
`ReduceAt`, `SwapAxes`, `FlattenAt` -- which is a menu. CuTe has composition,
complement, division and product, which is a calculus: the rest are derived, and
staticness is preserved through each.

The practical difference is not elegance, it is that a menu does not compose.
Every new operation is a new trait, and nothing guarantees two of them agree
about the same geometry. A calculus makes "transpose then flatten" a
composition with one set of rules rather than two traits that must be kept
consistent by hand.

**The prerequisite incin does not have** is hierarchical shapes. `DimCons<H: Dim,
T: Shape>` takes a `Dim` as its head, so shapes are flat. CuTe's nest --
`((_2,_3),_4)` -- *is* the tiling structure, which is how it expresses
thread-value partitioning at all. A flat shape can carry strides; it cannot
carry tiling.

**Already taken:** the congruence requirement between shape and stride, which is
what `LayoutOf<S>` states, and the mixed static/dynamic-per-mode design, which is
why `STATIC_EXTENTS` is `Option` per axis.

**Verdict:** the algebra is a 0.3.0-scale rewrite gated on nesting, not a
next step. The congruence idea was the portable part and it has been taken.

## 3. ATen's `MemoryFormat`: a named layout, not just a stride pattern

`channels_last` has no analogue here -- zero hits across the tree. ATen treats it
as a first-class property that propagates through operations and that kernels
dispatch on, because for convolutions the NHWC layout is not a micro-optimisation
but the difference between using tensor cores and not.

This is directly adjacent to the layout parameter that just landed. `RowMajor<S>`
is one named layout; `ChannelsLast<S>` would be another, with the same
congruence, and `Contiguous` would correctly refuse it. The machinery exists;
what is missing is the second layout and the conv kernels that would prefer it.

**Verdict:** the highest-value *new* layout to add, and the layout parameter is
what makes it expressible rather than a runtime flag.

## 4. ATen's structured kernels: incin already has a stronger version

`torchgen` emits a meta function (shape inference) separate from the impl, so a
kernel author writes only the inner loop. Incin's descriptor system does the same
split and enforces it harder: shape inference happens in `exec::catalog`,
`Validated<O>` is unforgeable outside the crate, and the backend receives a
descriptor it cannot have constructed itself. ATen's separation is a codegen
convention; incin's is a type.

**Verdict:** nothing to take. Worth knowing so it is not re-invented.

## 5. `derivatives.yaml`: incin's version is better and already landed

ATen declares derivatives in a YAML file and generates the backward code. That
removes the duplication between a forward kernel and a hand-written backward,
which is the same defect the `codegen` IR adoption fixed here -- except
`IrExpr::diff` *computes* the derivative from the forward rather than reading a
second declaration of it, so the two cannot disagree even in principle. The
incomplete `Gelu` derivative found this session is precisely the failure a YAML
file still permits.

**Verdict:** already ahead. The remaining gap is coverage -- the IR cannot yet
express `erf`, `asinh` and the rounding family.

## 6. Inductor / tinygrad: fusion as the default, not the exception

Both treat kernel fusion as the normal path: tinygrad builds a lazy graph and
fuses on realisation, Inductor fuses pointwise chains into generated Triton.
Incin's `compiled` module fails closed and has just gained an exclusivity proof
but no lowering (#112), and the eager path does not fuse at all.

The piece incin has that neither needs to invent is the fused-kernel
representation: `codegen::fragment` already lowers a chain of operations into one
kernel body with common subexpressions shared, and `unary_fused_backward`
demonstrates it end to end.

**Verdict:** the gap is the pass, not the representation. That is #112's step 2.

## What incin has that these do not

Worth recording so it is not traded away in pursuit of the list above.

- **A proof of what the compiler settled.** `ProofLevel` is a reified,
  three-valued record with a `meet`, carried across the backend boundary on
  `ShapeEvidence` and unforgeable outside the crate. CuTe has staticness as a
  property of types but no value-level notion of *how much was settled*; ATen has
  none of it, because its shapes are runtime facts.
- **Operation legality at compile time**, across the whole operation surface,
  not just for GEMM.
- **Typed distribution.** `Placement` with `Sharded<Mesh, Axis>` and a
  `ValidMesh` that rejects a positional swap. JAX's sharding is a runtime object;
  this is a type.

## Order I would take them

1. **`TensorIterator`-style reordering with per-operand output strides.**
   Bounded, measurable with existing tooling, and it makes the view path that
   just became reachable substantially better.
2. **`ChannelsLast` as a second layout.** Uses machinery that already exists and
   pays off in convolutions.
3. **Fusion lowering** (#112 step 2), using the fragment representation already
   built.
4. Layout algebra and hierarchical shapes: real, large, and gated on a decision
   about whether tiling belongs in the type system at all.

## Sources

- [PyTorch TensorIterator internals](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals)
  and the [2021 update](https://labs.quansight.org/blog/pytorch-tensoriterator-internals-update)
- [CuTe layout documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
  and [layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
