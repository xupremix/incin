typed layout: decisions taken, options kept open

A record of every choice made while implementing the layout parameter, the
alternatives that were live at the time, and why each was set aside. Written so
that a later reader can reopen any of them without re-deriving the reasoning.

## Settled

**One parameter, facts as traits.** Strides, offset, alignment and contiguity
are four facts but one type parameter, with `Contiguous` and `AlignedTo<N>` as
bounds. Rejected: a parameter per fact, which gives
`Tensor<S, B, K, G, P, L, A, Q>`. The precedent is `Shape`: one parameter, 74
traits over it. Bundling is also what makes rank congruence (`LayoutOf<S>`)
expressible at all, which is the rule taken from CuTe.

**`Unknown` is the default, and it claims nothing.** This is the same shape of
decision as `Dyn` for shapes and `ProofLevel::Dynamic` for proofs: the escape
hatch is the default, static knowledge is opt-in, and silence is never credited.
A tensor that has proven nothing keeps every runtime path it had. Rejected:
having no default (see below) and defaulting to `RowMajor` (would credit a claim
nobody checked).

**Layout facts are per-axis `Option`, not all-or-nothing.** Buys a real
asymmetry: a dynamic *outermost* axis voids no stride at all, because no stride
is a product that includes it, while a dynamic inner axis voids only those
enclosing it. All-or-nothing would discard the common transformer case, where
the batch axis is dynamic and everything inner is constant.

**Ranks past `MAX_STATIC_RANK` report nothing, not a prefix.** An intermediate
version panicked at const evaluation, reasoning that failing loudly beats
reporting a wrong geometry. That conflated a *truncated* geometry, which is a
miscompile, with *no* geometry, which is a missed optimisation -- and it broke
compilation of an existing rank-18 shape in `tensor_ops`. Deep shapes keep
working and simply forgo the specialisation.

**`into_row_major` is checked; there is no `assume_row_major`.** An unchecked
promotion is the single API that could make every downstream `L: Contiguous`
bound meaningless, and the check it saves is a stride comparison over the rank.
Defined only on `Unknown`, since a tensor that already carries a layout got it
from somewhere that knew.

**A gradient's layout is `Unknown`, not its source's.** A gradient is an
allocation the backend made on its own terms; carrying the source's claim across
would assert something about a buffer the function never inspected. The source
stays generic, so the proof simply does not transfer.

**Quantization block structure belongs with `K`, not `L`.** Alignment and
strides say *where* an element sits; block structure says *what an element is*.
The evidence is the `quantized_matmul` disagreement already in the tree -- three
components disagreeing about what a quantized operand *is* -- which filing under
layout would leave just as unchecked.

## Decided, but the alternative is worth revisiting

**Pointwise output carries the operand's `L`, rather than asserting
`RowMajor<S>`.**

Both are truthful: a pointwise op allocates a fresh dense buffer, and a binary
op under broadcast cannot alias either operand. `RowMajor<S>` is strictly more
informative -- it would *upgrade* an unproven operand, so contiguity would be
recoverable anywhere rather than only where someone called `into_row_major`.

It was tried first and reverted. Because `Unknown` and `RowMajor` are different
types, asserting `RowMajor` forces every downstream signature that says
`Tensor<S, B, K, G>` to be rewritten; `nn/lstm` failed on the first attempt.
Carrying `L` propagates exactly as much knowledge as the caller already had, so
existing signatures keep compiling and a proven chain stays proven.

**Reopen this once enough of the API is converted that the propagation is
cheap.** The upgrade is the more valuable end state; it is only the migration
cost that argues against it today. It also needs the operation *contract* to
state that outputs are dense, rather than relying on every current backend
happening to do so -- worth a conformance test asserting output contiguity
before the claim is made in a type.

**The migration order is wrong and was kept anyway.**

Adding the parameter defaulted lets every existing impl bind `L` silently: the
crate compiles, nothing looks wrong, and unconverted API is invisible until a
tensor carries a real layout. Undefaulted-first is correct -- the compiler
enumerates every site that must decide.

Rust does not allow it surgically: defaulted parameters must be trailing, so `L`
cannot be undefaulted while `K`, `G` and `P` have defaults. Stripping all of them
was measured at **650 errors in `incin-core`, all one class**, concentrated in
the ops and nn modules.

The compensation actually in place is `a_proven_tensor_keeps_the_ordinary_api`:
a test that holds a proven tensor and exercises the API, so a method that stops
compiling there names a module that still pins `L`. It is weaker than the
compiler enumerating everything, and it only covers what someone remembered to
call. **If the conversion stalls, restarting from the undefaulted parameter is
the better path.**

## Not attempted, and why

**Non-materialising views.** The largest prize and the original motivation.
Every CUDA operation that could produce a non-contiguous result materialises
instead -- measured: all nineteen `try_from_parts` call sites pass
`contiguous_strides`, and a 3x4 transposed to 4x3 comes back
`LayoutClass::Contiguous`. So every `transpose`, `narrow` and `broadcast` pays a
full copy for what is mathematically a relabelling.

Typed layout is the prerequisite: a view can only *be* a view if the consumer is
compiled against the stride pattern. But the gating measurement is still
unmade -- whether materialising actually costs anything on a real workload.
Materialising is not obviously wrong: a contiguous copy can beat repeated
strided reads and keeps everything downstream on the fast path. **Measure before
scoping.**

**Hierarchical shapes.** CuTe shapes nest (`((_2,_3),_4)`) and the nesting *is*
the tiling structure, which is how it expresses thread-value partitioning.
`DimCons<H: Dim, T: Shape>` takes a `Dim` as its head, so shapes here are flat.
A flat shape can carry strides; it cannot carry tiling. This is a deeper change
than the layout parameter and was not in scope.

**A layout algebra.** CuTe has four composable operations -- composition,
complement, division, product -- that generate the rest, and preserve staticness
through transformations. Incin has 74 individual shape traits: a menu, not a
calculus. Converting is a separate and much larger question, and the layout
parameter does not depend on it.

**`AlignedTo<N>` has no implementors or consumers yet.** It is defined because
it is part of the vocabulary the bundle exists to carry, and because
`select_unary_strategy` currently decides packing with
`offset.is_multiple_of(width)` at runtime -- a check a bound could replace. Not
wired, so today it is surface without a consumer, which is the thing #111
objects to elsewhere. Either wire it or remove it.

**Hover diagnostics.** Inlay hints are unaffected --
`shorten_collapsed_tensor_tail` already drops everything after the shape. Hover
shows the full type and will now display `Unknown`, a parameter that by
definition carries no information; eliding it there is a clear improvement.
Separately, rustc elides more aggressively with six parameters (`CpuBackendImpl`
became `...` in one refreshed fixture), so raw diagnostics are slightly harder
to read than before. That is work for `incin-diagnostics`.

**Uniqueness / aliasing as a type-level fact.** Considered as the way to unblock
compiled fusion, and it turned out not to be needed: `CapturedNode` already
carries `inputs` and `outputs` as edges, so exclusive consumption is countable
from the graph. Implemented that way instead. A type-level uniqueness parameter
would still be the answer for *eager* fusion, where there is no graph to count.

**Other facts that could be typed, in rough value order:** determinism (the flag
exists at runtime and the `DuplicateIndexRule::Accumulate` doc already describes
the tension); accumulation dtype, since `K` is the storage type and the
accumulator is resolved at runtime, so "f16 storage, f32 accumulate" is
invisible in the signature; and quantization block structure, under `K` per
above.

## Conversion status

Converted: accessors, local constructors, `Clone`, unary pointwise and both
unary descriptor helpers, binary pointwise and its descriptor helper.

Not converted: `reduce`, `manipulation/*`, `matmul`, `nn/*`. A proven tensor
cannot call those yet, and the symptom is a missing method rather than a clear
error -- which is why the book chapter states explicitly which modules are done.

## Reusing `Dyn` as the layout marker: tried, rejected on diagnostics

`Unknown` and `Dyn` express the same idea -- the compiler settled nothing, read
it at runtime -- so spelling the layout marker `Dyn` would remove a concept.
`impl Layout for Dyn` was written and compiles with no conflict, so nothing
technical stops it. Three things emerged only in the output.

**The diagnostic names the marker twice.** With `Dyn` in both the shape and
layout slots, a reader counts positions to tell which is which:

```text
expected struct `Tensor<Dyn, CpuBackendImpl, f32, NoGrad, _, Dyn>`
   found struct `Tensor<_, _, _, _, _, incin::shapes::Unknown>`
```

**The humanizer cannot separate them.** An unknown *layout* carries no
information and should be elided from a hover entirely; a `Dyn` *shape* carries
a great deal and must stay. Spelled the same, eliding one elides the other, and
`incin-diagnostics` has no way to tell the positions apart in the general case.
This is the deciding reason: the eliding is the ergonomic fix for the sixth
parameter making rustc's own output noisier, and reuse would foreclose it.

**`Dyn` is not a pure marker.** It is a `Shape` with `Arg = Vec<usize>` and its
own `resolve`, because a dynamic shape is *constructed* from runtime dimensions.
An unknown layout is constructed from nothing: it is the absence of a claim, not
a runtime value with a constructor. Giving one type both roles conflates
"determined at runtime from a value you supply" with "not determined".

Kept as siblings rather than one type. The relationship is documented on
`Unknown` so the equivalence is not lost.

### The associated-value half of the question was already right

The instinct that a marker needs a runtime counterpart when the type cannot
settle it is exactly how shapes work, and layout inherits it for free.
`Tensor` holds a `ShapeValue<S>` pairing the shape type with runtime dimensions;
`TensorMeta` already carries strides, offset and shape for every tensor.
`Layout` is the type half, `TensorMeta` the value half, so no new field is
needed -- and `into_row_major` is precisely the operation that reads the value
and promotes it into the type when the two agree.

The latent optimisation is the converse: a *fully static* layout makes the
runtime copy redundant. The CUDA pointwise path already does the kernel-side
version -- proven extents mean the `shape` array and its per-launch upload are
not emitted, because the values are in the source instead. The tensor-side
version, not storing strides a type already determines, is open and becomes
worthwhile once enough of the API carries a layout.
