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

## Conversion status: complete

Every tensor module accepts a layout-carrying operand: accessors, constructors,
`Clone`, pointwise unary and binary, reductions (per-axis and whole-tensor),
shape manipulation, matmul, concat/stack, conversions, and the `nn` layers.

The rule that emerged, and it is forced rather than chosen:

- **Shape-preserving** operations carry the operand's `L`.
- **Shape-changing** operations state theirs, as `Unknown`, because a layout
  describes one geometry and cannot be carried to another.

The second could state `RowMajor` -- every such result is a fresh allocation --
and that remains the more valuable end state. It is blocked on #113, not on
migration cost: CPU `transpose` returns a view where CUDA's returns a copy, so
the claim would be false on one backend.

### Things the completion surfaced

**Naming a parameter forces the ones before it to be named.** `L` is positional
after `P`, so writing it means writing `Local`, which means rustc prints
`Local` in diagnostics where it was previously elided as a default. That
exposed a path-rendering instability: rustc spells it `incin::prelude::Local`
under CI's feature set and `incin_core::dist::placement::Local` under workspace
defaults. Only one spelling can be stored in a trybuild snapshot. Recorded
under CI's invocation, with the divergence documented in the owning test.

The general lesson for any future parameter: appending to a defaulted list is
not free even when every existing signature keeps compiling, because it changes
what *diagnostics* print for everything to its left.

**`forward` should name `Self::Output`.** Several `nn` modules restated their
output type in the signature, which let it drift from `type Output` during the
conversion and produced a class of error that was pure duplication. Naming the
associated type makes them the same thing by construction.

**Spell the default by omitting it.** Several return types were written as
`Tensor<.., crate::dist::Local, crate::shapes::Unknown>` during conversion, which
is exactly what `Tensor<..>` already means. Removing the explicit form also
cleared a `clippy::type_complexity`.

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

## The backends disagree about whether `transpose` copies

Found by writing the conformance test that was supposed to *back* a type claim,
before making it. It refuted the claim instead.

Shape-changing operations cannot carry their operand's layout -- the shape
changes, and a layout is only meaningful against the shape it describes -- so
their result's layout has to be stated. Stating `RowMajor` is only honest if the
buffer is dense, and the CUDA measurement said every view materialises, so the
claim looked safe.

It is not. On CPU:

```
transpose_structural on s![3, 4]  ->  shape [4, 3], strides [1, 4]
```

That is a genuine non-contiguous **view** sharing the original buffer. On CUDA
the same operation returns shape `[4, 3]` with strides `[3, 1]` -- a fresh
contiguous copy.

**Same operation, different memory semantics per backend, and nothing states
which is correct.** Consequences:

- A type claiming `RowMajor` for a transpose would be false on CPU and true on
  CUDA. So shape-changing operations keep `Unknown` outputs until the contract
  is settled, and `into_row_major` remains the way to recover a proof.
- The earlier finding that "the strided path is unreachable" was correctly
  scoped to CUDA in the note, but it is not a framework-wide property. The
  strided kernel path *is* reachable on CPU. The extent-folding work is dead on
  CUDA specifically because CUDA copies, not because nothing is ever strided.
- It is a portability hazard independent of layouts: user code that transposes
  and then mutates through the original buffer observes different results per
  backend, and code that relies on a transpose being cheap silently pays a copy
  on CUDA.

Both behaviours are pinned by `shape_changing_operations_produce_dense_results`,
which asserts reductions and pointwise operations are dense and asserts the CPU
transpose is a view, so a backend changing its mind fails a test rather than
silently changing what a type would mean.

### Measured, and the framing was wrong

That framing -- pick one and make both backends do it -- does not survive the
measurement. On a GTX 1650, transpose plus one pointwise pass over the result:

| elements | materialise | strided view | view/mat |
|---:|---:|---:|---:|
| 4,096 | 42.2 | 42.2 | 1.00 |
| 1,048,576 | 830.5 | 731.8 | 0.88 |
| 4,194,304 | 3375.1 | 2998.5 | 0.89 |

And with the result consumed `k` times, at 1M elements:

| k | materialise | strided view | view/mat |
|---:|---:|---:|---:|
| 1 | 812.4 | 721.0 | 0.89 |
| 2 | 895.6 | 844.4 | 0.94 |
| 4 | 1064.8 | 1093.4 | 1.03 |
| 8 | 1367.6 | 1569.2 | 1.15 |

The copy's read-plus-write loses to the strided read for a single consumer at
scale, and wins from about four consumers on. The crossover is between two and
four; below ~64k elements launch overhead swamps both.

So the two are within roughly 11% in *opposite directions* depending on how
often the result is read -- a fact about the consumer, which the producing
operation cannot know. Unifying on either behaviour makes the framework
reliably wrong for half its callers, and choosing the one that wins a
microbenchmark would be choosing by the shape of the benchmark.

The resolution is to stop treating it as one operation: `transpose`
materialises and returns `RowMajor<S>`, `transpose_view` does not copy and
returns a permuted layout. The caller picks and the type records which they
got, so a downstream `L: Contiguous` bound is satisfiable in one case and
correctly refused in the other.

This is the strongest available argument for the layout parameter, and it is a
measurement rather than an aesthetic: the backends' disagreement is not a bug
to resolve by fiat, it is evidence that both behaviours are needed and that the
type should distinguish them. It also unblocks the deferred decision above --
a materialising `transpose` *can* honestly claim `RowMajor`, because it is the
operation that copies.

Caveats on the numbers: one GPU, one dtype, square shapes, a pointwise
consumer. A consumer that cannot take arbitrary strides -- matmul, generally --
will move the crossover. The shape of the conclusion should survive; the exact
`k` should not be treated as a constant.

## Implementing `transpose_view`: the spec, and why it was not started here

The measurement above says the resolution is two operations rather than one. That
is a catalog-level change, not a frontend one, and it is larger than it looks.

**What it touches.** A new operation is not a function; it is a row in a
taxonomy that several tables are checked against:

- `incin-core/src/operation_catalog.rs` -- the row itself, beside
  `(TransposeExact, "transpose", Storage, Shape, TransposeAttributes, 1, 1, "::transpose")`.
- `incin-backends/src/capability/declarations.rs` and `capability/rules.rs` --
  what each backend advertises for it.
- `incin-backends/src/conformance/fixtures/families.rs` -- operands the oracle
  builds to exercise it.
- Three executors: `cpu/canonical/shape_ops.rs`, `cuda/executor.rs`,
  `wgpu/executor.rs`.
- `docs/OPERATION_SEMANTICS.md` and `docs/capabilities.md`, both generated, so
  regenerated with `INCIN_DOCS=overwrite` rather than edited.
- `docs/public-api/*` baselines.
- The conformance oracle's covered-operation floor, which is a ratchet.

**What it should do.** The view produces no kernel launch at all: it is
metadata. On CUDA that is `CudaStorage::try_from_parts(buffer.clone(), permuted_shape,
permuted_strides, offset)`; the CPU backend already does exactly this inside
`transpose_structural`, which is why CPU's transpose is a view today. So the
executor bodies are small; the surface around them is not.

**Typing.** `transpose_view` returns `Unknown`, not a new `Strided` marker. A
marker meaning "known not contiguous" would behave identically to `Unknown`
everywhere that matters -- both fail a `Contiguous` bound, which is the whole
safety property -- so it would be a type with no distinguishing behaviour, and
the same objection #111 raises applies. Introduce one only when something needs
to *distinguish* an unproven layout from a known-permuted one; carrying the
permutation at the type level, so a consumer can specialise on it, would be that
reason.

Conversely, once `transpose` is unambiguously the materialising operation on
every backend, it *can* return `RowMajor<S::Output>` honestly, which is the
deferred output-layout decision recorded above.

**Order.** Add the operation and the CUDA executor first, since CUDA is the
backend that currently lacks the view and the one where the measurement was
taken; CPU's existing `transpose_structural` body is then reused for its
executor. Only after all three backends offer both should `transpose`'s output
layout be tightened to `RowMajor`, because tightening it while any backend still
returns a view would be exactly the false claim the earlier conformance test
caught.

## "Is there an approach that needs fewer type definitions?"

Asked after the conversion landed, and worth recording because two of the three
candidates were prototyped rather than reasoned about.

**Most of the verbosity was self-inflicted.** The conversion was scripted, so it
spelled every bound `L: crate::shapes::Layout` and every placement
`crate::dist::Local` -- 70 and 60 fully-qualified paths in positions where the
short name is unambiguous. Importing them turns

```rust
impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, L: crate::shapes::Layout>
    Tensor<S, B, K, G, crate::dist::Local, L>
```

into

```rust
impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, L: Layout>
    Tensor<S, B, K, G, Local, L>
```

No design change, and it accounts for a large share of what looked like the
parameter's cost.

**A proof-carrying wrapper would remove the parameter and lose propagation.**
Prototyped: `struct Dense<T>(T)` with `Deref<Target = T>` gives the entire
tensor API for free, with zero threading through any impl -- genuinely the
cheapest option on that axis. But `d.neg()` derefs, calls the inherent method,
and returns a plain tensor, so the proof dies at every operation and a chain
cannot end in `reshape_view`. `a_proof_survives_a_pointwise_chain` is exactly
the behaviour that would be lost. The wrapper is the better design *if* proofs
are meant to be established immediately before use and not carried; the
parameter is better if they are meant to survive a pipeline. The latter is what
makes the feature worth having.

**Bundling all six into `Tensor<M>` over a metadata trait** shortens
declarations and lengthens every bound, since each constraint routes through
`M::`, and it makes construction name a type nobody wants to write. It trades
verbosity at the declaration for verbosity at every use.

**What was adopted:** `AnyTensor`, which collapses the parameter list only where
it actually hurts -- generic code -- while leaving construction and inherent
methods alone. A helper writes `fn f<T: AnyTensor>(t: &T)` and reaches any
parameter it genuinely constrains as an associated type. It is one trait with
one method rather than a facade, because mirroring the tensor API here would
double the surface and drift from it immediately.

The general shape of the answer: the cost of a type parameter is not uniform.
It falls on declarations (fixed, one-time, and mostly formatting), on
propagation (which is the feature, not the cost), and on generic callers (which
is where it genuinely hurts and where a collapsing trait pays for itself).
