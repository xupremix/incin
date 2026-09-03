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

**`Dyn` is the default, and it claims nothing.** This is not merely the same
shape of decision as `Dyn` for shapes and `ProofLevel::Dynamic` for proofs --
it is the same marker: the escape
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
Defined only on `Dyn`, since a tensor that already carries a layout got it
from somewhere that knew.

**A gradient's layout is `Dyn`, not its source's.** A gradient is an
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

It was tried first and reverted. Because `Dyn` and `RowMajor` are different
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
shows the full type and will now display `Dyn`, a parameter that by
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
- **Shape-changing** operations state theirs, as `Dyn`, because a layout
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
`Tensor<.., crate::dist::Local, crate::shapes::Dyn>` during conversion, which
is exactly what `Tensor<..>` already means. Removing the explicit form also
cleared a `clippy::type_complexity`.

## Reusing `Dyn` as the layout marker: rejected, then reversed

The layout marker was first a distinct unit struct, `Unknown`. It is now `Dyn`,
the same marker the shape, dtype, device and placement slots use. Both the
original reasoning and what overturned it are recorded here, because the
objections were real and only one of them survived contact.

### Why it was rejected first

`Unknown` and `Dyn` express the same idea -- the compiler settled nothing, read
it at runtime -- so sharing the spelling removes a concept. `impl Layout for
Dyn` compiles with no conflict, so nothing technical stopped it. Three things
argued against:

1. **The diagnostic names the marker twice**, once per slot, so a reader counts
   positions to tell which is which.
2. **The humanizer cannot separate them.** An unproven *layout* carries no
   information and should be elided from a hover; a `Dyn` *shape* carries a
   great deal and must stay. Spelled the same, eliding one elides the other.
   This was called the deciding reason.
3. **`Dyn` is not a pure marker.** It is a `Shape` with `Arg = Vec<usize>` and
   its own `resolve`, because a dynamic shape is *constructed* from runtime
   dimensions, whereas an unproven layout is constructed from nothing.

### Why that was reversed

Objection 2, the deciding one, was **wrong about its own premise**. It assumed
the humanizer must identify the marker by name. It does not have to: `Tensor`
has exactly six parameters and the layout is the sixth, so a trailing `Dyn` is a
layout precisely when the argument list is fully spelled out. Keying on position
is not a workaround -- it is stricter than the name test it replaced, which
fired on any trailing argument called `Unknown` whatever the arity, and which a
downstream type of that name could have tripped. Both `Dyn`s in
`Tensor<Dyn, B, f32, NoGrad, Local, Dyn>` are now handled correctly in one pass:
the sixth is dropped, the first is kept.

Objection 3 is true and costs nothing. A trait impl does not oblige the layout
slot to use `Shape::resolve`, and the layout slot never constructs a value at
all -- it is `PhantomData`. The two roles coexist in one type without either
reaching the other.

Objection 1 is the one that stands, and it is a matter of taste: a fully
written-out tensor does say `Dyn` twice. Set against it is that a reader who has
learned `Dyn` means "decided at runtime" in five slots no longer has to learn a
sixth spelling for the same concept, and a `where` clause wanting "unproven
anything" names one type instead of two.

### The measured cost

Renaming the default changed one recorded rustc rendering.
`named_dim_identity_mismatch.stderr` previously abbreviated its `found` type to
`Tensor<Shape, CpuBackendImpl>`, eliding four arguments that all equalled their
defaults; it now spells out `f32, NoGrad, incin::dist::Local, Dyn`. Verified
that this is caused by the rename and not by pre-existing snapshot drift:
regenerating every snapshot on the unmodified tree produces no diff at all.

**Why rustc's default-elision changed behaviour was not pinned down.** A reduced
standalone case reproducing the parameter structure, the four defaults, the
unit-versus-tuple struct difference, the module layout and glob re-export, and
the marker implementing both `Shape` and `Layout`, still elides correctly. The
in-tree experiment that would have isolated it -- swapping the default for a
third, distinct marker -- does not compile, because `into_row_major` is defined
on the default layout and moving the default moves that impl with it. Recorded
as unexplained rather than guessed at; the practical impact is bounded, since
the humanizer strips the layout argument from the rendering either way.

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

## Widening the density claim past pointwise

Once pointwise operations stated `RowMajor` instead of carrying `L`, the same
question applied to everything else that allocates. Asking it turned up more
than the signatures it was aimed at.

### The bug was the operation that *could* carry a layout

`reduce.rs` already said, in its own module doc, that "the results are freshly
allocated dense buffers". Two signatures contradicted that sentence in opposite
directions:

- the axis reductions returned the default marker -- true but weak, they knew
  something and said nothing;
- `cumsum` returned `Self` -- and `Self` carries the *operand's* layout.

`cumsum` is the interesting one, and it is interesting for a structural reason:
it is the only reduction that preserves the shape. Every other reduction *had*
to state its result's layout, because a layout is congruent with one shape and
the shape changed. `cumsum` kept the shape, so carrying typechecked, so nobody
had to think about it. **The operations at risk of a false layout claim are
exactly the shape-preserving ones**, because those are the only ones where the
wrong answer compiles. That is worth checking at every future conversion.

### Widening one layer surfaces the next

The interesting part was not the reductions themselves but what fixing them
exposed. Once `sum` returned a proof, the *next* call in every chain stopped
compiling -- and each failure was an impl block that had pinned `L` to its
default and so had been silently unreachable from a proven tensor all along:
the six comparison operators, `logical_and`/`or`/`not`, `masked_fill`'s mask
parameter, `where_cond`, and the scalar `Mul`/`Add`/`Sub` operators for all four
scalar types.

None of these were found by looking. They were found because a value with a
proof had to flow through them, and `a_proven_tensor_keeps_the_ordinary_api`
only covers the surface someone remembered to list in it. **Returning a proof is
a better detector of a pinned parameter than a test enumerating methods**, since
it forces the whole reachable graph to typecheck rather than a hand-written
sample of it. Expect the same when the next operation starts returning one.

### The facade had the same gap one level up

`incin::Tensor` is a type alias with five parameters; the core struct has six.
So the alias fixed `L` to the default and a facade user could not name the type
of anything that returned a proof -- the mirror of the prelude gap found
earlier. Adding the parameter had one non-obvious consequence: **a type alias's
parameter defaults do not apply in expression position**, so
`Tensor::scaled_dot_product_attention(..)` stopped inferring, because that
associated function takes no `self` and nothing else constrained `L`. Fixed by
tying its `q` operand to the impl block's own layout parameter, which was the
right signature anyway -- it had been refusing proven operands.

### `matmul` and `Linear` were waiting on evidence, and the evidence was cheap

Both were deliberately left claiming nothing, on the recorded grounds that the
conformance tests backing the pointwise claim "cover the pointwise surface
only". That was the right call at the time and the wrong state to leave things
in, because the missing test was about twenty lines: transpose a `[3, 4]` into a
strided `[4, 3]`, multiply by a dense `[3, 2]`, check the result.

It found the same bug one more time. `addmm` returned `Self` -- shape-preserving
again, because the result takes the *bias* operand's shape -- so it handed the
bias's layout to a buffer a GEMM wrote. Third instance of the pattern after
`cumsum` and the pointwise surface, and the third time it was the only
shape-preserving member of its family.

The test checks the numbers, not only the strides. A GEMM that walked the
strided operand linearly would return a correctly shaped dense buffer of wrong
values, and a strides-only assertion would pass. The CUDA test multiplies the
same matrices and asserts the same product, so the two backends are compared
against one answer rather than each against itself.

One thing fell out for free: `Linear`'s bias-free `Module` impl had a
`forget_layout()` in it, with a comment explaining that the two arms disagreed
-- the bias path allocated through a pointwise add and was `RowMajor`, the
bias-free path handed back `matmul`'s result, which claimed nothing. Once
`matmul` claimed, both arms agreed and the weakening deleted itself. **A
`forget_layout` call is a marker for an operation upstream that has not made its
claim yet**, which makes the remaining ones worth reading as a to-do list rather
than as settled design. The one in `rnn.rs` is not: a loop variable can only
hold what every assignment satisfies, and the seed is the caller's.

### The one operation for which carrying *is* right

`Dropout` and `BatchNorm2d` were the last two shape-preserving `nn` layers
returning the operand's layout, and they turned out to be opposite cases.

`BatchNorm2d` has no identity path. Every call dispatches and writes a fresh
buffer, so carrying was the same mistake as everywhere else and `Dense` is the
answer.

`Dropout` is different, and it is the shape of exception worth naming. In eval
mode, or at `p == 0`, it returns the very tensor it was handed -- same buffer,
same strides. For that branch the operand's layout is not merely compatible with
the result, it *is* the result's. So carrying is right, and forcing `Dense` here
would be a lie in the other direction.

What makes the carry sound is the branch that *does* allocate. It writes a dense
buffer, so the layout carried across both branches has to be one a fresh dense
allocation also satisfies. That is exactly [`FreshDense<S>`], the sealed bound
the constructors already use, so the signature says it rather than relying on an
accident:

```rust,ignore
impl<.., L: FreshDense<S>> Module<Tensor<S, B, K, G, Local, L>> for Dropout {
    type Output = Tensor<S, B, K, G, Local, L>;
```

Bounding on `Layout` compiles today and would keep compiling for a long time,
because `Dyn` and `RowMajor` are the only layouts and a dense buffer satisfies
both. It starts lying the day a `ChannelsLast` exists, and nothing would ask.
**The general rule: an operation may carry its operand's layout exactly when
every branch either returns the operand itself or writes a buffer the carried
layout describes** -- and when the second kind of branch exists, the bound has to
say so.

### What CUDA contributes, and what it does not

The reduction claim rests on different evidence from the pointwise one, because
the CUDA capability table answers the strided question differently for the two:
`elementwise_layouts` advertises `Strided`, `reduction_layouts` does not. So a
strided CUDA reduction is refused before any kernel runs, and a `RowMajor`
result cannot be wrong for an operand the backend will not accept.

That is a real argument, but it is contingent on a table row, so the hardware
test asserts *both* halves -- dense results on a dense operand, and refusal on a
strided one. If the row is ever widened, the test fails and whoever widens it
has to show the strided reduction kernel writes a dense result before the claim
stands again. A claim that rests on a refusal has to pin the refusal.

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
  CUDA. So shape-changing operations keep `Dyn` outputs until the contract
  is settled, and `into_row_major` remains the way to recover a proof.
- The earlier finding that "the strided path is unreachable" was correctly
  scoped to CUDA in the note, but it was not a framework-wide property, and it
  has since stopped being true on CUDA either. It was never unreachable
  *because CUDA copies* -- it was unreachable because `CUDA_CAPABILITIES`
  declared `elementwise_layouts = CONTIGUOUS`, so the dispatcher refused a
  strided operand before any kernel was consulted. That row was true by vacuity
  when written (nothing could yet produce a strided CUDA tensor) and was never
  revisited once `transpose_view` landed. Widened in `76875c79`; the
  extent-folding work is reachable on both backends now. See issue #113's
  comment thread for the corrected premises.
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
| 4,096 | 31.0 | 27.4 | 0.88 |
| 65,536 | 30.9 | 27.2 | 0.88 |
| 1,048,576 | 227.6 | 126.8 | **0.56** |
| 4,194,304 | 921.0 | 527.8 | **0.57** |

And with the result consumed `k` times, at 1M elements:

| k | materialise | strided view | view/mat |
|---:|---:|---:|---:|
| 1 | 233.4 | 128.6 | **0.55** |
| 2 | 313.4 | 252.0 | 0.80 |
| 4 | 480.1 | 503.6 | 1.05 |
| 8 | 827.9 | 1016.8 | **1.23** |

The copy's read-plus-write loses badly to the strided read for a single
consumer at scale -- the view is about 45% faster -- and wins by about 23% by
eight consumers. The crossover is at roughly four reads.

So the two differ substantially in *opposite directions* depending on how often
the result is read, which is a fact about the consumer that the producing
operation cannot know. Unifying on either behaviour makes the framework
reliably wrong for half its callers.

The resolution is to stop treating it as one operation: `transpose`
materialises and returns `RowMajor<S>`, `transpose_view` does not copy and
returns a permuted layout. The caller picks and the type records which they
got, so a downstream `L: Contiguous` bound is satisfiable in one case and
correctly refused in the other.

This is the strongest available argument for the layout parameter, and it is a
measurement rather than an aesthetic.

**These numbers replace an earlier set that were wrong.** The first harness
forced completion by copying the whole output back to the host every iteration.
That is a valid barrier, but at a million floats it is four megabytes of
transfer added to both arms -- roughly 80% of the measured time -- which
compressed every ratio toward one. It reported the single-consumer view
advantage as 11% rather than 45%, and put the crossover between two and four
reads rather than at four. The qualitative conclusion survived; the magnitudes
did not. Synchronising the stream instead measures the work and nothing else.

The lesson is worth more than the numbers: a barrier that costs more than the
thing being measured does not merely add noise, it systematically biases every
ratio toward unity, which reads as "no difference" and is the easiest possible
result to believe.

Caveats that remain: one GPU, one dtype, square shapes, a pointwise consumer. A
consumer that cannot take arbitrary strides -- matmul, generally -- will move
the crossover. The shape of the conclusion should survive; the exact `k` should
not be treated as a constant.

### Does folding proven extents into the strided walk pay?

Yes, and it could not be measured until `transpose_view` existed, because
nothing could reach the strided path on CUDA before it. Same kernel body, same
strided view, differing only in whether the extents were proven:

| elements | loaded divisors | folded | folded/loaded |
|---:|---:|---:|---:|
| 256 | 26.6 | 23.3 | 0.88 |
| 4,096 | 26.9 | 25.0 | 0.90 |
| 65,536 | 26.7 | 24.4 | 0.90 |
| 1,048,576 | 124.3 | 120.9 | 0.94 |

A consistent 6-12%, largest at small sizes where the eliminated `shape` upload
is the bigger share of the launch. Under the original readback-bound harness
this measured as 0.97-1.03 -- indistinguishable from nothing -- which would have
been read as evidence the optimisation was pointless.

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

**Typing.** `transpose_view` returns `Dyn`, not a new `Strided` marker. A
marker meaning "known not contiguous" would behave identically to `Dyn`
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
