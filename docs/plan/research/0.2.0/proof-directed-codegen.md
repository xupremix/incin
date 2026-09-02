proof-directed codegen: the evidence is delivered and nobody opens it

Finding: `ShapeEvidence` crosses the backend boundary on every validated
descriptor, carries exactly the facts a kernel specializer wants, and has zero
readers in `incin-backends`.

The chain is complete and already built. `Shape::PROOF` derives a `ProofLevel`
from the shape *type*. `exec::catalog::validated` calls `ShapeEvidence::of::<S>()`
where the typed frontend still has `S` in hand, capturing `proof`, `static_rank`
and `static_numel`. `Validated::shape_evidence()` is a `pub const fn`. Every
backend `Execute<O>` receives `&Validated<O>` inside its `ExecutionRequest`. So a
backend can ask what the frontend proved, at no runtime cost, today.

`grep -rn ShapeEvidence crates/` returns three hits outside its defining file:
one re-export and two construction sites. `grep -rn ProofLevel
crates/incin-backends/` returns nothing at all.

This is not an oversight in the design, which states the intent plainly in
`exec/proof.rs`: "A shape proved entirely at compile time and one checked at
runtime a microsecond ago are both valid, but they justify different amounts of
work: the first can specialize a kernel on constants, the second cannot.
`ProofLevel` records which happened, and travels with the descriptor so a backend
can act on it." No backend acts on it. The road is built and nothing drives on
it.

Why this is the interesting direction rather than importing more compiler
technique. The `codegen` adoption that just landed applied one textbook idea --
SSA linearization with CSE by structural memoisation -- and that was the right
first move, but it is what every compiler does. The lowering literature's hard
problem is *recovering* static structure: TVM, Triton and XLA all specialise on
runtime shapes and recompile per shape bucket, because by the time they see a
tensor its geometry is a runtime fact. Incin's frontend knows the answer from the
type and is already shipping it across the boundary. That asymmetry is the thing
worth exploiting, and it is unavailable to a Python-fronted framework by
construction.

What the current fields buy, concretely, in the pointwise path.

`static_numel` makes the element count a compile-time constant. Two consequences.
The packed templates in `kernel::packed` emit a ragged-tail `else` branch for
`numel % width != 0`; when the numel is a static multiple of the vector width
that branch is provably dead and can be omitted, which removes a divergent branch
and shrinks the kernel. And `numel` stops being a kernel parameter and becomes a
literal, which lets the bounds check fold away entirely in the dense case.

`static_rank` makes the strided template's trip count known. That template
currently runs `for (int i = ndim - 1; i >= 0; i--)` with a runtime `%` and `/`
per axis per element -- integer division is among the most expensive operations
on the device -- and `launch_unary_body` uploads two `int` buffers per launch to
feed it. A known rank fully unrolls the loop.

What the current fields do *not* buy, and the natural extension. `ShapeEvidence`
carries rank and numel but not per-axis extents, so the strides themselves cannot
be folded to literals yet. That is the larger win: with literal extents nvcc
turns each `%` and `/` by a constant into a multiply-shift, and both `clone_htod`
uploads and both pointer parameters disappear. Adding static extents to
`ShapeEvidence` is the obvious follow-up, and it should be weighed against kernel
cache pressure, since specialising on exact extents rather than on
`ShapeBucket` multiplies the number of distinct compiled kernels.

Note that `KernelKey` already has `rank_class`, `shape_bucket` and `alignment`
fields, so the cache is prepared to distinguish specialisations. They are
currently populated from runtime metadata; a proof-directed path would populate
them from evidence and record which, so a statically specialised kernel cannot be
served to a dynamically shaped call.

The plumbing gap. `Execute<O>` has the `Validated`, but the pointwise launchers
in `cuda::ops::elementwise` receive only `CudaStorage`, whose `TensorMeta` is
entirely runtime. So evidence has to be threaded from the executor down to the
launcher and into `KernelKey`. That is the actual work, and it is the same shape
as the `ScalarFragment` change that just landed: add a parameter carrying the
richer fact, keep the existing entry points as wrappers that pass the "nothing
proven" value, and specialise only where the proof is present.

One caveat worth stating before anyone builds on this. `exec/proof.rs` notes that
`dispatch::execute` "is generic over the operation and the backend but not over
the operand shapes, so it has no `S` to read `Shape::PROOF` from and passes
`Dynamic` for everything". So the dynamic dispatch path will never carry a useful
proof, and any specialisation must be a pure optimisation over a correct dynamic
fallback rather than a path anything depends on. That is the right shape for it
anyway, but it means the win is only available to callers who came through the
typed frontend -- which should be measured before the extension is scoped, since
it bounds how much of a real workload can benefit.

Recommendation: adopt this the way codegen was adopted -- one operation family,
end to end, on a machine that can run it. CUDA pointwise again, using
`static_numel` to elide the packed tail branch, because that is the smallest
change with a result you can see in the emitted source and measure on the device.
If it lands, extending `ShapeEvidence` with static extents becomes the argument
for the rest.

**Update: that slice has landed.** `KernelSpecialization` is the backend-side
projection of the evidence; `Execute<op::*>` for the canonical unary family
builds one from `request.operation.shape_evidence()` and threads it to the
renderers, which drop the packed ragged tail when the count is a proven multiple
of the vector width. Absent or non-`Static` evidence specializes nothing.

Two things the implementation clarified that this note had guessed at.

The provenance seal is tighter than expected, and usefully so. `ShapeEvidence::of`
is `pub(crate)` to `incin-core`, so a backend cannot construct evidence at all --
it can only receive what a `Validated` carries. That means the test that a real
static shape type yields a usable count has to live in `incin-core`, and the
backend can only test the absent case. That is the seal working: there is no way
to write a backend test that fakes a proof, which is exactly the property
`exec::proof` was built for.

And the grid-overhang guard is not eliminable. `if (base >= numel) return;` looks
like it should fall to the same proof, but the launch rounds up to whole blocks
regardless of how well-known the element count is, so the overhang is a property
of the launch geometry rather than of the shape. Only the *ragged tail within a
packet* is what the divisibility proof settles. Worth stating because the two
look alike in the source.

**The reachability question is settled, in favour of extending this.** The
concern was that `dispatch::execute` reports `Dynamic`, so the win might only
reach a sliver of real calls. It does not: the typed tensor surface does not use
that entry point. `Tensor::relu` and its siblings call
`execute_unary_descriptor`, which calls `dispatch::execute_shaped::<O, B, S>`,
which calls `O::infer_invocation_typed` and builds the evidence from `S`. So
every operation reached through the typed frontend -- the primary API -- carries
its proof, and `dispatch::execute` is the escape hatch for callers who genuinely
have no shape type. `a_typed_invocation_carries_a_static_element_count_to_the_backend`
pins the end of that chain, because a break anywhere in it would silently drop
every kernel onto the general path with nothing failing.

Next along this line, in order of value:

1. Extend `ShapeEvidence` with static extents, so strides fold to literals, the
   `%` and `/` per axis per element become multiply-shifts, and both
   `clone_htod` uploads and both pointer parameters disappear.

   An earlier draft of this note claimed `&'static [usize]` "cannot be built
   from a `DimCons` chain in a const context" and offered a fixed
   `[Option<usize>; MAX_RANK]` field as the pragmatic option. That was wrong,
   and the mistake was conflating the construction vehicle with the payload.

   The working shape, verified on the pinned 1.97.1 toolchain with no unstable
   features:

   ```rust
   const BUF: [usize; MAX];                  // construction only
   const EXTENTS: &'static [usize] = Self::BUF.split_at(Self::RANK).0;
   ```

   A fixed-size array is how the recursive `prepend` is written, because
   prepending to a chain needs a length that does not depend on `Self::RANK`
   (that would want `generic_const_exprs`). But the array stays internal. What
   `ShapeEvidence` carries is the *slice*, promoted into `.rodata`: exact
   length, and 16 bytes rather than `MAX_RANK` elements' worth. That matters
   because `ShapeEvidence` is `Copy` and rides along on every dispatch.

   Use `Option<usize>` per axis over `usize`, so a `Mixed` shape can still
   contribute the axes it does know: a static inner dimension is enough to fold
   that axis's stride even when the batch axis is dynamic.

   **Correction, from implementing it.** An earlier revision of this note said
   exceeding the buffer should be a compile-time panic, "the correct failure
   mode for a proof, which must never quietly report less than it knows". That
   was wrong twice over. It conflated reporting a *truncated* geometry, which
   would be a miscompile, with reporting *no* geometry, which is only a missed
   optimisation — and the trait's whole convention is already that an
   unprovable fact is reported as absent. It was also immediately falsified:
   `tensor_ops` constructs a rank-18 shape, and the panic broke its
   compilation. A shape deeper than the buffer must keep working and simply
   forgo the specialisation. Landed that way, with the rank-18 case as a
   regression test.

   There is no const heap allocation to reach for here, and none is wanted.
   `Vec::push` is not const-stable on 1.97.1 (rust-lang/rust#143874). `'static`
   data is statically allocated, not heap allocated, and const promotion already
   provides it.
2. Apply the same projection to the binary family and to reductions. Mechanical
   once (1) settles the representation, since binary needs two operands' extents.
3. Weigh cache pressure once extents are in. Specialising on exact extents rather
   than on `ShapeBucket` multiplies distinct compiled kernels by the number of
   static shapes a program instantiates. That is bounded, but it is not small for
   a model with many differently-shaped layers, and the source-scoped cache key
   means each variant is a separate NVRTC compile.

Depends on: the `ScalarFragment` seam in `codegen-adoption-landed.md`, which is
where a specialised body would be emitted.

---

## Measurement: the strided path is unreachable, and that reframes typed strides

The question was what fraction of real work takes the strided pointwise path
versus the contiguous one, to decide whether folding its per-axis divisors was
worth extending. The answer is that on CUDA the fraction is zero, and not for a
subtle reason.

Every CUDA operation that could produce a non-contiguous result materialises
instead. `launch_transpose` runs a permutation kernel into a fresh buffer;
`launch_broadcast` and `launch_narrow` say so in their own doc comments. All
nineteen `CudaStorage::try_from_parts` call sites pass
`crate::layout::contiguous_strides(&out_shape)`. Probed directly: a 3x4 tensor
transposed to 4x3 comes back as `strides=[3, 1]`, `offset=0`,
`LayoutClass::Contiguous`.

So no public operation reaches the strided kernel. The extent folding and the
retired shape upload are correct and tested, but they are currently latent: they
optimise a path nothing takes.

This also caught a false claim. The test added alongside that work called
`transpose` and asserted it was exercising the strided path. It was not -- it ran
the contiguous kernel and passed for the wrong reason, and the commit message
said "verified against a genuinely non-contiguous tensor", which was untrue. The
test now builds the view from parts and asserts `LayoutClass::Strided` before
relying on anything, so it fails if the path stops being what it claims.

The useful part is what this says about putting strides in the type. The case
for it was going to be "fold the stride literals too, and the whole index walk
becomes constants". That case is weak: the walk is already unreachable, so
making it faster buys nothing.

The real cost is one layer up. Because views materialise, every `transpose`,
`narrow` and `broadcast` pays a full copy of the tensor -- an allocation, a
kernel launch and `numel` elements of bandwidth -- to produce a result that is
mathematically a relabelling of the same memory. An attention block transposes
constantly. That copy, not the divisions inside a kernel that never runs, is what
typed strides would remove: a layout in the type is what lets a view *be* a view,
because the consumer can then be compiled against the stride pattern instead of
discovering it at runtime.

So the ordering inverts. Typed strides are not an optimisation on top of the
strided kernel; the strided kernel is the prerequisite that makes non-copying
views possible, and it now exists and is verified. Whether to take the next step
is a question about the frontend's type surface, not about codegen.

Two caveats before anyone scopes that. The materialising design is not obviously
wrong -- a contiguous copy can beat a strided read for a consumer that touches
the data repeatedly, and it keeps every downstream kernel on the fast path -- so
the comparison needs measuring, not assuming. And the copy is only avoidable
where the consumer can handle arbitrary strides, which pointwise can and matmul
generally cannot.

## How CUTLASS does it, and how that differs from here

CuTe (CUTLASS 3.x) makes `Layout = (Shape, Stride)` a first-class type: a tuple
pair that maps a coordinate within `Shape` to an index via `Stride`. Both halves
are congruent -- same tuple profile, one stride integer per shape integer -- and
each integer independently may be a *static* integer (`Int<N>`, spelled `_N`,
carrying its value as a `constexpr` member) or an ordinary dynamic `int`. On top
of that sits a layout algebra -- composition, complement, division, product --
and the algebra is written to preserve staticness through transformations where
it can, so a statically known divisibility survives into the result rather than
being erased.

Against that, incin has half the structure. Shape is typed, and typed through
operations: `transpose_structural::<L, R>` returns
`Tensor<<S as SwapAxes<L, R>>::Output, ..>`, so the type system already knows
which axes swapped. Stride is absent from the type entirely, and `LayoutClass` is
a four-valued runtime classification computed by scanning
(`is_contiguous(&operand, dims)`), not a type.

The gap is therefore narrower than "adopt CuTe" and more specific: incin already
computes, at the type level, the fact from which the new strides follow. It
discards it. CuTe's contribution is not that strides can be static -- it is the
congruence requirement and the algebra that keeps shape and stride in step
through every transformation, which is exactly the part that makes a
`Tensor<S, L, ..>` tractable rather than a second set of parameters to keep
manually consistent.

The mixed static/dynamic-per-mode design is also directly applicable and matches
the choice already made here: `Shape::STATIC_EXTENTS` is `Option<usize>` per
axis for the same reason CuTe allows a dynamic integer in one mode of an
otherwise static shape.
